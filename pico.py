import logging
from typing import Optional

import einops
import lightning as L
import rich
import rich.box
import torch
import torch.nn.functional as F
import typer
from deepspeed.ops.lamb import FusedLamb
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from rich.logging import RichHandler
from rich.table import Table
from torch import nn
from torch.utils.data import DataLoader, Dataset

app = typer.Typer(pretty_exceptions_enable=False)
logger = logging.getLogger(__name__)

#################################
#            Config             #
#################################

config = {
    "context_len": 256,
    "dim": 128,
    "num_blocks": 8,
    "num_heads": 4,
    "query_capacity": 64,
    "batch_size": 128,
    "noise_levels": 16,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "max_epochs": 4,
}

#################################
#            Dataset            #
#################################


class DenoisingDataset(Dataset):
    def __init__(self, context_len: int, noise_levels: int):
        super().__init__()

        self.context_len = context_len
        self.noise_levels = noise_levels

        with open("shakespeare.txt", "rb") as file:
            self.data = torch.tensor(list(file.read()), dtype=torch.uint8).unfold(
                0, self.context_len, 1
            )

        self.bytes_freq = (
            torch.bincount(self.data.flatten(), minlength=256).float() + 1
        ) / self.data.numel()

    def __len__(self):
        return self.noise_levels * self.data.shape[0]

    def __getitem__(self, idx: int):
        sample_idx = idx // self.noise_levels

        sample = self.data[sample_idx]
        noisy_sample = sample.clone()

        noise_level = (idx % (self.noise_levels - 1)) + 1
        noise_rate = noise_level / self.noise_levels
        noise_rate = (1 - noise_rate) ** 0.5  # Sqrt schedule

        n_noisy_bytes = int(noise_rate * self.context_len)

        if n_noisy_bytes == 0:
            return sample, noisy_sample, noise_rate

        noise_indices = torch.randint(self.context_len, (n_noisy_bytes,))

        # Absolute noise
        # noise_values = torch.randint(256, (n_noisy_bytes,), dtype=torch.uint8)

        # Noise based on byte frequency distribution
        noise_values = torch.multinomial(
            self.bytes_freq, n_noisy_bytes, replacement=True
        ).to(torch.uint8)

        noisy_sample[noise_indices] = noise_values

        return sample, noisy_sample, noise_rate


#################################
#             Model             #
#################################


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base=10_000):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even."

        # (1, dim // 2)
        self.freqs = (
            torch.exp(torch.linspace(0, -1, dim // 2) * torch.log(torch.tensor(base)))
            .unsqueeze(0)
            .cuda()
        )

    def forward(
        self, x: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: A Tensor of shape (..., seq_len, dim)
            positions: A Tensor of shape (..., seq_len) containing the positions of the elements in the sequence.
        """

        if positions is None:
            # If no positions are provided, generate them
            # (1, seq_len)
            positions = torch.arange(x.size(-2)).unsqueeze(0).cuda()

        # (..., seq_len) -> (..., seq_len, 1)
        positions = positions.to(self.freqs.dtype).unsqueeze(-1)

        # (..., seq_len, dim // 2)
        angles = positions @ self.freqs
        # (..., seq_len, dim)
        rotary_matrix = torch.cat([angles.sin(), angles.cos()], -1)

        return x * rotary_matrix


class CapacitiveMHA(nn.Module):
    def __init__(
        self,
        query_seq_dim: int,
        value_seq_dim: int,
        num_heads: int,
        query_capacity: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = query_seq_dim // num_heads
        self.query_capacity = query_capacity

        self.router = nn.Linear(query_seq_dim, 1, bias=False)

        self.q_rope = RotaryEmbedding(self.emb_dim)
        self.k_rope = RotaryEmbedding(self.emb_dim)

        self.q_proj = nn.Linear(query_seq_dim, self.emb_dim * num_heads, bias=False)
        self.kv_proj = nn.Linear(
            value_seq_dim, self.emb_dim * num_heads * 2, bias=False
        )

        self.out_proj = nn.Linear(
            self.emb_dim * self.num_heads, query_seq_dim, bias=False
        )

    def forward(self, query_seq: torch.Tensor, value_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_seq: A Tensor of shape (batch_size, query_seq_len, query_seq_dim)
            value_seq: A Tensor of shape (batch_size, value_seq_len, value_seq_dim)
        """
        # First, we limit the number of tokens to consider in the query sequence
        # (capacity resampling) using a mixture-of-depths like mechanism

        # Weight most relevant tokens from query_seq
        # (batch_size, query_seq_len, 1)
        router_weights = self.router(query_seq).to(torch.float32)

        # Select top-k tokens
        # (batch_size, query_capacity, 1)
        top_router_weights, top_router_indices = torch.topk(
            router_weights, k=self.query_capacity, dim=1, sorted=False
        )

        # Reorder top_router_indices and top_router_weights in the order of original query_seq
        # NOTE: This is not necessary here as we do non-causal attention but I keep it for reference
        # top_router_indices, top_router_order = torch.sort(top_router_indices, dim=1)
        # top_router_weights = torch.gather(
        #     top_router_weights, dim=1, index=top_router_order
        # )

        # Duplicate top_router_indices over each emb_dim for gather/scatter operations
        top_router_indices_exp = einops.repeat(
            top_router_indices,
            "batch capacity 1 -> batch capacity emb_dim",
            emb_dim=query_seq.size(-1),
        )

        resampled_query_seq = torch.gather(
            query_seq, dim=1, index=top_router_indices_exp
        )

        # Then, we perform "standard" MHA

        # q, k and v are of shape (batch_size, seq_len, emb_dim * num_heads)
        # where seq_len is query_capacity for q, and value_seq_len for k and v
        q: torch.Tensor = self.q_proj(resampled_query_seq)
        kv: torch.Tensor = self.kv_proj(value_seq)
        k, v = kv.chunk(2, dim=-1)

        # Split heads so F.scaled_dot_product_attention can be applied separately to each head
        split_heads = (
            "batch seq_len (num_heads emb_dim) -> batch num_heads seq_len emb_dim"
        )
        q = einops.rearrange(q, split_heads, num_heads=self.num_heads)
        k = einops.rearrange(k, split_heads, num_heads=self.num_heads)
        v = einops.rearrange(v, split_heads, num_heads=self.num_heads)

        # Apply rotary embeddings
        q_pos = einops.rearrange(
            top_router_indices, "batch capacity 1 -> batch 1 capacity"
        )

        q = self.q_rope(q, q_pos)
        k = self.k_rope(k)

        # Efficiently compute attention
        att = F.scaled_dot_product_attention(q, k, v)

        # Concatenate heads output and project back
        att = einops.rearrange(
            att, "batch num_heads seq_len emb_dim -> batch seq_len (num_heads emb_dim)"
        )
        att = self.out_proj(att)

        # Finally, we create a sparse tensor with the resampled query tokens
        # at their original positions in the query sequence
        output = torch.zeros_like(query_seq)
        output = torch.scatter(
            output,
            dim=1,
            index=top_router_indices_exp,
            src=att * top_router_weights,
        )

        return output


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class Block(nn.Module):
    def __init__(
        self,
        query_seq_dim: int,
        value_seq_dim: int,
        num_heads: int,
        query_capacity: int,
    ):
        super().__init__()

        self.query_seq_dim = query_seq_dim
        self.value_seq_dim = value_seq_dim

        self.query_ln = nn.LayerNorm(query_seq_dim)
        self.value_ln = nn.LayerNorm(value_seq_dim)

        self.mha = CapacitiveMHA(
            query_seq_dim, value_seq_dim, num_heads, query_capacity
        )

        self.ffn = nn.Sequential(
            nn.Linear(query_seq_dim, query_seq_dim * 2, bias=False),
            SwiGLU(),  # SwiGLU divides dim by 2
            nn.Linear(query_seq_dim, query_seq_dim, bias=False),
        )

    def forward(
        self,
        query_seq: torch.Tensor,
        value_seq: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        query_seq_norm = self.query_ln(query_seq)
        value_seq_norm = self.value_ln(value_seq)

        att = self.mha(query_seq_norm, value_seq_norm)
        ffn = self.ffn(query_seq_norm)

        out = query_seq + att + ffn

        if return_attention:
            # Return attention output for visualization
            return out, att

        return out


class DenoisingModel(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)

        self.embedding = nn.Embedding(256, config["dim"])
        self.noise_rate_embedding = nn.Linear(1, config["dim"])

        self.blocks = nn.ModuleList(
            [
                Block(
                    query_seq_dim=config["dim"],
                    value_seq_dim=config["dim"],
                    num_heads=config["num_heads"],
                    query_capacity=config["query_capacity"],
                )
                for _ in range(config["num_blocks"])
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        noise_rate: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        x = self.embedding(x.long())

        noise_rate = einops.rearrange(noise_rate, "batch -> batch 1 1").float()
        x += self.noise_rate_embedding(noise_rate)

        block_attentions = []

        for block in self.blocks:
            if return_attention:
                x, att = block(x, x, return_attention=True)
                block_attentions.append(att)
                continue
            
            x = block(x, x)

        logits = x @ self.embedding.weight.T

        if return_attention:
            return logits, torch.stack(block_attentions, dim=1)

        return logits

    def compute_loss(
        self, sample: torch.Tensor, noisy_sample: torch.Tensor, noise_rate: torch.Tensor
    ) -> torch.Tensor:
        logits = self(noisy_sample, noise_rate.to(torch.float32))

        # F.cross_entropy expects (batch, num_classes, ...) and (batch, ...) as input
        logits = einops.rearrange(
            logits,
            "batch seq_len num_classes -> batch num_classes seq_len",
            num_classes=256,
        )

        return torch.mean((1 - noise_rate) * F.cross_entropy(logits, sample.long()))

    def training_step(self, batch, batch_idx):
        sample, noisy_sample, noise_rate = batch
        loss = self.compute_loss(sample, noisy_sample, noise_rate)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return FusedLamb(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )

    def on_before_optimizer_step(self, _):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)


#################################
#              CLI              #
#################################


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode."),
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    if verbose:
        logger.debug("Verbose mode enabled.")


@app.command()
def train(
    context_len: Optional[int] = None,
    dim: Optional[int] = None,
    num_blocks: Optional[int] = None,
    num_heads: Optional[int] = None,
    query_capacity: Optional[int] = None,
    batch_size: Optional[int] = None,
    noise_levels: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    max_epochs: Optional[int] = None,
    disable_wandb: bool = typer.Option(
        False, "--disable-wandb", help="Disable Weights & Biases logging."
    ),
):
    train_config = {
        key: value if value is not None else config[key]
        for key, value in locals().items()
        if key in config
    }

    # Print config as a table
    table = Table(title="Training Configuration")
    table.add_column("Parameter")
    table.add_column("Value")
    for key, value in train_config.items():
        table.add_row(key, str(value))

    rich.print(table)

    torch.set_float32_matmul_precision("medium")  # Enable Tensor Cores

    dataset = DenoisingDataset(
        train_config["context_len"], train_config["noise_levels"]
    )
    train_loader = DataLoader(
        dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=19
    )

    model = DenoisingModel(train_config)

    trainer = L.Trainer(
        max_epochs=train_config["max_epochs"],
        accelerator="gpu",
        precision="16-mixed",
        callbacks=[RichProgressBar()],
        logger=None if disable_wandb else WandbLogger(project="pico"),
        limit_train_batches=0.125,
    )

    trainer.fit(model, train_loader)


@app.command()
def test(version: str, epoch: int, step: int):
    model = DenoisingModel.load_from_checkpoint(
        f"pico/{version}/checkpoints/epoch={epoch}-step={step}.ckpt",
        # query_capacity=128
    )
    model.eval()

    with torch.no_grad():
        dataset = DenoisingDataset(1024, 30)
        x = (
            torch.multinomial(
                dataset.bytes_freq, config["context_len"], replacement=True
            )
            .unsqueeze(0)
            .cuda()
        )

        noise_rate = torch.tensor([1], device="cuda")

        logger.info(bytes(x.tolist()[0]))

        for level in range(config["noise_levels"] * 10):
            noise_rate[0] = (config["noise_levels"] - level) / config["noise_levels"]
            noise_rate = (1 - noise_rate) ** 0.5  # Sqrt schedule
            logits = F.softmax(model(x, noise_rate / 10), dim=-1)

            # Absolute sampling:
            # logits = logits.argmax(-1)

            # Relative sampling:
            logits = torch.tensor(
                [torch.multinomial(p, 1) for p in logits.squeeze()], device="cuda"
            ).unsqueeze(0)

            logger.info(bytes(logits.tolist()[0]))

            x = logits

        print(bytes(logits.tolist()[0]).decode("utf-8"))


if __name__ == "__main__":
    app()
