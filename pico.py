import logging
from collections import deque
from typing import Optional

import einops
import lightning as L
import rich
import torch
import torch.nn.functional as F
import typer
from deepspeed.ops.lamb import FusedLamb
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities import grad_norm
from rich.live import Live
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text
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
    "query_capacity": [1, 0.25, 0.25, 0.25],
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


# TODO: investigate if this implementation respects the original paper
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


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult=2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mult)
        self.fc2 = nn.Linear(dim * mult // 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        x = self.fc2(x)

        return x


class MHA(nn.Module):
    """
    Non-causal multi-head attention module with rotary embeddings.
    """

    def __init__(
        self,
        query_seq_dim: int,
        value_seq_dim: int,
        num_heads: int,
        query_capacity: float,
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

    def forward(
        self,
        query_seq: torch.Tensor,
        value_seq: torch.Tensor,
        query_seq_pos: Optional[torch.Tensor] = None,
        value_seq_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query_seq: A Tensor of shape (batch_size, query_seq_len, query_seq_dim)
            value_seq: A Tensor of shape (batch_size, value_seq_len, value_seq_dim)
            query_seq_pos: A Tensor of shape (batch_size, query_seq_len) containing the positions of the elements in the query sequence.
            value_seq_pos: A Tensor of shape (batch_size, value_seq_len) containing the positions of the elements in the value sequence.
        """
        # q, k and v are of shape (batch_size, seq_len, emb_dim * num_heads)
        # where seq_len is query_seq_len for q, and value_seq_len for k and v
        q: torch.Tensor = self.q_proj(query_seq)
        kv: torch.Tensor = self.kv_proj(value_seq)
        k, v = kv.chunk(2, dim=-1)

        # Split heads so F.scaled_dot_product_attention can be applied separately to each head
        split_heads = (
            "batch seq_len (num_heads emb_dim) -> batch num_heads seq_len emb_dim"
        )
        q = einops.rearrange(q, split_heads, num_heads=self.num_heads)
        k = einops.rearrange(k, split_heads, num_heads=self.num_heads)
        v = einops.rearrange(v, split_heads, num_heads=self.num_heads)

        # Add a dim for the heads to the positional encodings
        if query_seq_pos is not None:
            query_seq_pos = einops.rearrange(
                query_seq_pos, "batch seq_len -> batch 1 seq_len"
            )
        if value_seq_pos is not None:
            value_seq_pos = einops.rearrange(
                value_seq_pos, "batch seq_len -> batch 1 seq_len"
            )

        # Apply rotary embeddings
        q = self.q_rope(q, query_seq_pos)
        k = self.k_rope(k, value_seq_pos)

        # Efficiently compute attention
        att = F.scaled_dot_product_attention(q, k, v)

        # Concatenate heads output and project back
        att = einops.rearrange(
            att, "batch num_heads seq_len emb_dim -> batch seq_len (num_heads emb_dim)"
        )
        att = self.out_proj(att)

        return att


class Block(nn.Module):
    """
    A parallel-style transformer block with a mixture-of-depths like mechanism to limit the number of tokens
    to consider in the attention query sequence.
    """

    def __init__(
        self,
        seq_dim: int,
        num_heads: int,
        query_capacity: float,
    ):
        super().__init__()

        self.seq_dim = seq_dim
        self.query_capacity = query_capacity

        self.router = nn.Linear(seq_dim, 1, bias=False)

        self.query_ln = nn.LayerNorm(seq_dim)
        self.value_ln = nn.LayerNorm(seq_dim)

        self.mha = MHA(seq_dim, seq_dim, num_heads, query_capacity)
        self.ffn = SwiGLU(seq_dim)

    def forward(
        self,
        seq: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        #################################
        # MoD-like resampling mechanism #
        #################################
        # First, we limit the number of tokens to consider in the query sequence
        # (capacity resampling) using a mixture-of-depths like mechanism
        query_capacity_abs = int(self.query_capacity * seq.size(1))

        # Weight most relevant tokens from sequence
        # (batch_size, seq_len, 1)
        router_weights = self.router(seq)
        router_weights = F.softmax(
            router_weights, dim=1
        )  # This is not in the MoD paper, but I find it useful

        # Select top-k tokens
        # (batch_size, query_capacity_abs, 1)
        top_router_weights, top_router_indices = torch.topk(
            router_weights, k=query_capacity_abs, dim=1, sorted=False
        )

        # Reorder top_router_indices and top_router_weights in the order of original sequence
        # NOTE: This is not necessary here as we do non-causal attention but I keep it for reference
        # top_router_indices, top_router_order = torch.sort(top_router_indices, dim=1)
        # top_router_weights = torch.gather(
        #     top_router_weights, dim=1, index=top_router_order
        # )

        # Duplicate top_router_indices over each emb_dim for gather/scatter operations
        top_router_indices_exp = einops.repeat(
            top_router_indices,
            "batch capacity 1 -> batch capacity emb_dim",
            emb_dim=seq.size(-1),
        )

        # Create query sequence by resampling the original sequence
        query_seq = torch.gather(seq, dim=1, index=top_router_indices_exp)

        query_seq = self.query_ln(query_seq)
        value_seq = self.value_ln(seq)

        ##############################
        # Parallel-style transformer #
        ##############################

        # Multi-head attention
        # query sequence is resampled, value sequence is the full sequence
        att = self.mha(
            query_seq=query_seq,
            value_seq=value_seq,
            # query_seq_pos: (batch_size, query_capacity_abs)
            query_seq_pos=top_router_indices.squeeze(-1),
        )

        # In parallel, apply a feed-forward network
        ffn = self.ffn(query_seq)

        #############################
        # Update the input sequence #
        #############################

        # Update the original sequence with the attention and the feed-forward network    
        out = torch.scatter_add(
            seq,
            dim=1,
            index=top_router_indices_exp,
            src=(att + ffn) * top_router_weights,
        )

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

        # TODO: remove this, this is a temporary retrocompatibility fix
        # for previous query_capacity config
        if isinstance(config["query_capacity"], int):
            config["query_capacity"] = [1, 0.25, 0.25, 0.25]

        blocks = []
        block_capacity = deque(config["query_capacity"])

        for _ in range(config["num_blocks"]):
            blocks.append(
                Block(
                    seq_dim=config["dim"],
                    num_heads=config["num_heads"],
                    query_capacity=block_capacity[0],
                )
            )
            block_capacity.rotate(1)

        self.blocks = nn.ModuleList(blocks)

    @property
    def example_input_array(self):
        return (
            torch.randint(256, (1, self.hparams["context_len"])).cuda(),
            torch.tensor([1]).cuda(),
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
            x = block(x, return_attention)

            if return_attention:
                x, att = x
                block_attentions.append(att)

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
    query_capacity: Optional[list[float]] = None,
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

    if not disable_wandb:
        train_logger = WandbLogger(project="pico", log_model="all")
        train_logger.log_hyperparams(train_config)
        train_logger.log_graph(model)
        # train_logger.watch(model, log="all")
    else:
        train_logger = TensorBoardLogger(
            save_dir="tensorboard", name="pico", log_graph=True
        )

    trainer = L.Trainer(
        max_epochs=train_config["max_epochs"],
        accelerator="gpu",
        precision="16-mixed",
        callbacks=[RichProgressBar()],
        logger=train_logger,
        limit_train_batches=0.125,
    )

    trainer.fit(model, train_loader)


@app.command()
def test(
    version: str,
    epoch: int,
    step: int,
    prompt: str = "",
    denoise_rate: float = 0.1,
    temperature: float = 1.0,
):
    model = DenoisingModel.load_from_checkpoint(
        f"pico/{version}/checkpoints/epoch={epoch}-step={step}.ckpt",
        # query_capacity=[1]
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
        # Insert prompt
        prompt_tensor = torch.tensor(
            list(bytes(prompt, encoding="utf-8")), dtype=torch.uint8
        )
        x[:, : prompt_tensor.size(0)] = prompt_tensor

        noise_rate = torch.tensor([1.0], device="cuda")

        with Live() as live:
            for level in range(int(config["noise_levels"] / denoise_rate)):
                noise_rate[0] = level / config["noise_levels"] * denoise_rate
                logits = F.softmax(model(x, noise_rate) * temperature, dim=-1)

                # Absolute sampling:
                # logits = logits.argmax(-1)

                # Relative sampling:
                logits = torch.tensor(
                    [torch.multinomial(p, 1) for p in logits.squeeze()], device="cuda"
                ).unsqueeze(0)

                x = logits
                # Force prompt to stay at each steps
                x[:, : prompt_tensor.size(0)] = prompt_tensor

                # Log
                table = Table(style="green", width=100)
                table.add_column("Rate")
                table.add_column("Sample", ratio=0.9)

                rate_text = Text(
                    f"{round(100 * noise_rate[0].cpu().tolist(), 2)}%",
                    style="bold blue",
                )

                try:
                    logits_text = Text(bytes(logits.tolist()[0]).decode("utf-8"))
                    logits_text.stylize("yellow", 0, len(prompt))  # Highlight prompt
                # In case we can't decode the bytes, try to print them as is
                except UnicodeDecodeError:
                    logits_text = Text(bytes(logits.tolist()[0]).__repr__())
                    prompt_repr_len = (
                        len(bytes(prompt, encoding="utf-8").__repr__()) - 3
                    )
                    logits_text.stylize("yellow", 2, prompt_repr_len + 2)

                table.add_row(rate_text, logits_text)
                live.update(table)


if __name__ == "__main__":
    app()
