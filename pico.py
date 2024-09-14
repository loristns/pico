import os
from collections.abc import Iterable
from typing import Optional

import einops
import torch
import typer
from flash_attn import flash_attn_func

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

app = typer.Typer(pretty_exceptions_enable=False)

# Hyperparameters
params = {
    "dim": 384,
    "context_len": 8*1024,
    "train": {
        "file": "data/wikitext-103-v1-train.txt",
        "batch_size": 8,
        "learning_rate": 1e-3,
        "grad_accumulation_steps": 16,
        "num_epochs": 1,
        "dropout": 0.2,
    },
    "encoder": {
        "num_blocks": 2,
        "att_q_heads": 12,
        "att_kv_heads": 6,
        "att_window_size": 8,
    },
    "mod": {
        "capacity_factor": 0.125,
        "num_blocks": 16,
        "att_q_heads": 12,
        "att_kv_heads": 6,
        "att_window_size": 512,
    },
    "decoder": {
        "num_blocks": 1,
        "att_q_heads": 12,
        "att_kv_heads": 6,
        "att_window_size": 8,
    },
}


# Model definition
class GQA(nn.Module):
    def __init__(
        self,
        dim: int,
        q_heads: int,
        kv_heads: int,
        window_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.dim = dim
        self.window_size = window_size
        self.head_dim = dim // q_heads
        self.dropout = dropout

        self.fused_qkv = nn.Linear(dim, (q_heads + 2 * kv_heads) * self.head_dim, bias=False)
        self.proj = nn.Linear(q_heads * self.head_dim, dim, bias=False)

        self.alibi_slopes = torch.arange(q_heads)
        self.alibi_slopes = torch.exp2(-((self.alibi_slopes + 1) * 8.0 / q_heads))
        self.alibi_slopes = self.alibi_slopes.cuda()

    def forward(self, x: torch.Tensor):
        qkv = self.fused_qkv(x)
        qkv = einops.rearrange(
            qkv,
            "batch seq_len (qkv_heads head_dim) -> batch seq_len qkv_heads head_dim",
            head_dim=self.head_dim,
        )

        # Split query, key and value heads
        # q: [batch, seq_len, q_heads, head_dim]
        # k: [batch, seq_len, kv_heads, head_dim]
        # v: [batch, seq_len, kv_heads, head_dim]
        q, k, v = torch.split(qkv, [self.q_heads, self.kv_heads, self.kv_heads], dim=2)

        att = flash_attn_func(
            q,
            k,
            v,
            causal=True,
            window_size=(self.window_size, self.window_size),
            alibi_slopes=self.alibi_slopes,
            dropout_p=self.dropout,
        )
        att = einops.rearrange(
            att, "batch seq_len q_heads head_dim -> batch seq_len (q_heads head_dim)"
        )

        out = self.proj(att)
        return out


class SwiGLU(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)

        x, gate = torch.chunk(x, 2, dim=-1)
        x = F.silu(gate) * x

        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        att_q_heads: int,
        att_kv_heads: int,
        att_window_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.att = GQA(dim, att_q_heads, att_kv_heads, att_window_size, dropout)
        self.glu = SwiGLU(dim, dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.att(self.norm1(x))
        x = x + self.glu(self.norm2(x))
        return x


class BlockSeq(nn.Module):
    def __init__(self, blocks: Iterable[Block]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)

        return x


class MoD(nn.Module):
    def __init__(self, blocks: Iterable[Block], dim: int, capacity_factor: int):
        super().__init__()
        self.dim = dim
        self.capacity_factor = capacity_factor

        self.block_seq = BlockSeq(blocks)
        self.router = nn.Linear(dim, 1, bias=False)
        self.router_scale = nn.Parameter(torch.tensor(1.0))
        self.router_shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # [batch, seq_len, 1]
        router_weights = self.router(x)
        router_weights = F.sigmoid(router_weights)

        # Capacity is the number of elements to keep in the resampled sequence
        if self.training:
            # During training, the capacity is relative to the sequence length
            capacity = int(seq_len * self.capacity_factor)
        else:
            # During inference, we use router as a classifier to determine the capacity
            # Two sequences of same length can have different capacity so batch_size can't be > 1
            assert batch_size == 1
            capacity = int(router_weights.gt(0.5).sum().item())

        # Skip computations if capacity is 0
        if capacity == 0:
            return torch.zeros_like(x), torch.tensor(0.0)

        # [batch, capacity, 1]
        router_top_weights, router_top_indices = torch.topk(
            router_weights, k=capacity, dim=1, sorted=False
        )

        # Keep original sequence order
        router_top_indices, router_top_order = torch.sort(router_top_indices, dim=1)
        router_top_weights = torch.gather(router_weights, dim=1, index=router_top_order)
        router_top_weights = (router_top_weights - self.router_shift) * self.router_scale

        # Expand indices over each dimensions for gather/scatter operations
        router_top_indices_exp = einops.repeat(
            router_top_indices, "batch capacity 1 -> batch capacity dim", dim=self.dim
        )

        # Create resampled sequence
        # [batch, capacity, dim]
        mod_x = torch.gather(x, dim=1, index=router_top_indices_exp)

        # Apply blocks
        mod_x = self.block_seq(mod_x)

        # Apply router weights
        mod_x = mod_x * router_top_weights

        # Scatter back to original sequence (filling the rest with zeros)
        # [batch, seq_len, dim]
        out = torch.scatter(
            torch.zeros_like(x),
            dim=1,
            index=router_top_indices_exp,
            src=mod_x,
        )

        # Mixture-of-Depth auxiliary loss
        mod_loss = F.binary_cross_entropy(
            router_weights,
            torch.scatter(
                torch.zeros_like(router_weights),
                dim=1,
                index=router_top_indices,
                src=torch.ones_like(router_weights),
            ),
        )

        return out, mod_loss


class Pico(nn.Module):
    def __init__(self, params: dict):
        super().__init__()

        self.embedding = nn.Embedding(256, params["dim"])

        self.encoder = BlockSeq(
            Block(
                params["dim"],
                params["encoder"]["att_q_heads"],
                params["encoder"]["att_kv_heads"],
                params["encoder"]["att_window_size"],
                params["train"]["dropout"],
            )
            for _ in range(params["encoder"]["num_blocks"])
        )
        self.mod = MoD(
            (
                Block(
                    params["dim"],
                    params["mod"]["att_q_heads"],
                    params["mod"]["att_kv_heads"],
                    params["mod"]["att_window_size"],
                    params["train"]["dropout"],
                )
                for _ in range(params["mod"]["num_blocks"])
            ),
            params["dim"],
            params["mod"]["capacity_factor"],
        )
        self.decoder = BlockSeq(
            Block(
                params["dim"],
                params["decoder"]["att_q_heads"],
                params["decoder"]["att_kv_heads"],
                params["decoder"]["att_window_size"],
                params["train"]["dropout"],
            )
            for _ in range(params["decoder"]["num_blocks"])
        )

        self.norm = nn.LayerNorm(params["dim"])

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)

        x = self.encoder(x)
        mod_out, aux_loss = self.mod(x)
        x = x + mod_out
        x = self.decoder(x)

        x = self.norm(x)
        x = x @ self.embedding.weight.T

        return x, aux_loss


# Dataloader
class PicoDataset(Dataset):
    def __init__(self, file: str, context_len: int):
        self.file = open(file, "rb")
        self.context_len = context_len

        self.num_bytes = os.path.getsize(file)

    def __len__(self) -> int:
        return self.num_bytes // self.context_len - 1

    def __getitem__(self, index):
        self.file.seek(index * self.context_len)
        chunk = list(self.file.read(self.context_len + 1))
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:], dtype=torch.long)

    def _collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x)
        y = torch.stack(y)
        return x, y

    def dataloader(self, batch_size: int):
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self._collate_fn,
            shuffle=True,
            generator=torch.Generator(device="cuda"),
        )


# Training loop function
def train(model: Pico, params, on_step_end=None):
    torch.manual_seed(0)
    dataset = PicoDataset(params["train"]["file"], params["context_len"])

    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params["train"]["learning_rate"], fused=True
        )

        for epoch in range(params["train"]["num_epochs"]):
            print(f"Epoch: {epoch}")

            for step, (x, y) in enumerate(
                dataset.dataloader(params["train"]["batch_size"])
            ):
                out, aux_loss = model(x)

                out = einops.rearrange(
                    out, "batch seq_len probs -> (batch seq_len) probs"
                )
                y = einops.rearrange(y, "batch seq_len -> (batch seq_len)")

                loss = 0.9 * F.cross_entropy(out, y) + 0.1 * aux_loss
                print(
                    f"Step: {step}, Loss: {loss.item()} - Aux Loss: {aux_loss.item()}"
                )

                loss = loss / params["train"]["grad_accumulation_steps"]
                loss.backward()

                if step % params["train"]["grad_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                if on_step_end:
                    on_step_end(step, loss.item(), aux_loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()


# Inference function
def infer(model: Pico, text: str):
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        with torch.no_grad():
            model.eval()
            x = torch.tensor([ord(c) for c in text], dtype=torch.long)
            x = x.unsqueeze(0).to("cuda")

            i = len(text)

            print(text, end="", flush=True)

            while i < params["context_len"]:
                out, _ = model(x)
                out = out[:, i - 1, :]

                out = F.softmax(out, dim=-1)

                next_char = torch.multinomial(out, 1)
                x = torch.cat([x, next_char], dim=1)
                i += 1

                print(chr(next_char.item()), end="", flush=True)

# CLI
@app.command("train")
def train_command():
    model = Pico(params).bfloat16().cuda()
    infer(model, "T")
    train(model, params)
    while True:
        print("\n\n----------------- INFERENCE -----------------\n\n")
        infer(model, "T")


if __name__ == "__main__":
    app()
