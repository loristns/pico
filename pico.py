"""
The Pico byte-level model library.

This file contains the implementation of the Pico model, a byte-level model
that uses mixture of depth and local attention to reduce the computational cost
of long byte sequences.
"""

import dataclasses
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Optional, Union

import einops
import torch
import typer
from flash_attn import flash_attn_func
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

app = typer.Typer(pretty_exceptions_enable=False)


#######################
#   Building blocks   #
#######################
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

        self.fused_qkv = nn.Linear(
            dim, (q_heads + 2 * kv_heads) * self.head_dim, bias=False
        )
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
    def __init__(
        self,
        num_blocks: int,
        dim: int,
        att_q_heads: int,
        att_kv_heads: int,
        att_window_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            Block(dim, att_q_heads, att_kv_heads, att_window_size, dropout)
            for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)

        return x


class MoDBlockSeq(BlockSeq):
    def __init__(
        self,
        num_blocks: int,
        capacity_factor: int,
        dim: int,
        att_q_heads: int,
        att_kv_heads: int,
        att_window_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__(
            num_blocks,
            dim,
            att_q_heads,
            att_kv_heads,
            att_window_size,
            dropout,
        )

        self.dim = dim
        self.capacity_factor = capacity_factor

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
        router_top_weights = (
            router_top_weights - self.router_shift
        ) * self.router_scale

        # Expand indices over each dimensions for gather/scatter operations
        router_top_indices_exp = einops.repeat(
            router_top_indices, "batch capacity 1 -> batch capacity dim", dim=self.dim
        )

        # Create resampled sequence
        # [batch, capacity, dim]
        mod_x = torch.gather(x, dim=1, index=router_top_indices_exp)

        # Apply blocks
        mod_x = super().forward(mod_x)

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


########################
#   Model definition   #
########################
@dataclass
class PicoHyperparameters:
    # Model hyperparameters
    dim: int = 384
    context_len: int = 8 * 1024
    att_q_heads = 12
    att_kv_heads = 6

    # - Full bytes blocks (FB)
    fb_num_blocks: int = 2
    fb_att_window_size: int = 8

    # - Mixture of Depth blocks (MoD)
    mod_num_blocks: int = 16
    mod_att_window_size: int = 512
    mod_capacity_factor: float = 0.125

    # Train hyperparameters
    file: str = "data/wikitext-103-v1-train.txt"
    batch_size: int = 8
    learning_rate: float = 1e-3
    grad_accumulation_steps: int = 16
    num_epochs: int = 1
    dropout: float = 0.2


class Pico(nn.Module):
    def __init__(self, params: PicoHyperparameters):
        super().__init__()
        self.params = params

        self.embedding = nn.Embedding(256, params.dim)

        fb_params = {
            "num_blocks": params.fb_num_blocks,
            "dim": params.dim,
            "att_q_heads": params.att_q_heads,
            "att_kv_heads": params.att_kv_heads,
            "att_window_size": params.fb_att_window_size,
            "dropout": params.dropout,
        }
        mod_params = {
            **fb_params,
            "num_blocks": params.mod_num_blocks,
            "capacity_factor": params.mod_capacity_factor,
            "att_window_size": params.mod_att_window_size,
        }

        self.fb_in = BlockSeq(**fb_params)
        self.mod = MoDBlockSeq(**mod_params)
        self.fb_out = BlockSeq(**fb_params)

        self.norm = nn.LayerNorm(params.dim)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)

        x = self.fb_in(x)
        mod_out, aux_loss = self.mod(x)
        x = x + mod_out
        x = self.fb_out(x)

        x = self.norm(x)
        x = x @ self.embedding.weight.T

        return x, aux_loss


#####################
#   Serialization   #
#####################
def save(model: Pico, path: Union[pathlib.Path, str], checkpoint: bool = False):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    model_path = path / "model.safetensors"
    hyperparams_path = path / "hyperparams.json"

    # Rename the current "model.safetensors" to "model_{checkpoint}.safetensors"
    # (if checkpointing is enabled)
    if checkpoint and model_path.exists():
        checkpoint_nums = [
            int(p.stem.split("_")[-1]) for p in path.glob("model_*.safetensors")
        ]
        next_checkpoint_num = max(checkpoint_nums, default=0) + 1

        checkpoint_path = path / f"model_{next_checkpoint_num}.safetensors"
        model_path.rename(checkpoint_path)

    # Save model and hyperparameters
    save_file(model.state_dict(), model_path)

    if not hyperparams_path.exists() or not checkpoint:
        with open(hyperparams_path, "w") as f:
            json.dump(dataclasses.asdict(model.params), f, indent=2)


def load(path: Union[pathlib.Path, str], checkpoint: int = -1):
    path = pathlib.Path(path)

    model_path = path / "model.safetensors"
    if checkpoint >= 0:
        model_path = path / f"model_{checkpoint}.safetensors"

    hyperparams_path = path / "hyperparams.json"

    with open(hyperparams_path, "r") as f:
        params = PicoHyperparameters(**json.load(f))

    model = Pico(params)
    with safe_open(model_path, framework="pt") as f:
        model.load_state_dict({key: f.get_tensor(key) for key in f.keys()})

    return model


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
def train(model: Pico, params: PicoHyperparameters, on_step_end=None):
    torch.manual_seed(0)
    dataset = PicoDataset(params.file, params.context_len)

    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params.learning_rate, fused=True
        )

        for epoch in range(params.num_epochs):
            print(f"Epoch: {epoch}")

            for step, (x, y) in enumerate(dataset.dataloader(params.batch_size)):
                out, aux_loss = model(x)

                out = einops.rearrange(
                    out, "batch seq_len probs -> (batch seq_len) probs"
                )
                y = einops.rearrange(y, "batch seq_len -> (batch seq_len)")

                loss = 0.9 * F.cross_entropy(out, y) + 0.1 * aux_loss
                print(
                    f"Step: {step}, Loss: {loss.item()} - Aux Loss: {aux_loss.item()}"
                )

                loss = loss / params.grad_accumulation_steps
                loss.backward()

                if step % params.grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                if on_step_end:
                    on_step_end(step, loss.item(), aux_loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()


# Inference function
def infer(model: Pico, text: str, params: PicoHyperparameters):
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        with torch.no_grad():
            model.eval()
            x = torch.tensor([ord(c) for c in text], dtype=torch.long)
            x = x.unsqueeze(0).to("cuda")

            i = len(text)

            print(text, end="", flush=True)

            while i < params.context_len:
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
    params = PicoHyperparameters()
    model = Pico(params)
    model = model.bfloat16().cuda()
    infer(model, "T", params)
    train(model, params)
    while True:
        print("\n\n----------------- INFERENCE -----------------\n\n")
        infer(model, "T", params)


if __name__ == "__main__":
    app()
