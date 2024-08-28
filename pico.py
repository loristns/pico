from functools import lru_cache, partial
from typing import Iterable, Optional

import einops
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import flex_attention

# Hyperparameters
params = {
    "dim": 64,
    "encoder": {
        "num_blocks": 2,
        "att_q_heads": 16,
        "att_kv_heads": 8,
        "att_window_size": 128,
    },
    "mod": {
        "capacity_factor": 0.25,
        "num_blocks": 16,
        "att_q_heads": 16,
        "att_kv_heads": 8,
        "att_window_size": 128,
    },
    "decoder": {
        "num_blocks": 2,
        "att_q_heads": 16,
        "att_kv_heads": 8,
        "att_window_size": 128,
    },
}


# Model definition
@lru_cache(maxsize=16)
def _block_mask(seq_len: int, window_size: Optional[int] = None):
    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx

        if window_size is None:
            return causal_mask

        # If a window size is provided, apply a window mask (local attention)
        window_mask = (q_idx - kv_idx) <= window_size
        return causal_mask & window_mask

    return flex_attention.create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True
    )


class GQA(nn.Module):
    def __init__(
        self, dim: int, q_heads: int, kv_heads: int, window_size: Optional[int] = None
    ):
        super().__init__()

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.dim = dim
        self.window_size = window_size

        self.fused_qkv = nn.Linear(dim, (q_heads + 2 * kv_heads) * dim, bias=False)
        self.proj = nn.Linear(q_heads * dim, dim, bias=False)

    def attention_fn(self, seq_len: int):
        block_mask = _block_mask(seq_len, self.window_size)

        def score_mod(score, b, h, q_idx, kv_idx):
            alibi = (q_idx - kv_idx) * torch.exp2(-((h + 1) * 8.0 / self.q_heads))
            return score + alibi

        attention_fn = partial(
            flex_attention.flex_attention,
            score_mod=score_mod,
            block_mask=block_mask,
            enable_gqa=True,
        )
        attention_fn = torch.compile(attention_fn)

        return attention_fn

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[1]
        attention_fn = self.attention_fn(seq_len)

        qkv = self.fused_qkv(x)
        qkv = einops.rearrange(
            qkv,
            "batch seq_len (qkv_heads dim) -> batch qkv_heads seq_len dim",
            dim=self.dim,
        )

        # Split query, key and value heads
        # q: [batch, q_heads, seq_len, dim]
        # k: [batch, kv_heads, seq_len, dim]
        # v: [batch, kv_heads, seq_len, dim]
        q, k, v = torch.split(qkv, [self.q_heads, self.kv_heads, self.kv_heads], dim=1)

        att = attention_fn(q, k, v)
        att = einops.rearrange(
            att, "batch q_heads seq_len dim -> batch seq_len (q_heads dim)"
        )

        out = self.proj(att)
        return out


class SwiGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)

        x, gate = torch.chunk(x, 2, dim=-1)
        x = F.silu(gate) * x

        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, att_q_heads: int, att_kv_heads: int, att_window_size: Optional[int] = None):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.att = GQA(dim, att_q_heads, att_kv_heads, att_window_size)
        self.glu = SwiGLU(dim)

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


class Embedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(256, dim))

    def forward(self, x: torch.Tensor):
        return F.one_hot(x, num_classes=256).float() @ self.weights


class Pico(nn.Module):
    def __init__(self, params: dict):
        super().__init__()

        self.embedding = Embedding(params["dim"])

        self.encoder = BlockSeq(
            Block(
                params["dim"],
                params["encoder"]["att_q_heads"],
                params["encoder"]["att_kv_heads"],
                params["encoder"]["att_window_size"],
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
            )
            for _ in range(params["decoder"]["num_blocks"])
        )

        self.norm = nn.LayerNorm(params["dim"])

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)

        x = self.encoder(x)
        x, aux_loss = self.mod(x)
        x = self.decoder(x)

        x = self.norm(x)
        x = x @ self.embedding.weights.T

        return x, aux_loss


# Dataloader


# Training loop function
def train():
    pass


# Inference function
def infer():
    pass


# CLI
if __name__ == "__main__":
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        pico = Pico(params)

        print(pico)

        # Count parameters
        print(sum(p.numel() for p in pico.parameters() if p.requires_grad))

        x = torch.randint(0, 256, (1, 128*1024))
        out, aux_loss = pico(x)
        print(out.shape, aux_loss)
