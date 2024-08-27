from functools import lru_cache, partial
from typing import Optional

import einops
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import flex_attention

# Hyperparameters
hyperparams = {}


# Model definition
@lru_cache(maxsize=16)
def _block_mask(seq_len: int, window_size: Optional[int] = None):
    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx

        if window_size is None:
            return causal_mask

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
    def __init__(self, dim: int, att_q_heads: int, att_kv_heads: int):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.att = GQA(dim, att_q_heads, att_kv_heads)
        self.glu = SwiGLU(dim)

    def forward(self, x: torch.Tensor):
        x = x + self.att(self.norm1(x))
        x = x + self.glu(self.norm2(x))
        return x


class MoD(nn.Module):
    def __init__(self, module: nn.Module, dim: int, capacity_factor: int):
        super().__init__()
        self.dim = dim
        self.capacity_factor = capacity_factor

        self.module = module
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
        mod_x = torch.gather(x, dim=1, index=router_top_indices_exp)

        # Apply module
        mod_out = self.module(mod_x)

        # Apply router weights
        mod_out = mod_out * router_top_weights

        # Scatter back to original sequence
        out = torch.scatter(
            torch.zeros_like(x),
            dim=1,
            index=router_top_indices_exp,
            src=mod_out,
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
    def __init__(self):
        super().__init__()

    def train_step(self):
        pass


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
        a = GQA(dim=16, q_heads=4, kv_heads=2)

        mod = MoD(a, dim=16, capacity_factor=0.5)
        mod = mod.eval()

        x = torch.randn(1, 10, 16)
        x, aux_loss = mod(x)
        print(x, x.shape)
        print(aux_loss, aux_loss.shape)
        
