from dataclasses import dataclass

import einops
import torch
from flash_attn import flash_attn_func
from torch import nn
from torch.nn import functional as F

############################
#   Pico hyperparameters   #
############################


@dataclass
class PicoHyperparameters:
    # Model hyperparameters
    dim: int = 128
    next_tokens: int = 8
    att_q_heads: int = 9
    att_kv_heads: int = 3
    att_dropout: float = 0.2

    # - Full bytes blocks (FB)
    fb_num_blocks: int = 2  # * 2 (before and after latent blocks)
    fb_att_window_size: int = 16

    # - Latent blocks (MoD)
    latent_num_blocks: int = 12
    latent_att_window_size: int = 512
    latent_capacity_factor: float = 0.25

    # Train hyperparameters
    context_len: int = 6 * 1024
    batch_size: int = 32
    grad_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    warmup_steps: int = 150
    max_steps: int = 3600
    weight_decay: float = 0.1

    # Special sequences
    start_seq: str = "<pico:seq>"
    end_seq: str = "</pico:seq>"


#######################
#   Building blocks   #
#######################


class GQA(nn.Module):
    def __init__(
        self,
        dim: int,
        q_heads: int,
        kv_heads: int,
        window_size: int,
        dropout: float,
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
            dropout_p=self.dropout if self.training else 0.0,
        )

        att = einops.rearrange(
            att, "batch seq_len q_heads head_dim -> batch seq_len (q_heads head_dim)"
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
    def __init__(
        self,
        dim: int,
        att_q_heads: int,
        att_kv_heads: int,
        att_window_size: int,
        att_dropout: float,
    ):
        super().__init__()

        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        self.att = GQA(dim, att_q_heads, att_kv_heads, att_window_size, att_dropout)
        self.glu = SwiGLU(dim)

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
        att_window_size: int,
        att_dropout: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            Block(dim, att_q_heads, att_kv_heads, att_window_size, att_dropout)
            for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)

        return x


class LatentBlockSeq(BlockSeq):
    def __init__(
        self,
        num_blocks: int,
        capacity_factor: int,
        dim: int,
        att_q_heads: int,
        att_kv_heads: int,
        att_window_size: int,
        att_dropout: float,
    ):
        super().__init__(
            num_blocks,
            dim,
            att_q_heads,
            att_kv_heads,
            att_window_size,
            att_dropout,
        )

        self.dim = dim
        self.capacity_factor = capacity_factor

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
            return (
                torch.zeros_like(x),
                router_weights,
                torch.zeros_like(router_weights),
            )

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
        latent_x = torch.gather(x, dim=1, index=router_top_indices_exp)

        # Apply blocks
        latent_x = super().forward(latent_x)

        # Apply router weights
        latent_x = latent_x * router_top_weights

        # Scatter back to original sequence (filling the rest with zeros)
        # [batch, seq_len, dim]
        pred = torch.scatter(
            torch.zeros_like(x),
            dim=1,
            index=router_top_indices_exp,
            src=latent_x,
        )

        # During training: ground truth for Mixtures-of-Depth auxiliary loss
        # During inference: binary vector for router decisions
        router_decisions = torch.scatter(
            torch.zeros_like(router_weights),
            dim=1,
            index=router_top_indices,
            src=torch.ones_like(router_weights),
        )

        return pred, router_weights, router_decisions


########################
#   Model definition   #
########################


class Pico(nn.Module):
    def __init__(self, params: PicoHyperparameters):
        super().__init__()
        self.params = params

        self.embedding = nn.Embedding(256, params.dim)
        self.unembedding = nn.Linear(params.dim, 256 * params.next_tokens)

        fb_params = {
            "num_blocks": params.fb_num_blocks,
            "dim": params.dim,
            "att_q_heads": params.att_q_heads,
            "att_kv_heads": params.att_kv_heads,
            "att_dropout": params.att_dropout,
            "att_window_size": params.fb_att_window_size,
        }
        latent_params = {
            **fb_params,
            "num_blocks": params.latent_num_blocks,
            "capacity_factor": params.latent_capacity_factor,
            "att_window_size": params.latent_att_window_size,
        }

        self.fb_in = BlockSeq(**fb_params)
        self.latent = LatentBlockSeq(**latent_params)
        self.fb_out = BlockSeq(**fb_params)

        self.norm = nn.RMSNorm(params.dim)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)

        x = self.fb_in(x)

        latent_pred, router_weights, router_decisions = self.latent(x)
        x = x + latent_pred

        x = self.fb_out(x)

        x = self.norm(x)
        x = self.unembedding(x)

        x = einops.rearrange(
            x,
            "batch seq_len (next_tokens probs) -> batch seq_len next_tokens probs",
            next_tokens=self.params.next_tokens,
        )

        return x, router_weights, router_decisions
