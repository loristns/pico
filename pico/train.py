import math
from typing import Dict, Union

import einops
import torch
from accelerate import Accelerator
from datasets import Dataset, IterableDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .model import Pico, PicoHyperparameters


def format_dataset(
    dataset: Union[Dataset, IterableDataset], params: PicoHyperparameters
):
    def _preprocess(batch: Dict[str, list[bytes]]):
        chunks = []
        current_buffer = []

        window_len = params.context_len + params.next_tokens

        start_seq = list(params.start_seq.encode("utf-8"))
        end_seq = list(params.end_seq.encode("utf-8"))

        doc_separation_seq_len = len(start_seq) + len(end_seq)

        for bytes in batch["bytes"]:
            current_buffer.extend(start_seq)
            current_buffer.extend(bytes)

            # Split the buffer into chunks of window_len
            while len(current_buffer) >= window_len:
                chunk = current_buffer[:window_len]
                chunks.append(chunk)
                current_buffer = current_buffer[window_len:]

            # Handle the edge case where appending the end sequence + start sequence would exceed the window length
            while len(current_buffer) + doc_separation_seq_len > window_len:
                # Drop prefix
                current_buffer = current_buffer[doc_separation_seq_len - 1 :]

            current_buffer.extend(end_seq)

        chunks = torch.tensor(chunks).unfold(1, params.context_len, 1)
        chunks = einops.rearrange(
            chunks, "batch next_tokens seq_len -> batch seq_len next_tokens"
        )

        results = {
            "x": chunks[:, :, 0],
            "y": chunks[:, :, 1:],
        }

        return results

    return (
        dataset.shuffle()
        .map(_preprocess, batched=True, remove_columns=["bytes"])
        .with_format("numpy")
    )


def lr_schedule(step: int, params: PicoHyperparameters):
    if step < params.warmup_steps:
        return params.learning_rate * (step + 1) / params.warmup_steps

    if step >= params.max_steps:
        return 0.1 * params.learning_rate

    decay_ratio = (step - params.warmup_steps) / (
        params.max_steps - params.warmup_steps
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return 0.1 * params.learning_rate + coeff * 0.9 * params.learning_rate


@torch.compile
def loss_fn(
    predictions: torch.Tensor,
    router_weights: torch.Tensor,
    router_decisions: torch.Tensor,
    targets: torch.Tensor,
):
    _, seq_len, next_tokens, _ = predictions.shape

    # Weight tokens heads loss more heavily for immediate next tokens than for later ones such that:
    # - token_weights[head] ~= token_weight[head + 1] + token_weight[head + 2]
    # - token_weights.sum() = 1
    phi = 1.618
    exponents = torch.arange(0, next_tokens, step=1).flip(dims=(0,))
    token_weights = phi**exponents
    token_weights /= token_weights.sum()
    token_weights = token_weights.to(predictions.device)

    flat_predictions = einops.rearrange(
        predictions,
        "batch seq_len next_tokens probs -> (batch seq_len next_tokens) probs",
    )
    flat_targets = einops.rearrange(
        targets, "batch seq_len next_tokens -> (batch seq_len next_tokens)"
    )
    lm_losses = einops.rearrange(
        F.cross_entropy(flat_predictions, flat_targets, reduction="none"),
        "(batch seq_len next_tokens) -> batch seq_len next_tokens",
        seq_len=seq_len,
        next_tokens=next_tokens,
    )

    lm_loss = (
        (
            lm_losses * token_weights  # Weighted loss
        )
        .sum(dim=-1)  # Sum over next_tokens
        .mean()  # Average over batch/seq_len
    )

    # Also return next token loss for logging purposes
    next_token_lm_loss = lm_losses[:, :, 0]
    next_token_lm_loss = next_token_lm_loss.mean()

    router_weights = einops.rearrange(
        router_weights, "batch seq_len 1 -> (batch seq_len)"
    )
    router_decisions = einops.rearrange(
        router_decisions, "batch seq_len 1 -> (batch seq_len)"
    )

    aux_loss = F.binary_cross_entropy(router_weights, router_decisions)

    lm_loss = 0.9 * lm_loss + 0.1 * aux_loss
    return lm_loss, aux_loss, next_token_lm_loss


################
#   Training   #
################


def train(model: Pico, dataset: Union[Dataset, IterableDataset]):
    # Configure training
    accelerator = Accelerator()
    device = accelerator.device

    model.train()
    model.to(device)

    dataloader = DataLoader(
        format_dataset(dataset, model.params),
        batch_size=model.params.batch_size,
        pin_memory=True,
    )

    trainable_params = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }

    scaler = torch.GradScaler(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    param for param in trainable_params.values() if param.dim() >= 2
                ],
                "weight_decay": model.params.weight_decay,
            },
            # No weight decay for less than 2D parameters tensors (bias, RMSNorm, etc.)
            {
                "params": [
                    param for param in trainable_params.values() if param.dim() < 2
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=model.params.learning_rate,
        fused=True,
    )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Training loop
    for step, data in enumerate(dataloader):
        x = data["x"]
        y = data["y"]

        with torch.autocast(device.type, dtype=torch.bfloat16):
            pred, router_weights, router_decisions = model(x)

        loss, aux_loss, next_token_lm_loss = loss_fn(
            pred, router_weights, router_decisions, y
        )

        yield {
            "step": step,
            "loss": loss.item(),
            "next_token_lm_loss": next_token_lm_loss.item(),
            "aux_loss": aux_loss.item(),
        }

        loss = loss / model.params.grad_accumulation_steps

        accelerator.backward(
            scaler.scale(loss),
        )

        if (step + 1) % model.params.grad_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update learning rate according to schedule before next optimizer step
            lr = lr_schedule(step, model.params)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
