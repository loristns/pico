import math
from typing import Dict, Union

import einops
import torch
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

        window_len = params.context_len + 1

        eot_seq = list(params.eot_seq.encode("utf-8"))
        eot_seq_len = len(eot_seq)

        for bytes in batch["bytes"]:
            current_buffer.extend(eot_seq)
            current_buffer.extend(bytes)

            # Split the buffer into chunks of window_len
            while len(current_buffer) >= window_len:
                chunk = current_buffer[:window_len]
                chunks.append(chunk)
                current_buffer = current_buffer[window_len:]

            # Handle the case where appending eot_seq would exceed the window_len
            while len(current_buffer) + eot_seq_len > window_len:
                current_buffer = current_buffer[eot_seq_len - 1 :]  # Drop prefix

        return {
            "x": torch.tensor(chunks)[:, :-1],
            "y": torch.tensor(chunks)[:, 1:],
        }

    return dataset.map(_preprocess, batched=True, remove_columns=["bytes"]).with_format(
        "numpy"
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
    mod_weights: torch.Tensor,
    mod_decisions: torch.Tensor,
    targets: torch.Tensor,
):
    predictions = einops.rearrange(
        predictions, "batch seq_len probs -> (batch seq_len) probs"
    )
    mod_weights = einops.rearrange(mod_weights, "batch seq_len 1 -> (batch seq_len)")
    mod_decisions = einops.rearrange(
        mod_decisions, "batch seq_len 1 -> (batch seq_len)"
    )
    targets = einops.rearrange(targets, "batch seq_len -> (batch seq_len)")

    aux_loss = F.binary_cross_entropy(mod_weights, mod_decisions)
    loss = 0.9 * F.cross_entropy(predictions, targets) + 0.1 * aux_loss

    return loss, aux_loss


################
#   Training   #
################


def train(model: Pico, dataset: Union[Dataset, IterableDataset]):
    # Configure training
    device = torch.device("cuda")

    model.train()
    model.to(device)

    dataloader = DataLoader(
        format_dataset(dataset, model.params),
        batch_size=model.params.batch_size,
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
            # No weight decay for less than 2D parameters tensors (bias, LayerNorm, etc.)
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

    # Training loop
    for step, data in enumerate(dataloader):
        x = data["x"].to(device)
        y = data["y"].to(device)

        with torch.autocast(device.type, dtype=torch.bfloat16):
            pred, mod_weights, mod_decisions = model(x)

        loss, aux_loss = loss_fn(pred, mod_weights, mod_decisions, y)

        yield {
            "step": step,
            "loss": loss.item(),
            "aux_loss": aux_loss.item(),
        }

        loss = loss / model.params.grad_accumulation_steps

        scaler.scale(loss).backward()

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
