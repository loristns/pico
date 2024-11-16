import math
from typing import Dict, Optional, Union

import einops
import torch
from accelerate import Accelerator
from datasets import Dataset, IterableDataset
from pydantic import BaseModel, computed_field
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .model import Pico, PicoMeta
from .third_party.soap import SOAP

################################
#   Training hyperparameters   #
################################


class TrainingMeta(BaseModel):
    context_len: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_steps: int
    warmup_steps: int
    grad_accumulation_steps: int


DEFAULT_TRAINING_META = TrainingMeta(
    context_len=6 * 1024,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=0.1,
    max_steps=3600,
    warmup_steps=150,
    grad_accumulation_steps=1,
)


#############
#   Utils   #
#############


def format_dataset(
    dataset: Union[Dataset, IterableDataset],
    pico_meta: PicoMeta,
    training_meta: TrainingMeta,
):
    def _preprocess(batch: Dict[str, list[bytes]]):
        START_SEQ = list("<pico:seq>".encode("utf-8"))
        END_SEQ = list("</pico:seq>".encode("utf-8"))

        chunks = []
        current_buffer = []

        window_len = training_meta.context_len + pico_meta.next_tokens

        doc_separation_seq_len = len(START_SEQ) + len(END_SEQ)

        for bytes in batch["bytes"]:
            current_buffer.extend(START_SEQ)
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

            current_buffer.extend(END_SEQ)

        chunks = torch.tensor(chunks).unfold(1, training_meta.context_len, 1)
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


def lr_schedule(step: int, training_meta: TrainingMeta):
    if step < training_meta.warmup_steps:
        return training_meta.learning_rate * (step + 1) / training_meta.warmup_steps

    if step >= training_meta.max_steps:
        return 0.1 * training_meta.learning_rate

    decay_ratio = (step - training_meta.warmup_steps) / (
        training_meta.max_steps - training_meta.warmup_steps
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return 0.1 * training_meta.learning_rate + coeff * 0.9 * training_meta.learning_rate


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


class TrainingStep(BaseModel):
    i: int

    loss: float
    next_token_lm_loss: float
    aux_loss: float

    @computed_field
    def bit_per_bytes(self) -> float:
        return math.log2(math.exp(self.next_token_lm_loss))

    @computed_field
    def perplexity(self) -> float:
        return math.exp(self.next_token_lm_loss)


def train(
    model: Pico,
    dataset: Union[Dataset, IterableDataset],
    training_meta: TrainingMeta = DEFAULT_TRAINING_META,
    tracker_project_name: Optional[str] = None,
):
    # Configure training
    accelerator = Accelerator(log_with="all")
    device = accelerator.device

    if tracker_project_name is not None:
        accelerator.init_trackers(
            tracker_project_name, config=training_meta.model_dump()
        )

    model.train()
    model.to(device)

    dataloader = DataLoader(
        format_dataset(dataset, model.metadata, training_meta),
        batch_size=training_meta.batch_size,
        pin_memory=True,
    )

    trainable_params = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }

    scaler = torch.GradScaler(device)
    optimizer = SOAP(
        [
            {
                "params": [
                    param for param in trainable_params.values() if param.dim() >= 2
                ],
                "weight_decay": training_meta.weight_decay,
            },
            # No weight decay for less than 2D parameters tensors (bias, RMSNorm, etc.)
            {
                "params": [
                    param for param in trainable_params.values() if param.dim() < 2
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=training_meta.learning_rate,
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

        training_step = TrainingStep(
            i=step,
            loss=loss.item(),
            next_token_lm_loss=next_token_lm_loss.item(),
            aux_loss=aux_loss.item(),
        )

        accelerator.log(training_step.model_dump(exclude=["i"]), step=step)
        yield training_step

        loss = loss / training_meta.grad_accumulation_steps

        accelerator.backward(
            scaler.scale(loss),
        )

        if (step + 1) % training_meta.grad_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update learning rate according to schedule before next optimizer step
            lr = lr_schedule(step, training_meta)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    accelerator.end_training()
