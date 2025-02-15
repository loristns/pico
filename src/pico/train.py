import math
from typing import Literal

import einops
import torch
from datasets import Dataset, IterableDataset
from pydantic import BaseModel, computed_field
from torch.amp import GradScaler
from torch.nn import DataParallel
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
    epochs: int
    max_steps: int
    warmup_steps: int
    grad_accumulation_steps: int
    validation_interval: int


DEFAULT_TRAINING_META = TrainingMeta(
    context_len=6 * 1024,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=0.1,
    epochs=1,
    max_steps=3600,
    warmup_steps=150,
    grad_accumulation_steps=1,
    validation_interval=100,
)


#############
#   Utils   #
#############


class TrainingStepMetrics(BaseModel):
    loss: float
    next_token_lm_loss: float
    aux_loss: float

    @computed_field
    @property
    def bits_per_byte(self) -> float:
        return math.log2(math.exp(self.next_token_lm_loss))

    @computed_field
    @property
    def perplexity(self) -> float:
        return math.exp(self.next_token_lm_loss)


class TrainingStep(BaseModel):
    i: int
    epoch: int
    train: TrainingStepMetrics
    validation: TrainingStepMetrics | None = None


def format_dataset(
    dataset: Dataset | IterableDataset,
    pico_meta: PicoMeta,
    training_meta: TrainingMeta,
):
    start_seq = torch.tensor(bytearray(b"<pico:seq>"), dtype=torch.uint8)
    end_seq = torch.tensor(bytearray(b"</pico:seq>"), dtype=torch.uint8)
    chunk_window_len = training_meta.context_len + pico_meta.next_tokens

    def _preprocess(batch: dict[Literal["bytes"], list[bytes]]):
        token_list = []

        for doc in batch["bytes"]:
            doc_tensor = torch.tensor(bytearray(doc), dtype=torch.uint8)
            token_list.extend([start_seq, doc_tensor, end_seq])

        tokens = torch.cat(token_list)

        # Determine the number of full chunks we can form.
        n_chunks = tokens.shape[0] // chunk_window_len
        if n_chunks == 0:
            # Not enough tokens for a full chunk -> return empty tensors.
            return {
                "x": torch.empty(0, dtype=torch.long),
                "y": torch.empty(0, dtype=torch.long),
            }

        # Only use the tokens that form complete chunks.
        tokens = tokens[: n_chunks * chunk_window_len]

        # Create chunks
        # [n_chunks, chunk_window_len]
        chunks = tokens.to(torch.long).view(n_chunks, chunk_window_len)

        # Unfold to create next_tokens sliding windows
        # [n_chunks, next_tokens + 1, context_len]
        chunks = chunks.unfold(dimension=1, size=training_meta.context_len, step=1)

        chunks = einops.rearrange(
            chunks, "batch next_tokens context_len -> batch context_len next_tokens"
        )

        # For each sliding window, use the first token as input (x) and the rest as targets (y)
        x = chunks[:, :, 0]
        y = chunks[:, :, 1:]
        return {"x": x, "y": y}

    return dataset.shuffle().map(_preprocess, batched=True, remove_columns=["bytes"])


def lr_schedule(step: int, training_meta: TrainingMeta):
    if step < training_meta.warmup_steps:
        return training_meta.learning_rate * (step + 1) / training_meta.warmup_steps

    if step >= training_meta.max_steps - 1:
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


@torch.no_grad()
def get_validation_metrics(
    model: Pico,
    dataloader: DataLoader,
    device: torch.device,
) -> TrainingStepMetrics:
    total_loss = 0.0
    total_aux_loss = 0.0
    total_next_token_loss = 0.0
    num_steps = 0

    for data in dataloader:
        x = data["x"].to(device)
        y = data["y"].to(device)

        with torch.autocast(device.type, dtype=torch.bfloat16):
            pred, router_weights, router_decisions, _ = model(x)

        loss, aux_loss, next_token_lm_loss = loss_fn(
            pred, router_weights, router_decisions, y
        )

        total_loss += loss.item()
        total_aux_loss += aux_loss.item()
        total_next_token_loss += next_token_lm_loss.item()
        num_steps += 1

    return TrainingStepMetrics(
        loss=total_loss / num_steps,
        aux_loss=total_aux_loss / num_steps,
        next_token_lm_loss=total_next_token_loss / num_steps,
    )


################
#   Training   #
################


def train(
    model: Pico,
    dataset: Dataset | IterableDataset,
    validation_dataset: Dataset | IterableDataset | None = None,
    training_meta: TrainingMeta = DEFAULT_TRAINING_META,
):
    dataloader = DataLoader(
        format_dataset(dataset, model.metadata, training_meta),
        batch_size=training_meta.batch_size,
        pin_memory=True,
        num_workers=dataset.num_shards,
    )

    validation_dataloader = None
    if validation_dataset is not None:
        validation_dataloader = DataLoader(
            format_dataset(validation_dataset, model.metadata, training_meta),
            batch_size=training_meta.batch_size,
            pin_memory=True,
            num_workers=validation_dataset.num_shards,
        )

    device = torch.device("cuda")
    model = DataParallel(model)
    model = model.to(device)
    model.train()

    trainable_params = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }

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

    scaler = GradScaler()

    # Training loop
    step = 0
    for epoch in range(training_meta.epochs):
        for data in dataloader:
            x = data["x"].to(device)
            y = data["y"].to(device)

            with torch.autocast(device.type, dtype=torch.bfloat16):
                pred, router_weights, router_decisions, _ = model(x)

            loss, aux_loss, next_token_lm_loss = loss_fn(
                pred, router_weights, router_decisions, y
            )

            validation_metrics = None
            if (
                validation_dataloader is not None
                and step % training_meta.validation_interval == 0
                and step > 0
            ):
                validation_metrics = get_validation_metrics(
                    model, validation_dataloader, device=device
                )

            training_step = TrainingStep(
                i=step,
                epoch=epoch,
                train=TrainingStepMetrics(
                    loss=loss.item(),
                    next_token_lm_loss=next_token_lm_loss.item(),
                    aux_loss=aux_loss.item(),
                ),
                validation=validation_metrics,
            )

            yield training_step

            scaler.scale(loss).backward()

            if (step + 1) % training_meta.grad_accumulation_steps == 0:
                # Update learning rate according to schedule before next optimizer step
                lr = lr_schedule(step, training_meta)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            step += 1
