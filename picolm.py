import codecs
import logging
import pathlib
import time
from typing import Annotated

import torch
import typer
from datasets import IterableDataset, IterableDatasetDict, load_dataset
from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler

from pico.infer import infer
from pico.model import PICO_XS_PRESET, Pico
from pico.serialization import load, load_metadata, save
from pico.train import DEFAULT_TRAINING_META, TrainingMeta, TrainingStep, train

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("cli")

app = typer.Typer(pretty_exceptions_enable=False)


#################
#    Utility    #
#################


class TrainingLogFile(BaseModel):
    meta: TrainingMeta

    dataset_path: str
    dataset_column_name: str | None
    dataset_train_file: str | None
    dataset_validation_file: str | None
    dataset_train_split: str | None
    dataset_validation_split: str | None
    dataset_validation_size: int | None
    base_weights: pathlib.Path | None
    tracker_project_name: str | None

    checkpoints: list[TrainingStep] = []


def _update_if_set(obj: object, **kwargs):
    """Update object attributes only if they are explicitly set in kwargs (not None)."""
    for key, value in kwargs.items():
        if value is not None and hasattr(obj, key):
            setattr(obj, key, value)
    return obj


def _load_dataset(
    dataset_path: str,
    dataset_train_file: str | None = None,
    dataset_validation_file: str | None = None,
    dataset_train_split: str | None = None,
    dataset_validation_split: str | None = None,
    dataset_validation_size: int | None = None,
) -> tuple[IterableDataset, IterableDataset]:
    """Load and split dataset from HuggingFace."""
    if dataset_train_file and dataset_validation_file:
        dataset = load_dataset(
            dataset_path,
            data_files={
                "train": dataset_train_file,
                "validation": dataset_validation_file,
            },
            streaming=True,
        )
        return dataset["train"], dataset["validation"]

    dataset = load_dataset(dataset_path, streaming=True)

    # Handle single-split datasets
    if isinstance(dataset, IterableDatasetDict) and len(dataset.keys()) == 1:
        dataset = dataset[list(dataset.keys())[0]]

    # Create train/validation split
    if isinstance(dataset, IterableDataset) and dataset_validation_size:
        return dataset.skip(dataset_validation_size), dataset.take(
            dataset_validation_size
        )

    train_split = dataset_train_split or "train"
    val_split = dataset_validation_split or "validation"
    train_dataset = dataset[train_split]
    validation_dataset = dataset[val_split]

    # Handle validation size
    if dataset_validation_size:
        validation_dataset = validation_dataset.take(dataset_validation_size)

        # Handle the edge case where train and validation splits are the same
        # (do not train on validation data)
        if train_split == val_split:
            train_dataset = train_dataset.skip(dataset_validation_size)

    # Shuffle train dataset
    train_dataset = train_dataset.shuffle(seed=42)

    return train_dataset, validation_dataset


def _prepare_dataset(
    dataset: IterableDataset, column_name: str, shuffle: bool = False
) -> IterableDataset:
    """Prepare dataset for training by converting columns and optionally shuffling."""

    def _convert_column(x):
        value = x[column_name]
        if isinstance(value, bytes):
            return {"bytes": value}
        if isinstance(value, str):
            return {"bytes": value.encode("utf-8")}
        raise ValueError(f"Invalid column type: {type(value)}")

    dataset = dataset.map(
        _convert_column,
        remove_columns=dataset.column_names,
    )

    if shuffle:
        dataset = dataset.shuffle(seed=42, buffer_size=100_000)

    return dataset


#################
#    Command    #
#################


@app.command("init")
def init_command(
    path: Annotated[
        pathlib.Path,
        typer.Argument(help="Model directory path"),
    ],
    preset: Annotated[str, typer.Option(help="Model preset (supported: 'xs')")] = "xs",
    # Override preset
    dim: int | None = None,
    next_tokens: int | None = None,
    att_q_heads: int | None = None,
    att_kv_heads: int | None = None,
    fb_num_blocks: int | None = None,
    fb_att_window_size: int | None = None,
    latent_capacity_factor: float | None = None,
    latent_num_blocks: int | None = None,
    latent_att_window_size: int | None = None,
):
    # Initialize metadata
    match preset:
        case "xs":
            meta = PICO_XS_PRESET.model_copy()
        case _:
            raise ValueError(f"Invalid preset: {preset}")

    _update_if_set(
        meta,
        dim=dim,
        next_tokens=next_tokens,
        att_q_heads=att_q_heads,
        att_kv_heads=att_kv_heads,
        fb_num_blocks=fb_num_blocks,
        fb_att_window_size=fb_att_window_size,
        latent_capacity_factor=latent_capacity_factor,
        latent_num_blocks=latent_num_blocks,
        latent_att_window_size=latent_att_window_size,
    )

    path.mkdir(parents=True, exist_ok=False)
    meta_file = path / "model.json"
    meta_file.write_text(meta.model_dump_json(indent=2))

    logger.info(f"Initialized Pico model at {path}")


@app.command("train")
def train_command(
    path: Annotated[
        pathlib.Path,
        typer.Argument(help="Model directory"),
    ],
    run_name: Annotated[
        str,
        typer.Argument(help="Training run name"),
    ],
    dataset_path: Annotated[
        str,
        typer.Argument(help="HuggingFace dataset path"),
    ],
    dataset_column_name: Annotated[
        str,
        typer.Option(help="Name of the column in the dataset to use for training."),
    ] = "bytes",
    dataset_train_file: Annotated[
        str | None,
        typer.Option(help="Name of the training file in the dataset directory."),
    ] = None,
    dataset_validation_file: Annotated[
        str | None,
        typer.Option(help="Name of the validation file in the dataset directory."),
    ] = None,
    dataset_train_split: Annotated[
        str | None,
        typer.Option(help="Name of the training split in the dataset."),
    ] = None,
    dataset_validation_split: Annotated[
        str | None,
        typer.Option(help="Name of the validation split in the dataset."),
    ] = None,
    dataset_validation_size: Annotated[
        int | None, typer.Option(help="Number of samples to use for validation.")
    ] = None,
    base_weights: Annotated[
        pathlib.Path | None,
        typer.Option(help="Path to base weights to start training from."),
    ] = None,
    tracker_project_name: Annotated[
        str | None,
        typer.Option(help="Tracker project name (default to model directory name)."),
    ] = None,
    # Override default training meta
    context_len: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    epochs: int | None = None,
    max_steps: int | None = None,
    warmup_steps: int | None = None,
    grad_accumulation_steps: int | None = None,
    validation_interval: int | None = None,
):
    if not path.is_dir() or not (path / "model.json").exists():
        logger.error(f"Invalid model directory: {path}")
        raise typer.Exit(code=1)

    # Load datasets
    train_dataset, validation_dataset = _load_dataset(
        dataset_path,
        dataset_train_file,
        dataset_validation_file,
        dataset_train_split,
        dataset_validation_split,
        dataset_validation_size,
    )

    # Prepare datasets
    train_dataset = _prepare_dataset(train_dataset, dataset_column_name, shuffle=True)
    validation_dataset = _prepare_dataset(validation_dataset, dataset_column_name)

    # Init training metadata
    training_meta = DEFAULT_TRAINING_META.model_copy()
    _update_if_set(
        training_meta,
        context_len=context_len,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        grad_accumulation_steps=grad_accumulation_steps,
        validation_interval=validation_interval,
    )

    run_directory = path / run_name
    run_directory.mkdir(parents=True, exist_ok=True)

    training_logs = TrainingLogFile(
        meta=training_meta,
        dataset_path=dataset_path,
        dataset_column_name=dataset_column_name,
        dataset_train_file=dataset_train_file,
        dataset_validation_file=dataset_validation_file,
        dataset_train_split=dataset_train_split,
        dataset_validation_split=dataset_validation_split,
        dataset_validation_size=dataset_validation_size,
        base_weights=base_weights,
        tracker_project_name=tracker_project_name or path.name,
    )
    training_logs_file = run_directory / "training.json"
    training_logs_file.write_text(training_logs.model_dump_json(indent=2))

    # Load model
    if base_weights is None:
        model = Pico(load_metadata(path / "model.json"))
    else:
        metadata_file = base_weights.parent.parent / "model.json"
        model = load(base_weights, metadata_file)

    model = torch.compile(model)

    # Train loop
    tm1 = time.time()
    for step in train(
        model,
        train_dataset,
        validation_dataset=validation_dataset,
        training_meta=training_meta,
        tracker_project_name=tracker_project_name or path.name,
    ):
        tm2 = time.time()

        kb_per_sec = kb_per_sec = (
            training_meta.batch_size * training_meta.context_len / (tm2 - tm1)
        ) / 1000

        logger.info(
            f"[{step.epoch} - {step.i}]\t lm: {step.train.next_token_lm_loss:.4f} ({step.train.bits_per_byte:.2f} bpB, pplx: {step.train.perplexity:.2f}), aux: {step.train.aux_loss:.4f}\t ({kb_per_sec:.0f} kB/s)"
        )

        if step.i % 500 == 0:
            logger.info(f"Saving checkpoint at step: {step.i}")

            save(model, run_directory / f"{step.i:08d}.safetensors")
            training_logs.checkpoints.append(step)
            training_logs_file.write_text(training_logs.model_dump_json(indent=2))

        tm1 = time.time()

    # Save last step
    if step.i not in [s.i for s in training_logs.checkpoints]:
        save(model, run_directory / f"{step.i:08d}.safetensors")
        training_logs.checkpoints.append(step)
        training_logs_file.write_text(training_logs.model_dump_json(indent=2))

    logger.info(f"Training completed at step: {step.i}")


@app.command("run")
def run_command(
    path: Annotated[
        pathlib.Path,
        typer.Argument(help="Model directory"),
    ],
    run_name: Annotated[
        str,
        typer.Argument(help="Run name"),
    ],
    checkpoint: Annotated[
        int,
        typer.Option(help="Checkpoint to load (default: -1 for latest)"),
    ] = -1,
    prompt: Annotated[str, typer.Option(help="Prompt to start generation from")] = "",
    temperature: Annotated[float, typer.Option(help="Generation temperature")] = 1.0,
    show_router_decisions: Annotated[
        bool, typer.Option(help="Show router decisions")
    ] = False,
):
    if not path.is_dir() or not (path / "model.json").exists():
        logger.error(f"Invalid model directory: {path}")
        raise typer.Exit(code=1)

    run_directory = path / run_name
    training_logs_file = run_directory / "training.json"

    if not training_logs_file.exists():
        logger.error(f"Invalid run name: {run_name}")
        raise typer.Exit(code=1)

    training_logs = TrainingLogFile.model_validate_json(training_logs_file.read_text())
    checkpoint = training_logs.checkpoints[checkpoint]

    logger.info(f"Running checkpoint: {checkpoint.i}")
    logger.info(checkpoint)

    model = load(
        run_directory / f"{checkpoint.i:08d}.safetensors",
        path / "model.json",
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params} parameters")

    if prompt is not None:
        print(prompt, end="", flush=True)

    decoder = codecs.getincrementaldecoder("utf-8")()

    for iteration in infer(
        model,
        prompt=prompt.encode("utf-8") if prompt else None,
        temperature=temperature,
        stop_end_seq=False,
    ):
        if show_router_decisions and iteration.router_decision:
            print("_", end="", flush=True)

        try:
            char = decoder.decode(iteration.byte)
        except UnicodeDecodeError:
            char = "�"
            decoder.reset()

        print(char, end="", flush=True)


if __name__ == "__main__":
    app()
