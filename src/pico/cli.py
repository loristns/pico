import codecs
import logging
import pathlib
import time
from typing import Optional

import torch
import typer
from datasets import load_dataset
from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler

from .infer import infer
from .model import PICO_XS_PRESET, Pico
from .serialization import load, save
from .train import DEFAULT_TRAINING_META, TrainingMeta, TrainingStep, train

# Initialize Rich console and logger
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("cli")

app = typer.Typer(pretty_exceptions_enable=False)


class TrainingLogFile(BaseModel):
    meta: TrainingMeta

    dataset_path: str
    dataset_name: Optional[str]
    text_column: Optional[str]
    base_weights: pathlib.Path

    model_directory: pathlib.Path
    checkpoints: list[TrainingStep] = []


@app.command("init")
def init_command(
    path: pathlib.Path,
    preset: str = "xs",
    # Override preset
    dim: Optional[int] = None,
    next_tokens: Optional[int] = None,
    att_q_heads: Optional[int] = None,
    att_kv_heads: Optional[int] = None,
    fb_num_blocks: Optional[int] = None,
    fb_att_window_size: Optional[int] = None,
    latent_capacity_factor: Optional[float] = None,
    latent_num_blocks: Optional[int] = None,
    latent_att_window_size: Optional[int] = None,
):
    # Initialize metadata
    match preset:
        case "xs":
            meta = PICO_XS_PRESET.model_copy()
        case _:
            raise ValueError(f"Invalid preset: {preset}")

    meta.dim = dim or meta.dim
    meta.next_tokens = next_tokens or meta.next_tokens
    meta.att_q_heads = att_q_heads or meta.att_q_heads
    meta.att_kv_heads = att_kv_heads or meta.att_kv_heads
    meta.fb_num_blocks = fb_num_blocks or meta.fb_num_blocks
    meta.fb_att_window_size = fb_att_window_size or meta.fb_att_window_size
    meta.latent_capacity_factor = latent_capacity_factor or meta.latent_capacity_factor
    meta.latent_num_blocks = latent_num_blocks or meta.latent_num_blocks
    meta.latent_att_window_size = latent_att_window_size or meta.latent_att_window_size

    path.mkdir(exist_ok=False)
    model = Pico(meta)
    save(model, path / "default.safetensors", metadata_file=path / "model.json")

    logger.info(f"Model initialized at: {path}")
    logger.info(meta.model_dump_json(indent=2))


@app.command("train")
def train_command(
    run_name: str,
    dataset_path: str,
    dataset_name: Optional[str] = None,
    text_column: Optional[str] = None,
    model_directory: pathlib.Path = typer.Argument(
        default=pathlib.Path.cwd(), exists=True, file_okay=False
    ),
    base_weights: Optional[pathlib.Path] = None,
    # Override default training meta
    context_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    max_steps: Optional[int] = None,
    warmup_steps: Optional[int] = None,
    grad_accumulation_steps: Optional[int] = None,
):
    assert (
        model_directory.is_dir()
    ), f"Model directory is not a directory: {model_directory}"
    assert (
        model_directory / "model.json"
    ).exists(), f"Model metadata not found: {model_directory}"

    run_directory = model_directory / "runs" / run_name
    run_directory.mkdir(parents=True, exist_ok=True)

    if base_weights is None:
        base_weights = model_directory / "default.safetensors"
        logger.info(f"Using default base weights: {base_weights}")
    else:
        assert base_weights.exists(), f"Base weights not found: {base_weights}"
        assert (
            base_weights.suffix == ".safetensors"
        ), f"Invalid base weights format: {base_weights}"
        assert base_weights.is_file(), f"Base weights must be a file: {base_weights}"

        logger.info(f"Using base weights: {base_weights}")

    # Init training metadata
    training_meta = DEFAULT_TRAINING_META.model_copy()

    training_meta.context_len = context_len or training_meta.context_len
    training_meta.batch_size = batch_size or training_meta.batch_size
    training_meta.learning_rate = learning_rate or training_meta.learning_rate
    training_meta.weight_decay = weight_decay or training_meta.weight_decay
    training_meta.max_steps = max_steps or training_meta.max_steps
    training_meta.warmup_steps = warmup_steps or training_meta.warmup_steps
    training_meta.grad_accumulation_steps = (
        grad_accumulation_steps or training_meta.grad_accumulation_steps
    )

    logger.info(training_meta.model_dump_json(indent=2))

    training_log_file = run_directory / "training.json"
    training_log = TrainingLogFile(
        meta=training_meta,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        text_column=text_column,
        base_weights=base_weights,
        model_directory=model_directory,
    )
    training_log_file.write_text(training_log.model_dump_json(indent=2))

    logger.info(f"Training run initialized at: {run_directory}")

    dataset = load_dataset(dataset_path, dataset_name, streaming=True)

    if "train" in dataset:
        dataset = dataset["train"]

    if text_column is not None:
        dataset = dataset.map(
            lambda x: {"bytes": x[text_column].encode("utf-8")},
            remove_columns=dataset.column_names,
        )

    # Load model
    model = load(base_weights, model_directory / "model.json")
    model = model.to("cuda")
    model = torch.compile(model)

    # Train
    tm1 = time.time()
    for step in train(
        model, dataset, training_meta, tracker_project_name=model_directory.stem
    ):
        tm2 = time.time()

        kb_per_sec = (
            training_meta.batch_size * training_meta.context_len / (tm2 - tm1)
        ) / 1000

        logger.info(
            f"[{step.i}]\t lm: {step.next_token_lm_loss:.4f} ({step.bit_per_bytes:.2f} bPB, pplx: {step.perplexity:.2f}), aux: {step.aux_loss:.4f}\t ({kb_per_sec:.0f} kB/s)"
        )

        if step.i % 500 == 0:
            logger.info(f"Saving checkpoint at step: {step.i}")

            save(model, run_directory / f"{step.i:08d}.safetensors")
            training_log.checkpoints.append(step)
            training_log_file.write_text(training_log.model_dump_json(indent=2))

        tm1 = time.time()

    if step.i not in [s.i for s in training_log.checkpoints]:
        save(model, run_directory / f"{step.i:08d}.safetensors")
        training_log.checkpoints.append(step)
        training_log_file.write_text(training_log.model_dump_json(indent=2))

    logger.info("Training complete")


@app.command("run")
def run_command(
    run_name: str,
    model_directory: pathlib.Path = typer.Argument(
        default=pathlib.Path.cwd(), exists=True, file_okay=False
    ),
    checkpoint: int = -1,
    prompt: str = "",
    temperature: float = 1.0,
    show_router_decisions: bool = False,
):
    assert (
        model_directory.is_dir()
    ), f"Model directory is not a directory: {model_directory}"
    assert (
        model_directory / "model.json"
    ).exists(), f"Model metadata not found: {model_directory}"

    run_directory = model_directory / "runs" / run_name
    training_log_file = run_directory / "training.json"

    training_log = TrainingLogFile.model_validate_json(training_log_file.read_text())
    checkpoint = training_log.checkpoints[checkpoint]

    logger.info(f"Running checkpoint: {checkpoint.i}")
    logger.info(checkpoint)

    model = load(
        run_directory / f"{checkpoint.i:08d}.safetensors",
        model_directory / "model.json",
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
        if show_router_decisions and iteration["router_decision"]:
            print("_", end="", flush=True)

        try:
            char = decoder.decode(bytes([iteration["byte"]]))
        except UnicodeDecodeError:
            char = "�"

        print(char, end="", flush=True)
