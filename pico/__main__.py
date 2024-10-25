import codecs
import math
import time
from typing import Optional

import torch
import typer
from datasets import load_dataset

from .infer import infer
from .model import Pico, PicoHyperparameters
from .train import train
from .utils import load, save

app = typer.Typer()


@app.command("train")
def train_command(
    dataset_path: str,
    dataset_name: Optional[str] = None,
    text_column: Optional[str] = None,
    model_path: Optional[str] = None,
    # Training hyperparameters
    context_len: Optional[int] = 6 * 1024,
    batch_size: Optional[int] = 32,
    grad_accumulation_steps: Optional[int] = 1,
    learning_rate: Optional[float] = 1e-3,
    warmup_steps: Optional[int] = 150,
    max_steps: Optional[int] = 3600,
    weight_decay: Optional[float] = 0.1,
    # Checkpointing
    checkpoint_path: Optional[str] = None,
    checkpoint_steps: Optional[int] = 500,
):
    # Load dataset
    dataset = load_dataset(dataset_path, dataset_name, streaming=True)
    if "train" in dataset:
        dataset = dataset["train"]

    if text_column is not None:
        dataset = dataset.map(
            lambda x: {"bytes": x[text_column].encode("utf-8")},
            remove_columns=dataset.column_names,
        )

    # Load model
    if model_path is not None:
        model = load(model_path, -1)
        params = model.params
    else:
        params = PicoHyperparameters(
            context_len=context_len,
            batch_size=batch_size,
            grad_accumulation_steps=grad_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            weight_decay=weight_decay,
        )
        model = Pico(params)

    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"🌟 Model has {param_count} parameters")

    model = torch.compile(model)

    # Checkpointing
    if checkpoint_path is None:
        checkpoint_path = model_path

    # Training loop
    tm1 = time.time()
    for step in train(model, dataset):
        print(step)
        tm2 = time.time()

        byte_per_sec = params.batch_size * params.context_len / (tm2 - tm1)

        print(
            f"Time: {tm2 - tm1}, Byte/sec: {byte_per_sec}, Bit/Byte: {math.log2(math.exp(step['next_token_lm_loss']))}"
        )
        tm1 = tm2

        if step["step"] % checkpoint_steps == 0:
            save(model, checkpoint_path, checkpoint=True)

    save(model, checkpoint_path, checkpoint=True)
    

@app.command("test")
def test_command(
    model_path: str,
    checkpoint: Optional[int] = -1,
    # Inference hyperparameters
    prompt: Optional[str] = None,
    temperature: Optional[float] = 1.5,
    show_router_decisions: Optional[bool] = False,
):
    model = load(model_path, checkpoint)

    if prompt is not None:
        print(prompt, end="", flush=True)

    decoder = codecs.getincrementaldecoder('utf-8')()

    for iteration in infer(
        model,
        prompt=prompt.encode("utf-8") if prompt is not None else None,
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


if __name__ == "__main__":
    app()
