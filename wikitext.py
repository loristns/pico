import time
from typing import Optional

import torch
import typer
from datasets import IterableDataset, load_dataset

from pico.infer import infer
from pico.model import Pico, PicoHyperparameters
from pico.train import train
from pico.utils import load, save

app = typer.Typer(pretty_exceptions_enable=False)


@app.command("download")
def download_wikitext():
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    with open("data/wikitext-103-v1-train.txt", "w") as f:
        for i in range(len(ds["train"])):
            print(f"Writing {i}th line over {len(ds['train'])}")

            if i % 1000 == 0:
                print(ds["train"][i])

            f.write(ds["train"][i]["text"])


@app.command("train")
def run_training():
    print("Training on wikitext-103")

    params = PicoHyperparameters()
    model = Pico(params)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count} parameters")

    model = torch.compile(model)

    def read_file(file):
        with open(file, "r") as f:
            for line in f:
                yield {"bytes": line.encode("utf-8")}

    dataset = IterableDataset.from_generator(
        read_file, gen_kwargs={"file": "data/wikitext-103-v1-train.txt"}
    )

    tm1 = time.time()
    for epoch in range(2):
        for step in train(model, dataset):
            print(step)
            tm2 = time.time()

            byte_per_sec = params.batch_size * params.context_len / (tm2 - tm1)

            print(f"Time: {tm2 - tm1}, Byte/sec: {byte_per_sec}")
            tm1 = tm2

            if step["step"] % 250 == 0:
                save(model, "./models/test-wikitext-103", checkpoint=True)

        save(model, "./models/test-wikitext-103", checkpoint=True)


@app.command("test")
def run_inference(
    prompt: Optional[str] = None,
    temperature: float = 1.5,
    checkpoint: int = -1,
    show_mod: bool = False,
):
    print("Testing on wikitext-103")

    model = load("./models/test-wikitext-103-v1", checkpoint)

    if prompt is not None:
        print(prompt, end="", flush=True)

    for iteration in infer(
        model,
        prompt=prompt.encode("utf-8") if prompt is not None else None,
        temperature=temperature,
        stop_at_eot=False,
    ):
        if show_mod and iteration["mod"]:
            print("_", end="", flush=True)

        print(chr(iteration["byte"]), end="", flush=True)


if __name__ == "__main__":
    app()
