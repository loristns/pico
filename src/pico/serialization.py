import json
import pathlib

from safetensors.torch import safe_open, save_file

from .model import Pico, PicoMeta


def save(model: Pico, file: pathlib.Path, metadata_file: pathlib.Path | None = None):
    save_file(
        model._orig_mod.state_dict()
        if hasattr(model, "_orig_mod")
        else model.state_dict(),
        file,
    )

    if metadata_file is not None:
        metadata_file.write_text(model.metadata.model_dump_json(indent=2))


def load_metadata(file: pathlib.Path):
    return PicoMeta(**json.loads(file.read_text()))


def load(file: pathlib.Path, metadata_file: pathlib.Path):
    model = Pico(load_metadata(metadata_file))

    with safe_open(file, framework="pt") as f:
        model.load_state_dict({key: f.get_tensor(key) for key in f.keys()})

    return model
