import dataclasses
import json
import pathlib
from typing import Union

from safetensors import safe_open
from safetensors.torch import save_file

from .model import Pico, PicoHyperparameters

#####################
#   Serialization   #
#####################


def save(model: Pico, path: Union[pathlib.Path, str], checkpoint: bool = False):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    model_path = path / "model.safetensors"
    hyperparams_path = path / "hyperparams.json"

    # Rename the current "model.safetensors" to "model_{checkpoint}.safetensors"
    # (if checkpointing is enabled)
    if checkpoint and model_path.exists():
        checkpoint_nums = [
            int(p.stem.split("_")[-1]) for p in path.glob("model_*.safetensors")
        ]
        next_checkpoint_num = max(checkpoint_nums, default=0) + 1

        checkpoint_path = path / f"model_{next_checkpoint_num}.safetensors"
        model_path.rename(checkpoint_path)

    # Save model and hyperparameters
    if hasattr(model, "_orig_mod"):
        save_file(model._orig_mod.state_dict(), model_path)
    else:
        save_file(model.state_dict(), model_path)

    if not hyperparams_path.exists() or not checkpoint:
        with open(hyperparams_path, "w") as f:
            json.dump(dataclasses.asdict(model.params), f, indent=2)


def load(path: Union[pathlib.Path, str], checkpoint: int = -1):
    path = pathlib.Path(path)

    model_path = path / "model.safetensors"
    if checkpoint >= 0:
        model_path = path / f"model_{checkpoint}.safetensors"

    hyperparams_path = path / "hyperparams.json"

    with open(hyperparams_path, "r") as f:
        params = PicoHyperparameters(**json.load(f))

    model = Pico(params)
    with safe_open(model_path, framework="pt") as f:
        model.load_state_dict({key: f.get_tensor(key) for key in f.keys()})

    return model
