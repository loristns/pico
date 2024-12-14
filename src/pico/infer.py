from typing import Optional

import torch
from pydantic import BaseModel
from torch.nn import functional as F

from .model import Pico

#################
#   Inference   #
#################


class InferenceStep(BaseModel):
    iteration: int
    byte: bytes
    seq: bytes
    router_decision: bool
    router_weight: float


def infer(
    model: Pico,
    prompt: Optional[bytes] = None,
    temperature: float = 1.0,
    max_iteration: int = -1,
    stop_end_seq: bool = True,
):
    device = torch.device("cuda")

    model.eval()
    model.to(device)

    if prompt is None:
        prompt = "<pico:seq>".encode("utf-8")

    init_seq = torch.tensor([*prompt], dtype=torch.long).unsqueeze(0).to(device)
    x = init_seq

    seq = prompt

    iteration = 0

    kv_caches = None

    while True:
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            pred, router_weights, router_decisions, kv_caches = model(
                x, kv_caches=kv_caches
            )

        pred = F.softmax(pred[:, -1, 0, :] * temperature, dim=-1)
        pred = torch.multinomial(pred, 1)

        x = pred
        byte = bytes([pred.item()])
        iteration += 1
        seq += byte
        router_decision = router_decisions[:, -1].item()

        yield InferenceStep(
            iteration=iteration,
            byte=byte,
            seq=seq,
            router_decision=router_decision == 1,
            router_weight=router_weights[:, -1, :].item(),
        )

        if max_iteration > 0 and iteration >= max_iteration:
            break

        if stop_end_seq and seq.endswith(model.params.end_seq.encode("utf-8")):
            break
