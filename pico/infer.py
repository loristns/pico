from typing import Optional

import torch
from torch.nn import functional as F

from .model import Pico

#################
#   Inference   #
#################


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
        prompt = model.params.start_seq.encode("utf-8")

    init_seq = torch.tensor([*prompt], dtype=torch.long).unsqueeze(0).to(device)
    seq = init_seq
    iteration = 0

    while True:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            pred, mod_weights, mod_decisions = model(seq)

        pred = F.softmax(pred[:, -1, :] * temperature, dim=-1)
        pred = torch.multinomial(pred, 1)

        mod = mod_decisions[:, -1].item()

        seq = torch.cat([seq, pred], dim=1)

        iteration += 1
        byte_seq = bytes(seq.squeeze(0).cpu().tolist())
        yield {
            "iteration": iteration,
            "byte": pred.item(),
            "seq": byte_seq,
            "mod": mod == 1,
            "mod_weights": mod_weights[:, -1, :].item(),
        }

        if max_iteration > 0 and iteration >= max_iteration:
            break

        if stop_end_seq and byte_seq.endswith(model.params.end_seq.encode("utf-8")):
            break
