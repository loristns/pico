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
        prompt = b"<pico:seq>"

    iteration = 0
    byte_sequence = prompt
    x = torch.tensor([*prompt], device=device, dtype=torch.long).unsqueeze(0)
    kv_caches = None

    while True:
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            logits, _, _, kv_caches = model(x, kv_caches=kv_caches)

        predicted_suffix = []

        for next_token in range(model.metadata.next_tokens):
            head_pred = F.softmax(logits[:, -1, next_token, :] * temperature, dim=-1)

            # First token head: predict immediate next token -> standard sampling
            if next_token == 0:
                head_pred = torch.multinomial(head_pred, 1)

            # Next token heads: predict next n tokens -> greedy sampling + validation
            else:
                head_pred = torch.argmax(head_pred)

            predicted_suffix.append(head_pred.item())

        # Validate next n tokens
        validated_suffix = [
            predicted_suffix[0]
        ]  # Immediate next token is always validated

        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            spec_x = torch.tensor(
                predicted_suffix, device=device, dtype=torch.long
            ).unsqueeze(0)
            verif_logits, router_weights, router_decisions, _ = model(
                spec_x, kv_caches=kv_caches
            )

        for next_token in range(model.metadata.next_tokens - 1):
            head_pred = F.softmax(
                verif_logits[:, next_token, 0, :] * temperature, dim=-1
            )
            head_pred = torch.multinomial(head_pred, 1)

            if head_pred.item() != predicted_suffix[next_token + 1]:
                break

            validated_suffix.append(head_pred.item())

        # Send every validated token
        for i, token in enumerate(validated_suffix):
            iteration += 1
            byte = bytes([token])
            byte_sequence += byte

            yield InferenceStep(
                iteration=iteration,
                byte=byte,
                seq=byte_sequence,
                router_decision=router_decisions[:, i].item() == 1,
                router_weight=router_weights[:, i, :].item(),
            )

            if max_iteration > 0 and iteration >= max_iteration:
                return

            if stop_end_seq and byte_sequence.endswith(b"</pico:seq>"):
                return

        # Next loop
        x = torch.tensor(validated_suffix, device=device, dtype=torch.long).unsqueeze(0)
