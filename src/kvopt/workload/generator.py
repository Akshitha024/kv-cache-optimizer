"""Synthetic request workload generators.

Two distributions are bundled:

  - `mixed`: prompt lengths drawn from a log-normal centered at 800 tokens, with
    a 30% tail of very long prompts (4-16k tokens). This is what real chat
    serving looks like.
  - `uniform`: equal prompt lengths (for sanity / regression tests).
"""

from __future__ import annotations

import random

from kvopt.types import Request


def mixed_workload(n: int = 200, seed: int = 17) -> list[Request]:
    rng = random.Random(seed)
    reqs: list[Request] = []
    for i in range(n):
        if rng.random() < 0.3:
            prompt = int(rng.uniform(4000, 16000))
        else:
            prompt = max(64, int(rng.lognormvariate(6.6, 0.35)))
        out = max(8, int(rng.lognormvariate(5.0, 0.4)))
        arrival = i // 4
        reqs.append(Request(id=i, arrival_step=arrival, prompt_tokens=prompt, output_tokens=out))
    return reqs


def uniform_workload(
    n: int = 50, prompt_tokens: int = 512, output_tokens: int = 64
) -> list[Request]:
    return [
        Request(id=i, arrival_step=i // 4, prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        for i in range(n)
    ]
