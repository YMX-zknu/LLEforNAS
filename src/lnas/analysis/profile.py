from __future__ import annotations

import statistics
import time
from typing import Dict, List

import torch
from torch import nn

from lnas.proxies.base import Proxy


def profile_proxy(
    proxy: Proxy,
    model: nn.Module,
    inputs: torch.Tensor,
    repeats: int = 5,
    warmup: int = 1,
) -> Dict[str, float]:
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup must be non-negative")
    device = inputs.device
    for _ in range(warmup):
        proxy.score(model, inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        proxy.score(model, inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "mean_seconds": statistics.mean(timings),
        "std_seconds": statistics.pstdev(timings),
        "peak_memory_mib": peak / (1024**2),
        "repeats": float(repeats),
    }
