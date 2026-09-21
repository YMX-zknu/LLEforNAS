from __future__ import annotations

from typing import List

import torch
from torch import nn

from .base import Proxy
from .result import ProxyResult


class LLEProxy(Proxy):
    name = "lle"

    def __init__(self, max_outputs: int = 10, epsilon: float = 1e-8, stable_only: bool = True):
        self.max_outputs = max_outputs
        self.epsilon = epsilon
        self.stable_only = stable_only

    def score(self, model: nn.Module, inputs: torch.Tensor) -> ProxyResult:
        model.eval()
        samples = inputs.detach().clone().requires_grad_(True)
        sequence = model.forward_sequence(samples)
        if sequence.ndim != 3:
            raise ValueError("forward_sequence must return [B,T,K]")
        classes = sequence.shape[-1]
        count = min(classes, self.max_outputs)
        indices = torch.linspace(0, classes - 1, count, device=sequence.device).round().long()
        exponents: List[torch.Tensor] = []
        spectral_norms: List[torch.Tensor] = []
        for timestep in range(sequence.shape[1]):
            rows = []
            for index in indices:
                gradient = torch.autograd.grad(
                    sequence[:, timestep, index].sum(),
                    samples,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                rows.append(gradient[:, : timestep + 1].flatten(1))
            jacobian = torch.stack(rows, dim=1)
            singular_values = torch.linalg.svdvals(jacobian.float())
            maximum = singular_values[:, 0].clamp_min(self.epsilon)
            spectral_norms.append(maximum.detach())
            exponents.append(maximum.log() / float(timestep + 1))
        exponent_tensor = torch.stack(exponents, dim=1)
        raw = float(exponent_tensor.mean().detach().cpu())
        stable = raw <= 0.0
        score = raw
        if self.stable_only and not stable:
            score = -1_000_000.0 - raw
        norms = torch.stack(spectral_norms, dim=1)
        return ProxyResult(
            name=self.name,
            score=score,
            raw_value=raw,
            stable=stable,
            details={
                "mean_spectral_norm": float(norms.mean().cpu()),
                "minimum_exponent": float(exponent_tensor.min().detach().cpu()),
                "maximum_exponent": float(exponent_tensor.max().detach().cpu()),
                "output_directions": float(count),
            },
        )
