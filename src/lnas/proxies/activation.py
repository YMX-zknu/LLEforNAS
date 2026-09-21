from __future__ import annotations

from typing import List

import torch
from torch import nn

from lnas.models.common import LIFCell

from .base import Proxy
from .result import ProxyResult


class ActivationKernelProxy(Proxy):
    def __init__(self, sparsity_aware: bool, epsilon: float = 1e-6) -> None:
        self.sparsity_aware = sparsity_aware
        self.epsilon = epsilon
        self.name = "sahd" if sparsity_aware else "hd"

    def score(self, model: nn.Module, inputs: torch.Tensor) -> ProxyResult:
        model.eval()
        kernel = torch.zeros(inputs.shape[0], inputs.shape[0], device=inputs.device)
        handles: List[torch.utils.hooks.RemovableHandle] = []

        def collect(_module, _arguments, output):
            nonlocal kernel
            spikes = output[0] if isinstance(output, tuple) else output
            binary = (spikes.flatten(1) > 0).to(inputs.dtype)
            if self.sparsity_aware:
                neurons = binary.shape[1]
                sparsity = binary.mean(dim=1, keepdim=True)
                expected = (sparsity @ (1.0 - sparsity).T + (1.0 - sparsity) @ sparsity.T) * neurons
                scale = 0.5 / (expected + self.epsilon)
                differences = binary @ (1.0 - binary).T + (1.0 - binary) @ binary.T
                kernel = kernel + neurons - scale * differences
            else:
                kernel = kernel + binary @ binary.T + (1.0 - binary) @ (1.0 - binary).T

        for module in model.modules():
            if isinstance(module, LIFCell):
                handles.append(module.register_forward_hook(collect))
        if not handles:
            raise ValueError("The model exposes no LIFCell activations")
        try:
            with torch.no_grad():
                model(inputs)
        finally:
            for handle in handles:
                handle.remove()
        regularized = kernel + torch.eye(kernel.shape[0], device=kernel.device) * self.epsilon
        sign, logabsdet = torch.linalg.slogdet(regularized.double())
        raw = float(logabsdet.cpu()) if float(sign.cpu()) > 0 else float("-inf")
        return ProxyResult(
            name=self.name,
            score=raw,
            raw_value=raw,
            stable=True,
            details={"activation_layers": float(len(handles))},
        )
