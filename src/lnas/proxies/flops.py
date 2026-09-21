from __future__ import annotations

from typing import List

import torch
from torch import nn

from .base import Proxy
from .result import ProxyResult


class FLOPsProxy(Proxy):
    name = "flops"

    def score(self, model: nn.Module, inputs: torch.Tensor) -> ProxyResult:
        total = 0
        handles: List[torch.utils.hooks.RemovableHandle] = []

        def conv_hook(module: nn.Conv2d, arguments, output):
            nonlocal total
            batch, out_channels, out_h, out_w = output.shape
            kernel_h, kernel_w = module.kernel_size
            operations = kernel_h * kernel_w * module.in_channels // module.groups
            total += batch * out_channels * out_h * out_w * operations

        def linear_hook(module: nn.Linear, arguments, output):
            nonlocal total
            vectors = output.numel() // output.shape[-1]
            total += vectors * module.in_features * module.out_features

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(linear_hook))
        try:
            with torch.no_grad():
                model.eval()(inputs)
        finally:
            for handle in handles:
                handle.remove()
        raw = float(total)
        return ProxyResult(
            name=self.name,
            score=raw,
            raw_value=raw,
            stable=True,
            details={"giga_operations": raw / 1e9},
        )
