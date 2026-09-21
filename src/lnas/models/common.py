from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ATanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(value)
        ctx.alpha = alpha
        return (value >= 0).to(value.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (value,) = ctx.saved_tensors
        alpha = ctx.alpha
        scale = alpha / (2.0 * (1.0 + (torch.pi * alpha * value / 2.0).square()))
        return grad_output * scale, None


class LIFCell(nn.Module):
    def __init__(self, tau: float, threshold: float, alpha: float) -> None:
        super().__init__()
        self.decay = 1.0 - 1.0 / tau
        self.threshold = threshold
        self.alpha = alpha

    def forward(
        self, current: torch.Tensor, membrane: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if membrane is None:
            membrane = torch.zeros_like(current)
        membrane = membrane * self.decay + current
        spike = ATanSpike.apply(membrane - self.threshold, self.alpha)
        membrane = membrane - spike.detach() * self.threshold
        return spike, membrane


class SpikingConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        tau: float,
        threshold: float,
        alpha: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.neuron = LIFCell(tau, threshold, alpha)

    def step(
        self, inputs: torch.Tensor, membrane: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        current = self.norm(self.conv(inputs))
        return self.neuron(current, membrane)


class TemporalClassifier(nn.Module):
    def forward_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(inputs).mean(dim=1)


def initialize_module(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
