from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from .common import LIFCell, TemporalClassifier, initialize_module


class AutoBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        name: str,
        tau: float,
        threshold: float,
        alpha: float,
    ) -> None:
        super().__init__()
        self.name = name
        if name == "skip_connect":
            self.neuron = None
            self.body = nn.Identity()
            self.residual = False
        else:
            kernel = int(name.split("_k")[1].split("_")[0])
            self.neuron = LIFCell(tau, threshold, alpha)
            expansion = 3 if "_e3" in name else 1
            hidden = channels * expansion
            self.body = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm2d(hidden),
                nn.Conv2d(hidden, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.residual = name.startswith("SRB") or name.startswith("SIB")

    def step(
        self, inputs: torch.Tensor, membrane: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if self.neuron is None:
            return inputs, membrane
        spikes, membrane = self.neuron(inputs, membrane)
        output = self.body(spikes)
        if self.residual:
            output = output + inputs
        return output, membrane


class AutoSNN(TemporalClassifier):
    def __init__(
        self,
        channels: int,
        classes: int,
        width: int,
        blocks: List[str],
        tau: float,
        threshold: float,
        alpha: float,
    ) -> None:
        super().__init__()
        if len(blocks) != 8:
            raise ValueError("AutoSNN architecture must contain eight stages")
        self.stem = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1, bias=False), nn.BatchNorm2d(width)
        )
        modules = []
        current = width
        for name in blocks:
            if name == "max_pool_k2":
                modules.append(nn.MaxPool2d(2))
            else:
                modules.append(AutoBlock(current, name, tau, threshold, alpha))
        self.blocks = nn.ModuleList(modules)
        self.output_neuron = LIFCell(tau, threshold, alpha)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(current, classes)
        self.apply(initialize_module)

    def forward_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        states: List[torch.Tensor | None] = [None] * len(self.blocks)
        output_state = None
        outputs = []
        for frame in inputs.unbind(dim=1):
            value = self.stem(frame)
            for index, block in enumerate(self.blocks):
                if isinstance(block, AutoBlock):
                    value, states[index] = block.step(value, states[index])
                else:
                    value = block(value)
            value, output_state = self.output_neuron(value, output_state)
            outputs.append(self.head(self.pool(value).flatten(1)))
        return torch.stack(outputs, dim=1)
