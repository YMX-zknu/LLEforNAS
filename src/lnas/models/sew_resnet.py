from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from .common import LIFCell, TemporalClassifier, initialize_module


class SEWBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        tau: float,
        threshold: float,
        alpha: float,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.neuron1 = LIFCell(tau, threshold, alpha)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.neuron2 = LIFCell(tau, threshold, alpha)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def step(
        self,
        inputs: torch.Tensor,
        state1: torch.Tensor | None,
        state2: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value, state1 = self.neuron1(self.norm1(self.conv1(inputs)), state1)
        value, state2 = self.neuron2(self.norm2(self.conv2(value)), state2)
        return value + self.shortcut(inputs), state1, state2


class SEWResNet(TemporalClassifier):
    def __init__(
        self,
        channels: int,
        classes: int,
        width: int,
        stage_depths: List[int],
        tau: float,
        threshold: float,
        alpha: float,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(channels, width, 3, padding=1, bias=False)
        self.stem_norm = nn.BatchNorm2d(width)
        self.stem_neuron = LIFCell(tau, threshold, alpha)
        blocks = []
        current = width
        for stage, depth in enumerate(stage_depths):
            output = width * (2**stage)
            for index in range(depth):
                stride = 2 if stage > 0 and index == 0 else 1
                blocks.append(
                    SEWBlock(current, output, stride, tau, threshold, alpha)
                )
                current = output
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(current, classes)
        self.apply(initialize_module)

    def step(
        self, frame: torch.Tensor, state: Tuple[torch.Tensor, ...] | None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        previous = list(state) if state is not None else [None] * (1 + 2 * len(self.blocks))
        value, stem_state = self.stem_neuron(self.stem_norm(self.stem(frame)), previous[0])
        next_state = [stem_state]
        offset = 1
        for block in self.blocks:
            value, state1, state2 = block.step(
                value, previous[offset], previous[offset + 1]
            )
            next_state.extend((state1, state2))
            offset += 2
        output = self.head(self.pool(value).flatten(1))
        return output, tuple(next_state)
