from __future__ import annotations

from typing import List

import torch
from torch import nn

from .common import SpikingConvBlock, TemporalClassifier, initialize_module


class HandcraftedSNN(TemporalClassifier):
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
        blocks = []
        current = channels
        for stage, depth in enumerate(stage_depths):
            output = width * (2**stage)
            for index in range(depth):
                stride = 2 if stage > 0 and index == 0 else 1
                blocks.append(SpikingConvBlock(current, output, 3, stride, tau, threshold, alpha))
                current = output
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(current, classes)
        self.apply(initialize_module)

    def forward_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        states: List[torch.Tensor | None] = [None] * len(self.blocks)
        outputs = []
        for frame in inputs.unbind(dim=1):
            value = frame
            for index, block in enumerate(self.blocks):
                value, states[index] = block.step(value, states[index])
            outputs.append(self.head(self.pool(value).flatten(1)))
        return torch.stack(outputs, dim=1)
