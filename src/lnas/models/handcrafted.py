from __future__ import annotations

from typing import List, Tuple

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

    def step(
        self, frame: torch.Tensor, state: Tuple[torch.Tensor, ...] | None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        previous = list(state) if state is not None else [None] * len(self.blocks)
        next_state: List[torch.Tensor] = []
        value = frame
        for index, block in enumerate(self.blocks):
            value, membrane = block.step(value, previous[index])
            next_state.append(membrane)
        output = self.head(self.pool(value).flatten(1))
        return output, tuple(next_state)
