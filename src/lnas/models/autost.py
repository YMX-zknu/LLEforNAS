from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .common import LIFCell, TemporalClassifier, initialize_module


class SpikingTransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        mlp_ratio: int,
        tau: float,
        threshold: float,
        alpha: float,
    ) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("Transformer dimension must be divisible by the number of heads")
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dimension)
        hidden = dimension * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, dimension)
        )
        self.neuron1 = LIFCell(tau, threshold, alpha)
        self.neuron2 = LIFCell(tau, threshold, alpha)

    def step(self, inputs: torch.Tensor, state: Dict[str, torch.Tensor]):
        normalized = self.norm1(inputs)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        value, state1 = self.neuron1(inputs + attended, state.get("attention"))
        projected = self.mlp(self.norm2(value))
        value, state2 = self.neuron2(value + projected, state.get("mlp"))
        return value, {"attention": state1, "mlp": state2}


class AutoST(TemporalClassifier):
    def __init__(
        self,
        channels: int,
        classes: int,
        dimension: int,
        depth: int,
        heads: int,
        mlp_ratio: int,
        tau: float,
        threshold: float,
        alpha: float,
    ) -> None:
        super().__init__()
        self.patch = nn.Conv2d(channels, dimension, kernel_size=4, stride=4, bias=False)
        self.patch_norm = nn.BatchNorm2d(dimension)
        self.patch_neuron = LIFCell(tau, threshold, alpha)
        self.blocks = nn.ModuleList(
            [
                SpikingTransformerBlock(dimension, heads, mlp_ratio, tau, threshold, alpha)
                for _ in range(depth)
            ]
        )
        self.head = nn.Linear(dimension, classes)
        self.apply(initialize_module)

    def forward_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        patch_state = None
        block_states: List[Dict[str, torch.Tensor]] = [{} for _ in self.blocks]
        outputs = []
        for frame in inputs.unbind(dim=1):
            value = self.patch_norm(self.patch(frame))
            value, patch_state = self.patch_neuron(value, patch_state)
            value = value.flatten(2).transpose(1, 2)
            for index, block in enumerate(self.blocks):
                value, block_states[index] = block.step(value, block_states[index])
            outputs.append(self.head(value.mean(dim=1)))
        return torch.stack(outputs, dim=1)
