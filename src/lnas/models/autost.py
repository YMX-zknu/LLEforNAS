from __future__ import annotations

from typing import Dict, List, Tuple

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
        self.heads = heads
        self.head_dimension = dimension // heads
        self.norm1 = nn.LayerNorm(dimension)
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.attention_projection = nn.Linear(dimension, dimension)
        self.norm2 = nn.LayerNorm(dimension)
        hidden = dimension * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, dimension)
        )
        self.neuron1 = LIFCell(tau, threshold, alpha)
        self.neuron2 = LIFCell(tau, threshold, alpha)

    def step(self, inputs: torch.Tensor, state: Dict[str, torch.Tensor]):
        normalized = self.norm1(inputs)
        batch, tokens, dimension = normalized.shape
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        query = query.reshape(batch, tokens, self.heads, self.head_dimension).transpose(1, 2)
        key = key.reshape(batch, tokens, self.heads, self.head_dimension).transpose(1, 2)
        value = value.reshape(batch, tokens, self.heads, self.head_dimension).transpose(1, 2)
        scale = self.head_dimension**-0.5
        attention = (query @ key.transpose(-2, -1) * scale).softmax(dim=-1)
        attended = (attention @ value).transpose(1, 2).reshape(batch, tokens, dimension)
        attended = self.attention_projection(attended)
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

    def step(
        self, frame: torch.Tensor, state: Tuple[torch.Tensor, ...] | None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        previous = list(state) if state is not None else [None] * (1 + 2 * len(self.blocks))
        next_state: List[torch.Tensor] = []
        value = self.patch_norm(self.patch(frame))
        value, patch_state = self.patch_neuron(value, previous[0])
        next_state.append(patch_state)
        value = value.flatten(2).transpose(1, 2)
        offset = 1
        for block in self.blocks:
            block_state = {
                "attention": previous[offset],
                "mlp": previous[offset + 1],
            }
            value, updated = block.step(value, block_state)
            next_state.extend((updated["attention"], updated["mlp"]))
            offset += 2
        output = self.head(value.mean(dim=1))
        return output, tuple(next_state)
