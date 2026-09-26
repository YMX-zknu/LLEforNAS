"""State-explicit spiking Transformer for AutoST architecture variables."""

from __future__ import annotations

import torch
from torch import nn

from .common import LIFCell, TemporalClassifier, initialize_module


class SpikingTransformerBlock(nn.Module):
    def __init__(
        self, dimension: int, heads: int, mlp_ratio: int,
        tau: float, threshold: float, alpha: float,
    ) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("embedding dimension must be divisible by attention heads")
        self.heads = heads
        self.head_dim = dimension // heads
        self.pre_neuron = LIFCell(tau, threshold, alpha)
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.q_neuron = LIFCell(tau, threshold, alpha)
        self.k_neuron = LIFCell(tau, threshold, alpha)
        self.v_neuron = LIFCell(tau, threshold, alpha)
        self.attention_neuron = LIFCell(tau, 0.5, alpha)
        self.projection = nn.Linear(dimension, dimension)
        self.mlp_neuron1 = LIFCell(tau, threshold, alpha)
        self.fc1 = nn.Linear(dimension, dimension * mlp_ratio)
        self.mlp_neuron2 = LIFCell(tau, threshold, alpha)
        self.fc2 = nn.Linear(dimension * mlp_ratio, dimension)

    def step(self, inputs: torch.Tensor, previous: tuple) -> tuple:
        batch, tokens, dimension = inputs.shape
        source, s0 = self.pre_neuron(inputs, previous[0])
        q, k, v = self.qkv(source).chunk(3, dim=-1)
        q, s1 = self.q_neuron(q, previous[1])
        k, s2 = self.k_neuron(k, previous[2])
        v, s3 = self.v_neuron(v, previous[3])
        q = q.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        attended = (q @ k.transpose(-2, -1)) * 0.125
        attended = (attended @ v).transpose(1, 2).reshape(batch, tokens, dimension)
        attended, s4 = self.attention_neuron(attended, previous[4])
        value = inputs + self.projection(attended)
        hidden, s5 = self.mlp_neuron1(value, previous[5])
        hidden = self.fc1(hidden)
        hidden, s6 = self.mlp_neuron2(hidden, previous[6])
        return value + self.fc2(hidden), (s0, s1, s2, s3, s4, s5, s6)


class AutoST(TemporalClassifier):
    def __init__(
        self, channels: int, classes: int, dimension: int, depth: int,
        heads: int | list[int], mlp_ratio: int | list[int],
        tau: float, threshold: float, alpha: float,
    ) -> None:
        super().__init__()
        if dimension % 8:
            raise ValueError("embedding dimension must be divisible by eight")
        self.stem1 = nn.Conv2d(channels, dimension // 4, 3, stride=2, padding=1, bias=False)
        self.stem1_norm = nn.BatchNorm2d(dimension // 4)
        self.stem1_neuron = LIFCell(tau, threshold, alpha)
        self.stem2 = nn.Conv2d(dimension // 4, dimension, 3, stride=2, padding=1, bias=False)
        self.stem2_norm = nn.BatchNorm2d(dimension)
        self.stem2_neuron = LIFCell(tau, threshold, alpha)
        heads = [heads] * depth if isinstance(heads, int) else heads
        mlp_ratio = [mlp_ratio] * depth if isinstance(mlp_ratio, int) else mlp_ratio
        if len(heads) != depth or len(mlp_ratio) != depth:
            raise ValueError("heads and mlp_ratio must have one entry per block")
        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(dimension, h, ratio, tau, threshold, alpha)
            for h, ratio in zip(heads, mlp_ratio, strict=True)
        ])
        self.head = nn.Linear(dimension, classes)
        self.apply(initialize_module)

    def step(self, frame: torch.Tensor, state: tuple | None) -> tuple:
        previous = state if state is not None else (None,) * (2 + 7 * len(self.blocks))
        value, s0 = self.stem1_neuron(self.stem1_norm(self.stem1(frame)), previous[0])
        value, s1 = self.stem2_neuron(self.stem2_norm(self.stem2(value)), previous[1])
        value = value.flatten(2).transpose(1, 2)
        next_state = [s0, s1]
        for index, block in enumerate(self.blocks):
            offset = 2 + 7 * index
            value, states = block.step(value, previous[offset:offset + 7])
            next_state.extend(states)
        return self.head(value.mean(dim=1)), tuple(next_state)
