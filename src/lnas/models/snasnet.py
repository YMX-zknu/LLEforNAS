from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from .common import LIFCell, SpikingConvBlock, TemporalClassifier, initialize_module


class ZeroOp(nn.Module):
    def step(self, inputs: torch.Tensor, state: torch.Tensor | None):
        return torch.zeros_like(inputs), state


class IdentityOp(nn.Module):
    def step(self, inputs: torch.Tensor, state: torch.Tensor | None):
        return inputs, state


class PoolOp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def step(self, inputs: torch.Tensor, state: torch.Tensor | None):
        return self.pool(inputs), state


class SpikeConvOp(nn.Module):
    def __init__(self, channels: int, kernel: int, tau: float, threshold: float, alpha: float):
        super().__init__()
        self.neuron = LIFCell(tau, threshold, alpha)
        self.conv = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def step(self, inputs: torch.Tensor, state: torch.Tensor | None):
        spikes, state = self.neuron(inputs, state)
        return self.norm(self.conv(spikes)), state


def _operation(code: int, channels: int, tau: float, threshold: float, alpha: float) -> nn.Module:
    operations = {
        0: lambda: ZeroOp(),
        1: lambda: IdentityOp(),
        2: lambda: SpikeConvOp(channels, 1, tau, threshold, alpha),
        3: lambda: SpikeConvOp(channels, 3, tau, threshold, alpha),
        4: lambda: PoolOp(),
    }
    if code not in operations:
        raise ValueError(f"Invalid SNASNet operation code: {code}")
    return operations[code]()


class SearchCell(nn.Module):
    def __init__(
        self,
        channels: int,
        matrix: List[List[int]],
        tau: float,
        threshold: float,
        alpha: float,
    ) -> None:
        super().__init__()
        if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            raise ValueError("SNASNet connection matrix must be 4x4")
        self.matrix = matrix
        self.edges = nn.ModuleDict()
        for source in range(4):
            for target in range(4):
                if source != target:
                    key = f"{source}_{target}"
                    self.edges[key] = _operation(
                        int(matrix[source][target]), channels, tau, threshold, alpha
                    )

    def step(
        self,
        inputs: torch.Tensor,
        previous_nodes: List[torch.Tensor] | None,
        edge_states: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        previous_nodes = previous_nodes or [torch.zeros_like(inputs) for _ in range(4)]
        new_states: Dict[str, torch.Tensor] = {}
        recurrent_input = inputs
        for source in range(1, 4):
            key = f"{source}_0"
            contribution, state = self.edges[key].step(previous_nodes[source], edge_states.get(key))
            recurrent_input = recurrent_input + contribution
            if state is not None:
                new_states[key] = state
        nodes = [recurrent_input]
        for target in range(1, 4):
            value = torch.zeros_like(inputs)
            for source in range(4):
                if source == target:
                    continue
                source_value = nodes[source] if source < target else previous_nodes[source]
                key = f"{source}_{target}"
                contribution, state = self.edges[key].step(source_value, edge_states.get(key))
                value = value + contribution
                if state is not None:
                    new_states[key] = state
            nodes.append(value)
        return nodes[-1], nodes, new_states


class SNASNet(TemporalClassifier):
    def __init__(
        self,
        channels: int,
        classes: int,
        width: int,
        matrix: List[List[int]],
        tau: float,
        threshold: float,
        alpha: float,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1, bias=False), nn.BatchNorm2d(width)
        )
        self.cell1 = SearchCell(width, matrix, tau, threshold, alpha)
        self.down = SpikingConvBlock(width, width * 2, 3, 2, tau, threshold, alpha)
        self.cell2 = SearchCell(width * 2, matrix, tau, threshold, alpha)
        self.final_neuron = LIFCell(tau, threshold, alpha)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(width * 2, classes)
        self.apply(initialize_module)

    def forward_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(f"SNASNet expects [B,T,C,H,W], received {tuple(inputs.shape)}")
        cell1_nodes = None
        cell2_nodes = None
        cell1_states: Dict[str, torch.Tensor] = {}
        cell2_states: Dict[str, torch.Tensor] = {}
        down_state = None
        final_state = None
        outputs = []
        for frame in inputs.unbind(dim=1):
            value = self.stem(frame)
            value, cell1_nodes, cell1_states = self.cell1.step(value, cell1_nodes, cell1_states)
            value, down_state = self.down.step(value, down_state)
            value, cell2_nodes, cell2_states = self.cell2.step(value, cell2_nodes, cell2_states)
            value, final_state = self.final_neuron(value, final_state)
            outputs.append(self.head(self.pool(value).flatten(1)))
        return torch.stack(outputs, dim=1)
