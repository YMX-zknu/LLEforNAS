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
        if any(matrix[i][i] != 0 for i in range(4)):
            raise ValueError("SNASNet matrix diagonal must be zero")
        if any(matrix[i][j] not in range(5) for i in range(4) for j in range(4)):
            raise ValueError("SNASNet operation codes must be integers from 0 to 4")
        if matrix[0][3] == 0:
            raise ValueError("SNASNet requires a direct input-to-output connection")
        if any(matrix[i][j] and matrix[j][i] for i in range(4) for j in range(i + 1, 4)):
            raise ValueError("SNASNet allows at most one direction per node pair")
        self.matrix = matrix
        self.edges = nn.ModuleDict()
        for source in range(4):
            for target in range(4):
                if source != target:
                    key = f"{source}_{target}"
                    self.edges[key] = _operation(
                        int(matrix[source][target]), channels, tau, threshold, alpha
                    )
        self.state_keys = [
            key for key, operation in self.edges.items() if isinstance(operation, SpikeConvOp)
        ]

    def step(
        self,
        inputs: torch.Tensor,
        previous_feedback: List[torch.Tensor] | None,
        edge_states: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        if previous_feedback is None:
            previous_feedback = [torch.zeros_like(inputs) for _ in range(3)]
        new_states: Dict[str, torch.Tensor] = {}

        def apply(source: int, target: int, value: torch.Tensor) -> torch.Tensor:
            key = f"{source}_{target}"
            contribution, state = self.edges[key].step(value, edge_states.get(key))
            if state is not None:
                new_states[key] = state
            return contribution

        base = inputs + previous_feedback[0]
        node1 = apply(0, 1, base)
        delayed1 = node1 + previous_feedback[1]
        node2 = apply(0, 2, base) + apply(1, 2, delayed1)
        delayed2 = node2 + previous_feedback[2]
        node3 = apply(0, 3, base) + apply(1, 3, delayed1) + apply(2, 3, delayed2)
        feedback = [
            apply(1, 0, delayed1) + apply(2, 0, delayed2) + apply(3, 0, node3),
            apply(2, 1, delayed2) + apply(3, 1, node3),
            apply(3, 2, node3),
        ]
        return node3, feedback, new_states


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

    def step(
        self, frame: torch.Tensor, state: Tuple[torch.Tensor, ...] | None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        count1 = len(self.cell1.state_keys)
        count2 = len(self.cell2.state_keys)
        if state is None:
            cell1_feedback = None
            cell1_states: Dict[str, torch.Tensor] = {}
            down_state = None
            cell2_feedback = None
            cell2_states: Dict[str, torch.Tensor] = {}
            final_state = None
        else:
            offset = 0
            cell1_feedback = list(state[offset : offset + 3])
            offset += 3
            cell1_states = dict(
                zip(self.cell1.state_keys, state[offset : offset + count1], strict=True)
            )
            offset += count1
            down_state = state[offset]
            offset += 1
            cell2_feedback = list(state[offset : offset + 3])
            offset += 3
            cell2_states = dict(
                zip(self.cell2.state_keys, state[offset : offset + count2], strict=True)
            )
            offset += count2
            final_state = state[offset]

        value = self.stem(frame)
        value, cell1_feedback, cell1_states = self.cell1.step(
            value, cell1_feedback, cell1_states
        )
        value, down_state = self.down.step(value, down_state)
        value, cell2_feedback, cell2_states = self.cell2.step(
            value, cell2_feedback, cell2_states
        )
        value, final_state = self.final_neuron(value, final_state)
        output = self.head(self.pool(value).flatten(1))
        next_state = (
            *cell1_feedback,
            *(cell1_states[key] for key in self.cell1.state_keys),
            down_state,
            *cell2_feedback,
            *(cell2_states[key] for key in self.cell2.state_keys),
            final_state,
        )
        return output, next_state
