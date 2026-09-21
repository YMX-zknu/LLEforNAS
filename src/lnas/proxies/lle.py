from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from .base import Proxy
from .result import ProxyResult

State = Tuple[torch.Tensor, ...]


def _state_norm(state: State) -> torch.Tensor:
    squared = torch.stack([value.float().square().sum() for value in state]).sum()
    return squared.sqrt()


def _normalized_random_state(state: State, epsilon: float) -> State:
    direction = tuple(torch.randn_like(value) for value in state)
    norm = _state_norm(direction).clamp_min(epsilon)
    return tuple(value / norm for value in direction)


class LLEProxy(Proxy):
    name = "lle"

    def __init__(self, probes: int = 1, warmup_steps: int = 1, epsilon: float = 1e-8):
        if probes < 1:
            raise ValueError("probes must be positive")
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        self.probes = probes
        self.warmup_steps = warmup_steps
        self.epsilon = epsilon

    def score(self, model: nn.Module, inputs: torch.Tensor) -> ProxyResult:
        if not hasattr(model, "step"):
            raise TypeError("LLE requires a model exposing step(frame, state)")
        if inputs.ndim != 5:
            raise ValueError(f"LLE expects [B,T,C,H,W], received {tuple(inputs.shape)}")
        if inputs.shape[1] <= self.warmup_steps:
            raise ValueError("timesteps must be greater than proxy.warmup_steps")

        model.eval()
        frames = inputs.detach().unbind(dim=1)
        state = None
        with torch.no_grad():
            for frame in frames[: self.warmup_steps]:
                _output, state = model.step(frame, state)
        state = tuple(value.detach() for value in state)
        directions = [_normalized_random_state(state, self.epsilon) for _ in range(self.probes)]
        log_growth = torch.zeros(self.probes, device=inputs.device, dtype=torch.float64)

        for frame in frames[self.warmup_steps :]:
            next_state = None
            next_directions: List[State] = []

            def transition(*values: torch.Tensor, current_frame: torch.Tensor = frame) -> State:
                _output, updated = model.step(current_frame, tuple(values))
                return updated

            for index, direction in enumerate(directions):
                primal, tangent = torch.autograd.functional.jvp(
                    transition,
                    state,
                    direction,
                    create_graph=False,
                    strict=False,
                )
                if next_state is None:
                    next_state = tuple(value.detach() for value in primal)
                growth = _state_norm(tangent).clamp_min(self.epsilon)
                log_growth[index] += growth.double().log()
                next_directions.append(tuple(value.detach() / growth for value in tangent))
            state = next_state
            directions = next_directions

        transitions = inputs.shape[1] - self.warmup_steps
        exponents = log_growth / float(transitions)
        raw = float(exponents.max().cpu())
        score = -abs(raw)
        state_elements = sum(value.numel() for value in state)
        return ProxyResult(
            name=self.name,
            score=score,
            raw_value=raw,
            stable=raw <= 0.0,
            details={
                "minimum_directional_exponent": float(exponents.min().cpu()),
                "mean_directional_exponent": float(exponents.mean().cpu()),
                "maximum_directional_exponent": raw,
                "probes": float(self.probes),
                "warmup_steps": float(self.warmup_steps),
                "transitions": float(transitions),
                "state_tensors": float(len(state)),
                "state_elements": float(state_elements),
            },
        )
