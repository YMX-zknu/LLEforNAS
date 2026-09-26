"""Matrix-free finite-time state sensitivity of an untrained SNN."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

State = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class FTLEResult:
    estimate: float
    score: float
    eligible: bool
    directional_estimates: tuple[float, ...]
    state_elements: int
    timesteps: int
    warmup: int
    probes: int

    def to_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "score": self.score if self.eligible else None,
            "eligible": self.eligible,
            "directional_estimates": list(self.directional_estimates),
            "state_elements": self.state_elements,
            "timesteps": self.timesteps,
            "warmup": self.warmup,
            "probes": self.probes,
        }


def _norm(state: State) -> torch.Tensor:
    return torch.stack([value.float().square().sum() for value in state]).sum().sqrt()


def _initial_direction(state: State, epsilon: float) -> State:
    direction = tuple(torch.randn_like(value) for value in state)
    norm = _norm(direction).clamp_min(epsilon)
    return tuple(value / norm for value in direction)


def estimate_ftle(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    k: int = 8,
    warmup: int = 1,
    epsilon: float = 1e-8,
) -> FTLEResult:
    """Estimate the largest finite-time growth rate from K random directions.

    The model must expose step(frame, state) -> (output, tuple_of_states).
    Inputs use [batch, time, channels, height, width]. No weights are trained.
    """
    if not callable(getattr(model, "step", None)):
        raise TypeError("model must implement step(frame, state)")
    if inputs.ndim != 5:
        raise ValueError("inputs must have shape [batch, time, channels, height, width]")
    if k < 1 or not 1 <= warmup < inputs.shape[1]:
        raise ValueError("require k >= 1 and 1 <= warmup < timesteps")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    model.eval()
    frames = inputs.detach().unbind(dim=1)
    state = None
    with torch.no_grad():
        for frame in frames[:warmup]:
            _, state = model.step(frame, state)
    if not isinstance(state, tuple) or not state or any(value is None for value in state):
        raise ValueError("step must return a nonempty tuple of recurrent state tensors")
    state = tuple(value.detach() for value in state)
    directions = [_initial_direction(state, epsilon) for _ in range(k)]
    accumulated = torch.zeros(k, device=inputs.device, dtype=torch.float64)

    for frame in frames[warmup:]:
        def transition(*current: torch.Tensor, current_frame: torch.Tensor = frame) -> State:
            _, updated = model.step(current_frame, tuple(current))
            return updated

        next_directions = []
        next_state = None
        for index, direction in enumerate(directions):
            primal, tangent = torch.autograd.functional.jvp(
                transition, state, direction, create_graph=False, strict=False
            )
            if next_state is None:
                next_state = tuple(value.detach() for value in primal)
            growth = _norm(tangent).clamp_min(epsilon)
            accumulated[index] += growth.double().log()
            next_directions.append(tuple(value.detach() / growth for value in tangent))
        state = next_state
        directions = next_directions

    transitions = inputs.shape[1] - warmup
    exponents = accumulated / transitions
    if not bool(torch.isfinite(exponents).all()):
        raise FloatingPointError("FTLE evaluation produced a nonfinite exponent")
    directional = tuple(float(value) for value in exponents.cpu())
    estimate = max(directional)
    eligible = estimate < 0.0
    return FTLEResult(
        estimate=estimate,
        score=estimate if eligible else -math.inf,
        eligible=eligible,
        directional_estimates=directional,
        state_elements=sum(value.numel() for value in state),
        timesteps=inputs.shape[1],
        warmup=warmup,
        probes=k,
    )
