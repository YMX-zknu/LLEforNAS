from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import nn

State = Tuple[torch.Tensor, ...]


def _norm(state: State) -> torch.Tensor:
    return torch.stack([value.float().square().sum() for value in state]).sum().sqrt()


def finite_perturbation_growth(
    model: nn.Module,
    inputs: torch.Tensor,
    epsilon: float = 1e-3,
    directions: int = 3,
    warmup_steps: int = 1,
    numerical_epsilon: float = 1e-12,
) -> Dict[str, List[float]]:
    if inputs.ndim != 5:
        raise ValueError(f"Expected [B,T,C,H,W], received {tuple(inputs.shape)}")
    if inputs.shape[1] <= warmup_steps:
        raise ValueError("timesteps must be greater than warmup_steps")
    if epsilon <= 0 or directions < 1:
        raise ValueError("epsilon and directions must be positive")

    model.eval()
    frames = inputs.detach().unbind(dim=1)
    state = None
    with torch.no_grad():
        for frame in frames[:warmup_steps]:
            _output, state = model.step(frame, state)
    reference = tuple(value.detach() for value in state)
    reference_norm = float(_norm(reference).cpu())
    if reference_norm <= numerical_epsilon:
        reference_norm = math.sqrt(sum(value.numel() for value in reference))
    initial_norm = epsilon * reference_norm

    trajectories = []
    for _ in range(directions):
        direction = tuple(torch.randn_like(value) for value in reference)
        direction_norm = _norm(direction).clamp_min(numerical_epsilon)
        perturbed = tuple(
            value + initial_norm * delta / direction_norm
            for value, delta in zip(reference, direction)
        )
        base_state = reference
        perturbed_state = perturbed
        values = []
        with torch.no_grad():
            for elapsed, frame in enumerate(frames[warmup_steps:], start=1):
                _base_output, base_state = model.step(frame, base_state)
                _perturbed_output, perturbed_state = model.step(frame, perturbed_state)
                difference = tuple(
                    shifted - base for shifted, base in zip(perturbed_state, base_state)
                )
                ratio = float(_norm(difference).cpu()) / max(initial_norm, numerical_epsilon)
                values.append(math.log(max(ratio, numerical_epsilon)) / elapsed)
        trajectories.append(values)

    tensor = torch.tensor(trajectories, dtype=torch.float64)
    return {
        "mean": tensor.mean(dim=0).tolist(),
        "std": tensor.std(dim=0, unbiased=False).tolist(),
        "terminal": tensor[:, -1].tolist(),
    }
