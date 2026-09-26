"""Random and evolutionary joint architecture and timestep search."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .ftle import FTLEResult
from .spaces import candidate_key


@dataclass(frozen=True)
class SearchRecord:
    architecture: dict
    timesteps: int
    result: FTLEResult

    def to_dict(self) -> dict:
        return {
            "architecture": self.architecture,
            "timesteps": self.timesteps,
            "ftle": self.result.to_dict(),
        }


@dataclass(frozen=True)
class SearchResult:
    best: SearchRecord | None
    records: tuple[SearchRecord, ...]


def _rank(record: SearchRecord) -> tuple:
    result = record.result
    return (result.eligible, result.score if result.eligible else -result.estimate)


def run_jatst(
    space,
    evaluate: Callable[[dict, int], FTLEResult],
    *,
    timesteps: tuple[int, ...],
    budget: int,
    method: str = "random",
    seed: int = 2025,
    population_size: int = 20,
    parents: int = 10,
) -> SearchResult:
    """Evaluate distinct architecture-timestep pairs under a common budget."""
    values = tuple(sorted(set(timesteps)))
    if budget < 1 or not values or any(t < 2 for t in values):
        raise ValueError("require budget >= 1 and timesteps >= 2")
    if method not in {"random", "evolution"}:
        raise ValueError("method must be random or evolution")
    if population_size < 2 or parents < 2 or parents > population_size:
        raise ValueError("require population_size >= parents >= 2")
    rng = random.Random(seed)
    seen: set[str] = set()
    history: list[SearchRecord] = []

    def evaluate_once(architecture: dict, timestep: int) -> bool:
        key = candidate_key(architecture, timestep)
        if key in seen:
            return False
        seen.add(key)
        history.append(SearchRecord(architecture, timestep, evaluate(architecture, timestep)))
        return True

    while len(history) < budget:
        accepted = False
        for _ in range(5000):
            if method == "random" or len(history) < min(population_size, budget):
                architecture, timestep = space.sample(rng), rng.choice(values)
            else:
                selected = sorted(history, key=_rank, reverse=True)[: min(parents, len(history))]
                if rng.random() < 0.5:
                    parent = rng.choice(selected)
                    architecture, timestep = parent.architecture, parent.timesteps
                    if rng.random() < 0.5:
                        architecture = space.mutate(architecture, rng)
                    else:
                        timestep = rng.choice([v for v in values if v != timestep] or values)
                else:
                    left, right = rng.sample(selected, 2)
                    architecture = space.crossover(left.architecture, right.architecture, rng)
                    timestep = rng.choice([left.timesteps, right.timesteps])
            if evaluate_once(architecture, timestep):
                accepted = True
                break
        if not accepted:
            raise RuntimeError("Unable to propose an unevaluated candidate pair")

    eligible = [record for record in history if record.result.eligible]
    best = max(eligible, key=lambda record: record.result.score, default=None)
    return SearchResult(best=best, records=tuple(history))
