from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List


@dataclass(frozen=True)
class SearchRecord:
    architecture: Dict[str, Any]
    score: float
    raw_value: float
    stable: bool
    details: Dict[str, float]

    def to_dict(self):
        return asdict(self)


def _unique_sample(space, rng, seen):
    for _ in range(1000):
        candidate = space.sample(rng)
        key = repr(candidate)
        if key not in seen:
            seen.add(key)
            return candidate
    raise RuntimeError("Could not sample a new architecture")


def run_search(
    space,
    evaluate: Callable[[Dict[str, Any]], SearchRecord],
    candidates: int,
    method: str,
    seed: int,
) -> List[SearchRecord]:
    if candidates < 1:
        raise ValueError("Search requires at least one candidate")
    candidates = min(candidates, space.size) if space.size is not None else candidates
    rng = random.Random(seed)
    seen = set()
    if method == "random":
        records = [evaluate(_unique_sample(space, rng, seen)) for _ in range(candidates)]
        return sorted(records, key=lambda item: item.score, reverse=True)
    if method != "evolution":
        raise ValueError("search.method must be random or evolution")
    population_size = min(20, candidates)
    records = [evaluate(_unique_sample(space, rng, seen)) for _ in range(population_size)]
    while len(records) < candidates:
        ranked = sorted(records, key=lambda item: item.score, reverse=True)
        parents = ranked[: max(2, population_size // 2)]
        if rng.random() < 0.5:
            parent = rng.choice(parents)
            child = space.mutate(parent.architecture, rng)
        else:
            first, second = rng.sample(parents, 2)
            child = space.crossover(first.architecture, second.architecture, rng)
        key = repr(child)
        if key in seen:
            continue
        seen.add(key)
        records.append(evaluate(child))
    return sorted(records, key=lambda item: item.score, reverse=True)
