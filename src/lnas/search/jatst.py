from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Sequence, Tuple

from .engine import SearchRecord
from .spaces import architecture_key

Candidate = Tuple[Dict[str, Any], int]


def _candidate_key(candidate: Candidate) -> str:
    architecture, timestep = candidate
    return f"{architecture_key(architecture)}@{timestep}"


def _sample_pair(space, timesteps: Sequence[int], rng: random.Random, seen: set) -> Candidate:
    for _ in range(2000):
        candidate = space.sample(rng), rng.choice(timesteps)
        key = _candidate_key(candidate)
        if key not in seen:
            seen.add(key)
            return candidate
    raise RuntimeError("Could not sample a new architecture-timestep pair")


def _record_with_timestep(record: SearchRecord, timestep: int) -> SearchRecord:
    details = dict(record.details)
    details["timestep"] = float(timestep)
    return SearchRecord(
        architecture=record.architecture,
        score=record.score,
        raw_value=record.raw_value,
        stable=record.stable,
        details=details,
    )


def run_jatst(
    space,
    evaluate: Callable[[Dict[str, Any], int], SearchRecord],
    candidates: int,
    timesteps: Sequence[int],
    method: str,
    seed: int,
) -> List[SearchRecord]:
    if candidates < 1:
        raise ValueError("JATST requires at least one candidate pair")
    if not timesteps:
        raise ValueError("JATST requires at least one timestep")
    if method not in {"random", "evolution"}:
        raise ValueError("search.method must be random or evolution")

    values = tuple(sorted(set(int(value) for value in timesteps)))
    if space.size is not None:
        candidates = min(candidates, space.size * len(values))
    rng = random.Random(seed)
    seen = set()

    def evaluate_pair(candidate: Candidate) -> SearchRecord:
        architecture, timestep = candidate
        return _record_with_timestep(evaluate(architecture, timestep), timestep)

    if method == "random":
        records = [
            evaluate_pair(_sample_pair(space, values, rng, seen)) for _ in range(candidates)
        ]
        return sorted(records, key=lambda item: item.score, reverse=True)

    population_size = min(20, candidates)
    records = [
        evaluate_pair(_sample_pair(space, values, rng, seen)) for _ in range(population_size)
    ]
    pairs = [(record.architecture, int(record.details["timestep"])) for record in records]
    failed_attempts = 0
    while len(records) < candidates:
        order = sorted(range(len(records)), key=lambda index: records[index].score, reverse=True)
        parent_indices = order[: max(2, min(population_size, len(order)) // 2)]
        if rng.random() < 0.5:
            architecture, timestep = pairs[rng.choice(parent_indices)]
            if rng.random() < 0.5:
                child = space.mutate(architecture, rng), timestep
            else:
                alternatives = [value for value in values if value != timestep]
                child = architecture, rng.choice(alternatives or values)
        else:
            first, second = rng.sample(parent_indices, 2)
            first_arch, first_timestep = pairs[first]
            second_arch, second_timestep = pairs[second]
            child = (
                space.crossover(first_arch, second_arch, rng),
                first_timestep if rng.random() < 0.5 else second_timestep,
            )
        key = _candidate_key(child)
        if key in seen:
            failed_attempts += 1
            if failed_attempts >= 1000:
                child = _sample_pair(space, values, rng, seen)
                pairs.append(child)
                records.append(evaluate_pair(child))
                failed_attempts = 0
            continue
        failed_attempts = 0
        seen.add(key)
        pairs.append(child)
        records.append(evaluate_pair(child))
    return sorted(records, key=lambda item: item.score, reverse=True)
