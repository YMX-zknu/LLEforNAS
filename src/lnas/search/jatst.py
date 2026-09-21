from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .engine import SearchRecord


class DiscreteBayesOptimizer:
    def __init__(self, lower: int, upper: int, trials: int, seed: int) -> None:
        self.values = list(range(lower, upper + 1))
        self.trials = min(trials, len(self.values))
        self.rng = random.Random(seed)

    @staticmethod
    def _kernel(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        distance = (first[:, None] - second[None, :]) ** 2
        return np.exp(-0.5 * distance / 4.0)

    def _propose(self, observed_x: List[int], observed_y: List[float]) -> int:
        available = [value for value in self.values if value not in observed_x]
        if len(observed_x) < 2:
            return self.rng.choice(available)
        x = np.asarray(observed_x, dtype=float)
        y = np.asarray(observed_y, dtype=float)
        query = np.asarray(available, dtype=float)
        covariance = self._kernel(x, x) + np.eye(len(x)) * 1e-6
        inverse = np.linalg.inv(covariance)
        cross = self._kernel(query, x)
        mean = cross @ inverse @ y
        variance = 1.0 - np.sum((cross @ inverse) * cross, axis=1)
        std = np.sqrt(np.clip(variance, 1e-12, None))
        improvement = mean - y.max()
        z = improvement / std
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
        expected = improvement * cdf + std * pdf
        return available[int(np.argmax(expected))]

    def optimize(self, objective: Callable[[int], float]) -> Tuple[int, float]:
        observed_x: List[int] = []
        observed_y: List[float] = []
        initial = [self.values[0], self.values[len(self.values) // 2], self.values[-1]]
        for timestep in initial[: self.trials]:
            if timestep not in observed_x:
                observed_x.append(timestep)
                observed_y.append(objective(timestep))
        while len(observed_x) < self.trials:
            timestep = self._propose(observed_x, observed_y)
            observed_x.append(timestep)
            observed_y.append(objective(timestep))
        best = int(np.argmax(observed_y))
        return observed_x[best], observed_y[best]


def run_jatst(
    space,
    evaluate: Callable[[Dict[str, Any], int], SearchRecord],
    candidates: int,
    timestep_min: int,
    timestep_max: int,
    timestep_trials: int,
    seed: int,
):
    rng = random.Random(seed)
    records = []
    seen = set()
    candidates = min(candidates, space.size) if space.size is not None else candidates
    for index in range(candidates):
        architecture = space.sample(rng)
        while repr(architecture) in seen:
            architecture = space.sample(rng)
        seen.add(repr(architecture))
        cache = {}

        def objective(timestep: int, architecture=architecture, cache=cache) -> float:
            record = evaluate(architecture, timestep)
            cache[timestep] = record
            return record.score

        optimizer = DiscreteBayesOptimizer(
            timestep_min, timestep_max, timestep_trials, seed + index
        )
        timestep, _score = optimizer.optimize(objective)
        record = cache[timestep]
        details = dict(record.details)
        details["timestep"] = float(timestep)
        records.append(
            SearchRecord(
                architecture=record.architecture,
                score=record.score,
                raw_value=record.raw_value,
                stable=record.stable,
                details=details,
            )
        )
    return sorted(records, key=lambda item: item.score, reverse=True)
