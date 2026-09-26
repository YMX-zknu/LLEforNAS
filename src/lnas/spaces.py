"""Candidate variables from the SNASNet and AutoST search spaces."""

from __future__ import annotations

import copy
import json
import random


class SNASNetSpace:
    name = "snasnet"

    @staticmethod
    def _path(matrix: list[list[int]], start: int, goal: int) -> bool:
        visited = {start}
        pending = [start]
        while pending:
            source = pending.pop()
            if source == goal:
                return True
            for target, operation in enumerate(matrix[source]):
                if operation and target not in visited:
                    visited.add(target)
                    pending.append(target)
        return False

    @classmethod
    def _prune(cls, matrix: list[list[int]]) -> list[list[int]]:
        for node in (1, 2):
            if not cls._path(matrix, 0, node) or not cls._path(matrix, node, 3):
                for neighbor in range(4):
                    matrix[node][neighbor] = 0
                    matrix[neighbor][node] = 0
        return matrix

    def sample(self, rng: random.Random) -> dict:
        while True:
            matrix = [[0] * 4 for _ in range(4)]
            for source in range(4):
                for target in range(source + 1, 4):
                    if rng.randrange(2):
                        matrix[source][target] = rng.randrange(5)
                    else:
                        matrix[target][source] = rng.randrange(5)
            if matrix[0][3]:
                return {"matrix": self._prune(matrix)}

    def mutate(self, architecture: dict, rng: random.Random) -> dict:
        child = copy.deepcopy(architecture)
        source, target = rng.choice([
            (i, j) for i in range(4) for j in range(i + 1, 4)
        ])
        options = [(value, 0) for value in range(1, 5)]
        if (source, target) != (0, 3):
            options += [(0, 0)] + [(0, value) for value in range(1, 5)]
        old = (child["matrix"][source][target], child["matrix"][target][source])
        forward, backward = rng.choice([pair for pair in options if pair != old])
        child["matrix"][source][target] = forward
        child["matrix"][target][source] = backward
        child["matrix"] = self._prune(child["matrix"])
        return child

    def crossover(self, first: dict, second: dict, rng: random.Random) -> dict:
        matrix = [[0] * 4 for _ in range(4)]
        for source in range(4):
            for target in range(source + 1, 4):
                parent = first if rng.random() < 0.5 else second
                matrix[source][target] = parent["matrix"][source][target]
                matrix[target][source] = parent["matrix"][target][source]
        return {"matrix": self._prune(matrix)}


class AutoSTSpace:
    name = "autost"
    choices = {
        "dimension": (192, 256, 320, 384),
        "depth": tuple(range(1, 11)),
        "heads": (4, 8),
        "mlp_ratio": (3, 4, 5),
    }

    def sample(self, rng: random.Random) -> dict:
        depth = rng.choice(self.choices["depth"])
        return {
            "dimension": rng.choice(self.choices["dimension"]),
            "depth": depth,
            "heads": [rng.choice(self.choices["heads"]) for _ in range(depth)],
            "mlp_ratio": [rng.choice(self.choices["mlp_ratio"]) for _ in range(depth)],
        }

    def mutate(self, architecture: dict, rng: random.Random) -> dict:
        child = copy.deepcopy(architecture)
        key = rng.choice(list(self.choices))
        if key == "depth":
            depth = rng.choice([v for v in self.choices[key] if v != child[key]])
            for field in ("heads", "mlp_ratio"):
                child[field] = child[field][:depth]
                child[field].extend(
                    rng.choice(self.choices[field]) for _ in range(depth - len(child[field]))
                )
            child[key] = depth
        elif key in {"heads", "mlp_ratio"}:
            index = rng.randrange(child["depth"])
            child[key][index] = rng.choice(
                [v for v in self.choices[key] if v != child[key][index]]
            )
        else:
            child[key] = rng.choice([v for v in self.choices[key] if v != child[key]])
        return child

    def crossover(self, first: dict, second: dict, rng: random.Random) -> dict:
        depth = rng.choice([first["depth"], second["depth"]])
        child = {"dimension": rng.choice([first["dimension"], second["dimension"]]),
                 "depth": depth}
        for field in ("heads", "mlp_ratio"):
            child[field] = [
                rng.choice([parent[field][i] for parent in (first, second)
                            if i < parent["depth"]])
                for i in range(depth)
            ]
        return child


def get_space(name: str):
    spaces = {"snasnet": SNASNetSpace, "autost": AutoSTSpace}
    if name not in spaces:
        raise ValueError(f"space must be one of {', '.join(spaces)}")
    return spaces[name]()


def candidate_key(architecture: dict, timestep: int) -> str:
    return json.dumps([architecture, timestep], sort_keys=True, separators=(",", ":"))
