"""Candidate variables from the SNASNet and AutoST search spaces."""

from __future__ import annotations

import copy
import json
import random


class SNASNetSpace:
    name = "snasnet"

    def sample(self, rng: random.Random) -> dict:
        matrix = [
            [0 if i == j else rng.randrange(5) for j in range(4)]
            for i in range(4)
        ]
        matrix[0][3] = rng.randrange(1, 5)
        return {"matrix": matrix}

    def mutate(self, architecture: dict, rng: random.Random) -> dict:
        child = copy.deepcopy(architecture)
        i, j = rng.choice([(i, j) for i in range(4) for j in range(4) if i != j])
        choices = list(range(1, 5)) if (i, j) == (0, 3) else list(range(5))
        child["matrix"][i][j] = rng.choice(
            [value for value in choices if value != child["matrix"][i][j]]
        )
        return child

    def crossover(self, first: dict, second: dict, rng: random.Random) -> dict:
        matrix = [
            [a if rng.random() < 0.5 else b for a, b in zip(row_a, row_b, strict=True)]
            for row_a, row_b in zip(first["matrix"], second["matrix"], strict=True)
        ]
        return {"matrix": matrix}


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
