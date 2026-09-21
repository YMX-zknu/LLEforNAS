from __future__ import annotations

import copy
import json
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class SearchSpace(ABC):
    size = None

    @abstractmethod
    def sample(self, rng: random.Random) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def mutate(self, architecture: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        raise NotImplementedError

    def crossover(
        self, first: Dict[str, Any], second: Dict[str, Any], rng: random.Random
    ) -> Dict[str, Any]:
        return copy.deepcopy(first if rng.random() < 0.5 else second)


class SNASSpace(SearchSpace):
    def sample(self, rng: random.Random) -> Dict[str, Any]:
        matrix = [[0 for _ in range(4)] for _ in range(4)]
        for source in range(4):
            for target in range(4):
                if source != target:
                    matrix[source][target] = rng.randrange(5)
        matrix[0][3] = rng.randrange(1, 5)
        return {"matrix": matrix}

    def mutate(self, architecture: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        result = copy.deepcopy(architecture)
        source, target = rng.choice([(i, j) for i in range(4) for j in range(4) if i != j])
        result["matrix"][source][target] = rng.randrange(5)
        if result["matrix"][0][3] == 0:
            result["matrix"][0][3] = rng.randrange(1, 5)
        return result

    def crossover(self, first, second, rng):
        matrix = []
        for row_first, row_second in zip(first["matrix"], second["matrix"]):
            matrix.append([a if rng.random() < 0.5 else b for a, b in zip(row_first, row_second)])
        if matrix[0][3] == 0:
            matrix[0][3] = rng.randrange(1, 5)
        return {"matrix": matrix}


class AutoSNNSpace(SearchSpace):
    candidates = [
        "SCB_k3",
        "SCB_k5",
        "SCB_k7",
        "SRB_k3",
        "SRB_k5",
        "SRB_k7",
        "SIB_k3_e1",
        "SIB_k3_e3",
        "SIB_k5_e1",
        "SIB_k5_e3",
        "skip_connect",
    ]
    pool_positions = {1, 4, 7}

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        blocks = [
            "max_pool_k2" if index in self.pool_positions else rng.choice(self.candidates)
            for index in range(8)
        ]
        return {"blocks": blocks}

    def mutate(self, architecture: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        result = copy.deepcopy(architecture)
        index = rng.choice([item for item in range(8) if item not in self.pool_positions])
        result["blocks"][index] = rng.choice(self.candidates)
        return result

    def crossover(self, first, second, rng):
        return {
            "blocks": [
                a if rng.random() < 0.5 else b for a, b in zip(first["blocks"], second["blocks"])
            ]
        }


class AutoSTSpace(SearchSpace):
    choices = {
        "dimension": [192, 256, 320, 384],
        "depth": list(range(1, 11)),
        "heads": [4, 8],
        "mlp_ratio": [3, 4, 5],
    }

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        while True:
            result = {key: rng.choice(value) for key, value in self.choices.items()}
            if result["dimension"] % result["heads"] == 0:
                return result

    def mutate(self, architecture: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        result = copy.deepcopy(architecture)
        key = rng.choice(list(self.choices))
        result[key] = rng.choice(self.choices[key])
        if result["dimension"] % result["heads"]:
            result["heads"] = 4
        return result

    def crossover(self, first, second, rng):
        result = {key: first[key] if rng.random() < 0.5 else second[key] for key in first}
        if result["dimension"] % result["heads"]:
            result["heads"] = 4
        return result


class FixedSetSpace(SearchSpace):
    def __init__(self, names: List[str]) -> None:
        self.names = names
        self.size = len(names)

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        return {"name": rng.choice(self.names)}

    def mutate(self, architecture: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        alternatives = [name for name in self.names if name != architecture["name"]]
        return {"name": rng.choice(alternatives or self.names)}


def get_search_space(name: str) -> SearchSpace:
    normalized = name.lower()
    if normalized == "snasnet":
        return SNASSpace()
    if normalized == "autosnn":
        return AutoSNNSpace()
    if normalized == "autost":
        return AutoSTSpace()
    if normalized == "hc-snn":
        return FixedSetSpace(
            [
                "spiking-vgg11",
                "spiking-vgg13",
                "spiking-vgg16",
                "spiking-resnet18",
                "spiking-resnet34",
                "spiking-resnet50",
                "sew-resnet18",
                "sew-resnet34",
                "sew-resnet50",
            ]
        )
    if normalized == "hc-st":
        return FixedSetSpace(["spikformer", "spikingformer", "spike-driven-transformer"])
    raise ValueError(f"Unknown search space: {name}")


def architecture_key(architecture: Dict[str, Any]) -> str:
    return json.dumps(architecture, sort_keys=True, separators=(",", ":"))
