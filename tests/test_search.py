import math
import random

from lnas.ftle import FTLEResult
from lnas.search import run_jatst
from lnas.spaces import AutoSTSpace, SNASNetSpace, candidate_key


def result(value):
    return FTLEResult(value, value if value < 0 else -math.inf, value < 0,
                      (value,), 2, 4, 1, 8)


def test_random_joint_search_chooses_negative_nearest_zero():
    space = SNASNetSpace()
    calls = []

    def evaluate(architecture, timestep):
        calls.append((architecture, timestep))
        return result((-0.5, 0.3, -0.01)[len(calls) - 1])

    search = run_jatst(space, evaluate, timesteps=(2, 4), budget=3, seed=7)
    assert search.best.result.estimate == -0.01
    assert len({candidate_key(*candidate) for candidate in calls}) == 3
    assert {record.timesteps for record in search.records} <= {2, 4}


def test_no_negative_candidates_returns_no_selection():
    search = run_jatst(AutoSTSpace(), lambda a, t: result(0.1),
                       timesteps=(2, 4), budget=3)
    assert search.best is None
    assert len(search.records) == 3


def test_evolutionary_search_uses_one_evaluation_per_pair():
    search = run_jatst(AutoSTSpace(), lambda a, t: result(-0.1),
                       timesteps=(2, 4), budget=23, method="evolution", seed=9)
    assert len(search.records) == 23
    assert len({candidate_key(r.architecture, r.timesteps) for r in search.records}) == 23


def test_autost_operator_choices_follow_depth():
    space = AutoSTSpace()
    assert space.choices == {
        "dimension": (192, 256, 320, 384),
        "depth": tuple(range(1, 11)),
        "heads": (4, 8),
        "mlp_ratio": (3, 4, 5),
    }
    rng = random.Random(19)
    first = space.sample(rng)
    second = space.sample(rng)
    for architecture in (first, space.mutate(first, rng), space.crossover(first, second, rng)):
        assert len(architecture["heads"]) == architecture["depth"]
        assert len(architecture["mlp_ratio"]) == architecture["depth"]
        assert all(architecture["dimension"] % h == 0 for h in architecture["heads"])


def test_snasnet_source_connection_constraints():
    space = SNASNetSpace()
    rng = random.Random(19)
    first = space.sample(rng)
    second = space.sample(rng)
    candidates = [space.sample(rng) for _ in range(100)]
    candidates.extend((first, space.mutate(first, rng), space.crossover(first, second, rng)))
    for candidate in candidates:
        matrix = candidate["matrix"]
        assert matrix[0][3] in (1, 2, 3, 4)
        assert all(not (matrix[i][j] and matrix[j][i])
                   for i in range(4) for j in range(i + 1, 4))
        for node in (1, 2):
            active = any(matrix[node]) or any(row[node] for row in matrix)
            if active:
                assert space._path(matrix, 0, node)
                assert space._path(matrix, node, 3)
