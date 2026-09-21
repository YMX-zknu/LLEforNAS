from lnas.search.engine import SearchRecord
from lnas.search.jatst import run_jatst
from lnas.search.spaces import FixedSetSpace


def _evaluate(architecture, timestep):
    score = -abs(timestep - 6)
    return SearchRecord(architecture, score, score, score <= 0, {})


def test_random_jatst_searches_architecture_timestep_pairs():
    records = run_jatst(
        FixedSetSpace(["a", "b"]),
        _evaluate,
        candidates=6,
        timesteps=[2, 4, 6],
        method="random",
        seed=3,
    )
    assert len(records) == 6
    assert records[0].details["timestep"] == 6


def test_evolutionary_jatst_returns_ranked_pairs():
    records = run_jatst(
        FixedSetSpace(["a", "b", "c"]),
        _evaluate,
        candidates=7,
        timesteps=[2, 4, 6, 8],
        method="evolution",
        seed=5,
    )
    assert len(records) == 7
    assert records == sorted(records, key=lambda item: item.score, reverse=True)
