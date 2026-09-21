from lnas.search.jatst import DiscreteBayesOptimizer


def test_discrete_optimizer_finds_best_sampled_value():
    optimizer = DiscreteBayesOptimizer(2, 8, 7, seed=3)
    timestep, score = optimizer.optimize(lambda value: -abs(value - 6))
    assert timestep == 6
    assert score == 0
