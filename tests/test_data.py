import torch

from lnas.data import EvaluationBatch


def test_synthetic_frames_are_stable_for_a_timestep_count():
    device = torch.device("cpu")
    first = EvaluationBatch("synthetic", "data", 1, 8, 2, 7, device).get(2)
    joint = EvaluationBatch("synthetic", "data", 1, 8, 14, 7, device).get(2)
    assert torch.equal(first, joint)
