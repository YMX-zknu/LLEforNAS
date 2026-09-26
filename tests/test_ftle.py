import math

import torch
from torch import nn

from lnas.ftle import estimate_ftle


class LinearState(nn.Module):
    def __init__(self, eigenvalues):
        super().__init__()
        self.register_buffer("matrix", torch.diag(torch.tensor(eigenvalues)))

    def step(self, frame, state):
        if state is None:
            state = (torch.ones(frame.shape[0], 2),)
        next_state = (state[0] @ self.matrix.T,)
        return next_state[0], next_state


def test_negative_estimate_is_selected_and_bounded_by_svd():
    torch.manual_seed(6)
    inputs = torch.zeros(1, 3, 2, 8, 8)
    result = estimate_ftle(LinearState([0.75, 0.5]), inputs)
    assert result.probes == 8
    assert result.eligible
    assert result.score == result.estimate
    assert math.log(0.5) <= result.estimate <= math.log(0.75) + 1e-6


def test_nonnegative_estimate_is_ineligible_and_json_safe():
    result = estimate_ftle(LinearState([1.2, 1.1]), torch.zeros(1, 2, 2, 8, 8))
    assert not result.eligible
    assert result.score == -math.inf
    assert result.to_dict()["score"] is None


def test_eight_probes_cannot_lower_maximum_with_same_seed():
    inputs = torch.zeros(1, 2, 2, 8, 8)
    model = LinearState([1.2, 0.7])
    torch.manual_seed(23)
    one = estimate_ftle(model, inputs, k=1)
    torch.manual_seed(23)
    eight = estimate_ftle(model, inputs, k=8)
    assert eight.estimate >= one.estimate
