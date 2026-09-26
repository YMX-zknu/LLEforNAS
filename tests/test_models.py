import torch

from lnas.ftle import estimate_ftle
from lnas.models.autost import AutoST
from lnas.models.snasnet import SearchCell, SNASNet
from lnas.spaces import SNASNetSpace


def test_snasnet_state_is_explicit():
    matrix = SNASNetSpace().sample(__import__("random").Random(7))["matrix"]
    model = SNASNet(2, 10, 4, matrix, 2, 1, 2).eval()
    with torch.no_grad():
        output, state = model.step(torch.randn(1, 2, 8, 8), None)
        output2, state2 = model.step(torch.randn(1, 2, 8, 8), state)
    assert output.shape == output2.shape == (1, 10)
    assert len(state) == len(state2)


def test_snasnet_feedback_uses_previous_timestep():
    matrix = [[0] * 4 for _ in range(4)]
    for source, target in ((0, 1), (1, 2), (2, 3), (0, 3), (3, 1)):
        matrix[source][target] = 1
    cell = SearchCell(2, matrix, 2, 1, 2)
    frame = torch.ones(1, 2, 4, 4)
    first, feedback, states = cell.step(frame, None, {})
    second, _, _ = cell.step(frame, feedback, states)
    assert torch.equal(first, 2 * frame)
    assert torch.equal(second, 4 * frame)


def test_autost_state_is_explicit():
    model = AutoST(2, 10, 16, 1, 4, 3, 2, 1, 2).eval()
    with torch.no_grad():
        output, state = model.step(torch.randn(1, 2, 8, 8), None)
    assert output.shape == (1, 10)
    assert len(state) == 9


def test_ftle_jvp_on_both_spiking_adapters():
    matrix = SNASNetSpace().sample(__import__("random").Random(2))["matrix"]
    frames = torch.randn(1, 2, 2, 8, 8)
    for model in (
        SNASNet(2, 10, 4, matrix, 2, 1, 2),
        AutoST(2, 10, 16, 1, 4, 3, 2, 1, 2),
    ):
        result = estimate_ftle(model, frames, k=2)
        assert result.state_elements > 0
        assert result.probes == 2
