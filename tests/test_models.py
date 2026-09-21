import pytest
import torch

from lnas.config import ModelConfig
from lnas.data import DatasetSpec
from lnas.models import build_model


@pytest.mark.parametrize("space", ["snasnet", "autosnn", "autost", "hc-snn", "hc-st"])
def test_search_space_output_shape(space):
    spec = DatasetSpec("test", 2, 7, True)
    config = ModelConfig(search_space=space, width=16)
    architectures = {
        "autost": {"dimension": 16, "depth": 1, "heads": 4, "mlp_ratio": 2},
        "hc-st": {"name": "spikformer"},
        "hc-snn": {"name": "spiking-vgg11"},
    }
    model = build_model(config, spec, architectures.get(space, {}))
    inputs = torch.rand(2, 3, 2, 16, 16)
    sequence = model.forward_sequence(inputs)
    assert sequence.shape == (2, 3, 7)
    assert model(inputs).shape == (2, 7)
