import math

import pytest
import torch

from lnas.config import ModelConfig, ProxyConfig
from lnas.data import DatasetSpec
from lnas.models import build_model
from lnas.proxies import build_proxy


@pytest.mark.parametrize("name", ["lle", "hd", "sahd", "flops"])
def test_proxy_returns_finite_score(name):
    torch.manual_seed(7)
    spec = DatasetSpec("test", 2, 4, True)
    model = build_model(ModelConfig(search_space="autosnn", width=8), spec)
    inputs = torch.rand(3, 2, 2, 12, 12)
    proxy = build_proxy(ProxyConfig(name=name, probes=2))
    result = proxy.score(model, inputs)
    assert math.isfinite(result.score)
    assert result.name == name


def test_lle_uses_state_space_without_materializing_a_jacobian():
    torch.manual_seed(11)
    spec = DatasetSpec("test", 2, 4, True)
    model = build_model(ModelConfig(search_space="autosnn", width=4), spec)
    inputs = torch.rand(2, 3, 2, 8, 8)
    result = build_proxy(ProxyConfig(name="lle", probes=2)).score(model, inputs)
    assert result.details["probes"] == 2
    assert result.details["state_elements"] > 0
    assert result.score == -abs(result.raw_value)


@pytest.mark.parametrize("space", ["snasnet", "autosnn", "autost", "hc-snn", "hc-st"])
def test_lle_supports_every_search_space(space):
    torch.manual_seed(13)
    spec = DatasetSpec("test", 2, 4, True)
    architectures = {
        "autost": {"dimension": 8, "depth": 1, "heads": 2, "mlp_ratio": 2},
        "hc-snn": {"name": "sew-resnet18"},
        "hc-st": {"name": "spikformer"},
    }
    model = build_model(
        ModelConfig(search_space=space, width=8), spec, architectures.get(space, {})
    )
    inputs = torch.rand(1, 3, 2, 8, 8)
    result = build_proxy(ProxyConfig(name="lle")).score(model, inputs)
    assert math.isfinite(result.raw_value)
