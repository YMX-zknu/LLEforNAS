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
    proxy = build_proxy(ProxyConfig(name=name, max_outputs=4))
    result = proxy.score(model, inputs)
    assert math.isfinite(result.score)
    assert result.name == name
