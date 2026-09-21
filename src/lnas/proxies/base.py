from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from .result import ProxyResult


class Proxy(ABC):
    name: str

    @abstractmethod
    def score(self, model: nn.Module, inputs: torch.Tensor) -> ProxyResult:
        raise NotImplementedError
