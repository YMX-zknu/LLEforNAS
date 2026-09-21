from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch

from lnas.config import ExperimentConfig
from lnas.data import DatasetSpec, prepare_batch
from lnas.models import build_model
from lnas.proxies import build_proxy
from lnas.proxies.result import ProxyResult
from lnas.runtime import seed_everything
from lnas.search.engine import SearchRecord


def collect_batches(
    loader: Iterable,
    count: int,
    spec: DatasetSpec,
    timesteps: int,
    device: torch.device,
) -> List[torch.Tensor]:
    batches = []
    for inputs, targets in loader:
        inputs, _targets = prepare_batch(inputs, targets, spec, timesteps, device)
        batches.append(inputs)
        if len(batches) >= count:
            break
    if not batches:
        raise RuntimeError("The data loader returned no batches")
    return batches


def _aggregate(results: List[ProxyResult]) -> ProxyResult:
    score = sum(item.score for item in results) / len(results)
    raw = sum(item.raw_value for item in results) / len(results)
    detail_keys = set().union(*(item.details for item in results))
    details = {
        key: sum(item.details.get(key, 0.0) for item in results) / len(results)
        for key in detail_keys
    }
    details["evaluations"] = float(len(results))
    return ProxyResult(results[0].name, score, raw, all(item.stable for item in results), details)


def evaluate_architecture(
    config: ExperimentConfig,
    spec: DatasetSpec,
    architecture: Dict[str, Any],
    batches: List[torch.Tensor],
    device: torch.device,
) -> SearchRecord:
    seed_everything(config.seed)
    model = build_model(config.model, spec, architecture).to(device)
    proxy = build_proxy(config.proxy)
    results = []
    for _ in range(config.proxy.repeats):
        for inputs in batches:
            results.append(proxy.score(model, inputs))
    aggregated = _aggregate(results)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    details = dict(aggregated.details)
    details["parameters"] = float(parameters)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return SearchRecord(
        architecture=architecture,
        score=aggregated.score,
        raw_value=aggregated.raw_value,
        stable=aggregated.stable,
        details=details,
    )
