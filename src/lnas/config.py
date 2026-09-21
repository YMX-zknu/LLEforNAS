from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class DatasetConfig:
    name: str = "synthetic"
    root: str = "data"
    timesteps: int = 10
    image_size: int = 32
    batch_size: int = 16
    workers: int = 4
    download: bool = True


@dataclass
class ModelConfig:
    search_space: str = "snasnet"
    architecture: Dict[str, Any] | None = None
    width: int = 64
    tau: float = 2.0
    threshold: float = 1.0
    surrogate_alpha: float = 2.0


@dataclass
class ProxyConfig:
    name: str = "lle"
    batches: int = 1
    repeats: int = 1
    max_outputs: int = 10
    epsilon: float = 1e-8
    stable_only: bool = True


@dataclass
class SearchConfig:
    method: str = "random"
    candidates: int = 1000
    timestep_min: int = 2
    timestep_max: int = 14
    timestep_trials: int = 5


@dataclass
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    optimizer: str = "sgd"
    momentum: float = 0.9


@dataclass
class ExperimentConfig:
    seed: int = 2025
    device: str = "auto"
    output_dir: str = "runs/default"
    dataset: DatasetConfig = None
    model: ModelConfig = None
    proxy: ProxyConfig = None
    search: SearchConfig = None
    training: TrainingConfig = None

    def __post_init__(self) -> None:
        self.dataset = self.dataset or DatasetConfig()
        self.model = self.model or ModelConfig()
        self.proxy = self.proxy or ProxyConfig()
        self.search = self.search or SearchConfig()
        self.training = self.training or TrainingConfig()
        validate_config(self)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _construct(cls: Any, values: Dict[str, Any]) -> Any:
    allowed = {item.name: item for item in fields(cls)}
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        field_type = allowed[name].type
        if is_dataclass(field_type) and isinstance(value, dict):
            value = _construct(field_type, value)
        kwargs[name] = value
    return cls(**kwargs)


def _set_nested(data: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _parse_override(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        normalized = value.lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        if normalized in {"none", "null"}:
            return None
        return value


def load_config(path: str | Path, overrides: List[str] | None = None) -> ExperimentConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must use key=value: {override}")
        key, raw = override.split("=", 1)
        _set_nested(data, key, _parse_override(raw))
    nested = {
        "dataset": DatasetConfig,
        "model": ModelConfig,
        "proxy": ProxyConfig,
        "search": SearchConfig,
        "training": TrainingConfig,
    }
    for key, cls in nested.items():
        if key in data and isinstance(data[key], dict):
            data[key] = _construct(cls, data[key])
    return _construct(ExperimentConfig, data)


def validate_config(config: ExperimentConfig) -> None:
    if config.dataset.timesteps < 1:
        raise ValueError("dataset.timesteps must be positive")
    if config.dataset.batch_size < 1:
        raise ValueError("dataset.batch_size must be positive")
    if config.dataset.image_size < 8:
        raise ValueError("dataset.image_size must be at least 8")
    if config.model.tau <= 1.0:
        raise ValueError("model.tau must be greater than 1")
    if config.search.timestep_min > config.search.timestep_max:
        raise ValueError("search timestep range is invalid")
    if config.proxy.max_outputs < 1:
        raise ValueError("proxy.max_outputs must be positive")
