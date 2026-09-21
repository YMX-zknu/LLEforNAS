from __future__ import annotations

from typing import Any, Dict

from lnas.config import ModelConfig
from lnas.data import DatasetSpec

from .autosnn import AutoSNN
from .autost import AutoST
from .handcrafted import HandcraftedSNN
from .snasnet import SNASNet

DEFAULT_SNASNET = [
    [0, 3, 0, 3],
    [0, 0, 3, 0],
    [3, 0, 0, 3],
    [0, 2, 0, 0],
]

DEFAULT_AUTOSNN = [
    "SRB_k5",
    "max_pool_k2",
    "SRB_k5",
    "skip_connect",
    "max_pool_k2",
    "SRB_k3",
    "SRB_k5",
    "max_pool_k2",
]

HC_SNN_DEPTHS = {
    "spiking-vgg11": [1, 1, 2, 2],
    "spiking-vgg13": [2, 2, 2, 2],
    "spiking-vgg16": [2, 2, 3, 3],
    "spiking-resnet18": [2, 2, 2, 2],
    "spiking-resnet34": [3, 4, 6, 3],
    "spiking-resnet50": [3, 4, 6, 3],
    "sew-resnet18": [2, 2, 2, 2],
    "sew-resnet34": [3, 4, 6, 3],
    "sew-resnet50": [3, 4, 6, 3],
}

HC_ST_CONFIGS = {
    "spikformer": {"dimension": 256, "depth": 2, "heads": 8, "mlp_ratio": 4},
    "spikingformer": {"dimension": 256, "depth": 2, "heads": 8, "mlp_ratio": 4},
    "spike-driven-transformer": {
        "dimension": 256,
        "depth": 2,
        "heads": 8,
        "mlp_ratio": 4,
    },
}


def build_model(config: ModelConfig, spec: DatasetSpec, architecture: Dict[str, Any] | None = None):
    architecture = architecture or config.architecture or {}
    common = {
        "channels": spec.channels,
        "classes": spec.classes,
        "tau": config.tau,
        "threshold": config.threshold,
        "alpha": config.surrogate_alpha,
    }
    space = config.search_space.lower()
    if space == "snasnet":
        return SNASNet(
            width=config.width,
            matrix=architecture.get("matrix", DEFAULT_SNASNET),
            **common,
        )
    if space == "autosnn":
        return AutoSNN(
            width=config.width,
            blocks=architecture.get("blocks", DEFAULT_AUTOSNN),
            **common,
        )
    if space == "autost":
        dimension = int(architecture.get("dimension", config.width))
        heads = int(architecture.get("heads", 4))
        if dimension % heads:
            raise ValueError("AutoST dimension must be divisible by heads")
        return AutoST(
            dimension=dimension,
            depth=int(architecture.get("depth", 4)),
            heads=heads,
            mlp_ratio=int(architecture.get("mlp_ratio", 4)),
            **common,
        )
    if space == "hc-snn":
        name = architecture.get("name", "spiking-resnet18")
        if name not in HC_SNN_DEPTHS:
            raise ValueError(f"Unknown HC-SNN architecture: {name}")
        return HandcraftedSNN(
            width=config.width,
            stage_depths=HC_SNN_DEPTHS[name],
            **common,
        )
    if space == "hc-st":
        name = architecture.get("name", "spikformer")
        if name not in HC_ST_CONFIGS:
            raise ValueError(f"Unknown HC-ST architecture: {name}")
        return AutoST(**HC_ST_CONFIGS[name], **common)
    raise ValueError("search_space must be one of: snasnet, autosnn, autost, hc-snn, hc-st")
