from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from lnas.config import DatasetConfig


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    channels: int
    classes: int
    event_based: bool


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    test: Dataset
    spec: DatasetSpec


SPECS: Dict[str, DatasetSpec] = {
    "synthetic": DatasetSpec("synthetic", 2, 10, True),
    "nmnist": DatasetSpec("nmnist", 2, 10, True),
    "cifar10dvs": DatasetSpec("cifar10dvs", 2, 10, True),
    "dvs128gesture": DatasetSpec("dvs128gesture", 2, 11, True),
    "ncaltech101": DatasetSpec("ncaltech101", 2, 101, True),
    "cifar10": DatasetSpec("cifar10", 3, 10, False),
    "imagenet": DatasetSpec("imagenet", 3, 1000, False),
}


def _event_dataset(name: str, root: Path, timesteps: int, train: bool):
    try:
        if name == "nmnist":
            from spikingjelly.datasets.n_mnist import NMNIST

            return NMNIST(
                root=str(root / name),
                train=train,
                data_type="frame",
                frames_number=timesteps,
                split_by="number",
            )
        if name == "dvs128gesture":
            from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

            return DVS128Gesture(
                root=str(root / name),
                train=train,
                data_type="frame",
                frames_number=timesteps,
                split_by="number",
            )
        if name == "cifar10dvs":
            from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

            return CIFAR10DVS(
                root=str(root / name),
                data_type="frame",
                frames_number=timesteps,
                split_by="number",
            )
        if name == "ncaltech101":
            from spikingjelly.datasets.n_caltech101 import NCaltech101

            return NCaltech101(
                root=str(root / name),
                data_type="frame",
                frames_number=timesteps,
                split_by="number",
            )
    except ImportError as error:
        raise RuntimeError(
            "Event datasets require spikingjelly==0.0.0.0.12; install the events extra"
        ) from error
    raise ValueError(f"Unsupported event dataset: {name}")


def _split(dataset: Dataset, seed: int):
    train_size = int(0.9 * len(dataset))
    sizes = [train_size, len(dataset) - train_size]
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, sizes, generator=generator)


def _synthetic(config: DatasetConfig, seed: int) -> DatasetBundle:
    spec = SPECS["synthetic"]
    generator = torch.Generator().manual_seed(seed)
    shape = (
        32,
        config.timesteps,
        spec.channels,
        config.image_size,
        config.image_size,
    )
    inputs = torch.rand(shape, generator=generator)
    targets = torch.randint(spec.classes, (shape[0],), generator=generator)
    train = TensorDataset(inputs[:24], targets[:24])
    test = TensorDataset(inputs[24:], targets[24:])
    return DatasetBundle(train, test, spec)


def _static(config: DatasetConfig) -> DatasetBundle:
    from torchvision import datasets, transforms

    spec = SPECS[config.name]
    transform = transforms.Compose(
        [transforms.Resize((config.image_size, config.image_size)), transforms.ToTensor()]
    )
    root = Path(config.root) / config.name
    if config.name == "cifar10":
        train = datasets.CIFAR10(root, train=True, transform=transform, download=config.download)
        test = datasets.CIFAR10(root, train=False, transform=transform, download=config.download)
    else:
        train = datasets.ImageFolder(root / "train", transform=transform)
        test = datasets.ImageFolder(root / "val", transform=transform)
    return DatasetBundle(train, test, spec)


def build_datasets(config: DatasetConfig, seed: int) -> DatasetBundle:
    name = config.name.lower()
    if name not in SPECS:
        raise ValueError(f"Unknown dataset: {config.name}; choose from {sorted(SPECS)}")
    if name == "synthetic":
        return _synthetic(config, seed)
    if not SPECS[name].event_based:
        return _static(config)
    root = Path(config.root)
    if name in {"nmnist", "dvs128gesture"}:
        train = _event_dataset(name, root, config.timesteps, True)
        test = _event_dataset(name, root, config.timesteps, False)
    else:
        dataset = _event_dataset(name, root, config.timesteps, True)
        train, test = _split(dataset, seed)
    return DatasetBundle(train, test, SPECS[name])


def build_loaders(bundle: DatasetBundle, config: DatasetConfig, seed: int):
    generator = torch.Generator().manual_seed(seed)
    common: Dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train = DataLoader(bundle.train, shuffle=True, generator=generator, **common)
    test = DataLoader(bundle.test, shuffle=False, **common)
    return train, test


def prepare_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    spec: DatasetSpec,
    timesteps: int,
    device: torch.device,
):
    inputs = torch.as_tensor(inputs, dtype=torch.float32)
    targets = torch.as_tensor(targets, dtype=torch.long)
    if spec.event_based:
        if inputs.ndim != 5:
            raise ValueError("Event input must have shape [B,T,C,H,W]")
        if inputs.shape[1] != timesteps and inputs.shape[0] == timesteps:
            inputs = inputs.transpose(0, 1)
        if inputs.shape[1] != timesteps:
            indices = torch.linspace(0, inputs.shape[1] - 1, timesteps).round().long()
            inputs = inputs.index_select(1, indices)
    else:
        if inputs.ndim != 4:
            raise ValueError("Static input must have shape [B,C,H,W]")
        inputs = inputs.unsqueeze(1).expand(-1, timesteps, -1, -1, -1)
    return inputs.contiguous().to(device), targets.to(device)
