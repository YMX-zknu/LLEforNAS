"""Fixed training mini-batches for candidate evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.nn import functional as F


class EvaluationBatch:
    def __init__(
        self, dataset: str, root: str, batch_size: int, image_size: int,
        max_timesteps: int, seed: int, device: torch.device,
    ) -> None:
        self.dataset = dataset
        self.root = Path(root)
        self.batch_size = batch_size
        self.image_size = image_size
        self.seed = seed
        self.device = device
        self.cache: dict[int, torch.Tensor] = {}
        if batch_size < 1 or image_size < 8 or max_timesteps < 2:
            raise ValueError("require batch_size >= 1, image_size >= 8 and timesteps >= 2")
        if dataset == "synthetic":
            self.synthetic = True
        elif dataset == "cifar10dvs":
            self.synthetic = False
        else:
            raise ValueError("dataset must be synthetic or cifar10dvs")

    def get(self, timesteps: int) -> torch.Tensor:
        if timesteps not in self.cache:
            if self.synthetic:
                generator = torch.Generator().manual_seed(self.seed + timesteps)
                frames = torch.rand(
                    self.batch_size, timesteps, 2, self.image_size, self.image_size,
                    generator=generator,
                )
            else:
                try:
                    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
                except ImportError as error:
                    raise RuntimeError("install the 'events' extra for CIFAR10-DVS") from error
                dataset = CIFAR10DVS(
                    root=str(self.root / "cifar10dvs"), data_type="frame",
                    frames_number=timesteps, split_by="number",
                )
                generator = random.Random(self.seed)
                indices = list(range(len(dataset)))
                generator.shuffle(indices)
                training_indices = indices[: int(0.9 * len(indices))]
                if self.batch_size > len(training_indices):
                    raise ValueError("batch size exceeds the CIFAR10-DVS training split")
                selected = training_indices[: self.batch_size]
                frames = torch.stack([
                    torch.as_tensor(dataset[index][0], dtype=torch.float32)
                    for index in selected
                ])
                if frames.shape[-2:] != (self.image_size, self.image_size):
                    batch, time, channels, height, width = frames.shape
                    frames = F.interpolate(
                        frames.reshape(batch * time, channels, height, width),
                        size=(self.image_size, self.image_size), mode="bilinear",
                        align_corners=False,
                    ).reshape(batch, time, channels, self.image_size, self.image_size)
            self.cache[timesteps] = frames.contiguous().to(self.device)
        return self.cache[timesteps]
