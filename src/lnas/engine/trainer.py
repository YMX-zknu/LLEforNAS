from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch import nn

from lnas.config import ExperimentConfig
from lnas.data import DatasetSpec, prepare_batch
from lnas.runtime import atomic_json


def _optimizer(config: ExperimentConfig, model: nn.Module):
    training = config.training
    if training.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=training.learning_rate,
            momentum=training.momentum,
            weight_decay=training.weight_decay,
        )
    if training.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
        )
    raise ValueError("training.optimizer must be sgd or adamw")


def _epoch(
    model: nn.Module,
    loader: Iterable,
    spec: DatasetSpec,
    config: ExperimentConfig,
    device: torch.device,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    criterion = nn.CrossEntropyLoss()
    for inputs, targets in loader:
        inputs, targets = prepare_batch(inputs, targets, spec, config.dataset.timesteps, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * targets.shape[0]
        total_correct += int((logits.argmax(dim=1) == targets).sum())
        total_examples += targets.shape[0]
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def train_model(
    model: nn.Module,
    train_loader: Iterable,
    test_loader: Iterable,
    spec: DatasetSpec,
    config: ExperimentConfig,
    architecture: Dict[str, Any],
    device: torch.device,
):
    optimizer = _optimizer(config, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.training.epochs, 1)
    )
    history = []
    best_accuracy = -1.0
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.training.epochs + 1):
        train_metrics = _epoch(model, train_loader, spec, config, device, optimizer)
        test_metrics = _epoch(model, test_loader, spec, config, device)
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "test": test_metrics}
        history.append(record)
        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            torch.save(
                {
                    "architecture": architecture,
                    "model": model.state_dict(),
                    "config": config.to_dict(),
                    "epoch": epoch,
                    "accuracy": best_accuracy,
                },
                output / "best.pt",
            )
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} test_acc={test_metrics['accuracy']:.4f}"
        )
    result = {"best_accuracy": best_accuracy, "history": history, "architecture": architecture}
    atomic_json(output / "training.json", result)
    return result
