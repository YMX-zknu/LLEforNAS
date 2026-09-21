from __future__ import annotations

import argparse
import importlib
import json
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import torch

from lnas.config import ExperimentConfig, load_config
from lnas.data import SPECS, build_datasets, build_loaders
from lnas.engine import collect_batches, evaluate_architecture, train_model
from lnas.models import build_model
from lnas.runtime import atomic_json, resolve_device, seed_everything
from lnas.search import run_jatst, run_search
from lnas.search.spaces import get_search_space


def _architecture(path: str | None, fallback: Dict[str, Any] | None) -> Dict[str, Any]:
    if path is None:
        return fallback or {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "architecture" in data:
        data = data["architecture"]
    if not isinstance(data, dict):
        raise ValueError("Architecture JSON must contain an object")
    return data


def _write_search(output_dir: str, records) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(item.to_dict(), sort_keys=True) for item in records) + "\n"
    (output / "search.jsonl").write_text(lines, encoding="utf-8")
    atomic_json(output / "best.json", records[0].to_dict())


def _load(args) -> ExperimentConfig:
    return load_config(args.config, args.set)


def command_score(args) -> None:
    config = _load(args)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    bundle = build_datasets(config.dataset, config.seed)
    train_loader, _ = build_loaders(bundle, config.dataset, config.seed)
    batches = collect_batches(
        train_loader,
        config.proxy.batches,
        bundle.spec,
        config.dataset.timesteps,
        device,
    )
    architecture = _architecture(args.architecture, config.model.architecture)
    record = evaluate_architecture(config, bundle.spec, architecture, batches, device)
    atomic_json(Path(config.output_dir) / "score.json", record.to_dict())
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))


def command_search(args) -> None:
    config = _load(args)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    bundle = build_datasets(config.dataset, config.seed)
    train_loader, _ = build_loaders(bundle, config.dataset, config.seed)
    batches = collect_batches(
        train_loader,
        config.proxy.batches,
        bundle.spec,
        config.dataset.timesteps,
        device,
    )
    space = get_search_space(config.model.search_space)

    def evaluate(architecture):
        record = evaluate_architecture(config, bundle.spec, architecture, batches, device)
        print(f"score={record.score:.6f} architecture={json.dumps(architecture, sort_keys=True)}")
        return record

    records = run_search(
        space, evaluate, config.search.candidates, config.search.method, config.seed
    )
    _write_search(config.output_dir, records)
    print(json.dumps(records[0].to_dict(), indent=2, sort_keys=True))


def command_jatst(args) -> None:
    config = _load(args)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    space = get_search_space(config.model.search_space)
    cache = {}

    def batches_for(timestep):
        if timestep not in cache:
            dataset_config = replace(config.dataset, timesteps=timestep)
            bundle = build_datasets(dataset_config, config.seed)
            train_loader, _ = build_loaders(bundle, dataset_config, config.seed)
            batches = collect_batches(
                train_loader, config.proxy.batches, bundle.spec, timestep, device
            )
            cache[timestep] = bundle.spec, batches
        return cache[timestep]

    def evaluate(architecture, timestep):
        spec, batches = batches_for(timestep)
        timestep_config = replace(config, dataset=replace(config.dataset, timesteps=timestep))
        record = evaluate_architecture(timestep_config, spec, architecture, batches, device)
        print(f"timestep={timestep:02d} score={record.score:.6f}")
        return record

    records = run_jatst(
        space,
        evaluate,
        config.search.candidates,
        config.search.timestep_min,
        config.search.timestep_max,
        config.search.timestep_trials,
        config.seed,
    )
    _write_search(config.output_dir, records)
    print(json.dumps(records[0].to_dict(), indent=2, sort_keys=True))


def command_train(args) -> None:
    config = _load(args)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    bundle = build_datasets(config.dataset, config.seed)
    train_loader, test_loader = build_loaders(bundle, config.dataset, config.seed)
    architecture = _architecture(args.architecture, config.model.architecture)
    model = build_model(config.model, bundle.spec, architecture).to(device)
    train_model(model, train_loader, test_loader, bundle.spec, config, architecture, device)


def command_doctor(_args) -> None:
    packages = ["torch", "torchvision", "numpy", "yaml", "spikingjelly"]
    report = {
        "python": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "packages": {},
    }
    for name in packages:
        try:
            module = importlib.import_module(name)
            report["packages"][name] = getattr(module, "__version__", "installed")
        except ImportError:
            report["packages"][name] = None
    print(json.dumps(report, indent=2, sort_keys=True))


def command_list(_args) -> None:
    print("datasets: " + ", ".join(sorted(SPECS)))
    print("search spaces: snasnet, autosnn, autost, hc-snn, hc-st")
    print("proxies: lle, hd, sahd, flops")
    print("search methods: random, evolution, jatst")


def _configured(subparsers, name, function, architecture=False):
    parser = subparsers.add_parser(name)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    if architecture:
        parser.add_argument("--architecture")
    parser.set_defaults(function=function)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lnas")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor")
    doctor.set_defaults(function=command_doctor)
    listing = subparsers.add_parser("list")
    listing.set_defaults(function=command_list)
    _configured(subparsers, "score", command_score, architecture=True)
    _configured(subparsers, "search", command_search)
    _configured(subparsers, "jatst", command_jatst)
    _configured(subparsers, "train", command_train, architecture=True)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    arguments.function(arguments)
