"""Small command-line interface for FTLE scoring and search."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .data import EvaluationBatch
from .ftle import estimate_ftle
from .models.autost import AutoST
from .models.snasnet import SNASNet
from .search import run_jatst
from .spaces import get_space


def _model(space: str, architecture: dict, width: int, tau: float) -> torch.nn.Module:
    common = dict(channels=2, classes=10, tau=tau, threshold=1.0, alpha=2.0)
    if space == "snasnet":
        return SNASNet(width=width, matrix=architecture["matrix"], **common)
    return AutoST(
        dimension=architecture["dimension"], depth=architecture["depth"],
        heads=architecture["heads"], mlp_ratio=architecture["mlp_ratio"], **common,
    )


def _load_architecture(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "best" in payload:
        payload = payload["best"]
    if payload is None:
        raise ValueError("search selected no eligible candidate; best is null")
    if "architecture" in payload:
        payload = payload["architecture"]
    if not isinstance(payload, dict):
        raise ValueError("architecture JSON must contain an architecture object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lnas")
    parser.add_argument("command", choices=("score", "search", "jatst"))
    parser.add_argument("--space", choices=("snasnet", "autost"), required=True)
    parser.add_argument("--dataset", choices=("synthetic", "cifar10dvs"), default="synthetic")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[4])
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--method", choices=("random", "evolution"), default="random")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--probes", type=int, default=8, help="K random tangent directions")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--width", type=int, default=16, help="SNASNet adapter width")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--architecture", help="architecture JSON for 'score'")
    parser.add_argument("--output", default="runs/ftle-search")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.command in {"score", "search"} and len(args.timesteps) != 1:
        raise ValueError("score and search require exactly one timestep count")
    if args.command == "score" and not args.architecture:
        raise ValueError("score requires --architecture")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tau = args.tau if args.tau is not None else (4 / 3 if args.space == "snasnet" else 2.0)
    space = get_space(args.space)
    batch = EvaluationBatch(
        args.dataset, args.data_root, args.batch_size, args.image_size,
        max(args.timesteps), args.seed, device,
    )

    def evaluate(architecture: dict, timestep: int):
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        model = _model(args.space, architecture, args.width, tau).to(device).eval()
        result = estimate_ftle(
            model, batch.get(timestep), k=args.probes,
            warmup=args.warmup, epsilon=args.epsilon,
        )
        del model
        return result

    if args.command == "score":
        architecture = _load_architecture(args.architecture)
        payload = {
            "space": args.space, "architecture": architecture,
            "ftle": evaluate(architecture, args.timesteps[0]).to_dict(),
        }
        print(json.dumps(payload, indent=2, allow_nan=False))
        return

    values = tuple(args.timesteps)
    result = run_jatst(
        space, evaluate, timesteps=values, budget=args.budget,
        method=args.method, seed=args.seed,
    )
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "search.jsonl").open("w", encoding="utf-8") as output:
        for record in result.records:
            output.write(json.dumps(record.to_dict(), allow_nan=False) + "\n")
    best = result.best.to_dict() if result.best is not None else None
    summary = {
        "space": args.space, "method": args.method, "dataset": args.dataset,
        "seed": args.seed, "probes": args.probes, "evaluations": len(result.records),
        "feasible": sum(record.result.eligible for record in result.records),
        "best": best,
    }
    (destination / "best.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
