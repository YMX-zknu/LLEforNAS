from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def architecture_key(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.scores).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        records = records[: args.limit]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        architecture = record["architecture"]
        key = architecture_key(architecture)
        identifier = hashlib.sha256(key.encode()).hexdigest()[:12]
        run_dir = output / identifier
        architecture_path = run_dir / "architecture.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        architecture_path.write_text(json.dumps(architecture, indent=2), encoding="utf-8")
        command = [
            "lnas",
            "train",
            "--config",
            args.config,
            "--architecture",
            str(architecture_path),
            "--set",
            f"output_dir={run_dir}",
        ]
        for override in args.set:
            command.extend(("--set", override))
        subprocess.run(command, check=True)
        result = json.loads((run_dir / "training.json").read_text(encoding="utf-8"))
        rows.append({"architecture": key, "accuracy": result["best_accuracy"]})

    with (output / "accuracies.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["architecture", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
