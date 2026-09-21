from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import kendalltau, spearmanr


def architecture_key(value) -> str:
    if isinstance(value, str):
        value = json.loads(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--accuracies", required=True)
    args = parser.parse_args()

    scores = {}
    for line in Path(args.scores).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        scores[architecture_key(record["architecture"])] = float(record["score"])

    accuracies = {}
    with Path(args.accuracies).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            accuracies[architecture_key(row["architecture"])] = float(row["accuracy"])

    keys = sorted(set(scores) & set(accuracies))
    if len(keys) < 2:
        raise ValueError("At least two matched architectures are required")
    score_values = [scores[key] for key in keys]
    accuracy_values = [accuracies[key] for key in keys]
    tau = kendalltau(score_values, accuracy_values, variant="b")
    rho = spearmanr(score_values, accuracy_values)
    print(
        json.dumps(
            {
                "candidates": len(keys),
                "kendall_tau_b": float(tau.statistic),
                "kendall_pvalue": float(tau.pvalue),
                "spearman_rho": float(rho.statistic),
                "spearman_pvalue": float(rho.pvalue),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
