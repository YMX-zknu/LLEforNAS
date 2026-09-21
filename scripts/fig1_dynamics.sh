#!/usr/bin/env bash
set -euo pipefail

lnas perturbation \
  --config configs/experiments/snasnet_cifar10dvs.yaml \
  --set model.width=64 \
  --set output_dir=runs/fig1_dynamics \
  --taus 1.5 2.0 2.5 3.0 \
  --epsilon 1e-3 \
  --directions 3 \
  --seeds 5 \
  "$@"
