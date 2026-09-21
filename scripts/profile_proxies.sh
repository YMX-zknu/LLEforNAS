#!/usr/bin/env bash
set -euo pipefail

lnas profile \
  --config configs/experiments/snasnet_cifar10dvs.yaml \
  --set output_dir=runs/profile \
  --proxies hd sahd flops lle \
  "$@"
