#!/usr/bin/env bash
set -euo pipefail

lnas search --config configs/experiments/snasnet_cifar10dvs.yaml "$@"
