#!/usr/bin/env bash
set -euo pipefail

lnas search --config configs/experiments/autosnn_cifar10dvs.yaml "$@"
