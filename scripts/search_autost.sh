#!/usr/bin/env bash
set -euo pipefail

lnas search --config configs/experiments/autost_cifar10dvs.yaml "$@"
