#!/usr/bin/env bash
set -euo pipefail

lnas jatst --config configs/experiments/snasnet_cifar10dvs.yaml "$@"
