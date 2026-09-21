#!/usr/bin/env bash
set -euo pipefail

lnas doctor
lnas list
lnas score --config configs/smoke.yaml
lnas search --config configs/smoke.yaml
