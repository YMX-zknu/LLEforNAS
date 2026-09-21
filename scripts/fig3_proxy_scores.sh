#!/usr/bin/env bash
set -euo pipefail

data_root="${1:?usage: scripts/fig3_proxy_scores.sh DATA_ROOT}"
candidates="${CANDIDATES:-2000}"

for space in snasnet autosnn autost hc-snn hc-st; do
  case "$space" in
    autosnn) config="configs/experiments/autosnn_cifar10dvs.yaml" ;;
    autost) config="configs/experiments/autost_cifar10dvs.yaml" ;;
    *) config="configs/experiments/snasnet_cifar10dvs.yaml" ;;
  esac
  for proxy in hd sahd flops lle; do
    lnas search \
      --config "$config" \
      --set dataset.root="$data_root" \
      --set model.search_space="$space" \
      --set proxy.name="$proxy" \
      --set search.method=random \
      --set search.candidates="$candidates" \
      --set output_dir="runs/fig3/${space}/${proxy}"
  done
done
