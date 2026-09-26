# L-NAS: training-free spiking architecture search

[![Python checks](https://github.com/YMX-zknu/LLEforNAS/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YMX-zknu/LLEforNAS/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An implementation of the finite-time Lyapunov exponent (FTLE) proxy and joint architecture and timestep search without training (JATST). This focused repository includes a convolutional SNASNet search-space adapter and a spiking Transformer AutoST search-space adapter.

## What the repository does

The FTLE proxy evaluates an untrained network on one fixed mini-batch. At each timestep, it propagates **eight independently initialized tangent vectors** with Jacobian-vector products, normalizes them, and accumulates their logarithmic growth. The maximum of the eight directional estimates is the raw FTLE estimate. Candidates are eligible for selection only if that estimate is **negative**; among eligible candidates, the estimate closest to zero ranks highest. No full Jacobian, Jacobian product, or SVD is constructed.

The estimator is a finite-direction approximation to the maximum finite-time growth rate, not an exact SVD. The reported proxy score is `null` when the estimate is nonnegative. JATST then selects no architecture if its entire evaluation budget produces no eligible candidate. A search evaluation does **not** train candidate weights; full final-model training is beyond this compact demonstration.

## Requirements

- Python 3.10 or 3.11; PyTorch 2.3.1 is the development reference.
- CPU for small synthetic examples; a CUDA GPU is recommended for event-data search.
- SpikingJelly 0.0.0.0.12 and locally prepared CIFAR10-DVS frames for real event data.

```bash
git clone https://github.com/YMX-zknu/LLEforNAS.git
cd LLEforNAS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

For CIFAR10-DVS, install the optional dataset support with `pip install -e '.[events,dev]'`. On an A100 CUDA 12.1 machine, the tested environment is specified in `environment.yml` and can be created with `conda env create -f environment.yml` followed by `conda activate lnas`.

## Quick start without downloads

The following commands build untrained networks and use a deterministic synthetic sequence to exercise the full pipeline. They are smoke tests, not the accuracy experiments reported in the paper.

```bash
lnas search --space snasnet --dataset synthetic --timesteps 4 \
  --budget 3 --image-size 8 --batch-size 1 --output runs/snasnet-smoke

lnas search --space autost --dataset synthetic --timesteps 4 \
  --budget 2 --image-size 8 --batch-size 1 --output runs/autost-smoke
```

Score the winning candidate again using its saved architecture. For instance:

```bash
lnas score --space snasnet --architecture runs/snasnet-smoke/best.json \
  --dataset synthetic --timesteps 4 --image-size 8 --batch-size 1
```

The command uses K=8, a one-step warm-up and an `epsilon` of `1e-8` by default. `--probes`, `--warmup`, `--epsilon`, and `--seed` expose these settings when running controlled checks. Search outputs are `search.jsonl` (one record per architecture–timestep evaluation) and `best.json` (the best eligible candidate or `null`).

## Search on CIFAR10-DVS

Prepare the event dataset in `<root>/cifar10dvs` using the format expected by SpikingJelly's `CIFAR10DVS` class. A seeded 90/10 partition reserves 10% of samples; proxy evaluation draws only from the training partition. Each timestep setting is independently converted into event frames; the same sample indices and weight-initialization seed are used for candidate comparisons. Proxy inputs have shape `[batch, time, 2, height, width]`.

```bash
lnas search --space snasnet --dataset cifar10dvs --data-root /datasets \
  --image-size 128 --batch-size 1 --timesteps 10 --budget 100 \
  --device cuda:0 --output runs/snasnet-cifar10dvs

lnas search --space autost --dataset cifar10dvs --data-root /datasets \
  --image-size 128 --batch-size 1 --timesteps 10 --budget 100 \
  --device cuda:0 --output runs/autost-cifar10dvs
```

Increase `--budget` according to the available hardware. AutoST configurations include depth up to ten and can require considerable GPU memory; start with a small mini-batch.

## Joint architecture and timestep search

JATST scores unique `(architecture, timesteps)` pairs with the same FTLE definition. `--method random` proposes pairs uniformly from each space and the listed timestep values. `--method evolution` starts with up to 20 random pairs, selects parents among the top ten, and proposes mutation or crossover with equal probability. The budget always counts evaluated pairs rather than training epochs.

```bash
lnas jatst --space snasnet --dataset cifar10dvs --data-root /datasets \
  --image-size 128 --batch-size 1 --timesteps 2 4 6 8 10 12 14 \
  --budget 100 --method evolution --device cuda:0 --output runs/jatst-snasnet

lnas jatst --space autost --dataset cifar10dvs --data-root /datasets \
  --image-size 128 --batch-size 1 --timesteps 2 4 6 8 10 12 14 \
  --budget 100 --method random --device cuda:0 --output runs/jatst-autost
```

The same commands accept `--dataset synthetic --image-size 8` for a small dry run. If `best` is `null`, the fixed budget contained no candidate with a negative estimate. All individual raw estimates remain in `search.jsonl`.

## Programmatic use

```python
from lnas import estimate_ftle

# model.step(frame, state) returns (prediction, tuple_of_recurrent_states).
# frames is a torch.Tensor of shape [batch, time, channels, height, width].
result = estimate_ftle(model, frames, k=8, warmup=1, epsilon=1e-8)
print(result.estimate, result.eligible, result.score)
```

Each search space uses the architecture variables described by its source project. AutoST samples embedding dimension and depth together with per-block attention head counts and MLP ratios. The local state-explicit models are **compact reference adapters** so that the proxy and JATST can be exercised without a separate training framework. They are not checkpoint-compatible reproductions of the upstream full backbones. The AutoST adapter uses spiking attention without softmax, and the SNASNet adapter includes forward and feedback cell connections. Consult [THIRD_PARTY.md](THIRD_PARTY.md) for source projects.

The supplied article source `NN-subsssss(2).tex` specifies K=1; **this repository uses the requested K=8**. Numerical scores, rankings, selected architectures, and published results must be rechecked under K=8 before they can be attributed to this implementation.

## Verification and license

```bash
ruff check .
pytest
python -m compileall -q src tests
```

Released under the [MIT License](LICENSE). Cite the manuscript information in [CITATION.cff](CITATION.cff).
