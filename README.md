# L-NAS

[![Python checks](https://github.com/YMX-zknu/LLEforNAS/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YMX-zknu/LLEforNAS/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **L-NAS: Lyapunov-Inspired Zero-Cost Proxy for Efficient Spiking Neural Architecture Search**.

L-NAS evaluates untrained spiking neural networks with a finite-time Lyapunov-inspired score and supports joint architecture-timestep search (JATST). The repository provides one configuration system and one command-line interface for the five search spaces used in the manuscript:

| Family | Search space | Repository name |
|---|---|---|
| Automatically designed CNN | SNASNet | `snasnet` |
| Automatically designed CNN | AutoSNN | `autosnn` |
| Automatically designed transformer | AutoST | `autost` |
| Hand-crafted CNN collection | HC-SNN Set | `hc-snn` |
| Hand-crafted transformer collection | HC-ST Set | `hc-st` |

The refactored code uses a single public tensor layout, `[batch, time, channel, height, width]`. Dataset metadata supplies the channel count, class count, resolution, and temporal encoding to every model. Switching datasets or search spaces no longer requires editing source files.

## Features

- LLE, HD, SAHD, and FLOPs proxies behind a common interface.
- Random search and evolutionary search for all five search spaces.
- JATST with discrete Bayesian optimization over timesteps.
- N-MNIST, CIFAR10-DVS, DVS128 Gesture, N-Caltech101, CIFAR-10, and ImageNet loaders.
- CPU-only synthetic smoke test with no dataset download.
- Deterministic seeding, validated YAML configuration, JSON results, and resumable model artifacts.
- Automatic input adaptation for static and event-based datasets.
- Unit tests and GitHub Actions checks for Python 3.9-3.11.

## Repository layout

```text
configs/                 Reproducible experiment configurations
scripts/                 Ready-to-run search commands
src/lnas/data/           Dataset registry and input normalization
src/lnas/models/         Unified SNNs and search-space builders
src/lnas/proxies/        LLE, HD, SAHD, and FLOPs
src/lnas/search/         Random, evolutionary, and JATST search
src/lnas/engine/         Scoring and training loops
tests/                   Configuration, shape, proxy, and JATST tests
```

## 1. Installation

The manuscript environment used PyTorch 2.3.1 and SpikingJelly 0.0.0.0.12. Python 3.10 is recommended.

### Conda with CUDA 12.1

```bash
git clone https://github.com/YMX-zknu/LLEforNAS.git
cd LLEforNAS
conda env create -f environment.yml
conda activate lnas
```

### Existing PyTorch environment

```bash
git clone https://github.com/YMX-zknu/LLEforNAS.git
cd LLEforNAS
python -m pip install --upgrade pip
pip install -e .[events,analysis]
```

For development:

```bash
pip install -e .[all]
```

Verify the installation:

```bash
lnas doctor
lnas list
```

`lnas doctor` reports the Python, PyTorch, CUDA, torchvision, NumPy, PyYAML, and SpikingJelly versions detected in the active environment.

## 2. Run the smoke test

The smoke configuration uses generated event frames and runs on CPU.

```bash
bash scripts/smoke_test.sh
```

Individual commands are also available:

```bash
lnas score --config configs/smoke.yaml
lnas search --config configs/smoke.yaml
```

The results are written to `runs/smoke/`.

## 3. Prepare datasets

Set `dataset.root` in a YAML file or override it from the command line. SpikingJelly downloads supported event datasets when their loader permits it. ImageNet must be prepared manually.

| Dataset | Configuration name | Classes | Input | Expected root |
|---|---:|---:|---|---|
| N-MNIST | `nmnist` | 10 | event frames | `<root>/nmnist` |
| CIFAR10-DVS | `cifar10dvs` | 10 | event frames | `<root>/cifar10dvs` |
| DVS128 Gesture | `dvs128gesture` | 11 | event frames | `<root>/dvs128gesture` |
| N-Caltech101 | `ncaltech101` | 101 | event frames | `<root>/ncaltech101` |
| CIFAR-10 | `cifar10` | 10 | static image | `<root>/cifar10` |
| ImageNet-1K | `imagenet` | 1000 | static image | `<root>/imagenet/{train,val}` |

Example override:

```bash
lnas score \
  --config configs/experiments/snasnet_cifar10dvs.yaml \
  --set dataset.root=/datasets \
  --set device=cuda:0
```

Event loaders return `[B,T,C,H,W]`. Static images are repeated along the time axis inside `prepare_batch`; models never contain dataset-specific transpose statements.

## 4. Score one untrained architecture

The default architecture for the selected search space is used when no JSON file is provided.

```bash
lnas score --config configs/experiments/snasnet_cifar10dvs.yaml
```

Select another proxy without editing code:

```bash
lnas score \
  --config configs/experiments/snasnet_cifar10dvs.yaml \
  --set proxy.name=sahd
```

Score an explicit architecture:

```bash
lnas score \
  --config configs/experiments/snasnet_cifar10dvs.yaml \
  --architecture runs/snasnet_cifar10dvs/best.json
```

The command writes `score.json` with the ranking score, raw value, stability flag, parameter count, and proxy-specific diagnostics.

## 5. Search an architecture

### Random search

```bash
bash scripts/search_snasnet.sh
bash scripts/search_autosnn.sh
```

### Evolutionary search

```bash
bash scripts/search_autost.sh
```

Any configuration can switch methods:

```bash
lnas search \
  --config configs/experiments/snasnet_cifar10dvs.yaml \
  --set search.method=evolution \
  --set search.candidates=200
```

For a quick validation before a full 2,000-candidate run:

```bash
lnas search \
  --config configs/experiments/snasnet_cifar10dvs.yaml \
  --set search.candidates=10 \
  --set proxy.max_outputs=4
```

Every candidate is evaluated with the same cached minibatch and the same initialization seed. Results are sorted in descending score order and saved as:

```text
<output_dir>/search.jsonl
<output_dir>/best.json
```

## 6. Joint architecture and timestep search

JATST evaluates a discrete timestep range with Bayesian optimization for each sampled architecture.

```bash
bash scripts/jatst_snasnet.sh
```

The default manuscript range is 2-14 timesteps with five evaluations per architecture. It can be changed from the command line:

```bash
lnas jatst \
  --config configs/experiments/snasnet_cifar10dvs.yaml \
  --set search.candidates=100 \
  --set search.timestep_min=2 \
  --set search.timestep_max=14 \
  --set search.timestep_trials=7
```

The selected timestep is stored in the `details.timestep` field of `best.json`.

## 7. Train the selected architecture

```bash
lnas train \
  --config configs/experiments/snasnet_cifar10dvs.yaml \
  --architecture runs/snasnet_cifar10dvs/best.json
```

Training uses cross-entropy loss, cosine learning-rate annealing, and the optimizer defined in the YAML file. Outputs are:

```text
<output_dir>/best.pt
<output_dir>/training.json
```

`best.pt` contains the architecture, model state, complete resolved configuration, epoch, and best test accuracy.

## 8. Reproduce the five-space proxy comparison

Run each search space with the same dataset, seed, candidate count, minibatch, and proxy settings. Only `model.search_space` and `proxy.name` should change.

```bash
for space in snasnet autosnn autost hc-snn hc-st; do
  for proxy in hd sahd flops lle; do
    lnas search \
      --config configs/experiments/snasnet_cifar10dvs.yaml \
      --set model.search_space="$space" \
      --set proxy.name="$proxy" \
      --set output_dir="runs/comparison/${space}/${proxy}"
  done
done
```

For the fixed HC-SNN and HC-ST sets, set `search.candidates` to 9 and 3, respectively, to cover every architecture. Random sampling is deterministic but can revisit a fixed member if a larger budget is requested.

## 9. Configuration reference

| Key | Meaning |
|---|---|
| `seed` | Global initialization and sampling seed |
| `device` | `auto`, `cpu`, `cuda`, or a device such as `cuda:1` |
| `dataset.timesteps` | Number of SNN simulation steps |
| `dataset.image_size` | Spatial size produced by the loader |
| `model.search_space` | One of the five registered search spaces |
| `model.width` | Base CNN width or default AutoST embedding dimension |
| `model.tau` | LIF membrane time constant |
| `proxy.batches` | Number of cached minibatches used per candidate |
| `proxy.max_outputs` | Output directions used by the LLE Jacobian estimator |
| `proxy.stable_only` | Penalize positive finite-time exponents |
| `search.candidates` | Total architecture evaluation budget |
| `search.timestep_trials` | Bayesian-optimization evaluations per architecture |

Any key can be overridden repeatedly with `--set key=value`. Unknown keys fail immediately instead of being silently ignored.

## LLE implementation note

The original research code constructed Jacobians with ambiguous batch/output axes and materialized large tensors before SVD. The refactored implementation defines the evaluated quantity explicitly:

1. For each timestep, obtain the time-resolved class logits.
2. Select up to `proxy.max_outputs` evenly spaced output directions.
3. Differentiate each selected logit with respect to the input prefix ending at that timestep.
4. Form the per-sample restricted Jacobian and compute its largest singular value with `torch.linalg.svdvals`.
5. Compute the finite-time exponent as `log(sigma_max + epsilon) / (t + 1)` and average across samples and timesteps.

The raw exponent is reported separately from the search score. With `stable_only: true`, a positive exponent receives a penalty; among non-positive values, candidates closer to zero rank higher. This makes the implemented ranking rule explicit and prevents a positive, unstable value from winning solely because it is numerically larger.

This corrected estimator is not numerically identical to the legacy script. Published tables produced with the legacy code should be regenerated before the manuscript and repository are treated as a single reproducible release.

## Architecture JSON formats

SNASNet:

```json
{"matrix": [[0,3,0,3],[0,0,3,0],[3,0,0,3],[0,2,0,0]]}
```

AutoSNN:

```json
{"blocks": ["SRB_k5","max_pool_k2","SRB_k5","skip_connect","max_pool_k2","SRB_k3","SRB_k5","max_pool_k2"]}
```

AutoST:

```json
{"dimension": 256, "depth": 6, "heads": 8, "mlp_ratio": 4}
```

HC sets:

```json
{"name": "spiking-resnet18"}
```

## Tests and code quality

```bash
ruff check .
pytest
python -m compileall -q src tests
```

The test suite checks YAML overrides, all five model interfaces, all four proxies, and discrete JATST optimization.

## Troubleshooting

### CUDA out of memory during LLE scoring

Reduce the batch size and number of output directions:

```bash
--set dataset.batch_size=4 --set proxy.max_outputs=4
```

### Event dataset import error

```bash
pip install spikingjelly==0.0.0.0.12
```

### ImageNet cannot be downloaded

ImageNet download is intentionally not automated. Place the standard `train/` and `val/` class directories under `<dataset.root>/imagenet/`.

### Reproducibility differs across GPUs

The package fixes Python, NumPy, and PyTorch seeds and disables cuDNN benchmarking. Some CUDA kernels may still vary by platform. Record `lnas doctor` output with each experiment.

## Related implementations

The search-space definitions were checked against the official AutoSNN, SNASNet, AutoST, and NASWOT repositories. See [THIRD_PARTY.md](THIRD_PARTY.md) for links and attribution.

## Citation

The bibliographic record will be updated when the manuscript is published. Until then, use the metadata in [CITATION.cff](CITATION.cff).
