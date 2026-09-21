# L-NAS

[![Python checks](https://github.com/YMX-zknu/LLEforNAS/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YMX-zknu/LLEforNAS/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **L-NAS: A Lyapunov-Inspired Training-Free Proxy for Efficient Spiking Neural Architecture Search**.

L-NAS ranks an untrained spiking neural network by matrix-free finite-time hidden-state perturbation growth. The implementation propagates random tangent vectors with Jacobian--vector products (JVPs), renormalizes them after every timestep, and accumulates logarithmic growth in FP64. It does not construct a full Jacobian, multiply Jacobian matrices, or perform an explicit SVD. In this repository, *zero-cost* means *training-free scoring at initialization*; runtime and memory are measured costs rather than zero.

The same configuration and `[batch, time, channel, height, width]` tensor convention are used for all supported datasets and search spaces. Dataset metadata supplies input channels, classes, image size, and static-to-temporal conversion, so changing spaces does not require source edits.

## Supported search spaces

| Family | Manuscript name | Configuration value |
|---|---|---|
| Automatically designed CNN | SNASNet | `snasnet` |
| Automatically designed CNN | AutoSNN | `autosnn` |
| Automatically designed transformer | AutoST | `autost` |
| Hand-crafted CNN set | HC-SNN Set | `hc-snn` |
| Hand-crafted transformer set | HC-ST Set | `hc-st` |

## Installation

Python 3.10 is recommended; CI checks Python 3.9--3.12.

```bash
git clone https://github.com/YMX-zknu/LLEforNAS.git
cd LLEforNAS
conda env create -f environment.yml
conda activate lnas
```

To install into an existing PyTorch environment:

```bash
python -m pip install --upgrade pip
pip install -e .[events,analysis]
```

For development and tests:

```bash
pip install -e .[all]
lnas doctor
pytest
ruff check .
```

The paper experiments used PyTorch 2.3.1 and SpikingJelly 0.0.0.0.12. The package accepts compatible PyTorch 2.3--2.4 releases.

## Quick verification

The smoke test is CPU-only and downloads no data.

```bash
bash scripts/smoke_test.sh
```

It validates environment discovery, one LLE score, and a three-candidate search under `configs/smoke.yaml`.

## Dataset layout

Set `dataset.root` in YAML or override it on the command line.

| Dataset | Name | Expected location |
|---|---|---|
| N-MNIST | `nmnist` | `<root>/nmnist` |
| CIFAR10-DVS | `cifar10dvs` | `<root>/cifar10dvs` |
| DVS128 Gesture | `dvs128gesture` | `<root>/dvs128gesture` |
| N-Caltech101 | `ncaltech101` | `<root>/ncaltech101` |
| CIFAR-10 | `cifar10` | `<root>/cifar10` |
| ImageNet-1K | `imagenet` | `<root>/imagenet/{train,val}` |

Event datasets are loaded as frame sequences. Static images are repeated along the temporal axis by the shared data adapter. ImageNet must be prepared manually.

## Score and search

Score the default SNASNet candidate:

```bash
lnas score \
  --config configs/experiments/snasnet_cifar10dvs.yaml \
  --set dataset.root=/datasets \
  --set device=cuda:0
```

Score an architecture stored in `best.json`:

```bash
lnas score \
  --config configs/experiments/snasnet_cifar10dvs.yaml \
  --architecture runs/snasnet_cifar10dvs/best.json
```

Run architecture-only random or evolutionary search:

```bash
bash scripts/search_snasnet.sh --set dataset.root=/datasets
bash scripts/search_autosnn.sh --set dataset.root=/datasets
bash scripts/search_autost.sh --set dataset.root=/datasets
```

Search outputs are written to the configured directory as `search.jsonl` and `best.json`. Every candidate in one run receives the same cached minibatch and initialization seed.

## Joint architecture--timestep search

JATST searches architecture--timestep pairs directly. Both random search and evolutionary search consume the same total number of proxy evaluations; no nested Bayesian optimizer is used.

```bash
bash scripts/jatst_snasnet.sh \
  --set dataset.root=/datasets \
  --set search.method=evolution \
  --set search.candidates=1000 \
  --set 'search.timesteps=[2,4,6,8,10,12,14]'
```

The selected timestep is stored as `details.timestep` in each result record.

## Train a selected architecture

```bash
lnas train \
  --config configs/experiments/snasnet_cifar10dvs.yaml \
  --architecture runs/snasnet_cifar10dvs/best.json \
  --set dataset.root=/datasets
```

The command writes `best.pt` and `training.json`. To train an entire scored candidate pool and create the accuracy CSV used by the rank-correlation calculation:

```bash
python scripts/train_pool.py \
  --config configs/experiments/snasnet_cifar10dvs.yaml \
  --scores runs/fig3/snasnet/lle/search.jsonl \
  --output runs/fig3/snasnet/training \
  --limit 100 \
  --set dataset.root=/datasets
```

## Reproduce manuscript analyses

| Manuscript item | Command | Primary output |
|---|---|---|
| Fig. 1 dynamical bridge | `bash scripts/fig1_dynamics.sh --set dataset.root=/datasets` | `runs/fig1_dynamics/perturbation.json` |
| Fig. 3 five-space scoring | `bash scripts/fig3_proxy_scores.sh /datasets` | `runs/fig3/<space>/<proxy>/search.jsonl` |
| Fig. 3 rank statistics | `python scripts/rank_correlation.py --scores ... --accuracies ...` | JSON on stdout |
| Fig. 4 robustness | commands in `docs/experiments.md` | one `score.json` per condition |
| Proxy time and memory | `bash scripts/profile_proxies.sh --set dataset.root=/datasets` | `runs/profile/profile.json` |
| JATST | `bash scripts/jatst_snasnet.sh --set dataset.root=/datasets` | `search.jsonl`, `best.json` |
| Final architecture training | `lnas train ... --architecture <best.json>` | `best.pt`, `training.json` |

Exact protocols, configuration overrides, output schemas, and the mapping from every paper table/figure to commands are in [docs/experiments.md](docs/experiments.md).

## LLE configuration

| Key | Meaning |
|---|---|
| `proxy.probes` | Number of independently initialized tangent directions |
| `proxy.warmup_steps` | State transitions completed before tangent propagation |
| `proxy.epsilon` | Lower clamp for tangent norms before logarithms |
| `proxy.batches` | Cached minibatches evaluated for each candidate |
| `proxy.repeats` | Repeated proxy evaluations averaged per candidate |
| `search.candidates` | Total candidate or architecture--timestep evaluation budget |
| `search.timesteps` | Explicit timestep choices used by JATST |

The reported ranking score is `-abs(raw_value)`, where `raw_value` is the maximum directional finite-time growth estimate over the configured probes. Both values and estimator diagnostics are preserved in result files.

## Repository layout

```text
configs/experiments/    Experiment configurations
docs/                   Reproduction protocols
scripts/                Search, analysis, profiling, and training entry points
src/lnas/analysis/      Perturbation-growth and resource measurements
src/lnas/data/          Dataset registry and temporal input normalization
src/lnas/models/        State-explicit SNNs and search-space builders
src/lnas/proxies/       LLE, HD, SAHD, and FLOPs proxies
src/lnas/search/        Architecture and joint architecture--timestep search
tests/                  Configuration, model, proxy, and search tests
```

## Citation

If this repository supports your work, cite the metadata in [CITATION.cff](CITATION.cff). Until the revised paper is publicly available, bibliographic fields marked as provisional should be checked before submission.

## License

Released under the [MIT License](LICENSE). Third-party code and datasets retain their original terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
