# Experiment reproduction

All commands are run from the repository root. Use one fixed device, software environment, candidate list, data split, preprocessing pipeline, and random seed when comparing proxies. Save the resolved YAML and raw JSON outputs with every reported result.

## 1. Environment verification

```bash
conda env create -f environment.yml
conda activate lnas
lnas doctor
bash scripts/smoke_test.sh
```

The synthetic smoke run checks execution only. It is not a paper result.

## 2. Fig. 1: controlled dynamical bridge

The revised analysis keeps the architecture, weights, input sequence, neuron family, warm-up step, and perturbation magnitude fixed. Only the LIF membrane time constant changes over `1.5, 2.0, 2.5, 3.0`. Each setting uses five initialization seeds and three normalized perturbation directions.

```bash
bash scripts/fig1_dynamics.sh --set dataset.root=/datasets --set device=cuda:0
```

The output `runs/fig1_dynamics/perturbation.json` contains the per-timestep mean, standard deviation, and terminal values. Plot these alongside the four already available final accuracies. This finite-difference experiment motivates the state-growth proxy; it is not itself an LLE computation or a proof of an accuracy theorem.

## 3. Fig. 3: proxy correlation in five spaces

Generate candidate scores for all four proxies and all five search spaces:

```bash
CANDIDATES=2000 bash scripts/fig3_proxy_scores.sh /datasets
```

For HC-SNN and HC-ST, the search engine automatically caps the budget at the fixed-set sizes of nine and three. For a strict paired comparison in sampled spaces, persist one architecture list and evaluate every proxy on that same list. The current script fixes the seed and sampling configuration; before producing a publication figure, verify identical architecture keys across proxy files.

Train the candidate pool used as ground truth:

```bash
python scripts/train_pool.py \
  --config configs/experiments/snasnet_cifar10dvs.yaml \
  --scores runs/fig3/snasnet/lle/search.jsonl \
  --output runs/fig3/snasnet/training \
  --set dataset.root=/datasets \
  --set device=cuda:0
```

Use the generated `accuracies.csv` with each proxy result:

```bash
python scripts/rank_correlation.py \
  --scores runs/fig3/snasnet/lle/search.jsonl \
  --accuracies runs/fig3/snasnet/training/accuracies.csv
```

Repeat for each space and proxy. Report candidate count, Kendall's tau-b, and Spearman's rho in every panel. The paper keeps the existing Fig. 3 scatter layout.

## 4. Fig. 4: estimator robustness

Use the same ten persisted SNASNet architecture JSON files for every condition. For one architecture file stored as `architecture.json`, vary the surrogate slope:

```bash
for alpha in 2.0 2.5 3.0 3.5 4.0; do
  lnas score --config configs/experiments/snasnet_cifar10dvs.yaml \
    --architecture architecture.json \
    --set dataset.root=/datasets \
    --set model.surrogate_alpha="$alpha" \
    --set output_dir="runs/fig4/surrogate_${alpha}"
done
```

Vary batch size while holding the data seed fixed:

```bash
for batch in 4 8 16 24 32; do
  lnas score --config configs/experiments/snasnet_cifar10dvs.yaml \
    --architecture architecture.json \
    --set dataset.root=/datasets \
    --set dataset.batch_size="$batch" \
    --set output_dir="runs/fig4/batch_${batch}"
done
```

Vary `seed` for minibatch or initialization sensitivity. Compare `proxy.probes=1` and `proxy.probes=4` on the identical inputs and candidates. Aggregate architecture-rank agreement rather than comparing unnormalized score magnitudes across unrelated candidate sets.

## 5. Table I: synchronized runtime and memory

```bash
bash scripts/profile_proxies.sh \
  --set dataset.root=/datasets \
  --set device=cuda:0 \
  --repeats 20 \
  --warmup 3
```

The profiler synchronizes CUDA before timing and reports peak allocated memory. Run every proxy with the same model, batch, timestep, device, and software stack. Repeat LLE with `proxy.probes=1` and `proxy.probes=4` as separate profiles. CPU memory is intentionally reported as zero because the profiler measures CUDA allocation only.

For an ImageNet-shaped scoring-only scalability check, override the dataset and architecture shape without training a candidate pool. This is a resource measurement and must not be described as ImageNet proxy correlation.

## 6. Table II: fixed timestep versus JATST

JATST with evolutionary search:

```bash
lnas jatst --config configs/experiments/snasnet_cifar10dvs.yaml \
  --set dataset.root=/datasets \
  --set device=cuda:0 \
  --set search.method=evolution \
  --set search.candidates=1000 \
  --set 'search.timesteps=[2,4,6,8,10,12,14]' \
  --set output_dir=runs/table2/es_jatst
```

Matched fixed-timestep search uses the same total evaluation budget:

```bash
lnas search --config configs/experiments/snasnet_cifar10dvs.yaml \
  --set dataset.root=/datasets \
  --set device=cuda:0 \
  --set dataset.timesteps=10 \
  --set search.method=evolution \
  --set search.candidates=1000 \
  --set output_dir=runs/table2/es_fixed_t10
```

Repeat with `search.method=random`. Train the selected `best.json` files with the same recipe. Report proxy-evaluation count, wall-clock search time, and final selected-model accuracy separately.

## 7. Tables III and IV: final selected architectures

Train a selected architecture on each target dataset:

```bash
lnas train --config <dataset-config.yaml> \
  --architecture <best.json> \
  --set dataset.root=/datasets \
  --set device=cuda:0
```

Static-dataset ImageNet and CIFAR-10 results are final selected-model evaluations. They are not part of the five-space candidate correlation study. The current public configuration set focuses on CIFAR10-DVS; add dataset-specific YAML files containing the exact published optimizer, augmentation, epoch count, batch size, and encoding before claiming exact reproduction of the final-model tables.

## 8. Output interpretation

`search.jsonl` stores one JSON object per candidate. The relevant fields are:

- `score`: ranking value, equal to `-abs(raw_value)` for LLE;
- `raw_value`: signed finite-time directional growth estimate;
- `stable`: whether the raw value is non-positive;
- `details`: probes, transition count, state size, parameters, and selected timestep when applicable;
- `architecture`: complete search-space representation.

Unavailable recomputed values in the revised manuscript remain blank until these protocols are run. Existing final accuracies are retained only where their original experimental provenance is clear.
