# Uncertainty Quantification (UQ)

This guide covers the Uncertainty Quantification features in `gracemaker`, specifically the GMM-UQ (Gaussian Mixture Model - Uncertainty Quantification) pipeline.

## Overview

The GMM-UQ method provides a per-atom uncertainty estimate by:
1.  Mapping atoms to a local environment "latent space" (the basis-RP feature, below).
2.  Clustering these environments using a Gaussian Mixture Model (GMM).
3.  Calculating the Mahalanobis distance of a new environment to its assigned cluster centroid.
4.  Calibrating these distances against the training data to provide a normalized uncertainty metric.

### Feature space (basis-RP)

The UQ feature **z** is a fixed, seeded **Johnson–Lindenstrauss random projection** of the model's
rotation-invariant ($\ell=0$) B-basis. Concretely, the build traces the model's compute graph upstream
from the atomic-energy output to the nearest invariant `FunctionReduceN` reduce(s), concatenates the
$\ell=0$ basis functions entering those reduces (dimension $D_\text{basis}$, typically several thousand),
and multiplies by a frozen matrix $R \in \mathbb{R}^{D_\text{basis}\times d}$ with
$R = \mathcal{N}(0,1)/\sqrt{d}$:

$$\mathbf{z} = \mathbf{b}_{\ell=0}\, R, \qquad d = \texttt{rp\_dim}\ (\text{default } 128).$$

$R$ is generated deterministically from `rp_seed` (default 42), so every build worker and the eval/export
paths use a byte-identical matrix; it is stored verbatim in the artifact (key `uq_rp_matrix`) and baked
into the exported SavedModel / kokkos weights, making the artifact fully self-describing. The feature is
rotationally invariant by construction and differentiable w.r.t. atomic positions (so HAL / `dsigma_dr`
work).

### Definitions

Given the feature vector **z** above for an atom of element *e*:

1.  **Cluster assignment** — assign to the nearest centroid by Euclidean distance:

$$k^* = \arg\min_k \| \mathbf{z} - \boldsymbol{\mu}_{e,k} \|^2$$

2.  **Sigma** (`atomic_sigma`) — Mahalanobis distance to the assigned centroid:

$$\sigma = \sqrt{(\mathbf{z} - \boldsymbol{\mu}_{e,k^*})^\top \, \Sigma_{e,k^*}^{-1} \, (\mathbf{z} - \boldsymbol{\mu}_{e,k^*})}$$

where $\boldsymbol{\mu}_{e,k}$ and $\Sigma_{e,k}$ are the centroid and covariance matrix of cluster *k* for element *e*.

3.  **Gamma** (`gamma`) — sigma normalized by the calibrated per-cluster threshold:

$$\gamma = \frac{\sigma}{\theta_{e,k^*}}$$

where $\theta_{e,k}$ is a robust upper fence on the training sigma for cluster *k* of element *e*, computed as $\mathrm{median} + 3 \cdot (1.4826 \cdot \mathrm{MAD})$ (a tail-immune ~3-sigma bound; deliberately *not* a p99 quantile, so a heavy outlier tail can't inflate the threshold and leave the bulk over-lenient). $\gamma \approx 1$ means the atom is at the boundary of the training distribution; $\gamma \gg 1$ indicates an out-of-distribution environment. `gamma` is the sole UQ signal — the dimensionless extrapolation grade used for screening, active learning, and HAL.

The pipeline has two parts:

-   **Post-training (required)**: fit the GMM on features extracted from a trained model, calibrate thresholds, and attach the artifact to a calculator. This is the `grace_uq` pipeline below.


## CLI tool: `grace_uq`

`grace_uq` is a multi-subcommand CLI built around the GMM-UQ pipeline:

| Subcommand | Purpose |
| :--- | :--- |
| `build`   | Generate UQ artifacts from training data (3-step pipeline + SavedModel export). |
| `info`    | Print a human-readable summary of an existing UQ `.npz` artifact. |
| `predict` | Run E/F/S + per-atom γ on a dataset; multi-worker; pkl.gz/extxyz I/O. |
| `select`  | Pick N structures from a candidate pool by extrapolation/diversity strategy. |

A subcommand is always required — invoking `grace_uq` with no subcommand prints this overview.

```bash
grace_uq <subcommand> --help    # see all options for a subcommand
```

---

## `grace_uq build`

The `build` subcommand implements a high-performance, parallelized three-step pipeline for generating UQ artifacts.

### Basic Usage

```bash
grace_uq build --model-yaml model.yaml \
               --checkpoint checkpoints/checkpoint.best_test_loss.index \
               --train-data training_set.pkl.gz \
               --n-workers 4 \
               --n-clusters 1 2 4 8 16 \
               --artifact-path gmm_artifacts.npz
```

This runs the 3-step UQ pipeline using 4 parallel workers, scans candidate cluster counts k=1,2,4,8,16 and selects the optimal k via the elbow method, then exports a SavedModel with `compute_uq` signature to `saved_model/` (next to the artifact file).

### Export-only Mode

If UQ artifacts already exist, you can export a SavedModel without re-running the pipeline:

```bash
grace_uq build --model-yaml model.yaml \
               --checkpoint checkpoints/checkpoint.best_test_loss.index \
               --artifact-path gmm_artifacts.npz
```

No `--train-data` is needed — only the model and existing artifacts.

### Command-Line Arguments

| Argument | Description |
| :--- | :--- |
| `--model-yaml` | (Required) Path to the model configuration file (`model.yaml`). |
| `--checkpoint` | (Required) Path to the model checkpoint (omit `.index` or include it). |
| `--train-data` | One or more paths to training data files. Supports pickled DataFrames (`.pkl.gz`, `.pkl`) and sharded TF datasets. Required for UQ-artifact generation; not needed for export-only mode. Mutually exclusive with `--train-data-weighted`. |
| `--train-data-weighted` | Repeatable. First token is a per-source weight (float > 0), remaining tokens are training shard paths. Atoms from those shards carry that weight in centroid placement, covariance accumulation, and the calibration histogram. See [Weighted multi-source builds](#weighted-multi-source-builds) below. Mutually exclusive with `--train-data`. Pickle/df shards only — not supported for sharded TF datasets. |
| `--filter-fn` | Dotted import path `module.path:function_name` to a callable `filter_atoms(ase.Atoms) -> bool`. False-returning structures are dropped during ingest, before `--frac` subsampling. The module must be importable on the worker's `PYTHONPATH`. Pickle/df shards only. See [Structure filtering](#structure-filtering). |
| `--n-workers` | Number of parallel processes to spawn for feature extraction and accumulation (default: 1). |
| `--n-clusters` | Number of GMM clusters per chemical element (default: `1 2 4 8 16`). Pass a single value to use it directly, or multiple values to run the elbow method and automatically select the optimal k. |
| `--max-neighbours-per-batch` | Target number of neighbour pairs per batch for streaming (pickle) input (default: 15000). Controls GPU memory usage. Ignored for sharded TF datasets (batches are pre-padded). |
| `--frac` | Float (0.0 to 1.0). Use a random fraction of the training data to speed up artifact generation. |
| `--seed` | Random seed for data shuffling and KMeans initialization (default: 42). |
| `--artifact-path` | Path for the `.npz` UQ artifact file (default: `gmm_artifacts.npz`). If a directory is given, `gmm_artifacts.npz` is appended. Alias: `--output-path`. |
| `--export-path` | Path for the exported SavedModel directory. Default: `saved_model/` next to the artifact file. |
| `--no-export` | Skip SavedModel export after artifact generation. |
| `--gpus` | Comma-separated list of GPU indices to distribute workers (e.g., `0,1,2,3`). Default is `0`. |
| `--threads-per-worker` | CPU threads per worker (OMP, MKL, TF). `auto` (default) = physical_cores / n_workers. Set an integer to override, or `0` to disable limiting. |
| `--rp-dim` | Projection dimension *d* of the basis-RP feature (default: `128`). |
| `--rp-seed` | Seed for the random projection matrix *R* (default: `42`). The same seed reproduces a byte-identical *R* across workers and at eval/export. |
| `--restart` | If set, deletes all intermediate `.step*.npz` files, re-exports the SavedModel, and restarts the pipeline from scratch. |
| `--verbose` | Enable persistent progress tracking and internal logging. |


### Weighted multi-source builds

When building a foundational UQ artifact for a model trained on multiple datasets at different mixing ratios, the artifact should mirror the *training* mix — not the on-disk shard counts. For example, if the model was trained on 320 OMAT shards + 32 SMAX shards but you only want to process 16 OMAT shards (to keep wall time reasonable) alongside all 32 SMAX shards, the OMAT atoms must be "weighted up" so they retain their training-time influence on centroid placement and covariance.

Use `--train-data-weighted W FILE [FILE...]` (repeatable). Pick `W = original_shard_count / used_shard_count`:

```bash
grace_uq build --model-yaml model.yaml --checkpoint checkpoint.index \
  --train-data-weighted 20.0 /OMAT/filtered_df_shard_*.pkl.gz \
  --train-data-weighted  1.0 /SMAX/TRAIN_shard_*.pkl.gz \
  --n-workers 32 --gpus 0,1,2,3 \
  --artifact-path UQ/gmm_artifacts.npz
```

Effects on the pipeline:

-   **Step 1** (centroid placement): atoms are weighted in `MiniBatchKMeans.partial_fit(sample_weight=...)`, so KMeans centroids are pulled toward heavily-weighted sources.
-   **Step 2** (covariance): per-cluster scatter accumulates `(w·δ)ᵀ δ` and the effective count accumulates `Σ w`; the covariance divisor is the effective count, not the raw atom count.
-   **Step 3** (thresholds): the master computes both a raw robust-threshold matrix (`interp_thresholds`) and an effective (weighted) one (`eff_interp_thresholds`), each the per-cluster $\mathrm{median} + 3 \cdot (1.4826 \cdot \mathrm{MAD})$ fence on its histogram. The raw atom count still drives the `min_atoms_for_p99` reliability gate — statistical confidence depends on sample size, not weight. Both matrices are persisted; inference prefers the effective thresholds when present.
-   **Bit-exact back-compat**: `--train-data-weighted 1.0 ...` produces an artifact identical (to floating-point round-off) to the equivalent `--train-data ...` build. The fast path in the hot loops skips the `sample_weight=` kwarg, the `w·δ` multiply, and the second weighted bincount whenever every atom in a batch carries weight 1.0.
-   **Diagnostics**: `grace_uq info` shows an extra `eff_count` column (and `eff_thr_med`) when they diverge from the raw values. Each artifact also stores both `counts_{e}` and `effective_count_{e}` per element, plus `hist_{e}` (raw int64) and `eff_hist_{e}` (weighted float64), so the artifact can be inspected from either side.

A path that appears in more than one weighted group with different weights is rejected at parse time.


### Structure filtering

Some datasets contain structures you'd rather not influence the UQ artifact — e.g. transition-metal oxides/fluorides that the model handles poorly, or rare-earth chemistries you intend to exclude from the calibrated distribution. `--filter-fn module.path:function_name` loads a callable `filter_atoms(atoms: ase.Atoms) -> bool` at build time. Returning `False` drops the structure; `True` keeps it. The filter runs on each shard before `--frac` random subsampling.

```python
# ~/.cache/grace/checkpoints/<MODEL>/UQ/uq_filter.py
TM_OXIDIZABLE = frozenset({"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"})
OXIDIZERS = frozenset({"O", "F"})
RARE_EARTHS = frozenset({"Gd", "Eu", "Yb"})

def keep(atoms) -> bool:
    symbols = set(atoms.get_chemical_symbols())
    if symbols & RARE_EARTHS:
        return False
    if (symbols & TM_OXIDIZABLE) and (symbols & OXIDIZERS):
        return False
    return True
```

```bash
PYTHONPATH=~/.cache/grace/checkpoints/<MODEL>/UQ:$PYTHONPATH \
  grace_uq build --model-yaml model.yaml --checkpoint checkpoint.index \
                 --train-data /path/shards/*.pkl.gz \
                 --filter-fn uq_filter:keep \
                 --artifact-path UQ/gmm_artifacts.npz
```

The filter module must be importable on the *worker's* `PYTHONPATH` — workers inherit the master's environment, so setting `PYTHONPATH` on the parent shell suffices. Each shard logs a one-line `filter <shard>: N_before → N_after structures` summary at `INFO` level so over-aggressive filters are easy to spot.

Element coverage caveat: if a filter drops every structure containing some element of interest (or a shard set excludes it entirely), that element is silently absent from the artifact. At inference time `GMMUQModel.from_artifacts` back-fills missing elements with a large isotropic inverse covariance and emits a `UserWarning` naming them — but you should always check `grace_uq info` after a filtered build to confirm every expected element appears.


### Pipeline Steps

Each of steps 1–3 splits the training data across `--n-workers` parallel processes. Each worker streams its shard through the model on a separate GPU and writes per-worker checkpoint files. The master process then merges the results. Steps are resumable — if a run is interrupted, re-running picks up from the last completed checkpoint.

1.  **Global Clustering**:
    -   Each worker extracts the basis-RP feature **z** (random projection of the invariant B-basis) from its data shard and fits a local `MiniBatchKMeans` (streaming, single pass) **for every candidate k** simultaneously.
    -   Each worker also accumulates per-element within-cluster sum of squares (WCSS) against its current centroids — these are the real data-level inertias used for elbow selection.
    -   The master collects all per-worker centroids (weighted by cluster counts) and runs a global `KMeans` on them for each candidate k, producing **k** global centroids per element.
    -   If multiple k values were requested, the master runs the elbow method per element (max perpendicular distance to chord on the WCSS curve) and selects the **global optimal k = max over per-element optima**. An elbow report (`.txt`) and plot (`.png`) are saved alongside the artifact.
    -   **Per-element cluster capping**: each cluster needs at least D samples (D = feature dimension) for a meaningful covariance estimate. If an element has fewer than k×D training atoms, the effective number of clusters for that element is reduced to k_eff = max(1, n_atoms // D). The remaining cluster slots are padded with sentinel centroids (value `1e10`) that never attract atoms at inference time, keeping tensor shapes uniform across elements. A per-element summary is always printed showing effective clusters, atom counts, and sentinel slots.
    -   Outputs `.step1_final.npz` with global centroids for the selected k.

2.  **Covariance Accumulation**:
    -   Each worker streams through its data shard again, assigns each atom's feature vector to the nearest global centroid (Euclidean distance), and accumulates per-cluster **scatter matrices**:
        $S_{e,k} = \sum_{i \in \text{cluster } k} (\mathbf{z}_i - \boldsymbol{\mu}_{e,k})(\mathbf{z}_i - \boldsymbol{\mu}_{e,k})^\top$
        and cluster counts $n_{e,k}$.
    -   The master **sums** scatter matrices and counts across all workers (scatter and counts are additive), then computes the per-cluster covariance and its regularized inverse:
        $\Sigma_{e,k} = S_{e,k} / n_{e,k} + \epsilon I, \qquad \Sigma_{e,k}^{-1} = \text{pinv}(\Sigma_{e,k})$
    -   After finalization, the master prints **covariance diagnostics** per element: condition number, effective rank, and number of truncated eigenvalues. High condition numbers (> 1e10) indicate ill-conditioned clusters that may produce unreliable uncertainty estimates.
    -   Outputs `.step2_final.npz` with centroids, inverse covariance matrices, and counts.

3.  **Threshold Calibration**:
    -   Each worker evaluates the Mahalanobis distance $\sigma$ for every atom in its data shard using the finalized centroids and inverse covariances, and accumulates per-element, per-cluster **histograms** of $\sigma$ values (250 bins over [0, 100]). When `--train-data-weighted` is in effect a parallel weighted histogram is also accumulated.
    -   The master sums histograms across workers and computes the **robust threshold** $\sigma$ for each cluster: $\theta_{e,k} = \mathrm{median} + 3 \cdot (1.4826 \cdot \mathrm{MAD})$ of $\sigma$ over element *e*, cluster *k* (a tail-immune ~3-sigma fence, not a tail quantile). Clusters whose raw atom count is below the reliability floor (`min_atoms_for_p99`) inherit the element-wide maximum threshold; the gate uses the raw count regardless of weighting, since statistical confidence depends on sample size. Under weighting a second robust threshold is computed from the weighted histogram and saved as `eff_interp_thresholds`; inference prefers it when present.
    -   Generates the final artifact file (default: `gmm_artifacts.npz`).

4.  **SavedModel Export** (default, skip with `--no-export`):
    -   Loads the trained model and calibrated UQ artifacts.
    -   Appends the basis-RP feature instruction and **bakes the projection matrix $R$** (recovered verbatim from the artifact) into the graph as a constant, so the exported signatures reproduce the exact feature.
    -   Exports a TF SavedModel with `compute`, `compute_uq` (full, includes `dsigma_dr`), and `compute_uq_gamma_only` (faster, no uncertainty-force backward pass) signatures.
    -   Output directory defaults to `saved_model/` next to the artifact file.

> **Artifact format (schema v3).** The `.npz` stores the GMM statistics as **float32** (the model
> runs float32, so this is lossless and halves the file; raw and effective counts stay float64 for
> exact round-trip), plus the self-describing basis-RP spec: `uq_feature_mode`, `uq_rp_matrix` ($R$),
> `uq_rp_dim`, `uq_rp_seed`. `grace_utils export_kokkos --uq-artifacts` likewise bakes $R$
> (`uq_rp_matrix`/`uq_rp_dim`/`uq_rp_seed`) into the kokkos weights.


### Internal Arguments (Master/Worker Interaction)
These arguments are typically managed by the master process but can be used for manual step-by-step debugging:

| Argument | Description |
| :--- | :--- |
| `--step` | Pipeline step to execute (1, 2, or 3). |
| `--worker-id` | Index of the parallel worker (0 to `n_workers - 1`). |
| `--step1-artifacts` | Path to merged Step 1 artifacts (centroids). |
| `--step2-artifacts` | Path to merged Step 2 artifacts (covariances). |

---

## Python API

For integration into custom scripts or Active Learning loops, `tensorpotential.uq` provides a modular API.

### `GMMUQArtifactBuilder`
The core builder for generating artifacts from feature streams.

A `feature_iterator` is any iterable yielding 3-tuples `(features [N, D], element_indices [N], weights [N])` of numpy arrays. The weights array carries per-atom UQ weights and is `np.ones(N)` for unweighted builds — the builder takes a fast path when every weight equals 1.0, so back-compat 2-tuple callers should pass `np.ones(N, dtype=np.float64)` for the third element.

Use `batched_feature_iterator` to create one from a trained model and a list of ASE `Atoms`:

```python
import pandas as pd
from tensorpotential.uq import GMMUQArtifactBuilder, batched_feature_iterator
from tensorpotential.uq.factories import load_uq_model
from tensorpotential.uq import constants as uq_constants
from tensorpotential.tpmodel import ComputeEnergy, extract_cutoff_and_elements

# Load model with feature extraction enabled
tp, instructions = load_uq_model(
    model_yaml="model.yaml",
    checkpoint="checkpoint.best_test_loss",
    model_compute_function=ComputeEnergy(extra_return_keys=[uq_constants.FEATURES]),
)
cutoff, symbols, indices = extract_cutoff_and_elements(instructions)
element_map = {s: int(i) for s, i in zip(symbols, indices)}

# Load training structures
df = pd.read_pickle("training_set.pkl.gz", compression="gzip")
atoms_list = list(df["ase_atoms"])

def make_feature_iter():
    # Yields (features, element_indices, weights) — weights default to ones
    # when no per-atom UQ weight is set via atoms.info[UQ_WEIGHT].
    return batched_feature_iterator(atoms_list, tp.model, element_map, cutoff)

# Build artifacts (feature_dim is inferred from the first batch)
feats0, _, _ = next(make_feature_iter())
builder = GMMUQArtifactBuilder(n_clusters=32, feature_dim=feats0.shape[1])

builder.fit_centroids(make_feature_iter())        # Pass 1: KMeans centroids
builder.accumulate_scatter(make_feature_iter())    # Pass 2: scatter matrices

artifacts = builder.finalize(element_names={int(i): s for s, i in element_map.items()})

# Stamp the self-describing basis-RP spec (regenerates R from the model graph and
# the default seed) so the artifact carries uq_rp_matrix / uq_feature_mode and is
# loadable by get_gmm_uq_calculator and the SavedModel/kokkos export paths.
from tensorpotential.uq.factories import make_basis_rp_spec
builder.save("artifacts.npz", artifacts, **make_basis_rp_spec("model.yaml"))
```

> `load_uq_model` automatically appends the basis-RP feature instruction (default `rp_dim=128`,
> `rp_seed=42`), so `ComputeEnergy(extra_return_keys=[uq_constants.FEATURES])` returns the projected
> feature. Pass `feature_spec={"out_dim": ..., "seed": ...}` to override.

### `GMMUQModel`
A TensorFlow-differentiable model for evaluating uncertainties. Compatible with `tf.GradientTape` for Active Learning (HAL).

```python
from tensorpotential.uq import GMMUQModel

uq_model = GMMUQModel("artifacts.npz")

# Evaluate on features
sigma, total_sigma, cluster_assign = uq_model(features, element_indices)

# Gamma (normalized uncertainty) directly from features, using the calibrated
# per-element interpolation thresholds stored in the artifact.
gamma, sigma = uq_model.gamma_from_features(features, element_indices)
```

### SavedModel export with UQ signature

To ship a model with UQ baked in, export it with `save_model_with_aux_computes` so a `compute_uq` signature is added alongside the standard `compute` signature:

```python
from tensorpotential import TensorPotential
from tensorpotential.uq import GMMUQModel

tp = TensorPotential(...)
uq_model = GMMUQModel("artifacts.npz")     # must have interp_thresholds set
tp.save_model_with_aux_computes(exact_path="saved_model", gmm_uq_model=uq_model)
```

When a `TPCalculator` loads this SavedModel it auto-detects the UQ signature and enables it. You can toggle between the plain and UQ compute paths at runtime:

```python
calc = TPCalculator(model="saved_model")

# UQ is on by default when the SavedModel contains a compute_uq signature
forces = atoms.get_forces()
sigma  = calc.results["atomic_sigma"]
gamma  = calc.results["gamma"]

calc.disable_uq()   # fall back to the standard (faster) signature
calc.enable_uq()    # re-enable
```

---

## Workflow Example

### Loading a UQ-Enabled Calculator
Use the `get_gmm_uq_calculator` factory to load a model with UQ artifacts attached.

```python
from tensorpotential.uq.factories import get_gmm_uq_calculator

hal_calc = get_gmm_uq_calculator(
    model_yaml="model.yaml",
    checkpoint="checkpoint.best_test_loss",
    gmm_artifact_path="UQ/gmm_artifacts.npz"
)

atoms.calc = hal_calc
energy = atoms.get_potential_energy()
# Uncertainties are available via the calculator's extra results
uncertainties = hal_calc.results["gamma"]
```

### Calculator Results

When using a UQ-enabled calculator (e.g., from `get_gmm_uq_calculator`), the following keys are available in `calc.results`:

| Key | Description |
| :--- | :--- |
| `atomic_sigma` | Per-atom raw Mahalanobis distances (unnormalized). |
| `total_sigma` | Sum of per-atom uncertainties. |
| `dsigma_dr` | Analytical gradients of uncertainty with respect to atomic positions (required for HAL). |
| `virial_sigma` | Contribution of uncertainty to the virial tensor. |
| `features` | Basis-RP environment features (random projection of the invariant B-basis) used for clustering. Only present when the compute function extracts them. |
| `gamma` | Normalized uncertainty metric (sigma relative to the per-cluster robust median+3·MAD training thresholds). Only present when the artifact contains calibrated thresholds. |

---

## Advanced: Incremental Updates
When running Active Learning (e.g., HAL), you can update UQ artifacts incrementally without a full rebuild.

```python
# Assuming uq_model is an instance of GMMUQModel
uq_model.update_one(new_features, new_element_indices)
uq_model.save("updated_artifacts.npz")
```
