# Uncertainty Quantification (UQ)

`gracemaker` has three ways to tell when a model is unsure:

1.  **CALM extrapolation grade** γ, for GRACE-1L/2L/3L models: one model call, calibrated so
    that γ ≈ 1 is the boundary of the training data
    ([arXiv:2609.40060](https://arxiv.org/abs/2609.40060)). Most of this page is about it.
2.  **D-optimality** extrapolation grade, for GRACE/FS models
    ([Phys. Rev. Materials 7, 043801 (2023)](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.7.043801);
    [arXiv:2212.08716](https://arxiv.org/abs/2212.08716)).
3.  **Ensembles** (query-by-committee), for any model.

!!! tip "New to UQ? Start with the tutorial"
    [UQ workflows](uq_workflows.md) goes step by step from building the CALM
    artifact to new training data, with small example models. This page is the
    reference: the method, all `grace_uq` options and the Python API.

## 1. CALM extrapolation grade {#calm}

CALM (calibrated Mahalanobis,
[Lysogorskiy, Bochkarev and Drautz, arXiv:2609.40060](https://arxiv.org/abs/2609.40060)) gives
every atom an uncertainty in four steps:

1.  Map the atom's local environment to a feature vector (the basis-RP feature, below).
2.  Cluster the training environments per element (streaming *k*-means).
3.  Measure the Mahalanobis distance of a new environment to its **nearest** cluster centre.
4.  Divide this distance by a threshold calibrated on the training data. The result is γ.

### Definitions

For an atom of element *e*:

1.  **Feature z.** A fixed, seeded **Johnson–Lindenstrauss random projection** of the model's
    rotation-invariant ($\ell=0$) basis **b**, plus log-norm **density channels**:

    $$\mathbf{z} = \Big[\ \frac{\mathbf{b}}{\lVert\mathbf{b}\rVert}\,R\ ,\ \ \log\lVert\mathbf{b}\rVert\ ,\ \
    \log\lVert\mathbf{b}^{(1)}\rVert\ ,\ \dots\Big]$$

    **b** holds the $\ell=0$ basis functions that enter the invariant reduce(s) before the atomic
    energy; $\mathbf{b}^{(k)}$ is the block of reduce $k$. $R = \mathcal{N}(0,1)/\sqrt{d}$ is
    generated from `rp_seed` and stored in the artifact, so the artifact describes its own feature.
    With `rp_dim` = $d$ (default 128), the feature dimension is D = d + 1 + (number of reduce
    blocks). The normalization keeps stretched environments visible, the density channels keep
    compressed ones visible, so both raise γ.

2.  **Cluster assignment.** The nearest centroid by Euclidean distance:

    $$k^* = \arg\min_k \| \mathbf{z} - \boldsymbol{\mu}_{e,k} \|^2$$

3.  **Sigma** (`atomic_sigma`). The Mahalanobis distance to that centroid, with the centroid
    $\boldsymbol{\mu}_{e,k}$ and covariance $\Sigma_{e,k}$ of the cluster:

    $$\sigma = \sqrt{(\mathbf{z} - \boldsymbol{\mu}_{e,k^*})^\top \, \Sigma_{e,k^*}^{-1} \, (\mathbf{z} - \boldsymbol{\mu}_{e,k^*})}$$

4.  **Gamma** (`gamma`). Sigma divided by the calibrated threshold of the cluster:

    $$\gamma = \frac{\sigma}{\theta_{e,k^*}}$$

The threshold $\theta_{e,k}$ is a robust upper fence on the training sigma of the cluster,
$\mathrm{median} + 3 \cdot (1.4826 \cdot \mathrm{MAD})$, about a 3-sigma bound. It is not a p99
quantile, so a heavy outlier tail cannot raise it. If many of your *training* atoms read
$\gamma > 1$, see [Calibrating the threshold](#calibrating-the-threshold).

$\gamma \approx 1$ means the atom is at the boundary of the training distribution; $\gamma \gg 1$
means an environment outside it. `gamma` is the one UQ signal to use, for screening, active
learning and HAL.

The workflow has two stages:

-   **Build** (once per set of model weights): fit the clusters on features of the training set,
    accumulate the covariances and calibrate the thresholds, with
    [`grace_uq build`](#grace_uq-build).
-   **Use**: load the artifact, or the exported SavedModel or Kokkos `.npz` that carries it, and
    read `gamma` per atom in ASE, in `grace_uq predict` or in LAMMPS.


## 2. D-optimality (GRACE/FS models)

GRACE/FS models have their own extrapolation grade, based on D-optimality, as for ACE
([Lysogorskiy et al., Phys. Rev. Materials 7, 043801 (2023)](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.7.043801);
[arXiv:2212.08716](https://arxiv.org/abs/2212.08716)). Build an
active set from the training data with `pace_activeset`
([how](quickstart.md#build-active-set-for-gracefs-only)) and use it in
[ASE](quickstart.md#ase-gracefs) or in [LAMMPS](quickstart.md#lammps-gracefs). The tools on this
page do not apply to it.

## 3. Ensembles (query-by-committee) {#alternative-ensembling-query-by-committee}

An **ensemble** asks whether independently trained models disagree. It works for **any** GRACE
model, also one without a CALM artifact. Fit the same input with several seeds
(`gracemaker ... --seed 1`, `--seed 2`, …) and pass the models together to one `TPCalculator`:

```python
from tensorpotential.calculator import TPCalculator

calc_ens = TPCalculator(model=[
    "fit/seed/1/saved_model/",
    "fit/seed/2/saved_model/",
    "fit/seed/3/saved_model/",
])

at.calc = calc_ens
at.get_potential_energy()

calc_ens.results["energy_std"]  # std. dev. of total energy over the ensemble
calc_ens.results["forces_std"]  # std. dev. of forces
calc_ens.results["stress_std"]  # std. dev. of stress
```

`energy`, `forces` and `stress` are the ensemble **mean**. An ensemble multiplies the cost of
fitting and inference by the number of models, and its spread is not calibrated: there is no
"γ ≈ 1 is the boundary". Use CALM for routine screening, active learning and HAL, and an ensemble
for a second, independent opinion or for a model without a CALM artifact.

---

## CLI tool: `grace_uq`

`grace_uq` works with CALM and has one subcommand per task:

| Subcommand | Purpose |
| :--- | :--- |
| `build`   | Build the CALM artifact from the training data and export a SavedModel with UQ. |
| `info`    | Print a summary of a CALM artifact (`.npz`). |
| `predict` | Energies, forces, stresses and per-atom γ for a dataset; several workers; reads pkl.gz, extxyz or a LAMMPS dump. |
| `select`  | Pick the structures for DFT from a pool of candidates. |
| `hal`     | Run uncertainty-biased (HAL) walkers, one or several (`--repeats`); see [UQ workflows](uq_workflows.md#3-biased-dynamics-hal). |

`grace_uq` without a subcommand prints this overview. Each subcommand lists
its options with `--help`:

```bash
grace_uq <subcommand> --help
```

---

## `grace_uq build`

`build` makes the CALM artifact in three steps, spread over several worker
processes, and then exports a SavedModel with UQ.

!!! tip "In a fit folder, no arguments are needed"
    gracemaker writes a fit into `seed/<N>/`. Run `grace_uq build` there
    without arguments:

    ```bash
    cd my_fit/seed/1
    grace_uq build
    ```

    It reads the files the fit left in this folder:

    | Input | Default |
    | :--- | :--- |
    | `--model-yaml` | `model.yaml` |
    | `--checkpoint` | `checkpoints/checkpoint.best_test_loss.index` |
    | `--train-data` | `training_set.pkl.gz`, the training split that gracemaker saves (unless `data: save_dataset: False`). If it is missing: `data.filename` of `../../input.yaml`, but not when that file splits the data (`train_size`, `test_size` or `test_fraction`); then the build stops and asks for `--train-data` |

    It writes `gmm_artifacts.npz`, the elbow report and `saved_model/` into
    the same folder, with the default options: 4 workers, GPU 0, k chosen from
    1 2 4 8 16, `--rp-dim 128`. If no training set is found but
    `gmm_artifacts.npz` exists, it only exports the SavedModel.

### Basic usage

```bash
grace_uq build --model-yaml model.yaml \
               --checkpoint checkpoints/checkpoint.best_test_loss.index \
               --train-data training_set.pkl.gz \
               --n-workers 4 \
               --n-clusters 1 2 4 8 16 \
               --artifact-path gmm_artifacts.npz
```

This runs the three steps with 4 workers, tries k = 1, 2, 4, 8 and 16
clusters and picks k with the elbow method. It then exports a SavedModel with
the `compute_uq` signatures to `saved_model/`, next to the artifact.

### Export-only mode

If the artifact exists already, leave out `--train-data` to export only the
SavedModel:

```bash
grace_uq build --model-yaml model.yaml \
               --checkpoint checkpoints/checkpoint.best_test_loss.index \
               --artifact-path gmm_artifacts.npz
```

For the [Kokkos LAMMPS pair styles](quickstart.md#lammps-grace-1l-grace-2l-grace-3l-kokkos-no-tensorflow)
(`grace/1l/kk`, `grace/2l/kk`, `grace/3l/kk`), bake the artifact
into the `.npz` weights with `grace_utils`
([details](utilities.md#baking-in-uq-uncertainty-quantification-artifacts)):

```bash
grace_utils -p model.yaml -c checkpoints/checkpoint.best_test_loss.index \
            export_kokkos -o grace_kokkos.npz --uq-artifacts gmm_artifacts.npz
```

??? note "All options of `grace_uq build`"
    | Argument | Description |
    | :--- | :--- |
    | `--model-yaml` | the model file (default: `model.yaml`). |
    | `--checkpoint` | the model checkpoint, with or without `.index` (default: `checkpoints/checkpoint.best_test_loss.index`). |
    | `--train-data` | one or more training-data files: pickled DataFrames (`.pkl.gz`, `.pkl`) or sharded TF datasets (default: `training_set.pkl.gz`). Not needed for export-only mode. |
    | `--train-data-weighted` | repeatable: a weight (float > 0), then training shard paths; the atoms of these shards count with this weight. For a training mix that differs from the shards on disk. Not together with `--train-data`; pickle shards only. |
    | `--filter-fn` | `module.path:function_name` of a function `filter_atoms(ase.Atoms) -> bool`; structures for which it returns `False` are dropped (see *Structure filtering* below). |
    | `--n-workers` | worker processes (default: 4). |
    | `--n-clusters` | clusters per element (default: `1 2 4 8 16`). One value is used as is; several values run the elbow method, which picks k. |
    | `--threshold-percentile` | put γ = 1 at the Q-th percentile (0 < Q ≤ 100) of each cluster's training sigma histogram, instead of the default fence. γ is then **not** comparable with default artifacts ([Calibrating the threshold](#calibrating-the-threshold)). |
    | `--max-neighbours-per-batch` | target number of neighbour pairs per batch for pickle input (default: 15000); sets the GPU memory use. |
    | `--frac` | use a random fraction (0.0 to 1.0) of the training data, for a faster build. |
    | `--seed` | seed for shuffling the data and for the KMeans start (default: 42). |
    | `--artifact-path` | the artifact file (default: `gmm_artifacts.npz`); its folder is created if needed. |
    | `--export-path` | the SavedModel folder (default: `saved_model/` next to the artifact). |
    | `--no-export` | do not export a SavedModel. |
    | `--gpus` | comma-separated GPU indices for the workers, e.g. `0,1,2,3` (default: `0`). |
    | `--threads-per-worker` | CPU threads per worker. `auto` (default) = cores / workers; an integer overrides it, `0` sets no limit. |
    | `--rp-dim` | projection dimension *d* of the feature (default: `128`). |
    | `--rp-seed` | seed of the projection matrix *R* (default: `42`). |
    | `--restart` | delete the intermediate `.step*.npz` files and build from scratch. |
    | `--verbose` | print the full worker output and progress. |

??? example "Structure filtering (`--filter-fn`)"
    To keep some structures out of the artifact, for example chemistries the
    model handles poorly, give `--filter-fn` a function that returns `False`
    for them. It runs on each shard before `--frac`:

    ```python
    # uq_filter.py
    RARE_EARTHS = frozenset({"Gd", "Eu", "Yb"})

    def keep(atoms) -> bool:
        return not (set(atoms.get_chemical_symbols()) & RARE_EARTHS)
    ```

    ```bash
    PYTHONPATH=/path/to/filter:$PYTHONPATH \
      grace_uq build --train-data /path/shards/*.pkl.gz --filter-fn uq_filter:keep
    ```

    The workers inherit `PYTHONPATH` from your shell. Each shard logs how
    many structures the filter removed. An element that the filter removes
    completely is missing from the artifact; check with `grace_uq info`.

### Pipeline steps

The method is described in [arXiv:2609.40060](https://arxiv.org/abs/2609.40060). Each step
spreads the training data over the workers, and the master merges their results:

1.  **Clustering**: streaming *k*-means of the features per element, for every candidate k at
    once. With several k, the elbow method picks one k; its report and plot are saved next to the
    artifact. An element with too few atoms gets fewer clusters.
2.  **Covariances**: every atom goes to its nearest centroid; the master sums the scatter matrices,
    inverts the regularized covariances and prints their condition numbers.
3.  **Thresholds**: a histogram of σ per cluster gives the threshold.
4.  **Export**: a SavedModel with the signatures `compute`, `compute_uq` (with `dsigma_dr`) and
    the faster `compute_uq_gamma_only`.

The workers save their results after every step, so an interrupted build continues where it
stopped when you run the same command again; `--restart` starts over.

---

## `grace_uq hal`

The `hal` subcommand runs one uncertainty-biased (HAL) MD walker: the force
is biased along the gradient of Σγ, frames with max γ above a trigger are
collected, and a live copy of the CALM model is updated after each collection
so that the walker keeps moving to new configurations. It takes the model as
`model.yaml` + checkpoint (not a SavedModel) and the training-set artifact:

```bash
grace_uq hal --model-yaml model.yaml \
             --checkpoint checkpoints/checkpoint.best_test_loss.index \
             --artifact-path gmm_artifacts.npz \
             --structure start.extxyz --ensemble nvt -T 300 --n-steps 1000 \
             --output-dir hal_w0
```

Periodic structures use `--ensemble nvt`, `nvt_ramp`, `npt` or `npt_stress`;
clusters and molecules (`pbc=False`) use `--ensemble cluster`. The collected
frames go to `hal_w0/hal_collected.extxyz`, together with the updated live
artifact, the per-step history, a summary and a diagnostics figure. A
step-by-step introduction with the tutorial models, the Python API
(`run_hal`), the LAMMPS counterpart and the selection of the collected frames
for DFT are on the [UQ workflows](uq_workflows.md#3-biased-dynamics-hal)
page. This section is the reference.

### HAL options

The first rows are required: the model (`--model-dir`, or `--model-yaml`,
`--checkpoint` and `--artifact-path`), the starting structure, the
temperature and the number of steps. The other options follow in
alphabetical order. `grace_uq hal --help` lists the same options:

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--model-dir` | — | directory to search for `model.yaml`, the checkpoint and the CALM artifact (search order in `--help`); explicit `--model-yaml`, `--checkpoint`, `--artifact-path` override what is found |
| `--model-yaml`, `--checkpoint` | (required without `--model-dir`) | the model; a SavedModel cannot be used, because the live tower is updated |
| `--artifact-path` | (required without `--model-dir`) | training-set CALM artifact: the frozen tower, and the start of the live tower |
| `--structure` | (required) | starting structure; any file ASE can read |
| `-T` | (required) | temperature, K; several values need `--repeats` ([grid](#hal-walker-grid)) |
| `--n-steps` | (required) | MD steps |
| `--alphas` | 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 | weights of the additive live update, tried in the given order (keep them ascending); must end in 1.0 while inflation is on |
| `--bulk-modulus` | `auto` | eV/Å³ (`npt_stress`) |
| `--cluster-wall-k` | 1.0 | eV/Å² (`cluster`) |
| `--cluster-wall-margin` | 2.0 | Å (`cluster`; [containment wall](#hal-containment-wall)) |
| `--ensemble` | `nvt` | `nvt`, `nvt_ramp`, `npt`, `npt_stress`, `cluster` ([Ensembles](#hal-ensembles)); several values need `--repeats` ([grid](#hal-walker-grid)) |
| `--equil-steps` | 0 | steps without collection at the start (the bias waits `--tau-ema` steps either way) |
| `--feature-gamma-min` | 1.0 | keep the CALM features of the atoms with γ<sub>pristine</sub> ≥ this of every collected frame in `hal_features.npz`, the input of `grace_uq select --hal-dirs`; `none` switches it off |
| `--fragment-mult` | 1.25 | bond cutoff for `n_fragments`: atoms closer than this × (r<sub>cov,i</sub> + r<sub>cov,j</sub>) are bonded (recorded only, never used to filter) |
| `--friction` | 0.01 | Langevin friction, 1/fs (`nvt`, `nvt_ramp`, `cluster`) |
| `--gpus` | — | sets `CUDA_VISIBLE_DEVICES` for the run |
| `--history-every` | 1000 | flush the history every N steps |
| `--inflate-max-factor` | 4.0 | cap on the covariance scale factor per round |
| `--inflate-max-rounds` | 3 | at most this many inflation rounds per collection |
| `--live-start-artifact-path` | — | start the live tower from an earlier walker's `hal_live_artifacts.npz` |
| `--n-collect` | — | stop after this many collected frames |
| `--nvt-ramp` | 1.0 1.4 | V/V<sub>0</sub> at start and end (`nvt_ramp`) |
| `--output-dir` | `hal_w0` | output directory; with `--repeats` the parent of the walker directories (default `hal_walkers`). The walker refuses to write into a directory that already holds HAL output unless `--overwrite` is given |
| `--pair-radii` | — | JSON file with per-pair overrides in Å for the ZBL switch and the rejection distance, e.g. `{"Al-Li": {"zbl": [1.3, 1.8], "reject": 1.6}}`; either key may be left out |
| `--param-dtype` | — | override the model dtype (`float32` / `float64`) |
| `--plot-only DIR` | — | redraw the diagnostics figure of a finished or killed walker |
| `--pool DIR …`, `--pool-output`, `--gamma-pristine-min` | — | pool the collected frames of these walker folders, or of all walker folders below them at any depth, instead of running a walker ([pool files](#select-pool-files)) |
| `--progress-every` | 100 | print a status line every N steps; 0 = quiet (`run_hal`: `progress_every`, default 0) |
| `--reject-bond-factor` | 0.65 | reject if a pair is closer than this × (r<sub>cov,i</sub> + r<sub>cov,j</sub>) |
| `--reject-gamma` | 10 | reject a trigger with max γ<sub>live</sub> or max γ<sub>pristine</sub> above this |
| `--remove-update-work` / `--no-remove-update-work` | on | after each CALM update, take the bias energy the update injects, τ<sub>r</sub>·(Σγ<sub>before</sub> − Σγ<sub>after</sub>), back out of the kinetic energy (`nvt`, `nvt_ramp`, `cluster`; recorded only for `npt`, `npt_stress`) |
| `--save-every` | 10 | save the live artifact every N collections |
| `--seed` | 42 | initial velocities and Langevin noise |
| `--stress-tau` | 0.02 | stress-bias strength τ<sub>s</sub> (`npt_stress`) |
| `--structure-index` | 0 | frame of a multi-frame `--structure` |
| `--tau` | 0.15 | force-bias strength τ; system dependent, see [Choosing τ](uq_workflows.md#choosing-tau). Several values run a τ scan: one walker per τ (K with `--repeats K`) into `OUTPUT_DIR/tau_<τ>/hal_w<r>/`, at most `--walkers-per-gpu` at a time, table in `OUTPUT_DIR/walkers.csv` ([grid](#hal-walker-grid)) |
| `--tau-ema` | 10 | EMA window and bias delay, steps |
| `--tdamp`, `--pdamp` | 100, 1000 | MTK damping times, fs (`npt`, `npt_stress`) |
| `--timestep` | `auto` | fs; `auto` = 2 fs, or 0.5 fs with H, He, Li, Be |
| `--trigger-gamma` | 1.5 | trigger threshold for γ<sub>live</sub> |
| `--trigger-gap` | 10 | minimum steps between triggers |
| `--trigger-min-atoms` | 1 | trigger only when at least this many atoms have γ<sub>live</sub> > `--trigger-gamma`; 1 = max γ<sub>live</sub> decides |
| `--update` | `alpha_search` | live-tower update; `none` keeps it fixed |
| `--update-inflate` / `--no-update-inflate` | on | bounded covariance inflation after an α = 1 update that misses the target |
| `--update-target` | 1.25 | α-search target for the frame's max γ |
| `--update-work-floor` | 0.5 | the removal never cools the walker below this fraction of `-T` |
| `--volume-guard` | 0.75 1.30 | stop when V/V<sub>0</sub> leaves this band (`npt`, `npt_stress`) |
| `--walker-id` | 0 | written into every collected frame |
| `--repeats K` | — | run K independent walkers for every setting (one combination of `--ensemble`, `-T` and `--tau`), each in its own process; K walkers in total when each has one value ([Many walkers](#hal-many-walkers)) |
| `--walkers-per-gpu M` | 4 | with `--repeats`: at most M walkers at once on each visible GPU, the others wait for a free slot |
| `--zbl` | `auto` | `auto` (universal ZBL) or `off` |
| `--zbl-switch` | 0.55 0.75 | ZBL fully on below IN, off above OUT, × (r<sub>cov,i</sub> + r<sub>cov,j</sub>) |

`--trigger-min-atoms` makes the trigger less sensitive to a single atom
with a high γ.

The same walker runs from Python with `run_hal`; see [HAL from Python](#hal-from-python) in the Python API.

### How the HAL driver works

HAL adds a force along the gradient of Σγ to the physical force:

$$\mathbf{F}_i^\text{HAL} = \mathbf{F}_i + \tau_r\,\nabla_i \sum_j \gamma_j,
\qquad
\tau_r = \tau\,\frac{\mathrm{EMA}\left(\sum_i |\mathbf{F}_i|\right)}{\mathrm{EMA}\left(\sum_i |\nabla_i \sum_j \gamma_j|\right)}$$

The moving averages run over `--tau-ema` steps, and the bias is off while they fill. The
reported energy is the physical one, without the bias. On every MD step:

- **One model call** gives the energy, forces, stress, γ, ∇Σγ and the CALM features.
- **Two CALM towers.** The *live* tower drives the bias and the trigger and is updated after
  every collected frame. The *frozen* tower is the training-set artifact; its γ
  (`gamma_pristine`) is checked at every trigger.
- **Trigger.** A frame triggers when at least `--trigger-min-atoms` atoms have
  γ<sub>live</sub> > `--trigger-gamma`, and `--trigger-gap` steps have passed since the last
  trigger.
- **Rejection.** A trigger is rejected (no frame, no update) when max γ<sub>live</sub> or max
  γ<sub>pristine</sub> is above `--reject-gamma`, or when two atoms are closer than
  `--reject-bond-factor` × the sum of their covalent radii.
- **Live update.** The driver takes the first weight α from `--alphas` that brings the frame's
  max γ below `--update-target`, and adds α times the frame's covariance to the clusters the
  frame touches. Centroids and thresholds stay fixed and the training data are never forgotten,
  so γ<sub>live</sub> ≤ γ<sub>pristine</sub>: chained walkers do not flag configurations inside
  the training distribution. If even α = 1 is not enough, the covariances of these clusters are
  inflated in a few bounded rounds (`--inflate-max-rounds`, `--inflate-max-factor`).
- **ZBL guard.** A switched ZBL repulsion keeps atoms apart below `--zbl-switch` × the sum of the
  covalent radii. It is not part of the bias normalization.

The walker stops after `--n-steps` or `--n-collect` frames, when the volume leaves
`--volume-guard` (`npt`, `npt_stress`), or on non-finite forces or γ; `hal_summary.json` gives
the reason. HAL re-inverts both towers in float64, so its γ<sub>pristine</sub> can differ
slightly from the γ of `grace_uq predict` or LAMMPS.

**Limit of the additive update.** With one cluster per element, an update along one direction
also covers further drift along it, so a walker that keeps moving the same way collects little.
A higher `--update-target` makes each update weaker, and the walker collects more.

### Temperature control

Each CALM update adds bias energy, which the walker would turn into heat. In the Langevin
ensembles (`nvt`, `nvt_ramp`, `cluster`) the driver therefore takes this energy out of the
kinetic energy right after each update, but never below `--update-work-floor` × `-T` (default
0.5). Walkers that collect often, small molecules most of all, can then run a little below the
target temperature. To switch this off, pass `--no-remove-update-work` (`run_hal`:
`remove_update_work=False`); the Langevin thermostat then removes the extra heat over time. In
`npt` and `npt_stress` the energy is only recorded, never removed.

### HAL ensembles {#hal-ensembles}

A structure must be either fully periodic (all three `pbc` true and a full
cell) or fully non-periodic (all `pbc` false). Mixed pbc, such as a slab with
`pbc=(True, True, False)`, is refused. Non-periodic structures must use
`cluster`, and `cluster` refuses periodic cells.

| `--ensemble` | Structures | Dynamics | Use it for |
| :--- | :--- | :--- | :--- |
| `nvt` (default) | periodic | Langevin, `--friction` 0.01 1/fs, total momentum reset every step | General exploration at fixed volume. |
| `nvt_ramp` | periodic | Langevin, cell volume ramped linearly from `--nvt-ramp` START to END × V<sub>0</sub> over the run (default 1.0 → 1.4) | Expanded, low-density and near-fracture configurations. The ramp costs no extra model call. |
| `npt` | periodic | MTK NPT at 1 bar, `--tdamp` 100 fs, `--pdamp` 1000 fs | Volume fluctuations and thermal expansion. The walker stops when V/V<sub>0</sub> leaves `--volume-guard` (default 0.75 1.30). Needs an ASE version with `ase.md.nose_hoover_chain.MTKNPT`. |
| `npt_stress` | periodic | as `npt`, plus an uncertainty stress bias σ − τ<sub>s</sub>·B·Ŵ (Ŵ: moving-average unit direction of the Σγ virial) | Pushing the cell shape and volume towards high γ. τ<sub>s</sub> = `--stress-tau` (default 0.02); B = `--bulk-modulus` in eV/Å³, or `auto` (5-point E–V fit with the model at the start). |
| `cluster` | non-periodic | as `nvt`, with zero rotation at the start, plus a containment wall | Clusters, molecules, nanoparticles. |

#### Containment wall {#hal-containment-wall}

Atoms that evaporate from a cluster would fly away and never come back. The
`cluster` ensemble keeps them with a flat-bottom harmonic restraint on the
distance of each atom from the initial centre of mass c<sub>0</sub>:

$$E_\text{wall} = \tfrac{1}{2}\,k \sum_i \max\left(0,\ |\mathbf{r}_i - \mathbf{c}_0| - R_\text{wall}\right)^2,
\qquad R_\text{wall} = \max_i |\mathbf{r}_i(0) - \mathbf{c}_0| + \text{margin}$$

with the margin `--cluster-wall-margin` (default 2.0 Å) and the stiffness
k = `--cluster-wall-k` (default 1.0 eV/Å²). Atoms inside R<sub>wall</sub>
feel nothing. The wall energy is reported separately (`wall_energy` in the
history), not in the physical energy, and its forces are added after the
bias. R<sub>wall</sub> is written to the settings as `cluster_wall_radius`.

### Collected frames

Every collected frame is appended to `hal_collected.extxyz` at once. It carries the per-atom
arrays `gamma_live`, `gamma_pristine` and `grace_forces`, and in `info` the step, walker id,
temperature, V/V<sub>0</sub> (or Rg/Rg<sub>0</sub> for cluster), α, τ<sub>r</sub>, `grace_energy`,
`grace_stress`, the max γ of both towers, the update work and `collection_index`, which links the
frame to `hal_features.npz`. Model predictions are stored only under `grace_*` keys, never as a
calculator result, so they cannot be mistaken for reference labels.

`n_fragments` counts bonded fragments (atoms closer than `--fragment-mult` × the sum of their
covalent radii). It is recorded only, never used to filter. A stretched bond in a hot molecule can
count as a split, and an expanded metal cell can read as many fragments with nothing broken.

### Output files

Per walker in `--output-dir`:

| File | Written | Content |
| :--- | :--- | :--- |
| `hal_collected.extxyz` | appended and flushed at every collection | the collected frames (keys listed [above](#collected-frames)) |
| `hal_live_artifacts.npz` | every `--save-every` collections and at the end (atomic write); not written with `--update none` | the updated live CALM artifact |
| `hal_features.npz` | at every history flush and at the end (atomic write); not written with `--feature-gamma-min none` | per collected frame, the CALM features, γ<sub>pristine</sub>, element and atom index of its atoms with γ<sub>pristine</sub> ≥ `--feature-gamma-min`, the element counts of the frame, and the sha256 of the training-set artifact; the input of `grace_uq select --hal-dirs` |
| `hal_history.npz` | every `--history-every` steps and at the end; float32, compressed (for diagnostics; `run_hal(...).history` keeps float64) | per step: max γ, Σγ, fraction of atoms with γ > 1, τ<sub>r</sub>, the two EMA norms, T, V/V<sub>0</sub> and pressure (or Rg/Rg<sub>0</sub> and wall energy), shortest pair distance and its ratio to the rejection distance, collected/rejected status, α, max γ<sub>pristine</sub> (trigger steps only), the number of atoms above `--trigger-gamma` and the update work (collection steps only) |
| `hal_settings.json` | after setup, before the first step | all settings, including the resolved model paths, the α grid, the training-set artifact sha256 `uq_artifact_sha256`, the tower re-inversion time `reinvert_s`, `n_fragments0` and `fragment_mult`, the time step, wall radius and bulk modulus actually used |
| `hal_summary.json` | at the end | stop reason, step and collection counts, rejections by reason (`gamma`, `gamma_pristine`, `short_bond`); per collection γ, α, `n_inflate`, `inflate_factor`, the update work, `update_ms` (time of the α search, update, inflation and update-work γ evaluation, ms) and `n_fragments`; `n_fragments0` and `n_fragmented` (collected frames with more fragments than the start); mean temperature (`temperature_mean_K`), total and removed update work, wall time, steps/s, number of model calls, settings |
| `hal_diagnostics.png` | at the end, or with `--plot-only` | the diagnostics figure |
| `walkers.csv` | several walkers only (a τ scan or `--repeats`), in the parent `--output-dir`, after every walker | one row per walker: its settings, collected, rejected (by reason), max γ<sub>live</sub>, max γ<sub>pristine</sub> of the collected frames, `n_fragmented`, stop reason, mean temperature, speed |

A killed walker (for example at the wall-time limit) keeps every frame it has
collected, the live artifact as of the last save, and the history as of the
last flush; `grace_uq hal --plot-only DIR` redraws its figure.

### Diagnostics figure

`hal_diagnostics.png` shows on one step axis: max γ<sub>live</sub> with the collected and
rejected triggers and max γ<sub>pristine</sub>; τ<sub>r</sub> and the temperature;
V/V<sub>0</sub> and pressure (for `cluster`: Rg/Rg<sub>0</sub> and the wall energy); the shortest
pair distance and α per collection; and a box with the settings and counts. An example is in
[UQ workflows, HAL from the command line](uq_workflows.md#command-line).

### What τ means {#what-tau-means}

By construction, the bias force is τ times the physical force, in summed per-atom magnitude
(averaged over `--tau-ema` steps). Only its size comes from the physical force; its direction
comes from the γ landscape.

Near equilibrium the bias points mostly against the physical force, because the training data sit
near energy minima. So the bias mainly softens the potential: sampling at T is roughly like
sampling at a higher effective temperature, T<sub>eff</sub> ≈ T/(1 − 0.8 τ). This is why a small τ
collects little on a well-trained model. At τ ≈ 1.25 the restoring force would cancel, and walkers
run away well before that. Where this happens depends on the system; find a good τ with a
[τ scan](uq_workflows.md#choosing-tau). On a GPU, runs with the same `--seed` are not
bit-identical, so the counts differ from run to run.

### Many walkers and parameter grids {#hal-many-walkers}

`--repeats K` runs K independent walkers that differ only in the seed (`--seed` + r) and the
start frame (frame `--structure-index` + r of a multi-frame `--structure`).
<a id="hal-walker-grid"></a>`--ensemble`, `-T` and `--tau` can take several values: K walkers
then run for **every combination**. All walkers run as separate processes, at most
`--walkers-per-gpu` (default 4) at a time on each visible GPU:

```bash
grace_uq hal --model-dir MODEL --structure start.extxyz --n-steps 2000 \
             --repeats 2 --ensemble nvt npt -T 300 --tau 0.35 0.5 --output-dir grid
```

This runs 2 × 2 × 2 = 8 walkers into `grid/<label>/hal_w<r>/`, where the label names the
settings that vary (`nvt_tau_0.35`, …). With one combination the folders are
`OUTPUT_DIR/hal_w<r>/` (default `OUTPUT_DIR` `hal_walkers`). Each walker logs to `walker.log` in
its folder; a failed walker does not stop the others, and `grace_uq hal` then exits with code 1.
At the end, a table with one row per walker is printed and saved as `walkers.csv`. Walker ids are
unique over the grid, so one call selects from all of them: `grace_uq select --hal-dirs grid`
(it searches the subfolders at any depth).

Several `--tau` values without `--repeats` are a τ scan with one walker per τ, the same as
`--repeats 1`. From Python, use `run_walkers` or `scan_tau`
([HAL from Python](#hal-from-python)).

??? info "Differences from the paper protocols"
    The driver differs on purpose from the drivers of Fig. 7 and 8 of
    [arXiv:2609.40060](https://arxiv.org/abs/2609.40060), so those figures are reproduced only
    approximately:

    - Additive covariance update, C<sub>new</sub> = C<sub>old</sub> + α C<sub>frame</sub>,
      instead of the paper's convex blend,
      C<sub>new</sub> = (1 − α) C<sub>old</sub> + α C<sub>frame</sub>, which stays available as
      `GMMUQModel.update_one(..., mode="convex")`. The bounded inflation after α = 1 follows the
      paper; there are no extra compounded α = 1 updates.
    - A fixed stress-bias strength instead of an adaptive controller, and a volume guard that
      stops the walker instead of reflecting walls.
    - The trigger is max γ by default; the count trigger of Fig. 8 is
      `--trigger-gamma 1.3 --trigger-min-atoms` max(3, ⌈0.15 N⌉).
    - Bond rejection by covalent radii, no per-atom bias switch-off and no cap on τ<sub>r</sub>.
    - τ = 0.15 with the moving average for `nvt_ramp`, instead of 2.0 without it, and one model
      call per step instead of two.

---

## HAL in LAMMPS {#hal-in-lammps-options-and-limits}

The TensorFlow pair style `grace` runs the HAL bias too
([example](uq_workflows.md#in-lammps)). `pair_style grace tau τ` adds
s·∇<sub>i</sub>Σγ to every GRACE force, with

$$s = \tau\,\frac{\sum_i |\mathbf{F}_i|}{\sum_i |\nabla_i \sum_j \gamma_j|}$$

over the GRACE forces only (not those of `zbl` or other `hybrid/overlay` styles). This is the
bias of `grace_uq hal` without the moving average.

All HAL keywords go on the `pair_style` line, right after `grace`:

```
group           li type 2                   # only needed for tau_group; define it before pair_style
pair_style      hybrid/overlay grace tau 0.4 tau_norm sum tau_group li stress_tau 0.02 bulk_modulus 65 zbl 1.3 1.8
pair_coeff      * * grace saved_model Al Li
pair_coeff      1 1 zbl 13 13
pair_coeff      1 2 zbl 13 3
pair_coeff      2 2 zbl 3 3
```

- `tau` τ: strength of the force bias.
- `stress_tau` τ<sub>s</sub> with `bulk_modulus` B (GPa, default 100) adds the stress bias of
  `npt_stress`. `estimate_bulk_modulus(atoms, calc)` in `tensorpotential.uq.hal` gives the
  model's own B (in eV/Å³; divide by `ase.units.GPa`).
- `tau_norm sum` (default) or `max`: the sums in s, or the largest per-atom norms.
- `zbl IN OUT` through `hybrid/overlay` guards against too-close atoms; take IN and OUT at about
  0.55 and 0.75 × the sum of the covalent radii.
- `fix pair` gives `gamma`, `atomic_sigma`, `gmm_cluster` and ∇Σγ (`uncertainty_force`).

!!! tip "HAL on a part of the system"
    `tau_group <group>` on the `pair_style` line applies the bias only to the atoms of a LAMMPS
    group, and the normalisation sums run over that group only. The rest of the system follows
    plain MD. Use it to explore, for example, an interface, a defect or a molecule on a surface.
    `pair_modify` changes the keywords between two `run` commands: `pair_modify tau 0.2
    tau_group all` with a plain `pair_style grace`, `pair_modify pair grace tau 0.2 tau_group all`
    under `hybrid/overlay`.

Compared with `grace_uq hal`: one MPI rank only; the CALM model stays fixed, so the bias keeps
pushing to regions already dumped; no moving average, no trigger gap, no rejection and no volume
guard; `bulk_modulus` is set by hand. Frames are written by the dump condition alone, so filter
them with `grace_uq select`. The keywords are in the `grace` branch of the GRACE LAMMPS sources
(lammps-ace).

---

## `grace_uq predict`

The `predict` subcommand evaluates a dataset with a UQ-enabled model: energies,
forces, stresses and the per-atom `gamma` column.

```bash
grace_uq predict --model UQ/saved_model \
                 --artifact UQ/gmm_artifacts.npz \
                 --dataset candidates.pkl.gz \
                 --output predicted.pkl.gz \
                 --features-gamma-min 1.0
```

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--model`, `--checkpoint`, `--artifact` | — | a SavedModel with UQ, or a `model.yaml` with its checkpoint and CALM artifact |
| `--dataset`, `--output` | —, `predicted.pkl.gz` | input and output; the extension selects the format (`.pkl.gz`, or `.xyz`/`.extxyz`). `--dataset` also takes a LAMMPS text dump (any file name; recognised by its first line `ITEM: TIMESTEP`) |
| `--dump-elements EL …` | — | for a LAMMPS dump without an `element` column: the element of atom type 1, 2, …; ignored when the dump has the column. A dump with neither is an error |
| `--features-gamma-min X` | off | keep the CALM features only of the atoms with γ ≥ X, as the candidate columns `cand_*` of a HAL pool, plus X and the sha256 of `--artifact`; `grace_uq select` reads such a file like a HAL pool. With a SavedModel, pass the artifact it was exported with as `--artifact` to record the hash. Not with `--no-ase-atoms` or extxyz output |
| `--save-features` | off | keep the features of every atom (column `features`); needed by `fps-all` |
| `--n-workers` | `auto` | worker processes; `auto` runs min(4, ⌈frames / 200⌉) workers, round robin over the visible GPUs (`--gpus`, else `CUDA_VISIBLE_DEVICES`), and a single worker in the same process. An explicit number is spread the same way over the visible GPUs |
| `--no-ase-atoms` | off | drop the structures from the output |

The output rows keep the order of the input for any number of workers.

---

## `grace_uq select`

The `select` subcommand picks the structures to compute with DFT from a pool
of candidates. It needs the training-set CALM artifact. From HAL walkers it
works on their `hal_features.npz` sidecars, without the model or a
prediction:

```bash
grace_uq select --hal-dirs hal_w0 hal_w1 hal_w2 \
                --artifact gmm_artifacts.npz \
                --output selected.pkl.gz
```

A file written by `grace_uq predict --features-gamma-min` (`--predicted`) or
a `grace_uq hal --pool` file (`--dataset`) is read the same way. For any
other pool, `select --dataset` predicts first, with `--model` plus
`--checkpoint` and `--artifact` for a `model.yaml`, or `--uq` for a
SavedModel; when the artifact is known, this prediction keeps only the
candidate features. The default strategy `calm-greedy` picks frames largest γ
first and updates a copy of the CALM model after every pick. Without `-n` it
returns the *natural set*, the frames needed until no remaining frame has max
γ above `--gamma-min` under the updated copy; `-n` caps it. The strategies, the natural set and the
element floors are described [below](#selection-strategies), and the route for
[large pools](#large-pools) after them;
a worked example is [step 4 of the UQ workflows page](uq_workflows.md#4-select-structures-for-dft).

!!! tip "Combine strategies"
    `--strategy` takes several parts, `NAME[:N]` each, run in order on the frames not yet
    picked. For example, `--strategy calm-greedy random-all:10` takes the natural set as the
    backbone and adds 10 random frames from the rest. Later parts avoid configurations close to
    earlier picks.

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--hal-dirs DIR …` | — | HAL walker folders, or folders that contain them at any depth: select from their `hal_features.npz`, read only the selected frames |
| `--dataset` | — | pool as pkl.gz (`ase_atoms` column), extxyz or LAMMPS text dump; predicted first, unless it holds candidate columns (`grace_uq hal --pool`, `grace_uq predict --features-gamma-min`) |
| `--dump-elements EL …` | — | element of atom type 1, 2, … for a LAMMPS dump `--dataset` without an `element` column (as in `grace_uq predict`) |
| `--predicted` | — | output of `grace_uq predict`, with `--features-gamma-min` (window strategies) or `--save-features` (all strategies) |
| `--model`, `--checkpoint`, `--artifact`, `--uq` | — | model for the prediction; the training-set CALM artifact (`--artifact` or `--uq`) is needed by `calm-greedy` and checked against the hash of candidate columns |
| `--model-dir DIR` | — | fills `--model`, `--checkpoint` and `--artifact` from one directory, searched like `grace_uq hal --model-dir`; explicit flags win. With `--hal-dirs` or `--predicted` only the artifact must be found |
| `-n` | none for `calm-greedy` | number of structures with a single `--strategy`; required for `fps-extrap`, `fps-all` and `random-*`; for `calm-greedy` a cap on the natural set. With several strategy parts, give N per part instead |
| `--strategy NAME[:N] …` | `calm-greedy` | `calm-greedy`, `fps-extrap`, `random-extrap`, `random-all`, `fps-all`; several parts run in the given order, each on the frames not yet picked, with N frames per part (N optional for `calm-greedy`). The output column `strategy` names the part that picked each frame |
| `--gamma-min`, `--gamma-max` | 1.0, no cap | γ window of the candidate atoms; for `calm-greedy`, `--gamma-min` is also the natural-set limit |
| `--update-target` | 1.25 | `calm-greedy`: α-search target of each update of the selection copy |
| `--element-stratified`, `--no-element-stratified` | off for `calm-greedy`, on for the others | per-element atom floor of the selection |
| `--min-per-element` | D + 1 | the floor in atoms per element, clamped to what `-n` frames of the pool hold; also switches the floor on for `calm-greedy` |
| `--fps-max-pool` | all candidates (`fps-extrap`), 50 000 (`fps-all`) | mini-batch FPS cap |
| `--n-workers`, `--gpus` | `auto` | workers of the prediction inside `select`, as for `grace_uq predict` |
| `--output` | `selected.pkl.gz` | output file; the extension selects the format (`.pkl.gz`, or `.xyz`/`.extxyz` without the selection columns) |


### Selection strategies

| `--strategy` | Picks |
| :--- | :--- |
| `calm-greedy` (default) | frames, largest γ first, each pick scored after updating a copy of the CALM model with the earlier ones; stops at the natural set |
| `fps-extrap` | farthest-point sampling over the atoms with γ in [`--gamma-min`, `--gamma-max`], in the CALM metric |
| `random-extrap` | random frames among those with at least one atom in the γ window |
| `random-all` | random frames of the whole pool, no γ filter |
| `fps-all` | farthest-point sampling over all atoms, no γ filter |

- **`calm-greedy`** picks the frame with the largest max γ, updates a copy of the training-set
  CALM model with it (as in HAL, `--update-target`), and scores the next pick on the updated
  copy, so frames similar to earlier picks drop out. It stops at the **natural set**: when no
  remaining frame has max γ above `--gamma-min` under the copy. `-n` caps it. This is γ under the
  selection copy, not after refitting: a refitted model with a rebuilt artifact can still
  read higher γ. Needs the artifact (`--artifact`,
  `--uq` or `--model-dir`).
- **`fps-extrap`** spreads the picks evenly over the extrapolative atoms (features whitened per
  training cluster, distances per element). No stopping point: needs `-n`.
- **`random-extrap`** is a cheap baseline over the frames with extrapolative atoms. Needs `-n`,
  no artifact.
- **`random-all`** draws frames at random from the whole pool: reference or validation frames
  that γ would not choose. Needs `-n`.
- **`fps-all`** spreads the picks over all atoms, without a γ filter. Needs `-n` and
  `--save-features`.

A `.pkl.gz` output has the columns `gamma_at_pick`, `pick_order` and `strategy`.

**γ window.** `--gamma-min` (default 1.0) and `--gamma-max` (default no cap) select candidate
atoms, not frames: a selected frame can still hold atoms above `--gamma-max`. To drop whole HAL
frames, pool with `--gamma-pristine-min` ([pool files](#select-pool-files)).

**Element floors.** A floor gives every element of the pool at least D + 1 atoms in the
selection (D: the CALM feature dimension), as far as the pool allows. It is off by default for
`calm-greedy`, which covers every element through γ, and on for the other strategies;
`--element-stratified`, `--no-element-stratified` and `--min-per-element` change it. The summary
says whether the floors were met.

### Large pools: keep only the extrapolative atoms, then select {#large-pools}

For a large pool, do not store the features of every atom. Keep only the candidates, the atoms
with γ above a threshold, and select from them:

- **MD frames, LAMMPS dumps or any other dataset:** `grace_uq predict --features-gamma-min 1.0`
  keeps the CALM features only of the atoms with γ ≥ 1.0. The file is much smaller than with
  `--save-features`, and `select --predicted` reads it like a HAL pool.
- **HAL walkers** keep these features already, in `hal_features.npz` (`--feature-gamma-min`,
  default 1.0). Select straight from the walker folders with `--hal-dirs`, with no model and no
  GPU, or pool them first with `grace_uq hal --pool`.

```bash
# any frames: predict once, keep the candidate atoms, then select
grace_uq predict --model saved_model --artifact gmm_artifacts.npz \
                 --dataset md_frames.pkl.gz --features-gamma-min 1.0 --output pool_pred.pkl.gz
grace_uq select --predicted pool_pred.pkl.gz --artifact gmm_artifacts.npz --output for_dft.extxyz

# HAL walkers: select from their sidecars directly
grace_uq select --hal-dirs hal_walkers --artifact gmm_artifacts.npz --output for_dft.extxyz
```

Selecting from such files runs on the CPU and stays fast for pools of many thousands of frames.

### Sources and checks

- `--hal-dirs` reads only `hal_features.npz` and the selected frames of
  `hal_collected.extxyz`: no model, prediction or GPU. Directories without
  frames or without a sidecar are skipped with a warning: a walker that
  collected nothing writes a sidecar but no `hal_collected.extxyz`, and a
  walker run with `--feature-gamma-min none` writes no sidecar.
- `select` refuses an artifact whose sha256 differs from the one the walkers
  (or `grace_uq predict --features-gamma-min`) used, a `--gamma-min` below
  their `--feature-gamma-min`, and sidecars written with different artifacts.
- When `select --dataset` predicts itself and the artifact is known
  (`--artifact`, `--uq` or `--model-dir`), the prediction keeps only the
  candidate features; otherwise it keeps the features of every atom. With a
  SavedModel, pass `--model saved_model --uq gmm_artifacts.npz`.
- The output extension selects the format; the selected structures are
  written without the model predictions.

<a id="select-pool-files"></a>

??? info "Pool files: one file from many walkers"
    `--hal-dirs` reads the walkers directly, so a pool file is optional. It helps to:

    - **mix HAL frames with other sources** in one selection: a pool and a
      `grace_uq predict --features-gamma-min` file have the same candidate columns `cand_*`, so
      you can join them with `pd.concat` and run one `select` over all frames;
    - **keep one file** to copy or inspect: every frame with all its `info` keys (T,
      V/V<sub>0</sub>, step, γ). `--gamma-pristine-min` keeps only frames with max
      γ<sub>pristine</sub> at or above a value.

    ```bash
    grace_uq hal --pool hal_walkers --pool-output hal_pool.pkl.gz
    grace_uq select --dataset hal_pool.pkl.gz --artifact gmm_artifacts.npz
    ```

    `select` reads a pool without a prediction. Write it as `.pkl.gz`: an extxyz pool loses the
    candidate columns. From Python: `pool_frames(dirs, gamma_pristine_min=None)` in
    `tensorpotential.uq.hal`.

---

## Calibrating the threshold

There are two ways to set γ = 1:

-   **Robust fence (default):** median + 3·(1.4826·MAD) of the training σ of each cluster. On a
    well-behaved training set only about 0.1 % of the training atoms read γ > 1, and a heavy
    outlier tail cannot move the fence.
-   **Percentile:** `--threshold-percentile Q` puts γ = 1 at the Q-th percentile of each
    cluster's training σ, so at most about (100 − Q) % of the training atoms read γ > 1:

    ```bash
    grace_uq build ... --threshold-percentile 99
    ```

    γ then means something else and is **not comparable** with default artifacts; `grace_uq info`
    prints which `threshold estimator` was used. On a heavy tail the percentile follows the tail,
    so most atoms read γ far below 1.

If many training atoms (10–20 %) read γ > 1 with the default, the σ distribution of a cluster is
a sharp spike with a heavy tail. Without a `narrow_sigma` warning from `grace_uq info`, try more
clusters first (`--n-clusters` with one value); with it, use the percentile.

??? info "The `narrow_sigma` warning (small models)"
    The threshold is read from a σ histogram. Each bin is about 1 % of σ wide (artifacts built
    before October 2026: a fixed width of 0.4). When the σ spread of a cluster is narrower than
    about a bin, the threshold is unreliable, usually too high, so γ reads low.
    `grace_uq build` and `grace_uq info` warn about such clusters. With the 1 % bins this is
    rare; old artifacts of small models trained on few distinct environments often have it.
    Rebuild such an artifact with the current `grace_uq build`. If the warning stays, use fewer
    clusters (`--n-clusters 1`; more clusters make σ narrower) or `--threshold-percentile 99`,
    which hardly depends on the bin width.

## Python API

`tensorpotential.uq` has a Python API for your own scripts and active-learning loops.

### `GMMUQModel`

Evaluates σ and γ from features in TensorFlow. It works with `tf.GradientTape`, so it also gives
the gradients that HAL needs.

```python
from tensorpotential.uq import GMMUQModel

uq_model = GMMUQModel("artifacts.npz")

# Evaluate on features
sigma, total_sigma, cluster_assign = uq_model(features, element_indices)

# Gamma (normalized uncertainty) directly from features, using the calibrated
# per-element interpolation thresholds stored in the artifact (from `grace_uq build`).
gamma, sigma = uq_model.gamma_from_features(features, element_indices)
```

### HAL from Python {#hal-from-python}

`run_hal` in `tensorpotential.uq.hal` runs one walker, like `grace_uq hal`:

```python
from ase.io import read
from tensorpotential.uq.hal import run_hal

res = run_hal(
    read("start.extxyz"),
    model_dir="my_fit/seed/1",       # model.yaml, checkpoint, gmm_artifacts.npz
    ensemble="nvt_ramp",
    temperature_K=300,
    n_steps=2000,
    tau=0.5,
    output_dir="hal_py",
)
frames = res.frames                  # collected Atoms, also in hal_py/hal_collected.extxyz
```

Its keyword arguments follow the [HAL options](#hal-options), with these spellings:
`uq_artifact` (`--artifact-path`), `live_start_artifact` (`--live-start-artifact-path`),
`temperature_K` (`-T`), `timestep_fs` (`--timestep`, `None` = auto), `tdamp_fs` / `pdamp_fs`
(`--tdamp` / `--pdamp`) and `pair_overrides`, a dict with the content of the `--pair-radii` file.
It returns a `HALResult` with `frames` (the collected `Atoms`), `history` (the per-step arrays of
`hal_history.npz`), `summary` (the content of `hal_summary.json`) and `live_model`.

In the same module, `scan_tau(taus, atoms, output_dir=..., **run_hal_kwargs)` runs a τ scan
(`run_walkers` with one walker per τ),
`run_walkers(...)` runs [many walkers](#hal-many-walkers), and `pool_frames(dirs)` pools their
frames ([pool files](#select-pool-files)); each returns a DataFrame.

### Calculator from `model.yaml` {#loading-a-uq-enabled-calculator}

`get_gmm_uq_calculator` loads a model from `model.yaml` and its checkpoint, with the CALM artifact
attached:

```python
from tensorpotential.uq.factories import get_gmm_uq_calculator

calc = get_gmm_uq_calculator(
    model_yaml="model.yaml",
    checkpoint="checkpoint.best_test_loss",
    gmm_artifact_path="gmm_artifacts.npz",
)
atoms.calc = calc
atoms.get_potential_energy()
gamma = calc.results["gamma"]
```

### Calculator results {#calculator-results}

A UQ calculator puts these keys into `calc.results`. The default mode `gamma_only` gives the
first five; `calc.enable_uq(mode="full")` adds `dsigma_dr` and `virial_sigma`.

| Key | Description |
| :--- | :--- |
| `gamma` | per-atom γ. Only when the artifact has calibrated thresholds. |
| `atomic_sigma` | per-atom Mahalanobis distance σ. |
| `total_sigma` | sum of σ over the atoms. |
| `gmm_cluster` | per-atom index of the nearest cluster of the atom's element. |
| `features` | the basis-RP features. |
| `dsigma_dr` | gradient of Σγ with respect to the positions, pointing towards higher γ: the HAL bias direction. Without thresholds in the artifact, the gradient of `total_sigma`. |
| `virial_sigma` | uncertainty contribution to the virial. |

---

## Advanced: incremental updates

In active learning (for example HAL) you can update a CALM model with new environments without a
full rebuild:

```python
# Assuming uq_model is an instance of GMMUQModel
uq_model.update_one(new_features, new_element_indices)
uq_model.save("updated_artifacts.npz")
```
