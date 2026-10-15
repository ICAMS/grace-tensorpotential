## Stages of model fitting

1. [Data collection](#data-collection)
2. [Input file](#input-file-setup)
3. [Fit with `gracemaker`](#grace-model-parameterization), [checkpoints and restart](#restart-from-checkpoint)
4. [Export the model](#export-model)
5. [Extrapolation grade](#extrapolation-grade) (uncertainty)
6. Use in [ASE](#usage-in-ase) and [LAMMPS](#usage-in-lammps)

---

### Data collection

Use [`grace_collect`](utilities.md#grace_collect):

```bash
grace_collect
```

or build a `pandas.DataFrame` yourself, with these columns:

* `ase_atoms` - atomic structures as ASE `Atoms`
* `energy` - total energy (force-consistent), one number
* `energy_corrected` (optional) - cohesive energy, i.e. the total energy minus the free-atom
  energies, one number. _gracemaker_ computes it if you give `reference_energy`.
* `forces` - per-atom forces, shape (number of atoms, 3)
* `stress` - six components of the stress

Datasets in extended XYZ format can be used directly in the input file, or converted to this
format with [`extxyz2df`](utilities.md#extxyz2df).

---

### Input file setup

Run `gracemaker -t` for an interactive set-up of a simple input file, then edit it if needed.
See [model hyperparameters](presets.md#models-hyperparameters) and the full
[input file](inputfile.md) reference.

---

### GRACE model parameterization

```bash
gracemaker
```

The `seed` entry of the input file sets the random seed of the model parameters. The fit writes
into a folder named after it (`seed/1/` for `seed: 1`):

* **log.txt** - the log
* **train_metrics.yaml**, **test_metrics.yaml** - training and test metrics
* **checkpoints/** - model checkpoints
* **model.yaml** - model architecture and hyperparameters

### Checkpoints and restart {#restart-from-checkpoint}

_gracemaker_ saves a checkpoint each time the test loss reaches a new best value:
`seed/*/checkpoints/checkpoint.best_test_loss.*`. It also saves a regular checkpoint every
`checkpoint_freq` epochs and keeps only the last one, unless `save_all_regular_checkpoints: True`.

To continue a fit from `checkpoint.best_test_loss` and `model.yaml` in `seed/*/`:

```bash
gracemaker -r
```

For a fine-tuning fit, also give the model file: `gracemaker -r -p seed/1/model.yaml`.

Other files can be given on the command line or in `input.yaml`:

```bash
gracemaker -r -p /path/to/model.yaml -cn /path/to/checkpoint.index
```

```yaml
potential:
  filename: /path/to/model.yaml
  checkpoint_name: /path/to/checkpoint.index
```

---

### Export model

#### TensorFlow SavedModel {#tensorflows-savedmodel}

At the end of a fit, the model is saved in TensorFlow
[SavedModel](https://www.tensorflow.org/guide/saved_model) format in `seed/1/final_model/`.
To export it from a checkpoint at any time:

```bash
gracemaker -r -s     # best test loss; -rl instead of -r: latest checkpoint
```

This writes `seed/1/saved_model/`, for [ASE](#grace-tensorflow) and
[LAMMPS](#lammps-grace-tensorflow).

#### GRACE/FS {#gracefs}

Add `-sf` to also write the GRACE/FS model in YAML format, for the C++ implementation in
[ASE](#ase-gracefs) and [LAMMPS](#lammps-gracefs):

```bash
gracemaker -r -s -sf     # export from the checkpoint: writes seed/1/saved_model.yaml
gracemaker -sf           # or at the end of a fit: writes seed/1/FS_model.yaml
```

The two files are the same model; this page calls it `FS_model.yaml`.

#### LAMMPS Kokkos `.npz` (GRACE-1L / GRACE-2L / GRACE-3L)

For the TensorFlow-free [Kokkos pair styles](#lammps-grace-1l-grace-2l-grace-3l-kokkos-no-tensorflow),
export the weights with [`grace_utils export_kokkos`](utilities.md#export-to-npz-for-lammps-kokkos-pair-style).
The architecture (1L, 2L or 3L) is detected automatically.

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index export_kokkos -o grace_weights.npz
```

---

### Extrapolation grade

The extrapolation grade γ says, per atom, how far the structure is from the training data
(γ > 1: extrapolation). See [Uncertainty Quantification](uq.md) for the methods.

#### GRACE-1L / 2L / 3L: CALM artifact {#uncertainty-quantification-uq-artifact-for-grace-1l2l3l-models}

Build the CALM artifact with [`grace_uq build`](uq.md#grace_uq-build). In the fit folder no
arguments are needed:

```bash
cd seed/1
grace_uq build
```

This writes `gmm_artifacts.npz` and a `saved_model/` that also computes γ. Step by step, with γ in
[ASE](uq_workflows.md#1-evaluate-in-ase) and
[LAMMPS](uq_workflows.md#2-md-in-lammps-monitor-and-collect-extrapolative-frames): the
[UQ workflows](uq_workflows.md) tutorial. For the Kokkos pair styles, put the artifact into the
`.npz` with [`export_kokkos --uq-artifacts`](utilities.md#baking-in-uq-uncertainty-quantification-artifacts).

#### GRACE/FS: active set {#build-active-set-for-gracefs-only}

The GRACE/FS model computes γ with D-optimality from an active set (ASI) file. Build it with
`pace_activeset` from `python-ace` ([installation](install.md#gracefs-cpu)), as in
[ML-PACE](https://pacemaker.readthedocs.io/en/latest/pacemaker/active_learning/#extrapolation_grade_and_active_learning):

```bash
cd seed/1
pace_activeset -d training_set.pkl.gz FS_model.yaml
```

Use γ in [ASE](#ase-gracefs) or [monitor it in LAMMPS](#monitoring-the-extrapolation-grade-in-lammps-gracefs).

---

### Usage in ASE

#### GRACE (TensorFlow) {#grace-tensorflow}

```python
from tensorpotential.calculator import TPCalculator, grace_fm

calc = TPCalculator("/path/to/saved_model")   # your model
calc = grace_fm("name_of_the_model")          # a foundation model
```

Both take options for padding and a minimal allowed bond distance:

```python
calc = TPCalculator("/path/to/saved_model",
                    pad_neighbors_fraction=0.05,
                    pad_atoms_number=2,
                    min_dist=0.5)
```

With `min_dist`, the calculator raises an exception for a distance shorter than `min_dist` (useful
to stop a relaxation that runs into non-physical short distances).

`mode` selects how neighbours are summed. Match it to your workload:

| `mode` | Use it for |
|---|---|
| `"uniform"` | one structure, an MD or relaxation trajectory, or a dataset of structures of about the same size; up to ~1.7× faster |
| `"diverse"` (default) | many structures of very different sizes |

```python
calc = TPCalculator("path/to/saved_model", mode="uniform")
```

??? warning "Recompilation: why `mode` and padding matter"
    The calculator is XLA-compiled for one input shape: the padded number of atoms and of
    neighbours. Each new shape triggers a new compilation of **~20 seconds**; after that, the shape
    runs at full speed. If the shape keeps changing, recompilation can take most of the run time.
    Keep the shape stable: choose `mode` as above, tune the padding (`pad_neighbors_fraction`,
    `pad_atoms_number`), and in a `"diverse"` scan give the structures **largest first**, so the
    first compiled shape covers the rest (smallest first is the worst case: about ten times more
    compilations).

    `"uniform"` (dense engine) is faster for any given structure, but it pads tightly, so every
    new size compiles again. The calculator sees one structure per call and cannot detect the
    regime itself, so you choose it.

    If the model cannot run the requested engine (an in-memory model has one engine, fixed when it
    is built; a SavedModel may export only `compute`), the calculator uses the available one and
    logs a warning. SavedModels exported from a dense-capable model have both engines.

#### GRACE/FS {#ase-gracefs}

Needs `python-ace` ([installation](install.md#gracefs-cpu)) and the
[active set](#build-active-set-for-gracefs-only):

```python
from pyace.asecalc import PyGRACEFSCalculator

calc = PyGRACEFSCalculator("/path/to/FS_model.yaml")
calc.set_active_set("/path/to/FS_model.asi")

at.calc = calc
at.get_potential_energy()
at.calc.results["gamma"]  # per-atom extrapolation grade
```

---

### Usage in LAMMPS

All GRACE pair styles need `units metal`.

#### LAMMPS: GRACE (TensorFlow) {#lammps-grace-tensorflow}

`pair_style grace` loads a SavedModel folder; for foundation models it is
`~/.cache/grace/<model_name>/`. A GPU is optional but strongly recommended: TensorFlow is much
slower on CPU.

```
pair_style grace
pair_coeff * * /path/to/saved_model Al Li
```

!!! warning "2-layer models on several MPI ranks: use `grace/2layer/parallel` or `grace/2layer/chunk`"
    `pair_style grace` runs a GRACE-2L model in a single process only. For a 2-layer model on
    several MPI ranks (or GPUs), use `grace/2layer/parallel` or `grace/2layer/chunk`
    (see the table below).

Options of `grace`:

- `padding <frac>` - fraction of extra neighbours, fewer recompilations (default 0.01)
- `pad_verbose` - print a message at each recompilation
- `no_pair_forces` - switch off the pairwise forces (on by default). They are needed for per-atom
  stress, for several MPI ranks (switched on again automatically) and for the HAL keywords. Older
  LAMMPS builds had them off by default: there, add `pair_forces`.

```
pair_style grace padding 0.05 pad_verbose
```

Extrapolation grade γ: see [γ in LAMMPS](uq_workflows.md#2-md-in-lammps-monitor-and-collect-extrapolative-frames)
in the UQ tutorial.

??? info "Other pair styles: large structures, MPI, Kokkos, GRACE/FS"
    Choose the style by model and parallelization:

    | `pair_style` | Model | TensorFlow | Parallel | Virials/stress |
    |---|---|---|---|---|
    | `grace` | 1-layer | yes | MPI | yes |
    | `grace` | 2-layer | yes | single process | yes |
    | `grace/1layer/chunk` | 1-layer | yes | MPI | yes |
    | `grace/2layer/chunk` | 2-layer | yes | MPI | yes |
    | `grace/2layer/parallel` | 2-layer | yes | MPI | yes |
    | `grace/1l/kk`, `grace/2l/kk`, `grace/3l/kk` | 1/2/3-layer | no | MPI + GPU/OpenMP | yes |
    | `grace/fs` | FS | no | MPI | yes |
    | `grace/fs/kk` | FS | no | MPI + GPU/OpenMP | yes |

    For **large structures**, `grace/1layer/chunk` (1-layer) and `grace/2layer/chunk` (2-layer)
    process the atoms in blocks (`chunksize`, default 4096) and always give virials. For 2-layer
    models on several MPI ranks use `grace/2layer/parallel`.

    ```
    pair_style grace/1layer/chunk chunksize 2048 padding 0.05
    pair_coeff * * /path/to/saved_model Al Li
    ```

    The TensorFlow styles run with a CUDA-aware `mpirun` on several GPUs. Without CUDA-aware MPI,
    give each rank its own GPU:

    ```bash
    mpirun -np 4 --bind-to none bash -c \
      'CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_RANK % 4)) lmp -in in.lammps'
    ```

    The [Kokkos styles](#lammps-grace-1l-grace-2l-grace-3l-kokkos-no-tensorflow) and
    [GRACE/FS](#lammps-gracefs) are described below.

#### LAMMPS: GRACE/FS {#lammps-gracefs}

A native C++ implementation: no TensorFlow, no GPU needed, full MPI support. `grace/fs/kk` adds GPU
and OpenMP (needs `newton on`).

```
pair_style grace/fs        # or grace/fs/kk
pair_coeff * * FS_model.yaml Mo Nb Ta W
```

##### Monitoring the extrapolation grade in LAMMPS (GRACE/FS)

```
pair_style grace/fs extrapolation
pair_coeff * * FS_model.yaml FS_model.asi Mo Nb Ta W

fix grace_gamma all pair 100 grace/fs gamma 1
compute max_grace_gamma all reduce max f_grace_gamma
variable max_grace_gamma equal c_max_grace_gamma
fix extreme_extrapolation all halt 10 v_max_grace_gamma > 25
```

#### LAMMPS: GRACE-1L / 2L / 3L (Kokkos) {#lammps-grace-1l-grace-2l-grace-3l-kokkos-no-tensorflow}

`grace/1l/kk`, `grace/2l/kk` and `grace/3l/kk` run on GPU or OpenMP without TensorFlow. They read
the `.npz` file from [`export_kokkos`](#lammps-kokkos-npz-grace-1l-grace-2l-grace-3l):

```
pair_style grace/1l/kk    # or grace/2l/kk, grace/3l/kk
pair_coeff * * grace_weights.npz Mo Nb Ta W
```

`newton on` is required (the usual `lmp -k on g 1 -sf kk -pk kokkos newton on neigh half ...`
sets it).

Extrapolation grade γ: see [γ in LAMMPS](uq_workflows.md#2-md-in-lammps-monitor-and-collect-extrapolative-frames)
in the UQ tutorial.

??? warning "Standard architectures only"
    The Kokkos pair styles support the **standard GRACE-1L / GRACE-2L / GRACE-3L architectures**
    ([built-in presets](presets.md) and [foundation models](foundation.md)). Custom models with
    non-standard instruction graphs, unsupported activations, or dimensions above the Kokkos
    compile-time limits are not supported: `export_kokkos` rejects them with a clear error. Use the
    TensorFlow pair styles (`grace`, `grace/2layer/parallel`, ...) or GRACE/FS for them.

??? info "Runtime precision"
    `export_kokkos` always writes the `.npz` in float64; the pair style chooses the compute
    precision. The same `.npz` works with all variants.

    | `pair_style` | NN math | Geometry |
    |---|---|---|
    | `grace/{1l,2l}/kk`       | fp64 | fp64 |
    | `grace/{1l,2l}/kk/mixed` | fp32 | fp64 |
    | `grace/{1l,2l}/kk/fp32`  | fp32 | fp32 |
    | `grace/3l/kk`            | fp32 | fp64 |
    | `grace/3l/kk/fp32`       | fp32 | fp32 |

    `fp32` and `mixed` agree with `fp64` to about **1e-6 relative** on energies and forces, well
    within typical MD needs. 3L models are natively fp32, so `grace/3l/kk` is the mixed-precision
    style and there is no fp64 3L variant.
