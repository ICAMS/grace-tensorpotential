## Stages of Model Parameterization

1. [Data Collection](#data-collection)
2. [Input File Setup](#input-file-setup)
3. [Parameterization with `gracemaker`](#grace-model-parameterization)
4. [Export model](#export-model)
5. [Build active set (for GRACE/FS only)](#build-active-set-for-gracefs-only)  
6. Usage in [ASE](#usage-in-ase) and [LAMMPS](#usage-in-lammps)
_____
### Data Collection

You can either use [`grace_collect`](../utilities#grace_collect) 
```bash
grace_collect
```

or build a `pandas.DataFrame` on you own. It must contain the following columns:
 
* `ase_atoms` - Atomic structures represented as ASE Atoms
* `energy` - Total energy (should be force-consistent), shape: single number 
* `energy_corrected` (optional) - Cohesive energy, i.e., total energy after subtraction of free atoms energies, shape: single number
* `forces` - Per-atom forces, shape: (number_of_atoms, 3)
* `stress` - six components of stress

Note, that the `energy_corrected` column is not mandatory and can be constructed automatically by _gracemaker_, 
if you provide `reference_energy`.

Alternatively, datasets in extended xyz format can either be converted to the
format above using the `extxyz2df` tool shipped with this package or used
directly in the input file.

_____

### Input File Setup

You can use `gracemaker -t` for an interactive interface to configure simple input file
and then adjust it manually, if needed. Check [model hyperparameters](../presets/#models-hyperparameters) for more details.
For complete information about input file, see [Input file](../inputfile).

_____

### GRACE Model Parameterization

To start `gracemaker`, simply run
```bash
gracemaker
```

Please note that the first line of input file should be `seed: 1`, which sets the random see for initializing the model
parameters. When `gracemaker` starts, it creates a working subfolder, (e.g., _seed/1/_)  containing the following output files:

* **log.txt** — Redirected log output.
* **train_metrics.yaml** and **test_metrics.yaml** — Various training and testing metrics in YAML format.
* **checkpoints/** — Model checkpoints
* **model.yaml** — Stores model architecture and hyperparameters.

_____

### Checkpointing 

During fit _gracemaker_ saves current model state every time new best test_loss is observed.
This checkpoint can be found in _seed/\*/checkpoints/checkpoint.best_test_loss.\*_. 

In addition, regular checkpointing is performed and its frequency is controlled  via `checkpoint_freq` flag. 
Only the last one is kept by default, but every regular checkpoint can be kept specifying `save_all_regular_checkpoints: True`  

_____

### Restart from checkpoint 

To initialized fitting from previously saved state, simply run
```bash
gracemaker -r
```
This will try restarting fit from the `checkpoint.best_test_loss` and `model.yaml` from the default 
locations at `./seed/*/checkpoints/checkpoint.best_test_loss` and `./seed/*/` respectively.

You can also explicitly provide path to the checkpoint and model configuration file:
  
```bash
gracemaker -r -p /path/to/model.yaml -cn /path/to/chekpoint.index
```

or in `input.yaml`:

```yaml
potential:
  filename: /path/to/model.yaml
  checkpoint_name: /path/to/chekpoint.index
```
_____

### Export Model 

#### TensorFlow's SavedModel
After `gracemaker` completes successfully, the model is exported in TensorFlow [SavedModel](https://www.tensorflow.org/guide/saved_model) 
format and is saved in the `seed/1/final_model` directory.

To export GRACE model from a checkpoint at any time, use:

```bash
gracemaker -r -s
```
to export the model with the _best test loss_. Replace `-r` with `-rl` flag to export the model from the _latest saved checkpoint_.

In all cases, the model will be saved in the `seed/1/saved_model` folder and can be used with [ASE](#grace-tensorflow) or [LAMMPS](#lammps-grace-tensorflow).

#### GRACE/FS

To export the GRACE/FS model in _YAML_ format (for using with a custom C++ implementation), add the `-sf` flag to the above commands:
```bash
gracemaker -r -s -sf
```
Here `-r` flag stands for reading checkpoint with best test loss (usually, it is `seed/1/checkpoints/checkpoint.best_test_loss.*` ),
`-s` flag is for saving the model immediately (without running fit) into TensorFlow SavedModel format and `-sf` is for also saving into GRACE/FS YAML format.

This will generate the `seed/1/FS_model.yaml` file, which can be used in [ASE](#gracefs) or [LAMMPS](#lammps-gracefs).

#### LAMMPS Kokkos `.npz` (GRACE-1L / GRACE-2L)

For TensorFlow-free GPU/OpenMP runs with `pair_style grace/1l/kk` or `grace/2l/kk`,
export the model weights to `.npz` using
[`grace_utils export_kokkos`](../utilities/#export-to-npz-for-lammps-kokkos-pair-style):

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index export_kokkos -o grace_weights.npz
```

The architecture (1L vs 2L) is auto-detected. The resulting `.npz` is loaded directly
by the matching Kokkos pair style — see [LAMMPS: GRACE-1L / GRACE-2L (Kokkos, no TensorFlow)](#lammps-grace-1l-grace-2l-kokkos-no-tensorflow).

---

### Build Active Set (for GRACE/FS Only)

For the GRACE/FS model, you can generate an active set (ASI) file to compute the extrapolation grade using D-optimality. 
Before doing this, ensure that `python-ace` is installed (see the [installation guide](../install/#gracefs-cpu)).

To create active set, use the `pace_activeset` utility, following a process similar 
to [ML-PACE](https://pacemaker.readthedocs.io/en/latest/pacemaker/active_learning/#extrapolation_grade_and_active_learning).

Example:

```bash
cd seed/1
pace_activeset -d training_set.pkl.gz FS_model.yaml
```

---

### Usage in ASE

You can use the exported GRACE model with ASE-compatible calculators, such as `TPCalculator` or `PyGRACEFSCalculator`.

#### GRACE (TensorFlow)
To use it as an ASE calculator:

```python
from tensorpotential.calculator import TPCalculator
calc = TPCalculator("/path/to/saved_model")
```

To load foundation models:
```python
from tensorpotential.calculator import grace_fm

calc = grace_fm('name_of_the_model') 
```

For both functions (TPCalculator and grace_fm), you can adjust options for padding and minimal allowed bond distance:

```python
calc = TPCalculator('path/to/saved_model',
                     pad_neighbors_fraction = 0.05,
                     pad_atoms_number = 2,
                     min_dist=0.5)
```
If `min_dist` is given, calculator will raise an exception when it encounters a distance shorter than `min_dist`
(useful for preventing non-physical short distances during relaxation).

##### Aggregation engine: the `mode` argument

`TPCalculator` has two neighbor-aggregation engines, selected with `mode`. There is no single winner —
pick by your workload:

| `mode` | Engine | Use it for | Why |
|---|---|---|---|
| `"uniform"` | dense (reshape) | a **single structure**, an **MD / relaxation trajectory**, or a **uniform dataset** (all evaluated structures have ~the same size) | ~1.3–1.7× faster per structure, and *order-invariant*. Pads tight from the start. |
| `"diverse"` (default) | segment_sum | scanning **many very different structures** (a heterogeneous dataset) | tolerant of size variation without exploding the number of XLA compiles |

```python
# single structure / MD / relaxation / uniform dataset
calc = TPCalculator('path/to/saved_model', mode="uniform")

# scanning a heterogeneous dataset
calc = TPCalculator('path/to/saved_model', mode="diverse")  # the default
```

**Why two modes (and why no automatic switch):** the dense engine is faster for any *given* structure,
but it pads tightly, so a stream of differently-sized structures forces a fresh XLA compile per distinct
size. segment_sum tolerates size variation with far fewer compiles. The calculator evaluates one structure
per call and cannot see the whole sequence in advance, so it cannot reliably auto-detect the regime —
hence an explicit, documented switch.

**Tip for `mode="diverse"` scans — sort by decreasing atom count.** Feeding structures **largest first**
lets the first compiled shape cover all the rest, minimizing recompiles. Ascending (smallest first) is the
worst case: it can mint a fresh compile at nearly every size step (an order of magnitude more compiles than
descending). The dense (`"uniform"`) engine, by contrast, is order-invariant.

**Fallback:** if the model cannot run the requested engine (an in-memory model is single-engine, fixed at
build time; a SavedModel may export only `compute`), the calculator falls back to the available engine and
logs a warning. SavedModels exported from a dense-capable model carry both engines and switch freely.

---

#### GRACE/FS

**Note:** The `python-ace` package is required (see the [installation guide](../install/#gracefs-cpu)).

To use the **GRACE/FS** calculator (with C++ implementations), you need first build active set (see [here](#build-active-set-for-gracefs-only)) and then use the following code:

```python
from pyace.asecalc import PyGRACEFSCalculator
calc = PyGRACEFSCalculator("/path/to/FS_model.yaml")
calc.set_active_set("/path/to/FS_model.asi")

at.calc = calc
at.get_potential_energy()
at.calc.results['gamma'] # per-atom extrapolation grades
```

_____

### Usage in LAMMPS

All GRACE pair styles require `units metal`. The right style depends on your model type and parallelization needs:

| `pair_style` | Model | TF required | MPI parallel | Virials/stress |
|---|---|---|---|---|
| `grace` | 1-layer | yes | yes | needs `pair_forces` |
| `grace` | 2-layer | yes | single process | needs `pair_forces` |
| `grace/1layer/chunk` | 1-layer | yes | yes | always available |
| `grace/2layer/chunk` | 2-layer | yes | yes | always available |
| `grace/2layer/parallel` | 2-layer | yes | yes | always available |
| `grace/1l/kk` | 1-layer (Kokkos) | no | yes + GPU/OpenMP | always available |
| `grace/2l/kk` | 2-layer (Kokkos) | no | yes + GPU/OpenMP | always available |
| `grace/fs` | FS | no | yes | always available |
| `grace/fs/kk` | FS (Kokkos) | no | yes + GPU/OpenMP | always available |

#### LAMMPS: GRACE (TensorFlow)

The TensorFlow-based styles load a `saved_model` directory. For downloaded foundation models the path is `~/.cache/grace/<model_name>/`.

The simplest invocation:
```
pair_style grace
pair_coeff * * /path/to/saved_model Al Li
```

**Key options for `grace`:**

- `padding <frac>` — fraction of neighbors to pad to reduce JIT recompilations (default: 0.01)
- `pad_verbose` — print a message whenever recompilation is triggered
- `pair_forces` — compute pairwise forces; **required for virials/stress** and automatically enabled when running with more than one MPI rank for GRACE-1L models

```
pair_style grace padding 0.05 pad_verbose pair_forces
pair_coeff * * /path/to/saved_model Al Li
```

**For single-layer models with LARGE structures**, use `grace/1layer/chunk`. It processes atoms in blocks (`chunksize`, default 4096), always supports virials, and does not need `pair_forces`:

```
pair_style grace/1layer/chunk chunksize 2048 padding 0.05
pair_coeff * * /path/to/saved_model Al Li
```

**For two-layer models with LARGE structures and (optionally) MPI**, use `grace/2layer/chunk` or `grace/2layer/parallel`, correspondingly:

```
pair_style grace/2layer/chunk
pair_coeff * * /path/to/2layer_saved_model Al Li
```

> **Note:** GPU usage is optional but strongly recommended — TensorFlow is significantly less efficient on CPU-only runs.

#### Multi-GPU / MPI parallelization

* Use `grace` for single-process runs for both single-layer and two-layer models.
* For single-layer models parallization also use `grace` (with CUDA-aware mpirun command)
* For two-layer models parallization use `grace/2layer/parallel` (with CUDA-aware mpirun command)
* For memory-reduced version use `grace/1layer/chunk` or `grace/2layer/chunk` - both of them support MPI parallelism.
* For GRACE/FS models use `grace/fs` (or `grace/fs/kk` for GPU/OpenMP).
* For TensorFlow-free GPU/OpenMP runs of GRACE-1L or GRACE-2L models use
  `grace/1l/kk` / `grace/2l/kk` after exporting weights to `.npz` with
  [`grace_utils export_kokkos`](../utilities/#export-to-npz-for-lammps-kokkos-pair-style).

If you don't have CUDA-aware MPI, use
```bash
mpirun -np 4 --bind-to none bash -c \
  'CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_RANK % 4)) lmp -in in.lammps'
```

#### LAMMPS: GRACE/FS

GRACE/FS has a native C++ implementation — no TensorFlow or GPU required, with full MPI support:

```
pair_style grace/fs
pair_coeff * * FS_model.yaml Mo Nb Ta W
```

A Kokkos-accelerated variant (`grace/fs/kk`) adds GPU and OpenMP support (`newton on` required):

```
pair_style grace/fs/kk
pair_coeff * * FS_model.yaml Mo Nb Ta W
```

#### LAMMPS: GRACE-1L / GRACE-2L (Kokkos, no TensorFlow)

`grace/1l/kk` and `grace/2l/kk` run GRACE-1L / GRACE-2L models on
GPU/OpenMP without TensorFlow at LAMMPS runtime. They read a `.npz`
weights file produced from a fitted `model.yaml` + checkpoint by
[`grace_utils export_kokkos`](../utilities/#export-to-npz-for-lammps-kokkos-pair-style).

!!! warning "Standard architectures only"
    The Kokkos pair styles support the **standard GRACE-1L / GRACE-2L
    architectures** ([built-in presets](../presets/) and
    [foundation models](../foundation/)). Custom GRACE models with
    non-standard instruction graphs, unsupported activations, or
    dimensions exceeding the Kokkos compile-time caps are not supported —
    `export_kokkos` will reject them with a clear error. For arbitrary
    custom architectures use the TensorFlow-based pair styles
    (`grace`, `grace/2layer/parallel`, …) or GRACE-FS instead.

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index export_kokkos -o grace_weights.npz
```

Then in the LAMMPS input:

```
pair_style grace/1l/kk    # or grace/2l/kk
pair_coeff * * grace_weights.npz Mo Nb Ta W
```

`newton on` is required (set automatically by the standard `lmp_kk -k on g 1 -sf kk -pk kokkos newton on neigh half ...` invocation).

**Runtime precision.** `export_kokkos` always writes the `.npz` in float64, but
the LAMMPS pair style picks the compute precision:

| `pair_style` | NN math | Geometry |
|---|---|---|
| `grace/{1l,2l}/kk`       | fp64 | fp64 |
| `grace/{1l,2l}/kk/mixed` | fp32 | fp64 |
| `grace/{1l,2l}/kk/fp32`  | fp32 | fp32 |

The same `.npz` works with all variants — choose by your accuracy / throughput
trade-off. Empirically, `fp32` and `mixed` agree with `fp64` to roughly
**1e-6 relative precision** on energies and forces — well within typical MD
requirements.

To monitor extrapolation grade:

```
pair_style grace/fs extrapolation
pair_coeff * * FS_model.yaml FS_model.asi Mo Nb Ta W

fix grace_gamma all pair 100 grace/fs gamma 1
compute max_grace_gamma all reduce max f_grace_gamma
variable max_grace_gamma equal c_max_grace_gamma
fix extreme_extrapolation all halt 10 v_max_grace_gamma > 25
```




