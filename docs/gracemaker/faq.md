# Frequently Asked Questions

## Installation and TensorFlow

### How to prevent TensorFlow from reserving all GPU memory

By default, TensorFlow takes almost all GPU memory (about 90%) at the start. To take memory
only when it is needed, set before the run:

```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

This matters when several processes share one GPU.

---

### How to reduce TensorFlow verbosity level?

```bash
export TF_CPP_MIN_LOG_LEVEL=3
```

or in Python, before TensorFlow (or `tensorpotential`) is imported:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

---

### `TypeError: 'NoneType' object is not callable` in callbacks; do I need `TF_USE_LEGACY_KERAS=1`? {#resolving-the-typeerror-nonetype-object-is-not-callable-in-tensorflow-callbacks}

GRACE needs the legacy Keras API (the `tf_keras` package, installed as a dependency).
`import tensorpotential` sets `TF_USE_LEGACY_KERAS=1` itself, so `gracemaker`, `grace_uq` and the
other tools need nothing from you.

If TensorFlow uses Keras 3 instead, the fit fails after the first epoch with:

```python
...
    if self.monitor_op(current, self.best):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'NoneType' object is not callable
```

This happens only in two cases:

* `TF_USE_LEGACY_KERAS` is already set to another value, e.g. `0` (tensorpotential then prints a
  warning and does not change it). Unset it or set it to `1`.
* Your own script imports TensorFlow *before* tensorpotential. Import tensorpotential first, or
  `export TF_USE_LEGACY_KERAS=1`.

---

### LAMMPS KOKKOS build hangs for hours on `pair_grace_2l` files

**Symptom.** A KOKKOS+CUDA build of LAMMPS with `PKG_ML-PACE` makes no progress on

```
src/KOKKOS/pair_grace_2l_kokkos.cpp
src/KOKKOS/pair_grace_2l_cpu_kokkos.cpp
```

while all other `*grace*` files compile in a few minutes. There is no error message: the build
never finishes (12 h and more). On the node, one `cicc` process per file runs at 100% CPU and its
memory does not change:

```bash
ps -eo pid,stat,%cpu,%mem,etime,cmd | grep -E "cicc|ptxas" | grep -v grep
```

**Cause.** An `nvcc` bug, not a configuration problem: the NVVM optimizer inside `cicc` loops
forever on these two files. It is **fixed in CUDA 12.8**. We saw the hang with CUDA 12.2–12.6
(A100, `compute_80`). The host compiler does not matter (gcc 11.2 and 13.4 hang the same way).

**Fix: build with CUDA ≥ 12.8.** If your HPC modules do not have it, install only the compiler
into a conda/micromamba environment. Your modules stay as they are, and you need no admin rights:

```bash
micromamba create -p ./env-cuda128 -c conda-forge \
  cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8 \
  libcublas-dev=12.8 cuda-nvtx=12.8

export CUDA_ROOT=$PWD/env-cuda128   # nvcc_wrapper reads this
export PATH=$CUDA_ROOT/bin:$PATH

cmake ... -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc
```

CUDA minor version compatibility lets a 12.8 build run with an older 12.x runtime. Still, check
the driver version with `nvidia-smi` on a compute node first; if it is close to the limit, link
`cudart` from the environment as well. There is no CUDA 12.7 (NVIDIA went from 12.6 to 12.8).

!!! note "If you must stay with system modules"
    A newer CUDA module together with a plain (not GPU-aware) OpenMPI also works: run with
    `-pk kokkos gpu/aware off`. GPU-aware MPI only makes multi-GPU runs faster; it is not
    required. So the MPI module does not have to match the CUDA version.

**What does *not* help:**

* `-Xptxas -O1`: the loop is in `cicc`, one step *before* `ptxas`. A plain `-O` sets the
  *host* optimization level and does not reach `cicc` either.
* Splitting the fp64/fp32/mixed template instantiations into separate files: a file with only
  one of the six instantiations hangs the same way.
* Building with `-j 1`: it hangs at the same place.
* Changing the host compiler.

!!! warning "Compiler flags instead of a newer CUDA"
    `-Xcicc -O0` / `-Xcicc -O1` get past the hang, but `nvcc_wrapper` passes `-Xcicc` on to the
    host compiler, which stops with an unknown-option error. So this works only when you call
    `nvcc` directly, and nobody has measured how much slower the code becomes. `-G` also works
    through `nvcc_wrapper`, but it switches off all device optimization: never benchmark such a
    build.

**Normal build cost.** With a working toolchain, each `pair_grace_2l*` file takes about 5 minutes
and up to ~4 GB of memory. Keep this in mind when you choose `-j` on a node with little memory
per core.

See [issue #36](https://github.com/ICAMS/grace-tensorpotential/issues/36) for the tests with all
toolchains.

---

## Fitting

### How to continue a current fit?

In the folder of the fit:

* `gracemaker -r` continues from the best-test-loss checkpoint;
* `gracemaker -rl` continues from the latest checkpoint.

To continue from files in another place, give them with `-p` and `-cn`, see
[Checkpoints and restart](quickstart.md#restart-from-checkpoint).

---

### How to save regular checkpoints?

`fit: checkpoint_freq: N` saves a regular checkpoint every N epochs (default 10) and keeps only
the last one. `save_all_regular_checkpoints: True` keeps all of them.
[`checkpoint_freq_steps`](inputfile.md#checkpoints-within-an-epoch) also saves within an epoch.

---

### How to perform multi-GPU fit?

On a node with several GPUs, run `gracemaker ... -m`: data-parallel fitting on all visible GPUs.
`batch_size` is per GPU, so the global batch grows with the number of GPUs. For several nodes and
large datasets, see [Several GPUs and nodes](inputfile.md#several-gpus-and-nodes).

---

### How to fit on a dataset that does not fit in memory?

Use the streaming pipeline or precomputed shards (`grace_preprocess`), see
[Large datasets](inputfile.md#large-datasets).

---

### How to fine-tune a foundation model?

See [Fine-tuning foundation models](foundation.md#fine-tuning-foundation-models). `gracemaker -t`
can write the input file for it.

---

### Can I have different cutoffs for different bond types?

Yes, with `cutoff_dict` in `input.yaml`. For example:

```yaml
cutoff: 5
cutoff_dict: {Mo: 4, MoNb: 3, W: 5, Ta*: 7}
```

`Mo` means the Mo-Mo pair, `MoNb` the Mo-Nb pair (in both directions), and `Ta*` every pair with
Ta. A pair given explicitly (here `MoNb`) wins over a `*` entry. Pairs that are not in
`cutoff_dict` use `cutoff`. Built-in tables are also available, see
the `cutoff_dict` comment in [All options](inputfile.md#all-options).

---

### What are buckets (`train_max_n_buckets` and `test_max_n_buckets`)?

GRACE models are JIT-compiled, so each new batch shape costs a compilation. `gracemaker` therefore
pads the batches to a small set of shapes, the **buckets**. Fewer buckets mean more padding (wasted
compute); more buckets mean more compilations. The default `auto` picks the number (1–32) for
each split, so that the neighbour padding stays below `auto_bucket_max_padding` (default `0.3`,
i.e. 30%; `0.15` with [dense neighbour batching](inputfile.md#dense-neighbour-batching-gpu-speed)).
Usually you do not need to change it.

```yaml
fit:
  train_max_n_buckets: auto      # default
  test_max_n_buckets: auto       # default
  # auto_bucket_max_padding: 0.3 # default; only used in "auto" mode
```

The log shows the result for each split:

```
[TRAIN] dataset stats: num. batches: 18 | num. real structures: 576 (+2.78%) | num. real atoms: 10942 (+5.25%) | num. real neighbours: 292102 (+1.74%)
```

The `+x%` after each count is the padding. `+1.74%` for the neighbours is good. Set an integer
only if the neighbour padding is above ~15% (a larger number) or if the compilations take too
long (a smaller number). These options are for the default in-memory data; streaming and
[precomputed shards](inputfile.md#precomputed-shards) have their own settings.

---

## Data

### How to provide custom weights?

Add the columns `energy_weight` (one number per structure), `force_weight` (an array with one
number per atom) and, when fitting stress, `virial_weight` (one number per structure) to the
DataFrame. See [Custom weights](inputfile.md#custom-weights).

---

## Python and ASE

### What does "Adaptive padding grew margins for the first time" mean?

`TPCalculator` pads every structure to a shape it has compiled before, to avoid new XLA
compilations. With **adaptive padding** (on by default, `adaptive_padding=True`) it watches how
often a new structure does not fit into any known shape. When this happens too often, it makes the
padding larger (`pad_atoms_number`, `pad_neighbors_fraction`), so new shapes are reused more
often. The first time, it logs at INFO level:

```
Adaptive padding grew margins for the first time (miss_rate=… > …):
pad_atoms N -> N', pad_neighbors_frac F -> F'. Further growths are
silent unless debug_padding_verbose >= 1.
```

This is only information, not an error. `debug_padding_verbose=1` reports every growth;
`adaptive_padding=False` keeps the padding values given at construction. See also
[Recompilation](quickstart.md#grace-tensorflow) on the quickstart page.

---

### How to extract basis functions from GRACE models?

Load the model from its checkpoint with the `ExtractBasisFunctions` compute function. The example
uses a foundation model; download its checkpoint first with
`grace_models checkpoint GRACE-2L-SMAX-OMAT-medium`.

```python
import os
from ase.build import bulk

from tensorpotential.tensorpot import TensorPotential
from tensorpotential.calculator import TPCalculator
from tensorpotential.tpmodel import ExtractBasisFunctions
from tensorpotential.instructions.base import load_instructions


def create_calculator_with_basis_functions(model_path, extract_2L_basis=False):
    """model_path: folder with model.yaml and checkpoint.*"""
    instr = load_instructions(os.path.join(model_path, "model.yaml"))
    tp = TensorPotential(
        instr,
        model_compute_function=ExtractBasisFunctions(
            extract_2L_basis=extract_2L_basis,  # True: also the 2nd-layer basis (2L models)
        ),
    )
    tp.load_checkpoint(checkpoint_name=os.path.join(model_path, "checkpoint"), verbose=True)
    tp.model.decorate_compute_function(jit_compile=True)

    extra_properties = ["1L_basis"]  # fixed names
    if extract_2L_basis:
        extra_properties += ["2L_basis"]
    return TPCalculator(model=tp.model,
                        truncate_extras_by_natoms=True,
                        extra_properties=extra_properties)


calc = create_calculator_with_basis_functions(
    os.path.expanduser("~/.cache/grace/checkpoints/GRACE-2L-SMAX-OMAT-medium"),
)
at = bulk("Mo")
at.calc = calc
at.get_potential_energy()
projs1 = at.calc.results["1L_basis"]  # shape [n_atoms, n_basis_1L]
# projs2 = at.calc.results["2L_basis"]  # with extract_2L_basis=True: [n_atoms, n_basis_2L]
```

---

## LAMMPS

### Which LAMMPS `pair_style` should I use?

It depends on the model and on how you run:

* **GRACE-1L/2L/3L on GPU or many CPU cores:** the TensorFlow-free Kokkos styles `grace/1l/kk`,
  `grace/2l/kk`, `grace/3l/kk`. They read a `.npz` file from
  [`grace_utils export_kokkos`](utilities.md#export-to-npz-for-lammps-kokkos-pair-style), see
  [LAMMPS Kokkos](quickstart.md#lammps-grace-1l-grace-2l-grace-3l-kokkos-no-tensorflow).
* **TensorFlow SavedModel:** `grace` for 1-layer models; for 2-layer models on several MPI ranks
  use `grace/2layer/parallel` or `grace/2layer/chunk` (`grace` runs a 2-layer model in one process
  only). For large structures use `grace/1layer/chunk` or `grace/2layer/chunk`.
* **GRACE/FS:** `grace/fs`, or `grace/fs/kk` for GPU/OpenMP.

The full table is in [LAMMPS: GRACE (TensorFlow)](quickstart.md#lammps-grace-tensorflow).

---

### How to run GRACE models in parallel in LAMMPS?

**Kokkos styles** (`grace/{1l,2l,3l}/kk`, `grace/fs/kk`): run as usual with MPI and Kokkos,
e.g. one GPU per rank; see the [Kokkos build](install.md#kokkos-build).

**TensorFlow styles:** use `grace` (1-layer) or `grace/2layer/parallel` / `grace/2layer/chunk`
(2-layer). Without CUDA-aware MPI, give each rank its own GPU:

```bash
mpirun -np 4 --bind-to none bash -c \
  'CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_RANK % 4)) lmp -in in.lammps'
```

**GRACE/FS:** `grace/fs` with standard MPI.

---

### How to get atomic virials/stress with GRACE models in LAMMPS?

All GRACE pair styles compute the global stress (pressure). Per-atom virials (e.g.
`compute stress/atom` or `compute centroid/stress/atom`) need pairwise forces. In current builds
of the `grace` LAMMPS branch, `pair_style grace` computes them by default; `no_pair_forces`
switches them off. Older builds compute them only with the `pair_forces` keyword or on more than
one MPI rank:

```
pair_style grace pair_forces
pair_coeff * * /path/to/saved_model Al Li
```

`grace/1layer/chunk`, `grace/2layer/chunk` and `grace/2layer/parallel` always give per-atom
virials.

---

## Uncertainty (extrapolation grade γ) {#uncertainty}

### How do I get the extrapolation grade γ for my model? {#how-to-evaluate-uncertainty-extrapolation-grade-gamma-for-grace-models}

γ is a per-atom number: γ ≤ 1 means the atom's environment is inside the training data,
γ ≫ 1 means extrapolation (do not trust the prediction).

* **GRACE-1L/2L/3L:** build the CALM artifact once, in the fit folder:

    ```bash
    cd seed/1
    grace_uq build
    ```

    This writes `gmm_artifacts.npz` and a `saved_model/` that also computes γ. In ASE:

    ```python
    from tensorpotential.calculator import TPCalculator

    at.calc = TPCalculator("saved_model")
    at.get_potential_energy()
    at.calc.results["gamma"]  # per-atom γ
    ```

    See [Extrapolation grade](quickstart.md#extrapolation-grade) and the
    [calculator results](uq.md#calculator-results).

* **Foundation models** with ✓ in the **UQ** column of [foundation models](foundation.md) ship
  with UQ: `grace_fm(name, enable_uq=True)`, or `calc.enable_uq()`.
* **GRACE/FS:** D-optimality with an active set, see
  [GRACE/FS: active set](quickstart.md#build-active-set-for-gracefs-only).
* **Ensembles** of models fitted with different seeds work for any model, see
  [Ensembles](uq.md#alternative-ensembling-query-by-committee).

---

### How do I use γ in LAMMPS? {#gamma-in-lammps}

Put the artifact into the Kokkos `.npz` with
[`export_kokkos --uq-artifacts`](utilities.md#baking-in-uq-uncertainty-quantification-artifacts),
then monitor γ during MD as in
[UQ workflows, step 2](uq_workflows.md#2-md-in-lammps-monitor-and-collect-extrapolative-frames).
For GRACE/FS, see [monitoring γ (GRACE/FS)](quickstart.md#monitoring-the-extrapolation-grade-in-lammps-gracefs).

---

### How do I find new training structures with γ? {#new-training-structures}

Follow [UQ workflows](uq_workflows.md): collect high-γ frames from MD or from biased MD (HAL),
pick a small diverse set with `grace_uq select`, compute them with DFT and refit.

---

### Many of my training atoms have γ > 1. Is the artifact broken? {#many-training-atoms-above-1}

With the default threshold, only about 0.1% of the training atoms should have γ > 1. If many
more do, see [Calibrating the threshold](uq.md#calibrating-the-threshold): more clusters
(`--n-clusters`) or `--threshold-percentile`.
