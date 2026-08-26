# Frequently Asked Questions

## How to prevent TensorFlow from reserving all GPU memory

By default, TensorFlow maps nearly all of the available GPU memory (typically around 90%) to the process.
To prevent this behavior and ensure memory is only allocated as needed, set the following environment variable:

```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

---

## Resolving the `TypeError: 'NoneType' object is not callable` in TensorFlow callbacks

If you encounter a `TypeError: 'NoneType' object is not callable` error, typically after the first epoch, the traceback will look similar to this:

```python
...
    if self.monitor_op(current, self.best):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'NoneType' object is not callable
```

This issue is caused by a change in how TensorFlow/Keras handles callbacks in newer versions. To resolve it, force the legacy Keras backend before running `gracemaker`:

```bash
export TF_USE_LEGACY_KERAS=1
```

!!! tip "Recurring need for `TF_USE_LEGACY_KERAS=1`"
    The same flag is also needed for [multi-GPU fits](#how-to-perform-multi-gpu-fit) and whenever a separate `keras>=3.0.0` package is installed alongside TensorFlow. If in doubt, export it once in your shell rc.

---

## How to reduce TensorFlow verbosity level?

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

or

```bash
export TF_CPP_MIN_LOG_LEVEL=3
```

---

## How to continue a current fit?

- Run `gracemaker -r` in the folder of the original fit to restart from the previous best-test-loss checkpoint.
- Run `gracemaker -rl` in the folder of the original fit to restart from the latest checkpoint.
- To continue in a new folder, copy `seed/{number}/checkpoints` and `seed/{number}/model.yaml` into the new folder.

---

## How to save regular checkpoints?

Use `checkpoint_freq` to specify how frequently to save regular checkpoints (only the last state will be saved).
To keep all regular checkpoints, add the flag `input.yaml::fit::save_all_regular_checkpoints: True`.

---

## How to perform multi-GPU fit?

If you have a node with multiple GPUs, use the `gracemaker ... -m` option to enable data-parallel fitting. In this case, increase the batch size (global batch size).

!!! note "Legacy Keras flag may be required"
    Multi-GPU runs sometimes need `export TF_USE_LEGACY_KERAS=1` — see the [callback `TypeError` workaround](#resolving-the-typeerror-nonetype-object-is-not-callable-in-tensorflow-callbacks) above.

---

## Can I have different cutoffs for different bond types?

Yes, you can specify bond-specific cutoffs using the `input.yaml::cutoff_dict` option. For example:

```yaml
cutoff_dict: {Mo: 4, MoNb: 3, W: 5, Ta*: 7}
```

This can be used alongside `input.yaml::cutoff`.

---

## How to provide custom weights?

To assign custom weights to each structure, include the following columns in the DataFrame:

* `energy_weight`: A single value representing the weight for each structure.
* `force_weight`: A per-atom array with a size equal to the number of atoms in the structure.
* `virial_weight`: (optional): A six-component array representing the weight for virial terms.

---

## LAMMPS KOKKOS build hangs for hours on `pair_grace_2l` files

**Symptom.** A KOKKOS+CUDA build of LAMMPS with `PKG_ML-PACE` makes no progress on

```
src/KOKKOS/pair_grace_2l_kokkos.cpp
src/KOKKOS/pair_grace_2l_cpu_kokkos.cpp
```

while all other `*grace*` files compile in a couple of minutes. There is no
error message — the build simply never finishes (12 h and more). Inspecting the
node shows one `cicc` process per file pinned at 100% CPU with a flat memory
footprint:

```bash
ps -eo pid,stat,%cpu,%mem,etime,cmd | grep -E "cicc|ptxas" | grep -v grep
```

**Cause.** This is an `nvcc` bug, not a configuration problem: the NVVM
optimizer inside `cicc` loops forever on these two translation units. It is
**fixed in CUDA 12.8**. Observed to hang on CUDA 12.2–12.6 (A100/`compute_80`);
the host compiler is irrelevant (gcc 11.2 and 13.4 hang identically).

**Fix — build with CUDA ≥ 12.8.** If your HPC modules do not offer it, install
just the compiler into a conda/micromamba environment and leave your modules
untouched (no admin rights needed):

```bash
micromamba create -p ./env-cuda128 -c conda-forge \
  cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8 \
  libcublas-dev=12.8 cuda-nvtx=12.8

export CUDA_ROOT=$PWD/env-cuda128   # nvcc_wrapper honours this
export PATH=$CUDA_ROOT/bin:$PATH

cmake ... -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc
```

Building with 12.8 against an older 12.x runtime is covered by CUDA minor
version compatibility, but check the driver version with `nvidia-smi` on a
compute node first; if it is tight, link `cudart` from the environment as well.
Note that there is no CUDA 12.7 (NVIDIA went 12.6 → 12.8).

!!! note "If you must stay with system modules"
    A newer CUDA module combined with a plain (non-GPU-aware) OpenMPI also
    works — just run with `-pk kokkos gpu/aware off`. GPU-aware MPI is a
    multi-GPU throughput optimization here, not a requirement, so this
    decouples the MPI module from the CUDA version.

**What does *not* help:**

* `-Xptxas -O1` — the loop is in `cicc`, one stage *upstream* of `ptxas`.
  A plain `-O` sets the *host* optimization level and does not reach `cicc`
  either.
* Splitting the fp64/fp32/mixed template instantiations into separate
  translation units — a variant with only one of the six instantiations hangs
  the same way, so this is not template volume.
* Serializing the build (`-j 1`) — it hangs at the same spot.
* Changing the host compiler.

!!! warning "Flag workarounds instead of upgrading"
    `-Xcicc -O0` / `-Xcicc -O1` do get past the hang, but `nvcc_wrapper`
    forwards `-Xcicc` to the host compiler, which then dies on the
    unrecognized option — it only works when calling `nvcc` directly, and the
    runtime cost of the lower NVVM optimization level is not characterized.
    `-G` also works through `nvcc_wrapper`, but it disables device
    optimization entirely — never benchmark a build made that way.

**Expected build cost (healthy toolchain).** Each `pair_grace_2l*` file needs
roughly 5 minutes and ~4 GB peak RSS on its own — worth keeping in mind when
choosing `-j` on a node with limited memory per core.

See [issue #36](https://github.com/ICAMS/grace-tensorpotential/issues/36) for
the full bisection across toolchains.

---

## Which LAMMPS `pair_style` should I use?

| `pair_style` | Model | TF required | MPI | GPU/OpenMP | Virials/stress |
|---|---|---|---|---|---|
| `grace` | 1-layer | yes | yes | — | needs `pair_forces` |
| `grace` | 2-layer | yes | no | — | needs `pair_forces` |
| `grace/1layer/chunk` | 1-layer | yes | yes | — | always |
| `grace/2layer/chunk` | 2-layer | yes | yes | — | always |
| `grace/2layer/parallel` | 2-layer | yes | yes | — | always |
| `grace/1l/kk` | 1-layer | no | yes | yes (Kokkos) | always |
| `grace/2l/kk` | 2-layer | no | yes | yes (Kokkos) | always |
| `grace/3l/kk` | 3-layer | no | yes | yes (Kokkos) | always |
| `grace/fs` | FS | no | yes | — | always |
| `grace/fs/kk` | FS | no | yes | yes (Kokkos) | always |

Pick the row that matches your build and parallelism needs. The Kokkos rows
(`*/kk`) require a `.npz` produced by
[`grace_utils export_kokkos`](../utilities/#export-to-npz-for-lammps-kokkos-pair-style)
and run TensorFlow-free.

---

## How to run GRACE models in parallel in LAMMPS?

**Single-layer models** — use `grace` (or `grace/1layer/chunk` for large structures
and guaranteed virials) and assign one GPU per MPI rank:

```bash
mpirun -np 4 --bind-to none bash -c \
  'CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_RANK % 4)) lmp -in in.lammps'
```

**Two-layer models** — use `grace/2layer/chunk` or `grace/2layer/parallel` (MPI is handled natively, no shell tricks needed):

```
pair_style grace/2layer/chunk
pair_coeff * * /path/to/2layer_saved_model Al Li
```

**GRACE/FS** — standard MPI with `grace/fs`; for GPU/OpenMP use `grace/fs/kk` (requires `newton on`).

---

## How to get atomic virials/stress with GRACE models in LAMMPS?

With `pair_style grace`, pairwise forces (needed for per-atom virials) are not computed by default. Enable them with:

```
pair_style grace pair_forces
pair_coeff * * /path/to/saved_model Al Li
```

`pair_forces` is automatically enabled when running with more than one MPI rank.

Alternatively, use `grace/1layer/chunk`, `grace/2layer/chunk`, or `grace/2layer/parallel` — these always support virials without any extra keyword.

---

## How to evaluate uncertainty (extrapolation grade `gamma`) for GRACE models?

Use the per-atom extrapolation grade **`gamma`** — the single UQ signal
reported by GRACE models. It is the Mahalanobis distance of an atomic
environment to its nearest cluster in the model's own latent space,
normalized by a calibrated per-cluster threshold, so it is dimensionless:

* $\gamma \lesssim 1$ — the environment lies inside the training distribution.
* $\gamma \approx 1$ — the atom sits at the boundary of the training distribution.
* $\gamma \gg 1$ — extrapolation; treat the prediction as unreliable.

**GRACE-1L/2L/3L models** need an NCM-UQ artifact, built once from the training
set with [`grace_uq build`](../uq/#grace_uq-build):

```bash
grace_uq build --model-yaml model.yaml \
               --checkpoint checkpoints/checkpoint.best_test_loss.index \
               --train-data training_set.pkl.gz \
               --artifact-path UQ/gmm_artifacts.npz
```

Alongside the artifact this writes a `saved_model/` with UQ baked in — just
load it with the usual calculator and read `gamma` from the results:

```python
from tensorpotential.calculator import TPCalculator

at.calc = TPCalculator(model="UQ/saved_model")
at.get_potential_energy()
at.calc.results["gamma"]         # per-atom extrapolation grades
at.calc.results["atomic_sigma"]  # raw, unnormalized Mahalanobis distances
```

UQ is detected and enabled automatically; call `calc.disable_uq()` if you want
the faster non-UQ path (and `calc.enable_uq()` to switch back).

!!! tip "Foundation models often ship UQ already"
    Many distributed models come with `gmm_artifacts.npz` and a UQ head — no
    build step needed. Check the **UQ** column in the
    [foundation models](../foundation/) tables.

**GRACE/FS models** use extrapolation grades based on D-optimality instead:
[build an active set (ASI)](../quickstart/#build-active-set-for-gracefs-only),
then read `gamma` from
[`PyGRACEFSCalculator`](../quickstart/#gracefs_1) in ASE or from
[`pair_style grace/fs extrapolation`](../quickstart/#lammps-gracefs) in LAMMPS.

**Screening datasets and active learning:** `grace_uq predict` evaluates
energies/forces/stresses plus per-atom γ over a whole dataset, and
`grace_uq select` picks N structures from a candidate pool by
extrapolation/diversity strategy.

**In LAMMPS** with the Kokkos pair styles, bake the artifact into the weights
file with
[`export_kokkos --uq-artifacts`](../utilities/#baking-in-uq-uncertainty-quantification-artifacts) —
γ is then computed from the same `.npz` at runtime, with no separate UQ file.

For a model without a UQ artifact — or for a second, independent opinion — you
can also fit several models with different seeds and use their spread:
see [ensembling (query-by-committee)](../uq/#alternative-ensembling-query-by-committee).

See the [Uncertainty Quantification](../uq/) page for the full pipeline,
options, and the Python API.

---

## What are buckets (`train_max_n_buckets` and `test_max_n_buckets`)?

GRACE models are JIT-compiled, so every batch must share a shape — achieved
by padding inputs into a small set of **buckets**, each pre-padded to a
common shape. Fewer buckets means more padding (wasted compute); more
buckets means more JIT recompilation cost. By default, `gracemaker` picks
the count automatically per split — you usually do not need to touch
this.

```yaml
fit:
  train_max_n_buckets: auto      # default
  test_max_n_buckets: auto       # default
  # auto_bucket_max_padding: 0.3 # default; only used in "auto" mode
```

In `"auto"` mode, the number of buckets is chosen dynamically (clamped to
1–32) so that the neighbour-padding overhead stays under
`auto_bucket_max_padding` (default `0.3`, i.e. 30%). The chosen count and
the resulting padding are visible in the per-split log line:

```
[TRAIN] dataset stats: num. batches: 18 | num. real structures: 576 (+2.78%) | num. real atoms: 10942 (+5.25%) | num. real neighbours: 292102 (+1.74%)
```

The `+x%` after each count is the padding overhead — `+1.74%` for
neighbours here is well-tuned. Override only if you see overhead above ~15%
(set a larger integer) or if recompilation cost dominates (set a smaller
integer).

---

## What does “Adaptive padding grew margins for the first time” mean?

`TPCalculator` ships with **adaptive padding** enabled by default
(`adaptive_padding=True`). At inference time the calculator keeps an
ordered set of padded shape buckets and tries to reuse them across calls
to avoid XLA recompilation. When the recent miss-rate (new structures that
fall outside every existing bucket) exceeds a threshold, the calculator
grows the per-atom and per-neighbour padding margins so the next bucket it
adds is roomier and gets reused more often.

On the **first** growth event, the calculator logs at INFO:

```
Adaptive padding grew margins for the first time (miss_rate=… > …):
pad_atoms N → N', pad_neighbors_frac F → F'. Further growths are silent
unless debug_padding_verbose >= 1.
```

This is a one-time confirmation that adaptation is doing something —
subsequent growths stay silent. If you would rather see every growth,
construct the calculator with `debug_padding_verbose=1`. To disable
adaptation entirely, pass `adaptive_padding=False`; the calculator then
uses the fixed `pad_atoms_number` / `pad_neighbors_fraction` margins from
construction.

---

## How to extract basis functions from GRACE models?

```python
from tensorpotential.tensorpot import TensorPotential
from tensorpotential.calculator import TPCalculator
from tensorpotential.tpmodel import ExtractBasisFunctions
from tensorpotential.instructions.base import load_instructions

from ase.build import bulk

def create_calculator_with_basis_functions(
        model_path,
        extract_2L_basis=False
):
    # path to checkpoint/model.yaml

    instr = load_instructions(model_path + '/model.yaml')

    tp = TensorPotential(
        instr,
        model_compute_function=ExtractBasisFunctions(
            extract_2L_basis=extract_2L_basis # set to True if you want to extract 2L basis functions

            ### optional parameters
            #reduce_1L_instruction_name='I_out_0', # this name depends on the model
            #reduce_2L_instruction_name='I_out_1', # this name depends on the model
        )
    )
    tp.load_checkpoint(checkpoint_name=model_path + '/checkpoint', verbose=True)
    tp.model.decorate_compute_function(jit_compile=True) # compile model

    # these names are fixed
    extra_properties=['1L_basis']
    if extract_2L_basis:
        extra_properties+=['2L_basis']
    calc = TPCalculator(model=tp.model,
                        truncate_extras_by_natoms=True,
                        extra_properties=extra_properties
                        )
    return calc



import os
calc = create_calculator_with_basis_functions(
    model_path=os.path.expanduser("~/.cache/grace/checkpoints/GRACE-2L-UEA-OMAT-medium/"),
)
at = bulk('Mo')
at.calc = calc
at.get_potential_energy()
projs1 = at.calc.results['1L_basis'] # shape [n_atoms, n_basis_1L]
# projs2 = at.calc.results['2L_basis'] # shape [n_atoms, n_basis_2L]
```
