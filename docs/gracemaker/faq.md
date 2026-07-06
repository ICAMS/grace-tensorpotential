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
| `grace/fs` | FS | no | yes | — | always |
| `grace/fs/kk` | FS | no | yes | yes (Kokkos) | always |

Pick the row that matches your build and parallelism needs. The Kokkos rows
(`*/kk`) require a `.npz` produced by
[`grace_utils export_kokkos`](../utilities/#export-to-npz-for-lammps-kokkos-pair-style)
and run TensorFlow-free.

---

## How to run GRACE models in parallel in LAMMPS?

**Single-layer models** — use `grace/1layer/chunk` and assign one GPU per MPI rank:

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

## How to evaluate uncertainty indication for GRACE models?

**For all GRACE models:** Use naive ensembling (query-by-committee). Run parameterization with different seeds, e.g.,

```bash
gracemaker ... --seed 1
gracemaker ... --seed 2
```

This generates multiple models in `seed/{number}/`. Use these models with the ASE calculator:

```python
from tensorpotential.calculator import TPCalculator

calc_ens = TPCalculator(model=[
    "fit/seed/1/saved_model/",
    "fit/seed/2/saved_model/",
    "fit/seed/3/saved_model/",
])

at.calc = calc_ens
at.get_potential_energy()

calc.results['energy_std']  # Standard deviation of total energy predictions
calc.results['forces_std']  # Standard deviation of forces predictions
calc.results['stress_std']  # Standard deviation of stress predictions
```

**For GRACE/FS models:** In addition to the ensembling method, use extrapolation grades based on D-optimality in [ASE](../quickstart/#gracefs_1) and [LAMMPS](../quickstart/#lammps-gracefs).

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
