## Resolving the `TypeError: 'NoneType' object is not callable` in TensorFlow Callbacks

If you encounter a `TypeError: 'NoneType' object is not callable` error, typically after the first epoch, the traceback will look similar to this:
```python
...
    if self.monitor_op(current, self.best):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'NoneType' object is not callable
```
This issue often occurs due to a change in how TensorFlow/Keras handles callbacks in newer versions.
To resolve this, ensure that you set the following environment variable before running `gracemaker`:
```bash
export TF_USE_LEGACY_KERAS=1
```

Setting this variable forces the use of the legacy Keras backend, which resolves compatibility conflicts with certain callback implementations.

---

## How to Continue a Current Fit?

- Run `gracemaker -r` in the folder of the original fit to restart from the previous best-test-loss checkpoint.  
- Run `gracemaker -rl` in the folder of the original fit to restart from the latest checkpoint.  
- To continue in a new folder, copy `seed/{number}/checkpoints` and `seed/{number}/model.yaml` into the new folder.  

---

## How to Save Regular Checkpoints?

Use `checkpoint_freq` to specify how frequently to save regular checkpoints (only the last state will be saved).  
To keep all regular checkpoints, add the flag `input.yaml::fit::save_all_regular_checkpoints: True`.  

---

## Can I Have Different Cutoffs for Different Bond Types?

Yes, you can specify bond-specific cutoffs using the `input.yaml::cutoff_dict` option. For example:  
```yaml
cutoff_dict: {Mo: 4, MoNb: 3, W: 5, Ta*: 7}
```  
This can be used alongside `input.yaml::cutoff`.  

---

## What Are Buckets (`train_max_n_buckets` and `test_max_n_buckets`)?

GRACE models are JIT-compiled, which requires all batches to have the same size. This is achieved through padding. To improve efficiency, batches are split into buckets that are then padded.  

The parameters `train_max_n_buckets` and `test_max_n_buckets` define the maximum number of buckets for padding training and testing data, respectively. More buckets reduce padding.  

To estimate the optimal number of buckets, refer to the `[TRAIN] dataset stats:` log line, which shows padding information:  
```
[TRAIN] dataset stats: num. batches: 18 | num. real structures: 576 (+2.78%) | num. real atoms: 10942 (+5.25%) | num. real neighbours: 292102 (+1.74%)
```  
Here, the padding for neighbors is only +1.74%. It is recommended to keep this value below 15%. The same applies to `[TEST] dataset stats`.  

---

## Which LAMMPS `pair_style` Should I Use?

| `pair_style` | Model           | TF required | MPI parallel | Virials/stress |
|---|-----------------|---|---|---|
| `grace` | 1-layer/2-layer | yes | single process | needs `pair_forces` |
| `grace/1layer/chunk` | 1-layer         | yes | yes | always available |
| `grace/2layer/chunk` | 2-layer         | yes | yes | always available |
| `grace/2layer/parallel` | 2-layer         | yes | yes | always available |
| `grace/fs` | FS              | no | yes | always available |
| `grace/fs/kk` | FS (Kokkos)     | no | yes + GPU/OpenMP | always available |

* Use `grace` for single-process runs for both single-layer and two-layer models.
* For single-layer models parallization also use `grace` (with CUDA-aware mpirun command)
* For two-layer models parallization use `grace/2layer/parallel` (with CUDA-aware mpirun command)
* For memory-reduced version use `grace/1layer/chunk` or `grace/2layer/chunk` - both of them support MPI parallelism.
* For GRACE/FS models use `grace/fs` (or `grace/fs/kk` for GPU/OpenMP).

---

## How to Run GRACE Models in Parallel in LAMMPS?

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

## How to Evaluate Uncertainty Indication for GRACE Models?

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

## How to Perform Multi-GPU Fit?

If you have a node with multiple GPUs, use the `gracemaker ... -m` option to enable data-parallel fitting. In this case, increase the batch size (global batch size).  
Sometimes you need to use following env variable: `export TF_USE_LEGACY_KERAS=1`
---

## How to Reduce TensorFlow Verbosity Level?

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

or  

```bash
export TF_CPP_MIN_LOG_LEVEL=3
```  

## How to Provide Custom Weights

To assign custom weights to each structure, include the following columns in the DataFrame:

* `energy_weight`: A single value representing the weight for each structure.
* `force_weight`: A per-atom array with a size equal to the number of atoms in the structure.
* `virial_weight`: (optional): A six-component array representing the weight for virial terms.

## How to Get Atomic Virials/Stress with GRACE Models in LAMMPS?

With `pair_style grace`, pairwise forces (needed for per-atom virials) are not computed by default. Enable them with:

```
pair_style grace pair_forces
pair_coeff * * /path/to/saved_model Al Li
```

`pair_forces` is automatically enabled when running with more than one MPI rank.

Alternatively, use `grace/1layer/chunk`, `grace/2layer/chunk`, or `grace/2layer/parallel` — these always support virials without any extra keyword.

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



calc = create_calculator_with_basis_functions(
    model_path = "~/.cache/grace/checkpoints/GRACE-2L-UEA-OMAT-medium/",
)
at = bulk('Mo')
at.calc = calc
at.get_potential_energy()
projs1 = at.calc.results['1L_basis'] # shape [n_atoms, n_basis_1L]
# projs2 = at.calc.results['2L_basis'] # shape [n_atoms, n_basis_2L]
```
