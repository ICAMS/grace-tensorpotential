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
* `energy_corrected` (optional) - Cohesive energy, i.e., total energy after subtraction of free atoms' energies, shape: single number
* `forces` - Per-atom forces, shape: (number_of_atoms, 3) 

Note, that the `energy_corrected` column is not mandatory and can be constructed automatically by _gracemaker_, 
if you provide `reference_energy`

_____

### Input File Setup

You can use `gracemaker -t` for an interactive interface to configure simple input file (not yet implemented)
and then adjust it manually as needed. For more information, see [Input file](../inputfile).

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
* **checkpoints/** — Model checkpoints, including the best test loss checkpoint (suffix .best_test_loss) and the latest checkpoint (no suffix).

Additionally, a file named _model.yaml_ will be created in the current working directory. This file contains the architecture of the model.

_____

Here's a refined version with improved grammar and style:

---

### Export Model

After `gracemaker` completes successfully, the exported model in [saved_model](https://www.tensorflow.org/guide/saved_model) 
format (used by TensorFlow) will be saved in the `seed/1/final_model` directory.

To export the GRACE model from a checkpoint at any time, use:

* `gracemaker -r -s` to export the model with the _best test loss_.
* `gracemaker -rl -s` to export the model from the _latest saved checkpoint_.

In all cases, the model will be saved in the `seed/1/saved_model` folder and can be used with [ASE](#grace-tensorflow) or [LAMMPS](#lammps-grace-tensorflow).

To export the GRACE/FS model in _YAML_ format (for use with a custom C++ implementation), add the `-sf` flag to the above commands.
This will generate the `seed/1/FS_model.yaml` file, which can be used in [ASE](#gracefs) or [LAMMPS](#lammps-gracefs).

_____

### Build Active Set (for GRACE/FS Only)

For the GRACE/FS model, you can generate an active set (ASI) file to compute the extrapolation grade using D-optimality. 
Before doing this, ensure that `python-ace` is installed (see the [installation guide](../install/#gracefs-cpu)).

To create the active set, use the `pace_activeset` utility, following a process similar 
to [ML-PACE](https://pacemaker.readthedocs.io/en/latest/pacemaker/active_learning/#extrapolation_grade_and_active_learning).

Example:

```bash
cd seed/1
pace_activeset -d training_set.pkl.gz FS_model.yaml
```

_____

### Usage in ASE

You can use the exported GRACE model with ASE-compatible calculators, such as `TPCalculator` or `PyGRACEFSCalculator`.

#### GRACE (TensorFlow)
To use it as an ASE calculator:

```python
from tensorpotential.calculator import TPCalculator
calc = TPCalculator("/path/to/saved_model")
```

To load foundational models:
```python
from tensorpotential.calculator import grace_fm

calc=grace_fm('name_of_the_model') 
```

For both functions (TPCalculator and grace_fm), you can adjust options for padding and minimal distance:
```python
calc=TPCalculator(model_path,
        pad_neighbors_fraction = 0.05,
        pad_atoms_number = 2,
        min_dist=0.5
)
```
If `min_dist` is given, calculator will raise an exception when it encounters a distance shorter than `min_dist`
(useful for preventing non-physical short distances during relaxation).

---

#### GRACE/FS

**Note:** The `python-ace` package is required (see the [installation guide](../install/#gracefs-cpu)).

To use the **GRACE/FS** calculator (with C++ implementations), use the following code:

```python
from pyace.asecalc import PyGRACEFSCalculator
calc = PyGRACEFSCalculator("/path/to/FS_model.yaml")
calc.set_active_set("/path/to/FS_model.asi")
```
___

### Usage in LAMMPS
For big production runs, it is better to use LAMMPS. 

#### LAMMPS: GRACE (TensorFlow)
Use following pair_style 
```
pair_style	grace
pair_coeff	* * /path/to/saved_model Al Li
```
If you downloaded foundation models (run `grace_download`), path will be like
* `/path/to/home/.cache/grace/train_1.5M_test_75_grace_1layer_v2_7Aug2024/` for `mp-1layer`
* `/path/to/home/.cache/grace/train_1.5M_test_75_grace_2layer_8Aug2024/` for `mp-2layer`

**Technical Note:** GRACE models are JIT (just-in-time) compiled, meaning the first run will be slower. 
GRACE uses padding (enabled by default) to reduce the frequency of recompilations if the structure size or number of 
neighbors changes. However, occasional recompilations may still occur; this is expected behavior.

You can adjust the fraction of neighbors padded with:
```
pair_style grace padding 0.05
```
If `padding ...` is not specified, the default value is 0.01 (i.e., 1%).

To receive messages when new padding and recompilations occur, add the `pad_verbose` option:
```
pair_style grace ... pad_verbose
```
to get messages when new padding and recompilation are happening



**Warning**: Currently, only a single LAMMPS process (utilizing a maximum of one GPU) is supported.
While GPU usage is optional, TensorFlow is less efficient when running solely on the CPU.

#### LAMMPS: GRACE/FS

Only the GRACE/FS family of models currently has a native C++ implementation (product evaluator, and no KOKKOS support yet).
To use these models, use the following pair_style:
```
pair_style grace/fs 
pair_coeff      * * FS_model.yaml  Mo Nb Ta W 
```
or with extrapolation grade:
```
pair_style grace/fs extrapolation
pair_coeff      * * FS_model.yaml FS_model.asi Mo Nb Ta W 

fix grace_gamma all pair 100 grace/fs gamma 1
compute max_grace_gamma all reduce max f_grace_gamma
```
The advantage of these models is that they are lightweight and do not rely on TensorFlow, meaning no GPU is required.
However, you can still parallelize them with MPI as usual.

##### Multi-GPU parallelization 
Currently, only **1-LAYER** models can be parallelized using MPI with domain decomposition.
No need to modify LAMMPS input file, but here is an example of shell run command:  

```
 TF_CPP_MIN_LOG_LEVEL=1 mpirun -np 4 --bind-to none  bash -c 'CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_RANK % 4)) lmp -in in.lammps'
```





