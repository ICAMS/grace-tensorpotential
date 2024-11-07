## Stages of model parameterization

1. [Data collection](#data-collection)
2. [Input file setup](#input-file-setup)
3. [Parameterization with `gracemaker`](#grace-model-parameterization)
4. [Export model](#export-model)
5. Build active set (for GRACE/FS only)  
6. Use in [ASE](#usage-in-ase) and/or [LAMMPS](#usage-in-lammps)

### Data collection

You can either use [`grace_collect`](../utilities#grace_collect) 
```bash
grace_collect
```

or build a `pandas.DataFrame` by you own. It must contain following columns:

* `ase_atoms` - with atomic structures, represented as ASE Atoms
* `energy` - total energy (should be force-consistent), shape: single number 
* `energy_corrected` - cohesive energy, i.e. total energy, after subtraction of free atoms energies, shape: single number
* `forces` - per-atom forces as NumPy array, shape: (number_of_atoms, 3) 

Note, that `energy_corrected` column is not mandatory, and can be constructed automatically by _gracemaker_, if you provide `reference_energy`

### Input file setup

You can use `gracemaker -t` for interactive interface to configure simple input file (not yet implemented)
and adjust it further manually (see [Input file](../inputfile) for more information)

### GRACE model parameterization

To start gracemaker, simply run
```python
gracemaker
```

Please, note, that the first line of input file is `seed: 42`, which specifies the random initialization of the model
parameters. After `gracemaker` started, it creates a working subfolder, i.e.  _seed/42/_ with related output files:

* _log.txt_ with redirected log
* _train_metrics.yaml_ and _test_metrics.yaml_ with multiple metrics in YAML format
* _checkpoints/_ with best test loss (suffix `.best_test_loss`) and last checkpoints (no extra suffix)

In addition to that, in a current working directory,  a file `model.yaml` will be created, that contains model's architecture.


### Export model

After normal termination of `gracemaker`, exported model in _saved_model_ format (used by TensorFlow)
will be saved into `seed/1/final_model`.

In order to export GRACE model from the checkpoint at anytime, do:

* `gracemaker -r -s`  to export model with _best test loss_
* `gracemaker -rl -s`  to export model from _latest saved checkpoint_

In all cases you will get model saved into folder `seed/1/saved_model`, that can be used in [ASE](#grace-tensorflow) 
or [LAMMPS](#lammps-grace-tensorflow)

In order to export GRACE/FS model in _YAML_ format (used by custom C++ implementation), add `-sf` flag to previous commands.
In that case you will get additionally `seed/1/FS_model.yaml` file. This model can be used in [ASE](#gracefs) or 
[LAMMPS](#lammps-gracefs)

### Usage in ASE

You can use exported GRACE model via ASE compatible calculator `TPCalculator` or  `PyGRACEFSCalculator`

#### GRACE (TensorFlow)
Usage as ASE calculator
```python
from tensorpotential.calculator import TPCalculator
calc = TPCalculator("/path/to/saved_model")
```

loading foundational models:
```python
from tensorpotential.calculator import grace_fm

calc=grace_fm('name_of_the_model') 
```

for both functions (_TPCalculator_ and _grace_fm_), you can change options for padding and minimal distance:
```python
calc=TPCalculator(model_path,
        pad_neighbors_fraction = 0.05,
        pad_atoms_number = 2,
        min_dist=0.5
)
```
If `min_dist` if given, calc will raise an exception when encounter distance shorter than `min_dist`
(usefull when you do  relaxation, in order to prevent non-physical short distances)

#### GRACE/FS

**NOTE!** You need `python-ace` package installed (see [here](../install/#gracefs-cpu))

In order to use **GRACE/FS** calculator (with C++ implementations), you can use
```python
from pyace.asecalc import PyGRACEFSCalculator
calc = PyGRACEFSCalculator("/path/to/FS_model.yaml")
```

### Usage in LAMMPS

#### LAMMPS: GRACE (TensorFlow)
Usage with LAMMPS:
Just use following pair_style 
```
pair_style	grace
pair_coeff	* * MODEL_PATH Al Li
```
where MODEL_PATH is a path to saved model.
If you downloaded foundation models (run `grace_download`), path will be like
* `/path/to/home/.cache/grace/train_1.5M_test_75_grace_1layer_v2_7Aug2024/` for `mp-1layer`
* `/path/to/home/.cache/grace/train_1.5M_test_75_grace_2layer_8Aug2024/` for `mp-2layer`

TECHNICAL NOTE: GRACE model are JIT (just-in-time) compiled, i.e. first run will be slow.  We do use padding (default - ON) to prevent too-frequent recompilation if structure size or number of neighbour are changed. However, you can notice recompilations from time to time, this is ok. 

You can change fraction of neighbours padded as 
```
pair_style grace padding 0.05
```
If `padding ...` is not provided, then default value is 0.01 i.e. 1%
add option `pad_verbose`
```
pair_style grace ... pad_verbose
```
to get messages when new padding and recompilation are happening



** WARNING!!!! Only single lammps process (that will use max 1 GPU) will work now **
Usage without GPU is possible, but TF utilizes CPU less efficient

#### LAMMPS: GRACE/FS

Only `GRACE/FS`  family of models has now native C++ implementaion (still not-optimal, product evaluator, NO KOKKOS support yet). 
In order to use these models, use following pairstyle:

with extrapolation grade:
```
pair_style grace/fs extrapolation
pair_coeff      * * FS_model.yaml FS_model.asi Mo Nb Ta W 

fix grace_gamma all pair 100 grace/fs gamma 1
compute max_grace_gamma all reduce max f_grace_gamma
```


without extrapolation grade:
```
pair_style grace/fs 
pair_coeff      * * FS_model.yaml  Mo Nb Ta W 
```
This type of model does not rely on TF and thus no GPU will be used. But you can parallelize  with MPI as usual


For MPI-parallel run on GPU for **1-LAYER** models:
```
 TF_CPP_MIN_LOG_LEVEL=1 mpirun -np 4 --bind-to none  bash -c 'CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_RANK % 4)) lmp -in in.lammps'
```


[//]: # (## &#40;TODO&#41; Model presets and configuration)





