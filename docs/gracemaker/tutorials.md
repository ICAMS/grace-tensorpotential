# Tutorials  

### Video-tutorial

* !!NEW!! [Video tutorial](https://www.youtube.com/watch?v=rndnkiu9LGE)

### Prerequisites  

- **Operating System**: Linux or macOS (on macOS, you may need to install a custom `tensorflow`).  
  - **Note for Windows**: GPU support is available only on WSL.  
- **Hardware**: A GPU is highly recommended. Multicore CPUs are significantly slower but still functional.  
- **Software**: `gracemaker` must be installed (see [Installation](../install)).  

### Tutorial Materials  

All materials for these tutorials are available on [GitHub](https://github.com/ICAMS/grace-tutorial). Clone the repository with:  
```bash
git clone --depth=1 https://github.com/ICAMS/grace-tutorial
```  

## Tutorial 1: Parameterization of 2-layer GRACE for Al-Li
Working folder for this tutorial is `1-AlLi-GRACE-2LAYER`
```bash
cd 1-AlLi-GRACE-2LAYER
```

### 1.1. Unpack and collect DFT data
```bash
cd 0-data
tar zxvf AlLi_vasp_data.tar.gz
```
Now, let's collect DFT data recursively, by running 
```bash 
grace_collect
```
This will result in `collected.pkl.gz` file, that we will be used by `gracemaker`

### 1.2. Parameterization
#### 1.2.1. Input file

Now, switch to another folder and generate input file with `gracemaker -t`:
```bash
cd ../1-fit
gracemaker -t
```
You have to enter following information:

* **training dataset filename**:  ../0-data/collected.pkl.gz
* Test dataset filename: \[ENTER\]
* Test set fraction: \[ENTER\]
* List of elements: \[ENTER\]
* **Enter model preset, available options**: GRACE_2LAYER
* **Model complexity**: small
* Enter cutoff: \[ENTER\]
* Loss function type: \[ENTER\]
* For huber loss, please enter delta: \[ENTER\]
* Enter force loss weight: \[ENTER\]
* **Enter stress loss weight**: 0.1
* **Switch loss function E:F:S** ... : 350
* Learning rate after switching (default = 0.001): \[ENTER\] 
* Energy weight after switching (old value = 1, default = 5): \[ENTER\] 
* Force weight after switching (old value = 5, default = 2):  \[ENTER\]
* Stress weight after switching (old value = 0.1, default = same value): 
* **Enter weighting scheme type, available options: uniform, energy (default = uniform)**: energy
* Enter batch size (default = 32): \[ENTER\] 

This will produce `input.yaml` file, which you can further check and tune.

#### 1.2.2. Run gracemaker
Now, you can run the model parameterization with:  
```bash
gracemaker
```  

During this process:  
- **Preprocessing and Data Preparation**: Tasks such as building neighbor lists will be performed.  
- **JIT Compilation**: The first epoch (or iteration) may take additional time due to JIT compilation for each training and testing bucket.  

If you do not wish to wait, you can manually terminate the process: `Ctrl+Z` and `kill %1`  

To create multiple models for an ensemble, run additional parameterizations with different seeds:  
```bash
gracemaker --seed 2
gracemaker --seed 3
```  

#### 1.2.3. Analyze learning curves
Check `1-AlLi-GRACE-2LAYER/1-fit/visualize_metrics.ipynb` Jupyter notebook to analyze learning curves

#### 1.2.4. Save/export model
In order to export the model into TensorFlow's SavedModel format, do
```bash
gracemaker -r -s 
``` 
Check [here](../quickstart/#tensorflows-savedmodel) for more details.

### 1.3. Usage of the model

#### 1.3.1. Python/ASE
Please, check Jupyter notebook `1-AlLi-GRACE-2LAYER/2-python-ase/Al-Li.ipynb`

#### 1.3.2. LAMMPS

You need LAMMPS with GRACE be compiled (check [here](../install/#lammps-with-grace))

```bash
cd ../3-lammps/
/path/to/lmp -in in.lammps
``` 

To use the GRACE potential in SavedModel format, the following `pair_style` should be used:  
```bash
pair_style grace pad_verbose
pair_coeff * * ../1-fit/seed/1/saved_model/ Al
```  

- The `pad_verbose` option provides detailed information about padding during the simulation.  
- In this example, an Al FCC supercell (up to 100k atoms) will be simulated under NPT conditions at 300K and zero pressure.  

### Simulation Details  
- The simulation will first run for **20 steps** to JIT-compile the model.  
- Then, it will run for another **20 steps** to measure execution time.  

For example, on an A100 GPU, one of the final output lines might be:  
```
Loop time of 24.4206 on 1 procs for 20 steps with 108000 atoms
```  

This indicates that the current model (GRACE-2LAYER, small) achieves a performance of approximately **11 mcs/atom**, supporting simulations with up to **100k atoms**.  


---

## Tutorial 2. Parameterization of GRACE/FS for high entropy alloy HEA25S dataset
Dataset for this tutorial was taken from 
["Surface segregation in high-entropy alloys from alchemical machine learning: dataset HEA25S"](https://iopscience.iop.org/article/10.1088/2515-7639/ad2983) paper.

All tutorial materials can be found in `grace-tutorial/2-HEA25S-GRACE-FS/`: 
```bash
cd grace-tutorial/2-HEA25S-GRACE-FS/
```

### (optional) Dataset conversion from extxyz format
You can download complete dataset from [Materials Cloud](https://archive.materialscloud.org/record/2024.43) in an _extxyz_ format.

* Download `data.zip` file (in browser)
* Unpack the `data.zip`, go to any(all) subfolders and convert the _extxyz_ dataset with `extxyz2df`: 
```bash
unzip data.zip
cd data/dataset_O_bulk_random
extxyz2df bulk_random_train.xyz
```
As a result, you will get compressed pandas DataFrame (`bulk_random_train.pkl.gz`). 
Same procedure can be repeated for other files.


### 2.1. Parameterization
#### 2.1.1. Input file
Go to `1-fit` folder and run `gracemaker -t` for interactive dialogue:
```bash
cd 1-fit 
gracemaker -t
```
You have to enter following information:

* **training dataset filename**: ../0-data/bulk_random_train.pkl.gz
* Test dataset filename: \[ENTER\]
* Test set fraction: \[ENTER\]
* List of elements: \[ENTER\]
* **Enter model preset, available options**: FS
* **Model complexity**: small
* Enter cutoff: \[ENTER\]
* Loss function type: \[ENTER\]
* For huber loss, please enter delta: \[ENTER\]
* Enter force loss weight: \[ENTER\]
* **Enter stress loss weight**: 0.1
* Switch loss function E:F:S ... : \[ENTER\]
* Enter weighting scheme type ... : \[ENTER\]
* Enter batch size : \[ENTER\]

#### 2.1.2. Run `gracemaker`

Now, you can run the model parameterization with:  
```bash
gracemaker input.yaml
```  

During the run, you may notice multiple warning messages starting with `ptxas warning:` or similar. These messages indicate that JIT compilation is occurring for each training and testing bucket, and they are normal. They will disappear after the first iteration/epoch.  

If you prefer, you can [reduce](../faq/#how-to-reduce-tensorflow-verbosity-level) the verbosity level of TensorFlow to minimize these messages.  

#### 2.1.3. (optional) Manual continuation of the fit with new loss function

In order to continue the fit with **new** parameters, for example, add more weight onto energy in the loss function, do following steps:

* create new folder and copy `input.yaml` and `seed/1/checkpoints/*` from previous fit: 
```bash
cd ..
mkdir -p 1b-continue-fit 
cp 1-fit/input.yaml 1b-continue-fit/
mkdir -p 1b-continue-fit/seed/1
cp -r 1-fit/seed/1/checkpoints 1b-continue-fit/seed/1/
cd 1b-continue-fit/
```
* in the `input.yaml` file find and change following parameters:
```yaml
fit:
  loss: {
    energy: {weight: 5, type: huber, delta: 0.01},   # new energy weight
    forces: {weight: 5, type: huber, delta: 0.01},   # new forces weight
    stress: {weight: 0.1, type: huber, delta: 0.01}, # new stress weight
  }

 # ...
  opt_params: {
      learning_rate: 0.001,  # reduce learning rate from 1e-2 -> 1e-3
      # ...
  }
  
  learning_rate_reduction: { ..., resume_lr: False} # overwrite learning_rate from checkpoint with new 1e-3 value

```
- Run `gracemaker` with the `-r` flag to read the previous best-test-loss checkpoint:  
```bash
gracemaker -r
```  

**NOTE**: You can switch energy/forces/stress weights in the loss function during a single `gracemaker` run. To do this, you need to manually provide the `input.yaml::fit::loss::switch` option (see [here](../inputfile/#input-file-inputyaml) for more details) or provide a non-empty answer to the `Switch loss function E:F:S...` question in the `gracemaker -t` dialog.  

### 2.2. Save/Export Model

To export the model into both TensorFlow's SavedModel and GRACE/FS YAML formats, run:  
```bash
gracemaker -r -s -sf
```  

For more details, check [here](../quickstart/#gracefs).

### 2.3. Active Set Construction
To construct the active set (ASI) for uncertainty indication, run the following commands (more details [here](../quickstart/#build-active-set-for-gracefs-only)):  
```bash
cd seed/1
pace_activeset -d training_set.pkl.gz FS_model.yaml
```

### 2.4. Usage of the Model

#### 2.4.1. Python/ASE  
Please refer to the Jupyter notebook `2-HEA25S-GRACE-FS/HEA25-GRACE-FS.ipynb` for usage details.

#### 2.4.2. LAMMPS  
You need to compile LAMMPS with GRACE/FS (see [here](../install/#lammps-with-grace) for instructions).  

```bash
cd 3-lammps/grace-fs-with-extrapolation-grade/
mpirun -np 2 /path/to/lmp -in in.lammps
```  

In this simulation, the FCC(111) surface slab will be run under NPT conditions with an increasing temperature from 500K to 5000K. The extrapolation grade will be computed for each atom, and the configuration will be saved to `extrapolative_structures.dump` if the max gamma > 1.5.

To select the most representative structures for DFT calculations based on D-optimality, use the `pace_select` utility:  

```bash
pace_select extrapolative_structures.dump  -p ../../1-fit/seed/1/FS_model.yaml -a ../../1-fit/seed/1/FS_model.asi -e "Au"
```  

Find more details [here](https://pacemaker.readthedocs.io/en/latest/pacemaker/utilities/#d-optimality_structure_selection).

---

## Tutorial 3: Using Universal Machine Learning Interatomic Potentials

Universal machine learning interatomic potentials, also known as foundation models, are models capable of supporting a wide range of elements or even nearly the entire periodic table. These models are parameterized using large reference DFT datasets, such as the [Materials Project](https://next-gen.materialsproject.org/) or [Alexandria](https://alexandria.icams.rub.de/). Some of these models have been tested for high-throughput materials discovery, as demonstrated in [Matbench Discovery](https://matbench-discovery.materialsproject.org/). 

We have parameterized several GRACE-1LAYER and GRACE-2LAYER models on the MPTraj dataset (relaxation trajectories from the Materials Project).

### 3.1. Overview and Download 

You can view the available GRACE foundation models using the command:
```bash
grace_models list
```
To download a specific model, use:
```bash
grace_models download MP_GRACE_1L_r6_4Nov2024
```
To download all models at once, use:
```bash
grace_models download all
```

### 3.2. Usage in ASE
To load a model in ASE, use the following function:
```python
from tensorpotential.calculator.foundation_models import grace_fm, GRACEModels

calc = grace_fm("MP_GRACE_1L_r6_4Nov2024",
                pad_atoms_number=2,
                pad_neighbors_fraction=0.05, 
                min_dist=0.5)
# or 
calc = grace_fm(GRACEModels.MP_GRACE_1L_r6_4Nov2024) # for better code completion
```
Note that the additional parameters are optional. Default values are provided for `pad_atoms_number` and `pad_neighbors_fraction`.

For more details, refer to the Jupyter notebook `3-foundation-models/1-python-ase/using-grace-fm.ipynb`.

### 3.3. Usage in LAMMPS

The usage of foundation models in LAMMPS is the same as for custom-parameterized GRACE potentials. Examples are provided in the following directories:

- `3-foundation-models/2-lammps/1-Pt-surface`: Simulation of an oxygen molecule on a Pt (100) surface.
- `3-foundation-models/2-lammps/2-ethanol-water`: Simulation of ethanol and water. 

**Note:** The currently available GRACE-1LAYER models do not support MPI parallelization. Updated models with MPI support will be released in the future.

---

## Tutorial 4: Fine-Tuning Foundation GRACE Models

Fine-tuning foundation GRACE models can only be performed using checkpoints, which will be published at a later date.

---
