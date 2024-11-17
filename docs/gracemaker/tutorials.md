# Tutorials
Prerequisites:

* Linux or MacOS  (for latter you probably would like to install custom `tensorflow`).  
For WSL on Windows, no GPU will be supported.
* Graphical card (GPU) is preferential, multicore CPU will be much slower, but still ok.
* Installed `gracemaker` (see [Installation](../install))

Materials for these tutorials can be found on [Github](https://github.com/ICAMS/grace-tutorial)
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
Now, you can run model parameterization with
```bash
gracemaker input.yaml
```
After preprocessing and data preparation (i.e. building neighbour lists), parameterization process will start.
First epoch (or iteration) will take some time, because of JIT compilation for each train and test bucket. 

It will take some time, if you don't want to wait, you can terminate the process  manually (`Ctrl+Z` and `kill %1`)

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

In order to use GRACE potential in SavedModel format, we use following pair_style
```
pair_style grace  pad_verbose
pair_coeff * * ../1-fit/seed/1/saved_model/  Al
```
where, we add `pad_verbose` option that will print more detailed information about padding.
In this simulation  Al FCC supercell (with up to 100k atoms) will be run under NPT conditions at 300K and zero pressure.
Simulation will run, first, for 20 steps to JIT-compile model and then for another 20 steps to measure execution time.
If you run on A100 GPU, you can find one of the last lines:
```
Loop time of 24.4206 on 1 procs for 20 steps with 108000 atoms
```
which mean that current model (GRACE-2LAYER, small) can run up to **100k atoms** with approx **11 mcs/atom** performance.


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

#### 2.1.2. Run gracemaker
Now, you can run model parameterization with
```bash
gracemaker input.yaml
```
During the run, at the beginning, you will probably see multiple messages, starting with `ptxas warning :` or similar.
This demonstrates that JIT compilation works (for each train and test bucket) and is completely fine. 
These messages will go away after first iteration/epoch. However, you can try to [reduce](../faq/#how-to-reduce-verbosity-level-of-tensorflow) verbosity level of
TensorFlow. 
Now, you can either wait until completion of the process, or terminate it manually (`Ctrl+Z` and `kill %1`)

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

* run gracemaker with `-r` flag (to read previous best-test-loss checkpoint):
```bash
gracemaker -r
```

NOTE! You can switch energy/forces/stress weights in loss function in one gracemaker run.
For that you need to provide `input.yaml::fit::loss::switch` option manually (see [here](../inputfile/#input-file-inputyaml) for more details) or
provide non-empty answer to `Switch loss function E:F:S...` question of `gracemaker -t` dialog.

### 2.2. Save/export model
In order to export the model into both TensorFlow's SavedModel and GRACE/FS YAML formats, do
```bash
gracemaker -r -s -sf
``` 
Check [here](../quickstart/#gracefs) for more details.

### 2.3. Active set construction
In order to construct active set (ASI), that will be used for uncertainty indication, run following commands (more details [here](../quickstart/#build-active-set-for-gracefs-only))
```bash
cd seed/1
pace_activeset -d training_set.pkl.gz FS_model.yaml
```

### 2.4. Usage of the model

#### 2.4.1. Python/ASE
Please, check Jupyter notebook `2-HEA25S-GRACE-FS/HEA25-GRACE-FS.ipynb`

#### 2.4.2. LAMMPS
You need LAMMPS with GRACE/FS be compiled (check [here](../install/#lammps-with-grace))
```bash
cd 3-lammps/grace-fs-with-extrapolation-grade/
mpirun -np 2 /path/to/lmp -in in.lammps
``` 
In this simulation  FCC(111) surface slab will be run under NPT conditions, with increasing temperature from 500K to 5000K.
Extrapolation grade will be computed for each atom, and current configuration will be saved to `extrapolative_structures.dump` 
if max gamma > 1.5.

In order to select most representative structures for following DFT calculations based on D-optimality, use `pace_select` utility:
```bash
pace_select extrapolative_structures.dump  -p ../../1-fit/seed/1/FS_model.yaml -a ../../1-fit/seed/1/FS_model.asi -e "Au"
```
Find more [here](https://pacemaker.readthedocs.io/en/latest/pacemaker/utilities/#d-optimality_structure_selection).

---

## Tutorial 3: Usage of universal machine learning interatomic potential

Universal machine learning interatomic potentials, also called foundation models, are models that support wide range of 
elements, or even (almost) whole periodic table. These models are parameterized on large reference DFT datasets, such as 
[Materials Project](https://next-gen.materialsproject.org/) or [Alexandria](https://alexandria.icams.rub.de/).
Some of these models were tested for high-throughput materials discovery, see [Matbench Discovery](https://matbench-discovery.materialsproject.org/).
We parameterize few GRACE-1LAYER/2LAYER on MPTraj dataset (relaxation trajectories from Materials Project).

### 3.1. Overview and download 

You can check available GRACE foundation models with
```bash
grace_models list
```
and download given model with `grace_models download MP_GRACE_1L_r6_4Nov2024` or all models with `grace_models download all`. 

### 3.2. Use in ASE
Check `3-foundation-models/1-python-ase/using-grace-fm.ipynb` Jupyter notebook.

### 3.3. Use in LAMMPS
  *  Check folder `3-foundation-models/2-lammps/1-Pt-surface` for simulation of oxygen molecule on Pt (100) surface.

  *  Check folder `3-foundation-models/2-lammps/2-ethanol-water` for  simulation of ethanol and water. (TODO: check mpi)


---

[//]: # (## 4. &#40;work in progress&#41; Fine-tuning of GRACE models)

[//]: # (work in progress, will be available later)
