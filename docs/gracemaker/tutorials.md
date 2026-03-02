# Tutorials  

### Video-tutorial

* !!NEW!! [Video tutorial](https://www.youtube.com/watch?v=rndnkiu9LGE)

### Prerequisites  

* **Operating System**: Linux or macOS (on macOS, you may need to install a custom `tensorflow`).  
  * **Note for Windows**: GPU support is available only on WSL.  
* **Hardware**: A GPU is highly recommended. Multicore CPUs are significantly slower but still functional.  
* **Software**: `gracemaker` must be installed (see [Installation](../install)).  

This tutorial is for the MLPfits workshop, hosted on the Noctua2 cluster of PC2 (Paderborn Center for Parallel Computing)

### Tutorial Materials

All materials for these tutorials are available on [GitHub](https://github.com/ICAMS/grace-tutorial). Clone the repository with:

```bash
git clone --depth=1 https://github.com/ICAMS/grace-tutorial
```

In order to activate GRACE environment do
```bash
cd
source ~/load_GRACE.sh
```

## Tutorial 1: Parameterization of 2-layer GRACE for Al-Li


### 1.1. Unpack and collect DFT data

Unpack data:

```bash
sh unpack_data.sh
```

Working folder for this tutorial is `1-AlLi-GRACE-2LAYER`

```bash
cd 1-AlLi-GRACE-2LAYER/0-data
```

Now, let's collect DFT data recursively, by running

```bash
grace_collect
```

This will generate a `collected.pkl.gz` file that will be used by `gracemaker`

### 1.2. Parameterization

#### 1.2.1. Input file

Now, switch to another folder and generate input file with `gracemaker -t`:

```bash
cd ../1-fit
gracemaker -t
```

```bash
── Fit type
? Fit type: fit from scratch

── Dataset
  Tab ↹ autocompletes path  ·  ↑↓ navigates history
? Training dataset file (e.g. data.pkl.gz): ../0-data/collected.pkl.gz
  ✓ Train file: ../0-data/collected.pkl.gz
? Use a separate test dataset file? No
? Test set fraction (split from train) 0.05
  ✓ Test fraction: 0.05

── Model Details
? Model preset: GRACE_2LAYER_latest
  ✓ Preset: GRACE_2LAYER_latest
? Model complexity: small
  ✓ Complexity: small
? Cutoff radius (Å) 6
  ✓ Cutoff: 6.0 Å

── Optimizer
? Optimizer: Adam
  ✓ Optimizer: Adam

── Loss function
? Loss type: huber
  ✓ Loss type: huber
? Huber delta 0.01
? Energy loss weight 16
  ✓ Energy weight: 16
? Force loss weight 32
  ✓ Force weight: 32
? Include stress in the loss? Yes
? Stress loss weight 128.0
  ✓ Stress weight: 128.0
? Switch E/F/S weights mid-training? Yes
? Energy weight after switch (was 16) 128
? Force weight after switch (was 32) 32
? Stress weight after switch (was 128.0) 128.0

── Weighting & batch size
? Sample weighting scheme: uniform
  ✓ Weighting: uniform
? Batch size 8
  ✓ Batch size: 8  (test: 32)
? Target total updates 5000
  ✓ Total updates: 5000
```

This will produce `input.yaml` file, which you can further check and tune.

#### 1.2.2. Run gracemaker

Now you can submit the fitting job to the queue with
```bash
sbatch submit.sh
```
or run the model parameterization with:

```bash
gracemaker
```

During this process:
* **Preprocessing and Data Preparation**: Tasks such as building neighbor lists will be performed.
* **JIT Compilation**: The first epoch may take additional time due to JIT compilation for each training and testing bucket.

If you do not wish to wait, you can manually terminate the process: `Ctrl+Z` and `kill %1`

To create multiple models for an ensemble, run additional parameterizations with different seeds:

```bash
sbatch submit-seed-2.sh
sbatch submit-seed-3.sh
```
Or for local runs:
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

Please, check Jupyter notebook `1-AlLi-GRACE-2LAYER/1-fit/validate.ipynb` and `1-AlLi-GRACE-2LAYER/2-python-ase/Al-Li.ipynb`

#### 1.3.2. LAMMPS

You need LAMMPS with GRACE be compiled (check [here](../install/#lammps-with-grace)).

To use the GRACE potential in SavedModel format, apply the following pair_style:

```bash
pair_style grace pad_verbose
pair_coeff * * ../1-fit/seed/1/saved_model/ Al
```
By default, this `pair_style` attempts to process the entire structure at once. However, for very large systems, this may result in out-of-memory (OOM) errors.

To prevent this, you can use the "chunked" versions of GRACE: `grace/1layer/chunk` or `grace/2layer/chunk`. These options process the structure in fixed-size pieces (chunks) that can be tuned to fit your GPU memory:

```bash
pair_style grace/2layer/chunk chunksize 4096
pair_coeff * * ../1-fit/seed/1/saved_model/ Al
```

The `pad_verbose` option provides detailed information about padding during the simulation.


Submit the LAMMPS calculations to the queue:
```bash
cd ../3-lammps/
sbatch submit.sh
sbatch submit.big.sh
```

Or you can run it locally

```bash
cd ../3-lammps/
lmp -in in.lammps
lmp -in in.lammps.chunked
```

to compare the normal and chunked versions of the GRACE-2L models.

### Simulation Details

- The simulation will first run for **20 steps** to JIT-compile the model.
- Then it will run for another **20 steps** to measure execution time.

For example, on an A100 GPU, one of the final output lines might be:

```
Loop time of 24.4206 on 1 procs for 20 steps with 108000 atoms
```

This indicates that the current model (GRACE-2LAYER, small) achieves a performance of approximately **11 mcs/atom**, supporting simulations with up to **108k atoms**.

---

## Tutorial 2. Parameterization of GRACE/FS for high entropy alloy HEA25S dataset

The dataset for this tutorial was taken from
the paper ["Surface segregation in high-entropy alloys from alchemical machine learning: dataset HEA25S"](https://iopscience.iop.org/article/10.1088/2515-7639/ad2983).

All tutorial materials can be found in `grace-tutorial/2-HEA25S-GRACE-FS/`:

```bash
cd grace-tutorial/2-HEA25S-GRACE-FS/
```

### (optional) Dataset conversion from extxyz format

You can download the complete dataset from [Materials Cloud](https://archive.materialscloud.org/record/2024.43) in _extxyz_ format.

* Download the `data.zip` file (in browser)
* Unpack the `data.zip`, go to any (or all) subfolders and convert the _extxyz_ dataset with `extxyz2df`:

```bash
unzip data.zip
cd data/dataset_O_bulk_random
extxyz2df bulk_random_train.xyz
```

As a result, you will get a compressed pandas DataFrame (`bulk_random_train.pkl.gz`).
The same procedure can be repeated for other files.

### 2.1. Parameterization

#### 2.1.1. Input file

Go to `1-fit` folder and run `gracemaker -t` for interactive dialogue:

```bash
cd 1-fit
gracemaker -t
```

You have to enter following information:
```bash

── Fit type
? Fit type: fit from scratch

── Dataset
  Tab ↹ autocompletes path  ·  ↑↓ navigates history
? Training dataset file (e.g. data.pkl.gz): ../0-data/bulk_random_train.pkl.gz
  ✓ Train file: ../0-data/bulk_random_train.pkl.gz
? Use a separate test dataset file? No
? Test set fraction (split from train) 0.05
  ✓ Test fraction: 0.05

── Model Details
? Model preset: FS
  ✓ Preset: FS
? Model complexity: medium
  ✓ Complexity: medium
? Cutoff radius (Å) 7
  ✓ Cutoff: 7.0 Å

── Optimizer
  → FS from scratch: BFGS (full Hessian) is recommended for small/medium models.
  → If your FS model has many parameters (large lmax/order), prefer L-BFGS-B
instead.
? Optimizer: Adam
  ✓ Optimizer: Adam

── Loss function
? Loss type: huber
  ✓ Loss type: huber
? Huber delta 0.01
? Energy loss weight 16
  ✓ Energy weight: 16
? Force loss weight 32
  ✓ Force weight: 32
? Include stress in the loss? Yes
? Stress loss weight 128.0
  ✓ Stress weight: 128.0
? Switch E/F/S weights mid-training? Yes
? Energy weight after switch (was 16) 128
? Force weight after switch (was 32) 32
? Stress weight after switch (was 128.0) 128.0

── Weighting & batch size
? Sample weighting scheme: uniform
  ✓ Weighting: uniform
? Batch size 16
  ✓ Batch size: 16  (test: 64)
? Target total updates 50000
  ✓ Total updates: 50000
```

#### 2.1.2. Run `gracemaker`

Now you can run the model parameterization with one of the following options.

Submit job to the queue:
```bash
sbatch submit.sh
```

Or run locally:

```bash
gracemaker input.yaml
```

During the run, you may notice multiple warning messages starting with `ptxas warning:` or similar. These messages indicate that JIT compilation is occurring for each training and testing bucket, and they are normal. They will disappear after the first iteration or epoch.

If you prefer, you can [reduce](../faq/#how-to-reduce-tensorflow-verbosity-level) the verbosity level of TensorFlow to minimize these messages.

#### 2.1.3. (optional) Manual continuation of the fit with new loss function

To continue the fit with **new** parameters, for example, to add more weight to energy in the loss function, take the following steps:

* create new folder and run `gracemaker -t`:

```bash
── Fit type
? Fit type: continue fit

── Dataset
  Tab ↹ autocompletes path  ·  ↑↓ navigates history
? Training dataset file (e.g. data.pkl.gz): ../1-fit/seed/1/training_set.pkl.gz
  ✓ Train file: ../1-fit/seed/1/training_set.pkl.gz
? Use a separate test dataset file? Yes
? Test dataset file: ../1-fit/seed/1/test_set.pkl.gz
  ✓ Test file: ../1-fit/seed/1/test_set.pkl.gz
  ✓ Test fraction: 0.05

── Model Details
? Model config to continue from (e.g. model.yaml): ../1-fit/seed/1/model.yaml
  ✓ Found 3 checkpoints in ../1-fit/seed/1/checkpoints
? Which checkpoint to load? Auto (best test or latest)
  → Auto-selected: checkpoint.best_test_loss
  ✓ Previous model: ../1-fit/seed/1/model.yaml
  ✓ Checkpoint: ../1-fit/seed/1/checkpoints/checkpoint.best_test_loss
  → Note: reset_epoch_and_step is set to True (training will start from epoch 0)

── Optimizer
? Optimizer: Adam
  ✓ Optimizer: Adam

── Loss function
? Loss type: huber
  ✓ Loss type: huber
? Huber delta 0.01
? Energy loss weight 16
  ✓ Energy weight: 16
? Force loss weight 32
  ✓ Force weight: 32
? Include stress in the loss? Yes
? Stress loss weight 128.0
  ✓ Stress weight: 128.0
? Switch E/F/S weights mid-training? No

── Weighting & batch size
? Sample weighting scheme: uniform
  ✓ Weighting: uniform
? Batch size 16
  ✓ Batch size: 16  (test: 64)
? Target total updates 10000
  ✓ Total updates: 10000
```

submit to the queue or run as usual with `gracemaker`.

**Note**: You can switch energy/forces/stress weights in the loss function during a single `gracemaker` run. To do this, you need to manually provide the `input.yaml::fit::loss::switch` option (see [here](../inputfile/#input-file-inputyaml) for more details) or provide a non-empty answer to the `Switch loss function E:F:S...` question in the `gracemaker -t` dialog.

### 2.2. Save/Export Model

To export the model into both TensorFlow's SavedModel and GRACE/FS YAML formats, navigate to `2-HEA25S-GRACE-FS/1-fit` and run:

```bash
gracemaker -r -s -sf
```

For more details, check [here](../quickstart/#gracefs).

### 2.3. Active Set Construction

To construct the active set (ASI) for uncertainty indication, run the following commands (more details [here](../quickstart/#build-active-set-for-gracefs-only)):

```bash
cd seed/1
pace_activeset -d training_set.pkl.gz saved_model.yaml
```

### 2.4. Usage of the Model

#### 2.4.1. Python/ASE

Please refer to the Jupyter notebook `2-HEA25S-GRACE-FS/HEA25-GRACE-FS.ipynb` for usage details.

#### 2.4.2. LAMMPS

You need to compile LAMMPS with GRACE/FS and KOKKOS (see [here](../install/#lammps-with-grace) for instructions).

Navigate to `3-lammps/grace-fs-with-extrapolation-grade/` and submit to the queue:
```bash
sbatch submit.sh  # for CPU
sbatch submit_kk.sh  # for GPU/KOKKOS acceleration
```

In the log.lammps.* files you can see

for CPU:
```
Performance: 10.573 ns/day, 2.270 hours/ns, 122.376 timesteps/s, 4.895 katom-step/s
```

for GPU:
```
Performance: 42.786 ns/day, 0.561 hours/ns, 495.213 timesteps/s, 19.809 katom-step/s
```

In this simulation, a small FCC(111) surface slab runs under NPT conditions with an increasing temperature from 500K to 5000K. The extrapolation grade is computed for each atom, and the configuration is saved to `extrapolative_structures.dump` if the maximum gamma > 1.5.

To select the most representative structures for DFT calculations based on D-optimality, use the `pace_select` utility:

```bash
pace_select extrapolative_structures.dump  -p ../../1-fit/seed/1/saved_model.yaml -a ../../1-fit/seed/1/saved_model.asi -e "Au"
```
This will generate a `selected.pkl.gz` file with calculations that need to be performed with DFT.
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
grace_models download GRACE-1L-OMAT-medium-ft-E
```

To download all models at once, use:

```bash
grace_models download all
```

### 3.2. Usage with ASE

To load a model in ASE, use the following function:

```python
from tensorpotential.calculator.foundation_models import grace_fm, GRACEModels

calc = grace_fm("GRACE-1L-OMAT-medium-ft-E",
                #pad_atoms_number=2,
                #pad_neighbors_fraction=0.05,
                #min_dist=0.5
                )
# or
calc = grace_fm(GRACEModels.GRACE_1L_OMAT_medium_ft_E) # for better code completion
```

Note that the additional parameters are optional. Default values are provided for `pad_atoms_number` and `pad_neighbors_fraction`.

For more details, refer to the Jupyter notebook `3-foundation-models/1-python-ase/using-grace-fm.ipynb`.

### 3.3. Usage in LAMMPS

The usage of foundation models in LAMMPS is the same as for custom-parameterized GRACE potentials. Examples are provided in the following directories:

* `3-foundation-models/2-lammps/1-Pt-surface`: Simulation of an oxygen molecule on a Pt (100) surface.
* `3-foundation-models/2-lammps/2-ethanol-water`: Simulation of ethanol and water.

You can submit simulation jobs to the queue with `sbatch submit.sh`

---

## Tutorial 4: Fine-Tuning and distillation of  Foundation GRACE Models


### Finetuning foundation models with a new dataset

Navigate to `3-foundation-models/3a-finetuning` and
run `gracemaker` with the `-t` flag to start an interactive dialogue and select the following options:

```bash

── Fit type
? Fit type: finetune foundation model

── Dataset
  Tab ↹ autocompletes path  ·  ↑↓ navigates history
? Training dataset file (e.g. data.pkl.gz): ../../1-AlLi-GRACE-2LAYER/0-
data/collected.pkl.gz
  ✓ Train file: ../../1-AlLi-GRACE-2LAYER/0-data/collected.pkl.gz
? Use a separate test dataset file? No
? Test set fraction (split from train) 0.05
  ✓ Test fraction: 0.05

── Model Details
? Model tier: 2L  — two message-passing layers (most accurate)
? Training dataset: OMAT  — OMat24 only (base / ft-E variants)
? Model size: medium    — larger capacity
? Fine-tuning variant: ft-E   — fine-tuned on energies
  ✓ Foundation model: GRACE-2L-OMAT-medium-ft-E

── Optimizer
? Optimizer: Adam
  ✓ Optimizer: Adam

── Loss function
? Loss type: huber
  ✓ Loss type: huber
? Huber delta 0.01
? Energy loss weight 16
  ✓ Energy weight: 16
? Force loss weight 32
  ✓ Force weight: 32
? Include stress in the loss? Yes
? Stress loss weight 128.0
  ✓ Stress weight: 128.0
? Switch E/F/S weights mid-training? Yes
? Energy weight after switch (was 16) 128
? Force weight after switch (was 32) 32
? Stress weight after switch (was 128.0) 128.0

── Weighting & batch size
? Sample weighting scheme: uniform
  ✓ Weighting: uniform
? Batch size 16
  ✓ Batch size: 16  (test: 64)
? Target total updates 10000
  ✓ Total updates: 10000
```

After that, submit the job to the cluster with `sbatch submit.sh` or run `gracemaker` locally.

After the training is finished, you can find the final model in the `seed/1/final_model` folder.
If not, you can export the model using the `gracemaker -r -s` command to the `seed/1/saved_model` folder.

You can quickly validate the finetuned model in `3-foundation-models/3a-finetuning/validate.ipynb` notebook.

### Generating distilled data

Now you can use the fine-tuned model to generate distilled reference data.
See `3-foundation-models/3b-distillation/distill_data.ipynb` for details.
This will create a `distilled_AlLi_dataset.pkl.gz` file.

### Training a distilled model

To train a model on the distilled dataset, start `gracemaker` in interactive mode:

```bash
gracemaker -t
```

Answer the prompts as follows:

```bash

── Fit type
? Fit type: fit from scratch

── Dataset
  Tab ↹ autocompletes path  ·  ↑↓ navigates history
? Training dataset file (e.g. data.pkl.gz): distilled_AlLi_dataset.pkl.gz
  ✓ Train file: distilled_AlLi_dataset.pkl.gz
? Use a separate test dataset file? No
? Test set fraction (split from train) 0.05
  ✓ Test fraction: 0.05

── Model Details
? Model preset: FS
  ✓ Preset: FS
? Model complexity: medium
  ✓ Complexity: medium
? Cutoff radius (Å) 7
  ✓ Cutoff: 7.0 Å

── Optimizer
  → FS from scratch: BFGS (full Hessian) is recommended for small/medium models.
  → If your FS model has many parameters (large lmax/order), prefer L-BFGS-B instead.
? Optimizer: Adam
  ✓ Optimizer: Adam

── Loss function
? Loss type: huber
  ✓ Loss type: huber
? Huber delta 0.01
? Energy loss weight 16
  ✓ Energy weight: 16
? Force loss weight 32
  ✓ Force weight: 32
? Include stress in the loss? Yes
? Stress loss weight 128.0
  ✓ Stress weight: 128.0
? Switch E/F/S weights mid-training? Yes
? Energy weight after switch (was 16) 128
? Force weight after switch (was 32) 32
? Stress weight after switch (was 128.0) 128.0

── Weighting & batch size
? Sample weighting scheme: uniform
  ✓ Weighting: uniform
? Batch size 16
```

When the dialog finishes, an `input.yaml` file will be created; you can inspect and adjust it if needed.  Submit the fit by running

```bash
sbatch submit.sh
```

or execute locally:

```bash
gracemaker input.yaml
```

Once training is complete the model will reside in `seed/1/final_model`.  You may also export a grace/fs YAML format with

```bash
gracemaker -r -s -sf
```

and then generate an active set:

```bash
cd seed/1
pace_activeset saved_model.yaml -d training_set.pkl.gz
```

For validation examples, see `3-foundation-models/3b-distillation/validate.ipynb`.

#### LAMMPS simulations

With the distilled model ready, you can perform production‑style molecular dynamics in LAMMPS.  Go to the
`3-foundation-models/3c-lammps-distilled` directory and open
`prepare_lammps_input.ipynb`.  The notebook shows how to generate
input files for a 500‑atom Al‑FCC cell containing 5 at % Li.  After
an initial equilibration the system is driven with MD‑MC moves (atomic
swaps) at 500 K.

Two separate simulation setups are provided:

* `lammps-grace-2L` – uses the original teacher model (GRACE‑2L,
  TensorFlow backend).
* `lammps-grace-fs-dist` – uses the distilled student model
  (GRACE‑FS, KOKKOS/GPU backend).

Each folder includes its own `submit.sh` script; run the jobs by
submitting the appropriate script to your queue.  Both calculations are
configured to run on a GPU.  During the MC portion only energy
evaluations are required, which further speeds the runs.

Typical performance numbers obtained on our hardware (A100 40Gb) are:

* **GRACE‑FS + KOKKOS/GPU:** 20.3 ns/day (1.18 h/ns),
  234.97 tps, 117.49 katom‑step/s
* **GRACE‑2L + TensorFlow:** 3.26 ns/day (7.36 h/ns),
  37.76 tps, 18.88 katom‑step/s

That corresponds to roughly a **6.5× speed‑up** for the distilled model
compared with the original teacher.


---


## Tutorial 5. Learning vectorial/tensorial properties

This exercise demonstrates how to fit a model that predicts first‑rank
tensorial quantities (vectors).

Change into the `4-fit-tensors/1-vector` directory and launch the
interactive setup:

```bash
── Dataset
  Tab ↹ autocompletes path  ·  ↑↓ navigates history
? Training dataset file (e.g. data.pkl.gz): H2O_100.pkl.gz
  ✓ Train file: H2O_100.pkl.gz
? Use a separate test dataset file? No
? Test set fraction (split from train) 0.05
  ✓ Test fraction: 0.05

── Tensor type
? Tensor type: [1]       — First-rank vector (e.g. Forces as tensor)
  ✓ Tensor components: [1]  rank=1

── Per-structure
? Property granularity: per-atom      — one tensor per atom  (e.g. EFG, BEC, for
ces)
  ✓ Per structure: False

── Model
? Model preset: TENSOR_1L — 1 message-passing layer, lighter (lmax=4, embedding=
32)
  ✓ Preset: TENSOR_1L
? Cutoff radius (Å) 6.0
  ✓ Cutoff: 6.0 Å

── Energy & forces
? Fit energy alongside tensor? No
  ✓ Compute energy: False

── Loss
? Loss type: square
  ✓ Loss type: square
? Tensor loss weight 10.0
  ✓ Tensor weight: 10.0

── Batch & training
? Batch size 10
  ✓ Batch size: 10  (test: 50)
? Target total updates 10000
  ✓ Total updates: 10000

```

After the wizard finishes, inspect the generated `input.yaml` and
modify the optimizer block:

```yaml
fit:
 #...
 opt_params: {learning_rate: 0.05, use_ema: True, ema_momentum: 0.5,
               weight_decay: null, clipnorm: 1.0}
```

Submit the job with `sbatch submit.sh` (or run locally with
`gracemaker input.yaml`).

When training completes, open the notebook
`4-fit-tensors/1-vector/use_model.ipynb`.  It contains examples of
predicting vectorial properties on the test set and verifying rotational
equivariance by applying random rotations to the structures and
comparing the rotated outputs.
