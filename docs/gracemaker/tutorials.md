# Fitting tutorials

Hands-on tutorials: fit GRACE-2L and GRACE/FS models from scratch, use and fine-tune foundation
models, distill a fast model, and fit vector properties.

!!! note "Next step: uncertainty"
    After you have a model, continue with the [UQ workflows](uq_workflows.md) tutorial. It shows how
    to compute the extrapolation grade γ of GRACE-1L/2L/3L models, find extrapolative structures in
    MD, select them for DFT and refit.

There is also a [video tutorial](https://www.youtube.com/watch?v=rndnkiu9LGE).

## Before you start {#tutorial-materials}

* **Operating system**: Linux or macOS (on macOS you may need a custom `tensorflow`). On Windows,
  the GPU works only in WSL.
* **Hardware**: a GPU is strongly recommended. Multicore CPUs work, but are much slower.
* **Software**: `gracemaker` ([Installation](install.md)).

Get the tutorial files from [GitHub](https://github.com/ICAMS/grace-tutorial) and unpack the DFT data:

```bash
git clone --depth=1 https://github.com/ICAMS/grace-tutorial
cd grace-tutorial
sh unpack_data.sh
```

The repository already has fitted models in the `seed/1/` folders. If you do not want to wait for a
fit, use them and go on with the next step.

!!! note "Batch scripts"
    The `submit*.sh` scripts are for a SLURM cluster (`sbatch`). On your own machine, run the
    command from the script directly (`gracemaker`, `lmp -in in.lammps`, ...).

`gracemaker -t` asks questions and writes `input.yaml`. Below, each tutorial lists the answers.
For any question not in the list, press Enter to keep the default. At the end, choose
`Yes, write input.yaml`. All options of `input.yaml` are on the [input file](inputfile.md) page.

---

## Tutorial 1: Parameterization of 2-layer GRACE for Al-Li

Folder: `1-AlLi-GRACE-2LAYER`.

### 1.1. Collect DFT data

`grace_collect` reads all VASP calculations (`OUTCAR` or `vasprun.xml`) in the subfolders:

```bash
cd 1-AlLi-GRACE-2LAYER/0-data
grace_collect
```

It writes `collected.pkl.gz`, the training data for `gracemaker`. See [`grace_collect`](utilities.md#grace_collect).

### 1.2. Parameterization

#### 1.2.1. Input file

```bash
cd ../1-fit
gracemaker -t
```

| Question | Answer |
|---|---|
| Fit type | `fit from scratch` |
| Training dataset file | `../0-data/collected.pkl.gz` |
| Use a separate test dataset file? | No |
| Test set fraction | 0.05 |
| Model preset | `GRACE_2LAYER_latest` |
| Model complexity | `small` |
| Cutoff radius (Å) | 6 |
| Optimizer | `Adam` |
| Loss type | `huber` (delta 0.01) |
| Energy / force loss weight | 16 / 32 |
| Include stress in the loss? | Yes, weight 128 |
| Switch E/F/S weights mid-training? | Yes: energy 128, force 32, stress 128 |
| Batch size | 8 |
| Target total updates | 10000 |

Check `input.yaml` and change it if needed.

#### 1.2.2. Run gracemaker

```bash
gracemaker          # or: sbatch submit.sh
```

First, `gracemaker` prepares the data (for example, the neighbor lists). The first epoch is slower,
because each train and test bucket is JIT-compiled. To stop the fit, press `Ctrl+Z` and run
`kill %1`.

For an ensemble, fit more models with other seeds:

```bash
gracemaker --seed 2   # or: sbatch submit-seed-2.sh
gracemaker --seed 3   # or: sbatch submit-seed-3.sh
```

Each seed has its own folder: `seed/1/`, `seed/2/`, ...

#### 1.2.3. Analyze learning curves

Open the notebook `1-AlLi-GRACE-2LAYER/1-fit/visualize_metrics.ipynb`.

#### 1.2.4. Save/export model

At the end of the fit, the model is saved in `seed/1/final_model/` (TensorFlow SavedModel format).
To export it from the best checkpoint at any time (for example, after you stopped the fit):

```bash
gracemaker -r -s
```

This writes `seed/1/saved_model/`. More in [Export model](quickstart.md#tensorflows-savedmodel).

### 1.3. Usage of the model

#### 1.3.1. Python/ASE

See the notebooks `1-AlLi-GRACE-2LAYER/1-fit/validate.ipynb` and
`1-AlLi-GRACE-2LAYER/2-python-ase/Al-Li.ipynb`.

#### 1.3.2. LAMMPS

You need LAMMPS compiled with GRACE ([installation](install.md#lammps-with-grace)). The input
`3-lammps/in.lammps` uses:

```
pair_style grace pad_verbose
pair_coeff * * ../1-fit/seed/1/final_model/ Al
```

`pad_verbose` prints information about padding. `pair_style grace` processes the whole structure at
once. For very large systems this can run out of GPU memory. The chunked pair styles
`grace/1layer/chunk` and `grace/2layer/chunk` process the atoms in blocks of `chunksize` atoms
(default 4096); choose it to fit your GPU memory:

```
pair_style grace/2layer/chunk chunksize 4096
pair_coeff * * ../1-fit/seed/1/final_model/ Al
```

Run both versions and compare them:

```bash
cd ../3-lammps/
lmp -in in.lammps            # or: sbatch submit.sh
lmp -in in.lammps.chunked
```

`submit.big.sh` runs the same for a 108,000-atom cell (`in.lammps.big`, `in.lammps.chunked.big`).
Each simulation first runs 20 steps to JIT-compile the model, then 20 more steps to measure the
time. On an A100 GPU, the large cell gives, for example:

```
Loop time of 24.4206 on 1 procs for 20 steps with 108000 atoms
```

That is about **11 µs/atom/step** (24.4 s / 20 steps / 108,000 atoms) for GRACE-2L (small).
More on pair styles: [LAMMPS usage](quickstart.md#lammps-grace-tensorflow).

---

## Tutorial 2: Parameterization of GRACE/FS for high entropy alloy HEA25S dataset

The data come from the paper ["Surface segregation in high-entropy alloys from alchemical machine
learning: dataset HEA25S"](https://iopscience.iop.org/article/10.1088/2515-7639/ad2983).
Folder: `2-HEA25S-GRACE-FS`.

??? info "(optional) Convert the full dataset from extxyz"
    The full dataset is on [Materials Cloud](https://archive.materialscloud.org/record/2024.43) in
    extxyz format. Download `data.zip` in the browser, unpack it and convert a file with
    [`extxyz2df`](utilities.md#extxyz2df):

    ```bash
    unzip data.zip
    cd data/dataset_O_bulk_random
    extxyz2df bulk_random_train.xyz
    ```

    This writes `bulk_random_train.pkl.gz` (a compressed pandas DataFrame). Repeat for other files.
    The tutorial already has this file in `0-data/`.

### 2.1. Parameterization

#### 2.1.1. Input file {#fs-input-file}

```bash
cd 2-HEA25S-GRACE-FS/1-fit
gracemaker -t
```

| Question | Answer |
|---|---|
| Fit type | `fit from scratch` |
| Training dataset file | `../0-data/bulk_random_train.pkl.gz` |
| Use a separate test dataset file? | No |
| Test set fraction | 0.05 |
| Model preset | `FS` |
| Model complexity | `medium` |
| Cutoff radius (Å) | 7 |
| Optimizer | `Adam` (the wizard suggests BFGS for small FS models) |
| Loss type | `huber` (delta 0.01) |
| Energy / force loss weight | 16 / 32 |
| Include stress in the loss? | Yes, weight 128 |
| Switch E/F/S weights mid-training? | Yes: energy 128, force 32, stress 128 |
| Batch size | 16 |
| Target total updates | 50000 |

#### 2.1.2. Run `gracemaker`

```bash
gracemaker input.yaml   # or: sbatch submit.sh
```

You may see many messages that start with `ptxas warning:`. They come from the JIT compilation of
each train and test bucket. They are normal and stop after the first epoch. You can also
[reduce the TensorFlow verbosity](faq.md#how-to-reduce-tensorflow-verbosity-level).

#### 2.1.3. (optional) Continue the fit with a new loss function

To continue the fit with new settings (for example, a larger energy weight), go to the folder
`1b-continue-fit` (or any new folder) and run `gracemaker -t`:

| Question | Answer |
|---|---|
| Fit type | `continue fit` |
| Training dataset file | `../1-fit/seed/1/training_set.pkl.gz` |
| Use a separate test dataset file? | Yes: `../1-fit/seed/1/test_set.pkl.gz` |
| Model config to continue from | `../1-fit/seed/1/model.yaml` |
| Which checkpoint to load? | `Auto (best test or latest)` |
| Optimizer | `Adam` |
| Loss | as in 2.1.1, change the weights you want |
| Switch E/F/S weights mid-training? | No |
| Batch size | 16 |
| Target total updates | 10000 |

Then run `gracemaker` as usual. The first fit wrote `training_set.pkl.gz` and `test_set.pkl.gz`
because the wizard sets `save_dataset: True` when the test set is split from the training data.

!!! note
    To change the loss weights during a single run, answer Yes to `Switch E/F/S weights
    mid-training?`. This writes `fit: loss: switch` (see [Loss](inputfile.md#loss)).

### 2.2. Save/Export Model

Export the model as TensorFlow SavedModel and as GRACE/FS YAML file:

```bash
cd 2-HEA25S-GRACE-FS/1-fit
gracemaker -r -s -sf
```

This writes `seed/1/saved_model/` and `seed/1/saved_model.yaml`. More in [GRACE/FS](quickstart.md#gracefs).

### 2.3. Active Set Construction

The active set (ASI) is needed to compute the extrapolation grade γ of a GRACE/FS model.
`pace_activeset` comes from `python-ace` ([installation](install.md#gracefs-cpu)).
More in [Active set](quickstart.md#build-active-set-for-gracefs-only).

```bash
cd seed/1
pace_activeset -d training_set.pkl.gz saved_model.yaml
```

This writes `saved_model.asi`.

### 2.4. Usage of the Model

#### 2.4.1. Python/ASE

See the notebook `2-HEA25S-GRACE-FS/2-python-ase/HEA25-GRACE-FS.ipynb`.

#### 2.4.2. LAMMPS

You need LAMMPS compiled with GRACE/FS, and with KOKKOS for the GPU run
([installation](install.md#lammps-with-grace)).

```bash
cd 3-lammps/grace-fs-with-extrapolation-grade/
sbatch submit.sh      # CPU, in.lammps
sbatch submit_kk.sh   # GPU with KOKKOS, in.kk.lammps
```

The simulation heats a small FCC(111) Au surface slab from 500 K to 5000 K (NPT). Every 50 steps it
computes γ for each atom. If the maximum γ is 1.5 or more, it writes the structure to
`extrapolative_structures.dump` (`extrapolative_structures.kk.dump` for the GPU run). It stops if γ
goes above 25.

In the `log.lammps.*` files you find the speed. CPU:

```
Performance: 10.573 ns/day, 2.270 hours/ns, 122.376 timesteps/s, 4.895 katom-step/s
```

GPU:

```
Performance: 42.786 ns/day, 0.561 hours/ns, 495.213 timesteps/s, 19.809 katom-step/s
```

To select the most useful structures for DFT (D-optimality), use `pace_select` from `python-ace`:

```bash
pace_select extrapolative_structures.dump -p ../../1-fit/seed/1/saved_model.yaml -a ../../1-fit/seed/1/saved_model.asi -e "Au"
```

This writes `selected.pkl.gz` with the structures to compute with DFT. More in the
[pacemaker documentation](https://pacemaker.readthedocs.io/en/latest/pacemaker/utilities/#d-optimality_structure_selection).

---

## Tutorial 3: Using Universal Machine Learning Interatomic Potentials

Universal machine learning interatomic potentials (foundation models) cover many elements, up to
most of the periodic table. They are fitted on large DFT datasets, such as the
[Materials Project](https://next-gen.materialsproject.org/) or
[Alexandria](https://alexandria.icams.rub.de/). Some are tested for materials discovery in
[Matbench Discovery](https://matbench-discovery.materialsproject.org/).

GRACE foundation models exist with one, two and three layers (GRACE-1L, -2L, -3L), fitted on
OMat24 (`-OMAT`), on OMat24 and then sAlex + MPtraj (`-OAM`), and on the SMAX dataset (`-SMAX`).
The full list is on the [foundation models](foundation.md) page.
Folder: `3-foundation-models`.

### 3.1. Overview and Download

```bash
grace_models list                                 # list the models
grace_models download GRACE-1L-OMAT-medium-ft-E   # download one model
grace_models download all                         # download all models
```

### 3.2. Usage with ASE

```python
from tensorpotential.calculator.foundation_models import grace_fm, GRACEModels

calc = grace_fm("GRACE-1L-OMAT-medium-ft-E")
# or, with code completion:
calc = grace_fm(GRACEModels.GRACE_1L_OMAT_medium_ft_E)
```

`grace_fm` also accepts optional arguments, such as `pad_atoms_number`, `pad_neighbors_fraction`
and `min_dist`. The defaults are fine for most cases. See the notebook
`3-foundation-models/1-python-ase/using-grace-fm.ipynb`.

### 3.3. Usage in LAMMPS

Foundation models work in LAMMPS in the same way as your own GRACE models. Examples:

* `3-foundation-models/2-lammps/1-Pt-surface`: an oxygen molecule on a Pt(100) surface.
* `3-foundation-models/2-lammps/2-ethanol-water`: ethanol and water.

Run them with `sbatch submit.sh` (or `lmp -in in.lammps`).

---

## Tutorial 4: Fine-Tuning and distillation of Foundation GRACE Models

Here you fine-tune a foundation model on the Al-Li data of Tutorial 1, use it to generate more data
(distillation), and fit a fast GRACE/FS model on these data.

### Finetuning foundation models with a new dataset

```bash
cd 3-foundation-models/3a-finetuning
gracemaker -t
```

| Question | Answer |
|---|---|
| Fit type | `finetune foundation model` |
| Training dataset file | `../../1-AlLi-GRACE-2LAYER/0-data/collected.pkl.gz` |
| Use a separate test dataset file? | No |
| Test set fraction | 0.05 |
| Model tier / Training dataset / Model size | `2L` / `OMAT` / `medium` |
| Fine-tuning variant | `GRACE-2L-OMAT-medium-ft-E` |
| Finetuning precision | FP32 (default) |
| Finetuning mode | `naive` (all parameters); folder `3a-finetuning-freeze` uses `frozen weights` |
| Optimizer | `Adam` |
| Loss type | `huber` (delta 0.01) |
| Energy / force loss weight | 16 / 32 |
| Include stress in the loss? | Yes, weight 128 |
| Switch E/F/S weights mid-training? | Yes: energy 128, force 32, stress 128 |
| Batch size | 16 |
| Target total updates | 10000 |

Run `gracemaker` (or `sbatch submit.sh`). The final model is in `seed/1/final_model/`. If the fit
did not finish, export the best checkpoint with `gracemaker -r -s` (writes `seed/1/saved_model/`).
More on fine-tuning: [foundation models](foundation.md#fine-tuning-foundation-models).

To check the fine-tuned model, use the notebook `3-foundation-models/3a-finetuning/validate.ipynb`.

### Generating distilled data

Use the fine-tuned model to compute new reference data: see the notebook
`3-foundation-models/3b-distillation/distill_data.ipynb`. It writes `distilled_AlLi_dataset.pkl.gz`.

### Training a distilled model

Fit a GRACE/FS model on the distilled data:

```bash
cd 3-foundation-models/3b-distillation
gracemaker -t
```

The answers are the same as in [Tutorial 2, step 2.1.1](#fs-input-file), except:

| Question | Answer |
|---|---|
| Training dataset file | `distilled_AlLi_dataset.pkl.gz` |

Run `gracemaker input.yaml` (or `sbatch submit.sh`). The final model is in `seed/1/final_model/`.
Then export the GRACE/FS YAML file and build the active set:

```bash
gracemaker -r -s -sf
cd seed/1
pace_activeset -d training_set.pkl.gz saved_model.yaml
```

For validation, see the notebook `3-foundation-models/3b-distillation/validate.ipynb`.

#### LAMMPS simulations

Go to `3-foundation-models/3c-lammps-distilled`. The notebook `prepare_lammps_input.ipynb` makes
the input: a 500-atom Al FCC cell with 5 at.% Li. After equilibration, the system runs MD with Monte
Carlo atom swaps (MD-MC) at 500 K. There are two setups:

* `lammps-grace-2L`: the fine-tuned teacher model (GRACE-2L, TensorFlow).
* `lammps-grace-fs-dist`: the distilled student model (GRACE/FS, KOKKOS on GPU).

Run each with its `submit.sh` script. Both use a GPU. The Monte Carlo steps need only energies,
which makes them faster.

Typical speed on an A100 (40 GB):

* **GRACE/FS + KOKKOS/GPU:** 20.3 ns/day (1.18 h/ns), 234.97 timesteps/s, 117.49 katom-step/s
* **GRACE-2L + TensorFlow:** 3.26 ns/day (7.36 h/ns), 37.76 timesteps/s, 18.88 katom-step/s

The distilled model is about **6.5 times faster** than the teacher.

---

## Tutorial 5: Learning vectorial/tensorial properties

Fit a model that predicts a vector (first-rank tensor) for each atom. More on tensor fits:
[generic tensors](gen_tensor.md).

```bash
cd 4-fit-tensors/1-vector
gracemaker -t
```

| Question | Answer |
|---|---|
| Fit type | `Generic Tensors (L<=2)` |
| Training dataset file | `H2O_100.pkl.gz` |
| Use a separate test dataset file? | No |
| Test set fraction | 0.05 |
| Tensor type | `[1]` (first-rank vector) |
| Property granularity | `per-atom` |
| Model preset | `TENSOR_1L` |
| Cutoff radius (Å) | 6.0 |
| Fit energy alongside tensor? | No |
| Loss type | `square` |
| Tensor loss weight | 10.0 |
| Batch size | 10 |
| Target total updates | 10000 |

Then change the optimizer settings in `input.yaml` (larger learning rate, smaller EMA momentum):

```yaml
fit:
  opt_params: {learning_rate: 0.05, use_ema: True, ema_momentum: 0.5,
               weight_decay: null, clipnorm: 1.0}
```

Run `gracemaker input.yaml` (or `sbatch submit.sh`). After the fit, open the notebook
`4-fit-tensors/1-vector/use_model.ipynb`. It predicts the vectors for the test set and checks
rotational equivariance: it rotates the structures at random and compares the rotated outputs.

---

## Tutorial 6: Uncertainty and active learning with CALM

The [UQ workflows](uq_workflows.md) tutorial uses the small Al-Li and ethanol models in
`examples/data/` of the [grace-tensorpotential repository](https://github.com/ICAMS/grace-tensorpotential).
It shows how to build the CALM artifact, find extrapolative structures with LAMMPS or HAL, select
them for DFT, and refit.
