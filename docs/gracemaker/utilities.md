## `grace_collect`

Utility to collect VASP calculations from a top-level directory and store them
in a _*.pkl.gz_ file that can be used for fitting with _gracemaker_ or _pacemaker_.
The reference energies could be provided for each element (default value is zero) or extracted automatically from the
calculation with single atom and large enough (>500 Ang^3/atom) volume. 

```
usage: grace_collect [-h] [-wd WORKING_DIR] [--output-dataset-filename OUTPUT_DATASET_FILENAME] [--free-atom-energy [FREE_ATOM_ENERGY ...]] [--selection SELECTION]

options:
  -h, --help            show this help message and exit
  -wd WORKING_DIR, --working-dir WORKING_DIR
                        top directory where keep calculations
  --output-dataset-filename OUTPUT_DATASET_FILENAME
                        pickle filename, default is collected.pckl.gzip
  --free-atom-energy [FREE_ATOM_ENERGY ...]
                        dictionary of reference energies (auto for extraction from dataset), i.e. `Al:-0.123 Cu:-0.456 Zn:auto`, default is zero. If option is `auto`, then it will be extracted from dataset
  --selection SELECTION
                        Option to select from multiple configurations of single VASP calculation: first, last, all, first_and_last (default: last)
```

___

## `grace_models`
Utility to download (all) foundation models

```
usage: grace_models [-h] {list,download} ...

Download foundational GRACE models

positional arguments:
  {list,download}  Sub-command help
    list           List available models
    download       Download a model

options:
  -h, --help       show this help message and exit

```

Example:
```bash
grace_models
```
___

## `grace_utils`
Utility to convert, export and summarize GRACE models

```
usage: grace_utils [-h] -p POTENTIAL [--param_dtype float32] [-c CHECKPOINT_PATH] [-os OUTPUT_SUFFIX] {update_model,resave_checkpoint,reduce_elements,cast_model,export,export_kokkos,summary,aux_model} ...

CLI tool for model conversions and summarization

positional arguments:
  {update_model,resave_checkpoint,reduce_elements,cast_model_param,export,export_kokkos,summary,aux_model}
    update_model        Update model (model.yaml) and corresponding checkpoint.
    resave_checkpoint   Resave model's (model.yaml) checkpoint (no optimizer)
    reduce_elements     Reduce elements from the model.
    cast_model_param    Change model parameters' floating point precision.
    export              Export model to saved_model or FS/C++ format.
    export_kokkos       Export GRACE-1L/2L weights to .npz for LAMMPS Kokkos pair style.
    summary             Show info about the model
    aux_model           Upgrade model with different compute functions: parallel_2L, compute_energy_only,

options:
  -h, --help            show this help message and exit
  -p POTENTIAL, --potential POTENTIAL
                        Path to model.yaml
  -c CHECKPOINT_PATH, --checkpoint-path CHECKPOINT_PATH
                        Path to checkpoint
  -os OUTPUT_SUFFIX, --output-suffix OUTPUT_SUFFIX
                        Output suffix for converted
  --param_dtype,        float32 (default), float64
                        Expected model parameters' dtype


-------------------
Optional arguments for different commands:

--------
update_model:
  None
 
--------
reduce_elements: 

  -e ELEMENTS [ELEMENTS ...], --elements ELEMENTS [ELEMENTS ...]
                        Elements to select
--------
resave_checkpoint:
 None
--------
export:

  -sf                   Save to GRACE-FS/C++ YAML model format
  -n SAVED_MODEL_NAME, --saved-model-name SAVED_MODEL_NAME
--------
export_kokkos:

  -o OUTPUT, --output OUTPUT
                        Output .npz path (e.g. grace_2l_weights.npz)
  --arch {auto,1l,2l}   Override architecture (default: auto-detect from instructions)
  --uq-artifacts UQ_ARTIFACTS_PATH
                        Path to gmm_artifacts.npz (UQ). Bakes dense uq_* arrays
                        (centroids, inverse covariances, gamma thresholds, error
                        model) into the exported .npz.
--------
cast_model_param:

  -curr {fp32,fp64}  Current precision type to cast from
  -to {fp32,fp64}    New precision type to cast into
--------
summary:

-v {0,1,2}, --verbose {0,1,2}
                        Verbosity level: 0, 1 or 2
--------
aux_model:

  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Path to save the upgraded model
  -ck COMMUNICATED_KEYS [COMMUNICATED_KEYS ...], --communicated-keys COMMUNICATED_KEYS [COMMUNICATED_KEYS ...]
                        List of communicated keys
  --aux AUX [AUX ...]   List of aux functions to add: parallel_2L, energy_only, compute_local (all by default)
```
#### Update models
If a model was fitted with `gracemaker` version < 0.5, it will break in the newer versions due to the format change.
Conversion to the new format can be easily done via:
 
```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.best_test_loss.index  -os dict update_model
```
one needs to provide path `-p` to the previously fitted `model.yaml`, `-c` path to the `checkpoint` of the corresponding model.
New updated `checkpoint` files and `model.yaml` will be saved with the suffix provided via `-os`.

#### Reduce model's chemical complexity

If you have a model that was fitted for large number of chemical elements, for example one of the [foundation models](../foundation/#pretrained-grace-foundation-models), but
you're interested only in a few specific, you can reduce the large model to the specified chemistry.
For example, selecting only  Mo, Nb, Ta and W from a large model:
```bash
grace_utils -p /path/to/model-dict.yaml -c /path/to/checkpoint/checkpoint-dict.index  -os MoNbTaW reduce_elements -e Mo Nb Ta W
```

#### Change model parameters' floating point precision
GRACE models can be trained in both single and double floating point precision
for trainable parameters.
Conversion between the two can be done with the `cast_model_param` utility,
for example to convert from single to double precision:

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint.index -os "double" cast_model_param -curr fp32 -to fp64
```


#### Export model to saved_model or GRACE-FS/C++ format

Export model.yaml + checkpoint into saved_model format:
```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index export -n my_saved_model
```

For GRACE-FS model, one can export to GRACE-FS/C++ format(.yaml) by adding `-sf` flag:
```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index export -n my_GRACE-FS.yaml -sf
```

#### Export to .npz for LAMMPS Kokkos pair style

GRACE-1L and GRACE-2L models can be exported to a self-contained `.npz` blob
that the `pair_grace_1l_kokkos` / `pair_grace_2l_kokkos` LAMMPS pair styles
read directly — no TensorFlow at LAMMPS runtime. The architecture (1L vs 2L)
is auto-detected from the model's instructions.

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index export_kokkos -o grace_weights.npz
```

Use the resulting file in your LAMMPS input as:

```
pair_style grace/1l/kk    # or grace/2l/kk
pair_coeff * * grace_weights.npz <element1> <element2> ...
```

!!! warning "Standard architectures only"
    `grace/1l/kk` and `grace/2l/kk` only support the standard GRACE-1L / 2L
    architectures from the built-in [presets](../presets/) and
    [foundation models](../foundation/). Models with non-standard instructions,
    unsupported activations, or dimensions above the LAMMPS compile-time caps
    are rejected by `export_kokkos`. For custom architectures use the
    TensorFlow-based pair styles (`grace`, `grace/2layer/parallel`, …) or
    GRACE-FS instead.

Use `--arch 1l` / `--arch 2l` to override architecture auto-detection.

##### Baking in UQ (uncertainty quantification) artifacts

If the model has a GMM-based UQ artifact (`gmm_artifacts.npz`, schema v2),
pass it via `--uq-artifacts` to bake the dense `uq_*` arrays — cluster
centroids, per-cluster inverse covariances, gamma extrapolation-grade
thresholds, and the per-element force-error model — directly into the
exported `.npz`:

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index \
            export_kokkos -o grace_weights.npz \
            --uq-artifacts /path/to/gmm_artifacts.npz
```

The Kokkos pair style then computes the per-atom extrapolation grade `gamma`
and predicted force error directly from the same `.npz` — no separate UQ file
is needed at LAMMPS runtime. Without `--uq-artifacts`, a plain (non-UQ) `.npz`
is exported.

The same `.npz` works with three runtime-precision variants of the pair style
— pick the one that matches your accuracy / throughput trade-off:

- `grace/{1l,2l}/kk`        — full fp64 (default)
- `grace/{1l,2l}/kk/mixed`  — geometry in fp64, NN math in fp32
- `grace/{1l,2l}/kk/fp32`   — everything in fp32

Empirically, `fp32` and `mixed` agree with `fp64` to roughly **1e-6 relative
precision** on energies and forces — well within typical MD requirements.

#### Model summary
To print summary of the GRACE model with different level of verbosity (0 - least, 1 - moderate and 2 - most verbose):

```bash
grace_utils -p /path/to/model.yaml summary -v 1
```

#### Upgrade model with auxiliary compute functions
One can add auxiliary compute functions to the model, for example to compute energy only (using `energy_only` compute function).
Also, one can split the 2L model into two parts for parallel computation (using `parallel_2L` aux function).

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index aux_model -o /path/to/upgraded_model --aux energy_only parallel_2L
```

## `grace_predict`
Utility to predict energies, forces and stresses for given ASE Atoms structures in dataset.pkl.gzip

```
 grace_predict [-h] [-m MODEL_PATH] [-d DATASET_FILE] [-o OUTPUT]

options:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        provide path to the saved_model directory
  -d DATASET_FILE, --dataset DATASET_FILE
                        path to the dataset.pkl.gzip containing ase_atoms structures
  -o OUTPUT, --output OUTPUT
                        path to the OUTPUT dataset (pkl.gzip) containing energy_predicted and forces_predicted
```

## `grace_preprocess`

Helper utility to precompute neighbour lists and preprocess batches for the large dataset.
Data is saved into TF.Dataset format.
Usually used in distributed training and called by `compute_distributed_data.sh` script (location: tests/data_distrib/compute_distributed_data.sh)

```
grace_preprocess [-h] [-o OUTPUT] [--sharded-input] [-e ELEMENTS] [-b BATCH_SIZE] [-bu MAX_N_BUCKETS] [-c CUTOFF] [-cd CUTOFF_DICT] [--compression COMPRESSION] [--energy-col ENERGY_COL] [--forces-col FORCES_COL] [--stress-col STRESS_COL] [--is-fit-stress] [-s STRATEGY] [--task-id TASK_ID]
                        [--total-task-num TOTAL_TASK_NUM] [--rerun] [--stage-1] [--stage-2] [--stage-3] [--stage-4] [--remove_stage1]
                        input [input ...]

Precompute dataset and save into TF.Dataset format

positional arguments:
  input                 input pkl.gz file

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output file name
  --sharded-input       Flag to show that input files are sharded
  -e ELEMENTS, --elements ELEMENTS
                        List of elements. Possible presets: `ALL` (except last 23 elements), `Alexandria` or `MP`
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -bu MAX_N_BUCKETS, --max-n-buckets MAX_N_BUCKETS
  -c CUTOFF, --cutoff CUTOFF
  -cd CUTOFF_DICT, --cutoff_dict CUTOFF_DICT
  --compression COMPRESSION
  --energy-col ENERGY_COL
  --forces-col FORCES_COL
  --stress-col STRESS_COL
  --is-fit-stress
  -s STRATEGY, --strategy STRATEGY
                        Strategy to batch splitting. Possible values: structures (default), atoms, neighbours
  --task-id TASK_ID     ZERO based ID of task
  --total-task-num TOTAL_TASK_NUM
                        Total number of tasks
  --rerun               Enforce to rerun process
  --stage-1             Run stage 1, precompute samples (non-batched)
  --stage-2             Run stage 2, compute padding bounds
  --stage-3             Run stage 3, padding batches
  --stage-4             Run stage 4, compute statistics
  --remove_stage1       If True - during stage 3, remove corresponding shard from stage1 folder
```
