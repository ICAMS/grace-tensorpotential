# Utilities

Besides `gracemaker` itself (see [CLI](cli.md)), the `tensorpotential` package installs these
command-line tools:

| Tool | What it does |
| :--- | :--- |
| [`grace_collect`](#grace_collect) | Collect VASP calculations into a dataset (`.pkl.gz`). |
| [`extxyz2df`](#extxyz2df), [`df2extxyz`](#df2extxyz) | Convert between extended XYZ and a dataset (`.pkl.gz`), in either direction. |
| [`grace_models`](#grace_models) | List, inspect, download and update foundation models. |
| [`grace_utils`](#grace_utils) | Convert, export (SavedModel, GRACE/FS, LAMMPS Kokkos) and summarize a fitted model. |
| [`grace_predict`](#grace_predict) | Predict energies, forces and stresses for a dataset. |
| [`grace_preprocess`](#grace_preprocess) | Precompute a large dataset as sharded TensorFlow datasets. |
| [`grace_uq`](#grace_uq) | Uncertainty quantification: build, predict, select, HAL. |
| [`grace_dashboard`](#grace_dashboard) | Browse and compare fit results in a web browser. |

Every tool prints its help with `-h`.

___

## `grace_collect`

Collects VASP calculations from a directory tree into one `.pkl.gz` dataset for
_gracemaker_ or _pacemaker_. It subtracts free-atom reference energies, which you give
for each element or let it find in the data.

```
usage: grace_collect [-h] [-wd WORKING_DIR] [--output-dataset-filename OUTPUT_DATASET_FILENAME]
                     [--free-atom-energy [FREE_ATOM_ENERGY ...]] [--selection SELECTION]

options:
  -h, --help            show this help message and exit
  -wd WORKING_DIR, --working-dir WORKING_DIR
                        top directory where keep calculations
  --output-dataset-filename OUTPUT_DATASET_FILENAME
                        pickle filename, default is collected.pkl.gz
  --free-atom-energy [FREE_ATOM_ENERGY ...]
                        dictionary of reference energies (auto for extraction from dataset), i.e. `Al:-0.123 Cu:-0.456
                        Zn:auto`, default is zero. If option is `auto`, then it will be extracted from dataset
  --selection SELECTION
                        Option to select from multiple configurations of single VASP calculation: first, last, all,
                        first_and_last (default: last)
```

* **Input.** It walks through `-wd` (default: the current directory) and all its
  subdirectories. In each directory it reads `OUTCAR`; if there is none, `vasprun.xml`; if
  there is neither, every file whose name contains `OUTCAR` or `vasprun.xml` (e.g.
  `OUTCAR_001`). Files that cannot be read are logged and skipped.
* **Output.** The default file name is `collected.pkl.gz`. The columns are `name`, `ase_atoms`,
  `energy` (force-consistent free energy), `forces`, `stress`, `results`, `energy_corrected` and `energy_corrected_per_atom`.
* **Reference energies.** `energy_corrected` = `energy` minus the sum of the free-atom
  energies. Without `--free-atom-energy` all reference energies are zero.
  `--free-atom-energy auto` finds them for all elements; `El:auto` does it for one element.
  For `auto`, the tool looks for a periodic single-atom calculation of that element with a
  volume above 500 Å³ and takes the one with the largest volume. If there is none, it
  warns and uses zero.
* **Several steps per calculation** (relaxation, MD). `--selection` takes `first`, `last`
  (default), `first_and_last`, `all`, or a Python-style slice such as `::10`.

```bash
grace_collect                                         # current directory -> collected.pkl.gz
grace_collect -wd calcs --free-atom-energy auto       # find free-atom energies in the data
grace_collect -wd md_runs --selection ::10 --free-atom-energy Al:-0.123 Li:-0.456 \
              --output-dataset-filename md.pkl.gz     # every 10th MD step, given reference energies
```

___

## `extxyz2df`

Converts an extended XYZ file into the `.pkl.gz` dataset format. You do not always need it:
`input.yaml` also accepts `.xyz` / `.extxyz` files directly.

```
usage: extxyz2df [-h] [--output-dataset-filename OUTPUT_DATASET_FILENAME] extxyz_filename

positional arguments:
  extxyz_filename       Name of extxyz file

options:
  -h, --help            show this help message and exit
  --output-dataset-filename OUTPUT_DATASET_FILENAME
                        pickle filename, default is inferenced from extxyz_filename
```

```bash
extxyz2df train.xyz                                     # -> train.pkl.gz
extxyz2df train.extxyz                                  # -> train.pkl.gz
extxyz2df train.extxyz --output-dataset-filename my_train.pkl.gz
```

Energies, forces and stresses are read through ASE, so the file must store them under the
standard keys (`energy`, `forces`, `stress`). The output has the columns `ase_atoms`,
`energy`, `forces`, `stress` and `info` (the `atoms.info` dictionary). There is no
`energy_corrected` column; give `reference_energy` in `input.yaml` to subtract free-atom
energies (see [Data collection](quickstart.md#data-collection)).

___

## `df2extxyz`

The reverse conversion: from a `.pkl.gz` dataset back to extended XYZ, for example to look at
a dataset or to use it with other tools.

```
usage: df2extxyz [-h] [-e ENERGY_COLUMN] [-f FORCE_COLUMN] [-s STRESS_COLUMN] [-o OUTPUT] input

Conversion from df.pkl.gz to extxyz

positional arguments:
  input                 input pkl.gz file

options:
  -h, --help            show this help message and exit
  -e ENERGY_COLUMN, --energy-column ENERGY_COLUMN
                        name of energy column
  -f FORCE_COLUMN, --force-column FORCE_COLUMN
                        name of forces column
  -s STRESS_COLUMN, --stress-column STRESS_COLUMN
                        name of stress column
  -o OUTPUT, --output OUTPUT
                        output file name
```

The default columns are `energy_corrected`, `forces` and `stress` (stress is skipped if the
column is missing). Without `-o`, the output name is the input name with `.pkl.gz` replaced by
`.xyz`. The values are written twice: as `energy`/`forces`/`stress` and as
`REF_energy`/`REF_forces`/`REF_stress`.

```bash
df2extxyz train.pkl.gz                        # -> train.xyz
df2extxyz train.pkl.gz -o train.extxyz
df2extxyz from_extxyz2df.pkl.gz -e energy     # extxyz2df output has no energy_corrected column
```

___

## `grace_models`

Lists, inspects, downloads and updates the [foundation models](foundation.md).

```
usage: grace_models [-h] {list,info,download,checkpoint,update} ...

Download foundation GRACE models

positional arguments:
  {list,info,download,checkpoint,update}
                        Sub-command help
    list                List available models
    info                Show declared + downloaded-artifact capability flags for a model
    download            Download a model
    checkpoint          Download a checkpoint
    update              Re-download cached models/checkpoints whose origin URL is out of date

options:
  -h, --help            show this help message and exit
```

| Subcommand | Purpose |
| :--- | :--- |
| `list` | Table of all models and what they support. `-v`/`--verbose` prints the full block for each model (description, paths, license); `--ascii` uses ASCII signs instead of unicode. |
| `info <model>` | Declared features (precision, UQ, parallel) of one model (name or alias). If the model is downloaded, it also inspects the files. |
| `download <model>` | Download a model (`all`: every model). `--kokkos` also downloads its LAMMPS Kokkos export (`kokkos.npz`) into the model directory. |
| `checkpoint <model>` | Download only the checkpoint (`all`: every model). You need it for fine-tuning. |
| `update [<model> ...]` | Download again the cached copies that are out of date. `--all` checks every built-in model, `--dry-run` only reports, `--force` downloads even copies that are up to date. See [Updating cached models](#updating-cached-models). |

??? info "Full help of the subcommands"
    ```
    usage: grace_models list [-h] [-v] [--ascii]
      -v, --verbose  full per-model block (description, paths, license) instead of the table
      --ascii        use ASCII glyphs (Y/N/./-) instead of unicode in the table

    usage: grace_models info [-h] model_name
      model_name  Name (or alias) of the model to inspect

    usage: grace_models download [-h] [--kokkos] model_name
      model_name  Name of the model to download
      --kokkos    also download the LAMMPS-Kokkos export (kokkos.npz) into the model dir

    usage: grace_models checkpoint [-h] model_name
      model_name  Name of the model to download checkpoint for

    usage: grace_models update [-h] [--all] [--force] [--dry-run] [model_name ...]
      model_name  Model name(s) to update
      --all       check every built-in model
      --force     re-download even when the origin URL already matches
      --dry-run   report what would change, download nothing
    ```

```bash
grace_models list                                          # what is available and what is downloaded
grace_models info GRACE-2L-OMAT-medium-ft-AM               # features of one model
grace_models download GRACE-2L-OMAT-medium-ft-AM --kokkos  # model + kokkos.npz for LAMMPS Kokkos
```

See [foundation models](foundation.md#downloading-foundation-models) for the model tables and
the cache location.

### Updating cached models

The cache is keyed by model name, so a new version of `tensorpotential` alone never
refreshes a model you already downloaded. When a model is exported again upstream (as in
0.6.1, when the UQ-enabled models were rebuilt to fix a NaN in `virial_sigma`), use
`grace_models update`:

```bash
grace_models update GRACE-2L-OMAT-medium-ft-AM   # one model
grace_models update --all --dry-run     # report what is out of date, download nothing
grace_models update --all               # update everything that is out of date
```

Each cached model and checkpoint directory has a `.grace_origin.json` file with the URL it
came from. `update` compares it with the URL of the installed package and downloads only the
ones that differ. A directory without this file counts as out of date. The old copy is first
moved to `<path>.pre-update`; it is deleted only after the new download succeeds. If the
download fails, the old copy is put back.

!!! warning "Directories you manage yourself are never replaced"
    `update` does not touch a cache entry that is a symlink, or whose path comes from an
    explicit `path:` / `checkpoint_path:` in
    [`models_registry.yaml`](foundation.md#user-defined-models). It reports what it would do
    and goes on. Update such a directory yourself.

`kokkos.npz` is not part of the updated download. If you use the
[Kokkos pair styles](#export-to-npz-for-lammps-kokkos-pair-style), get it again with
`grace_models download <model> --kokkos` after an update. `update` reminds you when it
replaces a directory that had one.

___

## `grace_utils`

Converts, exports and summarizes a fitted model (`model.yaml` + checkpoint). The general
options (`-p`, `-c`, `-os`, `--param_dtype`) go **before** the subcommand; the options of
the subcommand go after it:

```bash
grace_utils -p model.yaml -c checkpoint/checkpoint.best_test_loss.index <subcommand> [options]
```

| Subcommand | Purpose |
| :--- | :--- |
| [`update_model`](#update-models) | Convert a model from _gracemaker_ < 0.5 to the current format. |
| `resave_checkpoint` | Save the checkpoint again without the optimizer state (smaller file). |
| [`reduce_elements`](#reduce-models-chemical-complexity) | Keep only some elements of the model. |
| [`cast_model_param`](#change-model-parameters-floating-point-precision) | Change the precision of the parameters (fp32 / fp64). |
| [`export`](#export-model-to-saved_model-or-grace-fsc-format) | Export to a TensorFlow SavedModel or to the GRACE/FS C++ format. |
| [`export_kokkos`](#export-to-npz-for-lammps-kokkos-pair-style) | Export GRACE-1L/2L/3L to `.npz` for the LAMMPS Kokkos pair styles. |
| [`summary`](#model-summary) | Print information about the model. |
| [`aux_model`](#upgrade-model-with-auxiliary-compute-functions) | Save a SavedModel with extra compute functions (energy only, 2L parallel). |

* `-p` (path to `model.yaml`) is always required. `-c` (checkpoint; the `.index` suffix is
  optional) is needed by every subcommand except `summary`.
* `--param_dtype` (`float32` or `float64`): if not given, it is read from `model.yaml`;
  old models without this information use `float64`.
* `-os` (default `converted`) is the suffix of the new files of `update_model`,
  `resave_checkpoint`, `reduce_elements` and `cast_model_param`.

??? info "Full help"
    ```
    usage: grace_utils [-h] -p POTENTIAL [--param_dtype PARAM_DTYPE] [-c CHECKPOINT_PATH] [-os OUTPUT_SUFFIX]
                       {update_model,resave_checkpoint,reduce_elements,cast_model_param,export,export_kokkos,summary,aux_model}
                       ...

    CLI tool for model conversions and summarization

    positional arguments:
      {update_model,resave_checkpoint,reduce_elements,cast_model_param,export,export_kokkos,summary,aux_model}
        update_model        Update model (model.yaml) and corresponding checkpoint.
        resave_checkpoint   Resave model's (model.yaml) checkpoint (no optimizer)
        reduce_elements     Reduce elements from the model.
        cast_model_param    Change model's floating point precision.
        export              Export model to saved_model or FS/C++ format.
        export_kokkos       Export GRACE-1L/2L/3L weights to .npz for LAMMPS Kokkos pair style.
        summary             Show info about the model
        aux_model           Upgrade model with different compute functions: parallel, compute energy only, compute local

    options:
      -h, --help            show this help message and exit
      -p POTENTIAL, --potential POTENTIAL
                            Path to model.yaml
      --param_dtype PARAM_DTYPE
                            Model parameters' dtype (float32 or float64). If not set, inferred from model.yaml metadata;
                            falls back to float64 for old models.
      -c CHECKPOINT_PATH, --checkpoint-path CHECKPOINT_PATH
                            Path to checkpoint
      -os OUTPUT_SUFFIX, --output-suffix OUTPUT_SUFFIX
                            Output suffix for converted

    ---------------- options of the subcommands ----------------

    usage: grace_utils update_model [-h]
    usage: grace_utils resave_checkpoint [-h]

    usage: grace_utils reduce_elements [-h] -e ELEMENTS [ELEMENTS ...]
      -e ELEMENTS [ELEMENTS ...], --elements ELEMENTS [ELEMENTS ...]
                            Elements to select

    usage: grace_utils cast_model_param [-h] -curr {fp32,fp64} -to {fp32,fp64}
      -curr {fp32,fp64}  Current precision type to cast from
      -to {fp32,fp64}    New precision type to cast into

    usage: grace_utils export [-h] [-sf] [-n SAVED_MODEL_NAME] [-ck COMMUNICATED_KEYS [COMMUNICATED_KEYS ...]]
      -sf                   Save to GRACE-FS/C++ YAML model format
      -n SAVED_MODEL_NAME, --saved-model-name SAVED_MODEL_NAME
                            Path where to save the SavedModel or YAML file
      -ck COMMUNICATED_KEYS [COMMUNICATED_KEYS ...], --communicated-keys COMMUNICATED_KEYS [COMMUNICATED_KEYS ...]
                            List of communicated keys for 2L model parallel split export

    usage: grace_utils export_kokkos [-h] -o OUTPUT [--arch {auto,1l,2l,3l}] [--uq-artifacts UQ_ARTIFACTS_PATH]
      -o OUTPUT, --output OUTPUT
                            Output .npz path (e.g. grace_2l_weights.npz)
      --arch {auto,1l,2l,3l}
                            Override architecture (default: auto-detect from instructions).
      --uq-artifacts UQ_ARTIFACTS_PATH
                            Path to gmm_artifacts.npz (UQ). If given, dense uq_* arrays (centroids, inverse covariances,
                            gamma thresholds, error model) are baked into the exported kokkos .npz.

    usage: grace_utils summary [-h] [-v {0,1,2}]
      -v {0,1,2}, --verbose {0,1,2}
                            Verbosity level: 0, 1 or 2

    usage: grace_utils aux_model [-h] -o OUTPUT_PATH [-ck COMMUNICATED_KEYS [COMMUNICATED_KEYS ...]]
      -o OUTPUT_PATH, --output-path OUTPUT_PATH
                            Path to save the upgraded model
      -ck COMMUNICATED_KEYS [COMMUNICATED_KEYS ...], --communicated-keys COMMUNICATED_KEYS [COMMUNICATED_KEYS ...]
                            List of communicated keys, used for parallelization of GRACE-2L model.
    ```

### Update models

A model fitted with _gracemaker_ < 0.5 does not load in newer versions, because the format
changed. Convert it with:

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.best_test_loss.index -os dict update_model
```

The new files get the suffix from `-os`: here `model-dict.yaml` and
`checkpoint.best_test_loss-dict.*`. A model that is already in the new format is left as it is.

### Reduce model's chemical complexity

A model fitted for many elements (for example a
[foundation model](foundation.md#pretrained-grace-foundation-models)) can be cut down to
the few elements you need. For example, keep only Mo, Nb, Ta and W:

```bash
grace_utils -p /path/to/model-dict.yaml -c /path/to/checkpoint/checkpoint-dict.index -os MoNbTaW reduce_elements -e Mo Nb Ta W
```

This writes `model-dict-MoNbTaW.yaml` and `checkpoint-dict-MoNbTaW.*`. The model must be in
the new format (run `update_model` first if needed). The tool checks that the dimer energies
of the kept elements do not change.

### Change model parameters' floating point precision

GRACE models can have their parameters in single (fp32) or double (fp64) precision. To
convert, for example from single to double precision:

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint.index -os double cast_model_param -curr fp32 -to fp64
```

The result is written in two forms:

* `casted-double/` next to the checkpoint, with `model.yaml` and `checkpoint.*`. This
  `model.yaml` records the new precision, so other tools (`export`, `export_kokkos`,
  `grace_uq`) pick it up without `--param_dtype`.
* `casted_model/` in the current directory: a SavedModel.

### Export model to saved_model or GRACE-FS/C++ format

Export `model.yaml` + checkpoint as a TensorFlow SavedModel (default name `saved_model`):

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index export -n my_saved_model
```

The SavedModel already contains the extra compute functions (energy only; for 2L models
also the parallel split used by `grace/2layer/parallel`). The communicated keys of the 2L
split are found automatically; `-ck` sets them by hand.

A GRACE/FS model can be exported to the GRACE/FS C++ format (`.yaml`) with the `-sf` flag:

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index export -n my_GRACE-FS.yaml -sf
```

### Export to .npz for LAMMPS Kokkos pair style

GRACE-1L, GRACE-2L and GRACE-3L models can be exported to one `.npz` file that the
`grace/1l/kk`, `grace/2l/kk` and `grace/3l/kk` LAMMPS pair styles read directly. No
TensorFlow is needed in LAMMPS. The architecture (1L, 2L or 3L) is found from the model;
`--arch 1l|2l|3l` overrides it.

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index export_kokkos -o grace_weights.npz
```

Use the file in your LAMMPS input:

```
pair_style grace/1l/kk    # or grace/2l/kk, grace/3l/kk
pair_coeff * * grace_weights.npz <element1> <element2> ...
```

For foundation models you do not need to export: `grace_models download <model> --kokkos`
downloads a ready `kokkos.npz`.

!!! warning "Standard architectures only"
    The Kokkos pair styles support only the standard GRACE-1L / 2L / 3L architectures from
    the built-in [presets](presets.md) and [foundation models](foundation.md).
    `export_kokkos` rejects models with non-standard instructions, unsupported activations,
    or sizes above the LAMMPS compile-time limits. For custom architectures use the
    TensorFlow pair styles (`grace`, `grace/2layer/parallel`, …) or GRACE/FS.

#### Baking in UQ (uncertainty quantification) artifacts

If the model has a CALM UQ artifact (`gmm_artifacts.npz`, made by
[`grace_uq build`](uq.md#grace_uq-build)), pass it with `--uq-artifacts`. The `uq_*`
arrays (cluster centroids, inverse covariances, the random projection of the basis and the
γ thresholds) are then stored in the exported `.npz`:

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index \
            export_kokkos -o grace_weights.npz \
            --uq-artifacts /path/to/gmm_artifacts.npz
```

The Kokkos pair style then computes the per-atom extrapolation grade γ from this one file;
LAMMPS needs no other UQ file. Without `--uq-artifacts` the `.npz` has no UQ.

#### Precision of the Kokkos pair styles

The same `.npz` works with several precision variants of the pair style. Choose by the
accuracy and speed you need:

- `grace/{1l,2l}/kk`        — everything in fp64 (default)
- `grace/{1l,2l}/kk/mixed`  — geometry in fp64, neural-network math in fp32
- `grace/{1l,2l}/kk/fp32`   — everything in fp32

3L models are fp32 by design: `grace/3l/kk` is the mixed-precision style (geometry in fp64)
and `grace/3l/kk/fp32` does everything in fp32. There is no fp64 3L style.

In tests, `fp32` and `mixed` agree with `fp64` to about **1e-6 relative** on energies and
forces, which is enough for normal MD.

### Model summary

Print a summary of the model; `-v` sets the detail (0: least, 1: medium, 2: most):

```bash
grace_utils -p /path/to/model.yaml summary -v 1
```

### Upgrade model with auxiliary compute functions

`aux_model` saves a SavedModel with an extra energy-only compute function. For 2L models it
also adds the parallel split used by `grace/2layer/parallel`; the communicated keys are set
with `-ck` (default `I_nl_LN I`). There is no `--aux` option.

```bash
grace_utils -p /path/to/model.yaml -c /path/to/checkpoint/checkpoint.index aux_model -o /path/to/upgraded_model
```

!!! note
    [`export`](#export-model-to-saved_model-or-grace-fsc-format) already adds the same
    functions and finds the communicated keys by itself, so in most cases you do not need
    `aux_model`.

___

## `grace_predict`

Predicts energies, forces and stresses with a SavedModel for all structures in a `.pkl.gz`
dataset (column `ase_atoms`). It writes a new dataset with the extra columns
`energy_predicted`, `forces_predicted` and `stress_predicted`.

```
usage: grace_predict [-h] [-m MODEL_PATH] [-d DATASET_FILE] [-o OUTPUT] [-e]

options:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        provide path to the saved_model directory
  -d DATASET_FILE, --dataset DATASET_FILE
                        path to the dataset.pkl.gzip containing ase_atoms structures
  -o OUTPUT, --output OUTPUT
                        path to the OUTPUT dataset (pkl.gzip) containing energy_predicted and forces_predicted
  -e, --raise-errors    Whether to NOT ignore errors and stop the program.
```

* Defaults: `-m saved_model`, `-d dataset.pkl.gz`, `-o predicted_dataset.pkl.gz`.
* The `ase_atoms` column is not written to the output; all other columns are kept.
* If a structure fails, the tool logs a warning and stores `None` for it. With `-e` it stops
  at the first error.
* If the dataset has a `mag_mom` column, these magnetic moments are set on the structures.

```bash
grace_predict -m seed/1/saved_model -d test.pkl.gz -o test_predicted.pkl.gz
```

For per-atom γ (UQ) over a dataset use [`grace_uq predict`](uq.md#grace_uq-predict).

### Python API: `predict_structures`

`grace_predict` is a thin wrapper around `predict_structures`, which evaluates any ASE
calculator on a list of structures. [`grace_uq predict`](uq.md#cli-tool-grace_uq) uses the
same code, so it is also the way to get per-atom `gamma` for a whole dataset from Python:

```python
from tensorpotential.calculator import TPCalculator, predict_structures

calc = TPCalculator(model="GRACE-2L-OMAT-medium-ft-AM")
out = predict_structures(structures, calc)
out["energy"]   # list, one entry per structure, in the order of `structures`
out["forces"]

# UQ-enabled model: get gamma together with the energy
out = predict_structures(structures, calc, properties=("energy",), extra=("gamma",))
```

The structures are copied, so your own objects never get a calculator attached. A property
the calculator cannot provide (`stress`, for example) comes back as `None` for that structure;
there is no error.

| Argument | Purpose |
| :--- | :--- |
| `properties` | Which of `energy`, `forces`, `stress` to compute. Default `("energy", "forces")`. |
| `extra` | Keys read from `calc.results` after each evaluation, e.g. `gamma`, `atomic_sigma`, `features` for a UQ-enabled model. A key the model does not provide gives `None`. |
| `optional` | Properties the calculator may not implement; they give `None`, not an error. Default `("stress",)`. |
| `sort_by_natoms` | Evaluate the largest structures first, so the biggest XLA shape is compiled only once. Default `True`. The results are in input order either way. |
| `on_error` | `"raise"` (default), or `"warn"`: log the failure and store `None` for that structure. |
| `progress` | `callable(done, total)`, called after each structure, e.g. for a progress bar. |
| `postprocess` | `callable(dict) -> dict`, applied to the `extra` values of each structure before they are stored, e.g. to keep the features of only some atoms. |

___

## `grace_preprocess`

Precomputes neighbour lists and padded batches of a large dataset and saves them as sharded
TensorFlow datasets, for fits on many GPUs or nodes. The workflow (four stages, the
`input.yaml` settings): [Precomputed shards](inputfile.md#precomputed-shards).

```
usage: grace_preprocess [-h] [-o OUTPUT] [--sharded-input] [-e ELEMENTS]
                        [-b BATCH_SIZE] [-bu MAX_N_BUCKETS] [-c CUTOFF]
                        [-cd CUTOFF_DICT] [--compression COMPRESSION]
                        [--energy-col ENERGY_COL] [--forces-col FORCES_COL]
                        [--stress-col STRESS_COL] [--is-fit-stress]
                        [--precision PRECISION] [-s STRATEGY]
                        [--task-id TASK_ID] [--total-task-num TOTAL_TASK_NUM]
                        [--rerun] [--stage-1] [--stage-2] [--stage-3]
                        [--stage-4] [--remove_stage1] [--dense-nbr]
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
                        List of elements. Possible presets: `ALL` (except last
                        23 elements), `Alexandria` or `MP`
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -bu MAX_N_BUCKETS, --max-n-buckets MAX_N_BUCKETS
  -c CUTOFF, --cutoff CUTOFF
  -cd CUTOFF_DICT, --cutoff_dict CUTOFF_DICT
  --compression COMPRESSION
  --energy-col ENERGY_COL
  --forces-col FORCES_COL
  --stress-col STRESS_COL
  --is-fit-stress
  --precision PRECISION
  -s STRATEGY, --strategy STRATEGY
                        Strategy to batch splitting. Possible values:
                        structures (default), atoms, neighbours
  --task-id TASK_ID     ZERO based ID of task
  --total-task-num TOTAL_TASK_NUM
                        Total number of tasks
  --rerun               Enforce to rerun process
  --stage-1             Run stage 1, precompute samples (non-batched)
  --stage-2             Run stage 2, compute padding bounds
  --stage-3             Run stage 3, padding batches
  --stage-4             Run stage 4, compute statistics
  --remove_stage1       If True - during stage 3, remove corresponding shard
                        from stage1 folder
  --dense-nbr           Mark the dataset for the dense (reshape) aggregation.
                        Reserved for the batched dense data prep (per-atom-
                        uniform layout bucketed by max_neigh, TODO); should
                        match the model (input.yaml::potential::dense_nbr).
```

Defaults: `-o tf_dataset`, `-b 8`, `-bu 5`, `-c 6`, `--compression GZIP`,
`--precision float64`, `--energy-col energy_corrected`. Each call runs exactly one stage
(`--stage-1` … `--stage-4`).

___

## `grace_uq`

Uncertainty quantification with CALM, the extrapolation grade γ. One subcommand per task:

| Subcommand | Purpose |
| :--- | :--- |
| `build` | Build the CALM artifact (`gmm_artifacts.npz`) from the training data and export a SavedModel with UQ. |
| `info` | Print a summary of a CALM artifact. |
| `predict` | Energies, forces, stresses and per-atom γ for a dataset. |
| `select` | Pick structures for DFT from a pool of candidates or from HAL walkers. |
| `hal` | Run an uncertainty-biased (HAL) MD walker that collects extrapolative frames. |

```bash
grace_uq build --model-yaml model.yaml \
               --checkpoint checkpoints/checkpoint.best_test_loss.index \
               --train-data training_set.pkl.gz
grace_uq predict --model UQ/saved_model --dataset candidates.pkl.gz
grace_uq <subcommand> --help       # options of one subcommand
```

The options of all subcommands: [Uncertainty Quantification](uq.md#cli-tool-grace_uq). A
step-by-step tutorial from γ to new training data: [UQ workflows](uq_workflows.md).

___

## `grace_dashboard`

A web dashboard to browse and compare fit results. It searches a base directory (and all
its subdirectories) for _fit folders_, i.e. directories that contain
`seed/<N>/test_metrics.yaml`, and shows three panels in the browser:

1. **Training curves**: metric vs. epoch, for each fit.
2. **Scatter / Pareto plot** with X and Y axes of your choice.
3. **Overview table**: all metrics at the epoch with the best test loss.

Several seeds of the same fit are shown as separate rows; a checkbox averages them.

```
usage: grace_dashboard [-h] [--base BASE] [--host HOST] [--port PORT] [--target-epochs TARGET_EPOCHS] [--debug]

options:
  -h, --help            show this help message and exit
  --base BASE           directory to scan for fit folders (default: cwd)
  --host HOST           bind host
  --port PORT           bind port
  --target-epochs TARGET_EPOCHS
                        target epochs used for extrapolation of incomplete runs
  --debug
```

Defaults: `--host 0.0.0.0`, `--port 5000`, `--target-epochs 160` (used to extrapolate the
curves of fits that are not finished). `--debug` runs Flask in debug mode.

!!! note "Install Flask"
    The dashboard needs [Flask](https://flask.palletsprojects.com/), which is not installed
    with `tensorpotential`: `pip install flask`.

```bash
grace_dashboard                                  # scan the current directory, open http://localhost:5000/
grace_dashboard --base fits/ --port 8050
grace_dashboard --host 127.0.0.1                 # only this machine; on a cluster use an SSH tunnel:
                                                 #   ssh -L 5000:localhost:5000 <cluster>
```

With the default `--host 0.0.0.0` other computers in the network can open the dashboard
too. Use `--host 127.0.0.1` if you do not want that.
