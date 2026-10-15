# `gracemaker` command line

`gracemaker` fits a GRACE model. Run it in the folder with your `input.yaml`:

```bash
gracemaker [options] [input.yaml]
```

All results go to `seed/<seed>/` (for example `seed/1/`). The input file is described in
[Input file](inputfile.md), the whole workflow in [Quick start](quickstart.md).

## Help text

Output of `gracemaker -h`:

```
usage: gracemaker [-h] [-l LOG] [-m] [-rl] [-r] [-rs RESTART_SUFFIX]
                  [--no-intra-epoch-redo] [-p POTENTIAL] [-s] [-sf] [-e] [-nj]
                  [--seed SEED] [-cm] [-t] [-cn CHECKPOINT_NAME]
                  [--reset-epoch-and-step]
                  [input]

Fitting utility for (graph) atomic cluster expansion potentials.

positional arguments:
  input                 input YAML file, default: input.yaml

options:
  -h, --help            show this help message and exit
  -l LOG, --log LOG     log filename, default: log.txt
  -m, --multigpu        Single host/multi GPU distributed fit
  -rl, --restart-latest
                        Restart from latest checkpoint (use separately from
                        -r/-rs)
  -r, --restart-best-test
                        Restart from latest best test checkpoint (use
                        separately from -rs/-rl)
  -rs RESTART_SUFFIX, --restart-suffix RESTART_SUFFIX
                        Suffix of checkpoint to restart from, i.e. .epoch_10
                        (use separately from -r/-rl)
  --no-intra-epoch-redo
                        When resuming from a mid-epoch checkpoint
                        (intra_epoch_save=True), skip the interrupted epoch's
                        remaining batches entirely and bump the step counter
                        to the nominal end of that epoch. This keeps the LR
                        scheduler aligned with the original maxiter budget at
                        the cost of under-training by (nominal_end -
                        saved_step) gradient updates. Default is to fast-
                        forward the iterator and train only the unseen tail,
                        which is usually cheaper and preserves full training
                        amount.
  -p POTENTIAL, --potential POTENTIAL
                        Potential configuration to load, model.yaml file
  -s, --save-model      Export model as TF saved model
  -sf, --save--fs       Export FS model as yaml to be loaded in CPP
  -e, --eager           Eager graph execution
  -nj, --no-jit         No JIT
  --seed SEED           Random seed (will overwrite value from input.yaml)
  -cm, --check-model    Check model consistency, without performing fit
  -t, --template        Generate a template 'input.yaml' file by dialog
  -cn CHECKPOINT_NAME, --checkpoint-name CHECKPOINT_NAME
                        Explicit name of the checkpoint (omit .index suffix)
  --reset-epoch-and-step
                        Reset epoch and step counters from prev. runs
```

## Output folder

| Path in `seed/<seed>/`                     | Content                                                     |
|--------------------------------------------|-------------------------------------------------------------|
| `log.txt`                                  | Log of all runs in this folder (new runs append to it; `-l` sets the name) |
| `model.yaml`                               | Model architecture                                          |
| `checkpoints/checkpoint.best_test_loss.*`  | Weights with the best test loss so far (`-r`)               |
| `checkpoints/checkpoint.*`                 | Latest regular checkpoint (`-rl`)                           |
| `final_model/`                             | SavedModel written at the end of the fit (best-test weights if there is a test set) |
| `saved_model/`                             | SavedModel written by `-s`                                  |
| `train_metrics.yaml`, `test_metrics.yaml`  | Metrics per epoch                                           |
| `training_set.pkl.gz`, `test_set.pkl.gz`   | The train and test split                                    |

## Common tasks

### Write an input file

```bash
gracemaker -t
```

An interactive dialog asks about the data, model, optimizer and loss. It writes `input.yaml` in
the current folder (an old file with this name is overwritten) and exits. See
[Input file](inputfile.md) for all keys and [Presets](presets.md) for the models.

### Check the model

```bash
gracemaker -cm
```

Builds the model from `input.yaml` for 89 elements (H to Ac), prints the number of trainable
parameters and exits. It does not read the data.

### Start a fit

```bash
gracemaker                  # reads input.yaml
gracemaker my_input.yaml    # another input file
gracemaker --seed 2         # same input, other seed; writes seed/2/
```

`--seed` overrides `seed` in the input file. Use it to fit several models from one input (for
example for an [ensemble](uq.md)). Note that `--seed 0` is ignored; use 1 or larger.

### Continue a fit (restart) {#restart}

Run the command in the folder of the old fit, with the same `input.yaml`. The model is taken from
`seed/<seed>/model.yaml`.

```bash
gracemaker -r               # from the best-test-loss checkpoint
gracemaker -rl              # from the latest regular checkpoint
gracemaker -rs .epoch_100   # from checkpoints/checkpoint.epoch_100
```

- Regular checkpoints are saved every `fit: checkpoint_freq` epochs (default 10). Only the last one
  is kept, unless `fit: save_all_regular_checkpoints: True`; then each one gets the suffix
  `.epoch_<N>` for `-rs`.
- The epoch and step counters continue from the checkpoint. If `maxiter` is already reached, raise
  it in `input.yaml`. To start the counters from 0 (for example a new learning-rate schedule on new
  data), add `--reset-epoch-and-step` (or `fit: reset_epoch_and_step: True`).
- `-p model.yaml` and `-cn path/to/checkpoint` load other files. They replace
  `potential: filename` and `potential: checkpoint_name` in the input file. With `-cn`, this
  checkpoint is loaded, even if `-r`, `-rl` or `-rs` is also given. A trailing `.index` is removed.

!!! warning "Restart of a fine-tuning fit"
    If `input.yaml` has `potential: finetune_foundation_model`, `gracemaker -r` (or `-rl`) stops
    with `Neither preset nor custom specified`. Give the model file of the fit:

    ```bash
    gracemaker -r -p seed/1/model.yaml
    ```

More details: [Checkpoints and restart](quickstart.md#restart-from-checkpoint).

### Restart from a checkpoint in the middle of an epoch {#intra-epoch-redo}

With `fit: checkpoint_freq_steps: N` (Adam only), a checkpoint is also saved every N steps inside
an epoch (see [Checkpoints within an epoch](inputfile.md#checkpoints-within-an-epoch)). It
replaces the latest regular checkpoint, so restart with `gracemaker -rl`. Then:

- **Default:** the fit goes back to the start of the interrupted epoch, skips the batches that are
  already trained and trains the rest of the epoch. The full number of updates is kept.
- **`--no-intra-epoch-redo`:** the rest of the interrupted epoch is not trained. The step counter
  jumps to the end of that epoch, so the learning-rate schedule ends where planned. The fit makes
  fewer updates than planned.

With a streaming dataset, the interrupted epoch is always trained again from its start, and this
option has no effect. `fit: reset_optimizer: True` also removes the mid-epoch state.

### Export the model {#export}

```bash
gracemaker -r -s            # best-test-loss weights -> seed/1/saved_model/
gracemaker -rl -s           # latest checkpoint      -> seed/1/saved_model/
```

The data is not loaded, so the export is fast. Use the SavedModel in
[ASE](quickstart.md#grace-tensorflow) and [LAMMPS](quickstart.md#lammps-grace-tensorflow).

!!! warning "Always give a checkpoint with `-s`"
    `gracemaker -s` alone loads **no** checkpoint and writes a model with untrained weights, with no
    error. Always combine `-s` with `-r`, `-rl`, `-rs` or `-cn`.

For [GRACE/FS](quickstart.md#gracefs) models (preset `FS`), `-sf` also writes the C++ YAML file:

```bash
gracemaker -r -s -sf        # seed/1/saved_model/ and seed/1/saved_model.yaml
gracemaker -sf              # fit; at the end also writes seed/1/FS_model.yaml
```

For other models, `-sf` fails. Export to `.npz` for the LAMMPS Kokkos pair styles, or export any
`model.yaml` and checkpoint, with [`grace_utils`](utilities.md#grace_utils).

### Fit on several GPUs {#multigpu}

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 gracemaker -m
```

`-m` (or `fit: strategy: mirrored`) trains on all visible GPUs of one machine. `batch_size` is per
GPU. For several nodes, start one `gracemaker` per node with `TF_CONFIG` set by the job script.
See [Several GPUs and nodes](inputfile.md#several-gpus-and-nodes).

### Debugging

- `-e`: run in eager mode (no `tf.function` graphs). Very slow; only for debugging.
- `-nj`: no XLA JIT compilation. Same as `fit: jit_compile: False`.
- `NUM_VIRTUAL_DEVICES=4 gracemaker -m` (environment variable): test a multi-GPU fit on the CPU
  with 4 virtual devices.

## Related pages

- [Quick start](quickstart.md): the steps of a fit, export and use in ASE and LAMMPS
- [Input file](inputfile.md): all keys of `input.yaml`
- [Tutorials](tutorials.md): complete examples
- [Foundation models](foundation.md): fine-tuning
- [Utilities](utilities.md): `grace_utils`, `grace_predict`, `grace_preprocess` and other tools
