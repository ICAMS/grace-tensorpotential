# Input file `input.yaml`

Generate the input file with `gracemaker -t` (interactive), then edit it if needed. This page
lists the options. Defaults are the values the code uses when a key is left out.

## Minimal input

A fit from scratch, as `gracemaker -t` writes it:

```yaml
seed: 1
cutoff: 6

data:
  filename: train.pkl.gz
  test_size: 0.05            # 5 % of the structures as test set
  reference_energy: 0

potential:
  preset: GRACE_1LAYER_latest
  kwargs: {n_rad_max: 32, prod_func_n_max: 64, n_mlp_dens: 12}   # see Presets
  scale: True

fit:
  loss: {
    energy: { type: huber, weight: 16, delta: 0.01 },
    forces: { type: huber, weight: 32, delta: 0.01 },
  }
  target_total_updates: 50000
  optimizer: Adam
  opt_params: { learning_rate: 0.008, use_ema: True, ema_momentum: 0.99, clipnorm: 1.0 }
  scheduler: cosine_decay
  scheduler_params: { minimum_learning_rate: 6.7e-4 }
  batch_size: 16
  test_batch_size: 64
  jit_compile: True
  checkpoint_freq: 2
```

The [presets](presets.md#models-hyperparameters) page describes `preset` and `kwargs`.

## All options

```yaml
seed: 42        # random seed: model initialization, data split; the fit folder is seed/<seed>/
cutoff: 6       # cutoff radius (Å)

# cutoff_dict: {Mo: 4, MoNb: 3, W: 5, Ta*: 7}   # cutoff per bond type (GRACE presets)
# cutoff_dict: DEFAULT_CUTOFF_1L   # or a built-in table: DEFAULT_CUTOFF_1L, DEFAULT_CUTOFF_2L,
#                                  # CUTOFF_2L, BIG_CUTOFF_2L (per-element radii; pair = mean of
#                                  # the two; other elements use `cutoff`)

######################
##       DATA       ##
######################
data:
  filename: /path/to/train.pkl.gz       # pkl.gz, extxyz, or a list of files
  # train_size: 0.5                     # use part of the training data: < 1 fraction, >= 1 number
  test_filename: /path/to/test.pkl.gz
  # test_size: 0.05                     # see "Train and test split" below
  reference_energy: 0                   # free-atom energies subtracted from `energy`
  # reference_energy: auto              # least-squares per-element E0 from the train set
  # reference_energy: {Al: -1.23, Li: -3.56}
  # save_dataset: True                  # save the processed train/test sets into seed/<seed>/
  # stress_units: eV/A3                 # eV/A3 (default), GPa, kbar or -kbar
  # max_workers: 6                      # parallel data builder

  # pipeline: in_memory                 # or streaming, see "Large datasets" below

######################
##    POTENTIAL     ##
######################
potential:
  # elements: [C, H, O]                 # default: all elements of the data

  ## Option 1. Preset
  preset: GRACE_1LAYER_latest           # FS, GRACE_1LAYER_latest, GRACE_2LAYER_latest
  # kwargs: {n_rad_max: 32}             # preset parameters

  ## Option 2. Custom model in a Python file (advanced)
  # custom: model.custom_model          # function custom_model from model.py

  ## Option 3. Model from model.yaml (e.g. to continue a fit)
  # filename: model.yaml                # architecture only, no weights
  # checkpoint_name: /path/to/checkpoint.index

  ## Option 4. Fine-tune a foundation model (see Foundation models)
  # finetune_foundation_model: GRACE-1L-OMAT-medium-ft-AM
  # reduce_elements: False              # True: keep only the elements of the data

  ## Other parameters
  # param_dtype: float32                # float32 (default) or float64 (float_dtype: deprecated alias)
  # shift: False                        # True: least-squares per-element energy shift (from scratch)
  #                                     # auto: per-element shift from foundation-model predictions
  #                                     #       vs your DFT data (fine-tuning)
  # scale: False                        # True: scale by the force RMSE, or a float (e.g. 2.0,
  #                                     # not 2: an integer is read as True)
  # avg_n_neigh: 40                     # average number of neighbours; default: from the data
  # dense_nbr: False                    # dense neighbour aggregation, see "Dense batching" below

######################
##       FIT        ##
######################
fit:
  loss: {
    energy: { type: huber, weight: 16, delta: 0.01 },   # or { type: square, weight: 1 }
    forces: { type: huber, weight: 32, delta: 0.01 },
    # stress: { type: huber, weight: 32, delta: 0.01 },
    # l2_reg: 1.0e-8                                    # L2 regularization of the weights
    # switch: { after_iter: 0.75, energy: { weight: 5.0 }, forces: { weight: 2.0 },
    #           learning_rate: 0.001 }                  # see "Loss" below
  }

  maxiter: auto                 # number of epochs; auto: ~50k updates from scratch,
                                # ~10k for fine-tuning (BFGS: ~500 / ~100 epochs), 10-5000 epochs
  # target_total_updates: 50000 # number of updates instead; overrides maxiter

  optimizer: Adam               # Adam, or L-BFGS-B / BFGS (scipy, CPU; up to ~10k parameters)
  opt_params: { learning_rate: 0.008, use_ema: True, ema_momentum: 0.99, weight_decay: 1.e-20,
                clipnorm: 1.0 }  # passed to the Keras Adam optimizer (amsgrad, beta_1,
                                 # clipvalue, global_clipnorm, ... work too)
                                 # default without opt_params: {learning_rate: 0.01, amsgrad: True}
  # opt_params: { maxcor: 100, maxls: 20 }   # for L-BFGS-B
  scheduler: cosine_decay       # see "Learning-rate schedulers" below
  scheduler_params: { minimum_learning_rate: 1.0e-4 }
  # reset_optimizer: False      # reset the optimizer state after loading a checkpoint
  # reset_epoch_and_step: False # reset the epoch and step counters stored in the checkpoint

  batch_size: 32                # default 8; important for Adam, ignored by (L-)BFGS
  test_batch_size: 200          # default 1: set it

  # weighting: { type: energy_based }   # see "Weighting" below
  # compute_convex_hull: False  # distance to the convex hull of train+test (needed by weighting)
  # eval_init_stats: False      # metrics before the first epoch

  jit_compile: True             # XLA compilation, keep it on
  train_max_n_buckets: auto     # number of padded batch shapes; auto: 1-32, chosen to keep the
  test_max_n_buckets: auto      #   neighbour padding below auto_bucket_max_padding
  # auto_bucket_max_padding: 0.3   # 0.3 = 30 % (0.15 with dense_nbr)

  checkpoint_freq: 10           # regular checkpoint every N epochs (the best-test-loss one is extra)
  # save_all_regular_checkpoints: False
  # checkpoint_freq_steps: 500  # also every N steps within an epoch (Adam), see below
  # progressbar: False
  # train_shuffle: False        # shuffle batches every epoch (True by default in streaming)
  # strategy: mirrored          # several GPUs on one machine, same as `gracemaker -m`

  # trainable_variable_names: ["rho/reducing_"]   # train only these weights (frozen-weights
  #                                               # fine-tuning, see Foundation models)
  # normalize_weights: False            # normalize per-structure weights to sum to one
  # normalize_force_per_structure: False  # divide force weights by the number of atoms
```

## Train and test split

| Keys | Test set |
|---|---|
| `test_filename` | that file |
| `test_filename` + `test_size: N` (N > 1) | N random structures of that file |
| `test_size: f` (f < 1), no `test_filename` | a fraction f of the training data, removed from it |
| neither | no test set |

`train_size` (fraction below 1, number from 1) reduces the training data first. With
`test_filename`, a fractional `test_size` is ignored.

## Loss

Loss types per component:

| Component | Types |
|---|---|
| `energy` (per atom) | `square`, `huber` (`delta`), `mae`, `piecewise_linear` |
| `forces` | `square`, `huber` (`delta`), `piecewise_linear` |
| `stress` | `square`, `huber` (`delta`), `mae`, `piecewise_linear` |
| `virial` | `square`, `huber` (`delta`) |

`switch` changes the loss weights once, after `after_iter` epochs (an integer) or a fraction of
`maxiter` (`0.75`, or `auto` = 0.75). With a test set, the fit then continues from the best
checkpoint so far (saved as `checkpoint.stage1.best_test_loss`). The new learning rate is `learning_rate`, or the
current one times `learning_rate_reduction_factor`. `switch` needs `scheduler: reduce_on_plateau`;
`gracemaker -t` sets it.

## Learning-rate schedulers

`scheduler` and its `scheduler_params`:

| `scheduler` | `scheduler_params` (default) |
|---|---|
| `cosine_decay`, `linear_decay`, `exponential_decay` | `minimum_learning_rate` (1e-4), `warmup_epochs` (0), `cold_learning_rate` (1e-7); `exponential_decay` also `staircase` (False) |
| `reduce_on_plateau` | `patience` (10), `reduction_factor` (0.8), `minimum_learning_rate` (1e-4), `stop_at_min` (False), `cooldown` (0), `resume_lr` (True), `monitor` (`test_loss`) |

* The decays go from `opt_params: learning_rate` to `minimum_learning_rate` over `maxiter`
  epochs. With `warmup_epochs` > 0 (can be < 1), they first rise from `cold_learning_rate` to
  `learning_rate`.
* `stop_at_min: True` stops the fit when `reduce_on_plateau` reaches `minimum_learning_rate`. It is
  the only early stopping.
* Unknown keys are ignored with a warning in the log, e.g. `minimal_learning_rate` for
  `minimum_learning_rate`.

??? info "Legacy `learning_rate_reduction` (deprecated)"
    `learning_rate_reduction: {patience: 5, factor: 0.98, min: 5.0e-4, stop_at_min: True,
    resume_lr: True}` still works (with a warning) as `reduce_on_plateau`. It cannot be combined
    with `scheduler`.

## Weighting

By default all structures have the same weight: 1 for the energy and 1 for each force
component. Let `fit: weighting` compute weights, or set your own in the training data.

### Energy-based weighting

Gives low-energy structures more weight:

```yaml
fit:
  compute_convex_hull: True     # needed for energy: convex_hull (the default)
  weighting: {type: energy_based,
    nfit: 10000,                # number of structures to select (default: all)
    DElow: 1.0,                 # eV/atom; structures in (E_min, E_min + DElow) are all kept
                                #   (if fewer than nfit)
    DEup: 10.0,                 # eV/atom; structures above E_min + DEup are dropped
    DFup: 50.0,                 # eV/Å; structures with a larger force are dropped (default: none)
    DE: 1.0,                    # energy shift of the weights
    DF: 1.0,                    # force shift of the weights
    wlow: 0.75,                 # 0 < wlow < 1: total weight of the low-energy range (default: none)
    energy: convex_hull,        # E_min from: convex_hull, cohesive or zero_formation_energy
    reftype: all,               # all, bulk or cluster
    seed: 42}
```

Without `compute_convex_hull: True`, use `energy: cohesive` or `zero_formation_energy`.

### Custom weights

Add these columns to the training (and test) DataFrame:

| Column | Value per structure | Default |
|---|---|---|
| `energy_weight` | a number: weight of the energy | 1 |
| `force_weight` | an array with one number per atom: weight of each atom's force | 1 per atom |
| `virial_weight` | a number: weight of the stress (only when `stress` is in the loss) | 1 periodic, 0 non-periodic |

```python
import numpy as np
import pandas as pd

df = pd.read_pickle("train.pkl.gz")
non_periodic = df["ase_atoms"].map(lambda at: not at.pbc.all())   # e.g. molecules, clusters
df["energy_weight"] = np.where(non_periodic, 5.0, 1.0)
df["force_weight"] = [np.full(len(at), w) for at, w in zip(df["ase_atoms"], df["energy_weight"])]
df.to_pickle("train_weighted.pkl.gz")
```

* `force_weight` must be one number **per atom**; a single number per structure stops the fit
  with a shape error.
* The weights multiply the `weight` of the loss component. They are used as given; with
  `normalize_weights: True` they are divided by their sum.
* `test_size` keeps the weights of the structures it moves to the test set; a separate
  `test_filename` needs its own columns (default 1).
* Weights in the `info` of an extxyz file are not read: convert it to a DataFrame
  ([`extxyz2df`](utilities.md#extxyz2df)) and add the columns.
* `fit: weighting` (above) replaces `energy_weight` and `force_weight`.

## Large datasets

There are three ways to feed the data to the fit:

| Mode | How | Use it for |
|---|---|---|
| in memory (default) | all batches and neighbour lists in RAM | small and medium datasets |
| streaming | neighbour lists computed on the fly, one batch in memory | datasets that do not fit in RAM |
| precomputed shards | `grace_preprocess` writes padded batches to disk once; the fit reads them | very large datasets, many GPUs or nodes |

### Streaming data pipeline

`data: pipeline: streaming` computes the neighbour lists on the fly and keeps one batch in
memory: for datasets that do not fit in RAM.
`normalize_weights` has no effect in streaming mode.

```yaml
data:
  pipeline: streaming
  streaming:
    target_metric: 3000          # total metric per batch (e.g. total neighbours)
    metric_strategy: neighbours  # neighbours, atoms or structures
    num_bins: 10                 # number of parallel accumulation bins
    sorting_buffer_size: 32      # sort structures every N arrivals (load balancing)
    growth_fraction: 0.1         # bucket growth on overflow: < 1 relative, >= 1 absolute (+N)
                                 # per axis: {atoms: 0.1, neighbours: 0.05, structures: 5}
    outlier_strategy: expand     # expand (grow the bucket), warn_skip (drop) or error
    verbose: false               # log bucket discovery
    prefetch_queue_size: 0       # > 0: threaded prefetching
```

### Precomputed shards (`grace_preprocess`) {#precomputed-shards}

[`grace_preprocess`](utilities.md#grace_preprocess) computes the neighbour lists, groups the
structures into padded batches and saves them as TensorFlow datasets, split into N shards. It
runs on CPU, in four stages; stages 1-3 run for each shard separately (in parallel, e.g. as a
job array), stage 4 once:

```bash
N=8   # number of shards
ARGS="train.pkl.gz -o tf_dataset --total-task-num $N -c 6 -e MP -s neighbours -b 3000 -bu 4"
for i in $(seq 0 $((N-1))); do grace_preprocess $ARGS --task-id $i --stage-1 & done; wait  # per-structure data
for i in $(seq 0 $((N-1))); do grace_preprocess $ARGS --task-id $i --stage-2 & done; wait  # padding bounds
for i in $(seq 0 $((N-1))); do grace_preprocess $ARGS --task-id $i --stage-3 & done; wait  # padded batches
grace_preprocess $ARGS --task-id 0 --stage-4                                                # statistics
```

* `-b 3000 -s neighbours`: each batch holds about 3000 neighbours (`-s`: `structures`
  (default), `atoms` or `neighbours`); `-bu`: number of padding buckets.
* `-c` and `-cd` (cutoff, `cutoff_dict`) must match `input.yaml`; add `--is-fit-stress` to fit
  stress. `-e`: the element list, or `MP`, `Alexandria`, `ALL`.
* Input: one or more `pkl.gz` files; with `--sharded-input`, each task reads whole files
  instead of every N-th row.
* `--remove_stage1` frees the disk space of stage 1 during stage 3; `--rerun` overwrites.
* The full script: [tests/data_distrib/compute_distributed_data.sh](https://github.com/ICAMS/grace-tensorpotential/blob/master/tests/data_distrib/compute_distributed_data.sh).

Then point the fit to the shards:

```yaml
data:
  distributed: True
  path: tf_dataset/stage3        # the shards (shard_*-of-*)
  # test_path: test/stage3       # shards of a separate test set, made the same way
  # test_shards: 1               # or: the last N shards of path as test set (< 1: fraction)
  # train_size: 1000             # use only N batches (here sizes count batches)
  # test_size: 50
```

The element list, the data scale and the average number of neighbours come from the statistics
of stage 4. `batch_size` and the bucket options of `fit:` do not apply: the batches are already
made. On several GPUs each GPU reads its own shards, so use at least as many shards as GPUs (in
all nodes).

### Checkpoints within an epoch

`checkpoint_freq_steps: N` (Adam) saves a checkpoint every N training steps, so a crash loses
at most N steps. It overwrites the default `checkpoint` file. On a `gracemaker -rl` restart the
fit rewinds to the start of the interrupted epoch and skips the batches already trained (the
gradients are in the restored optimizer state). `--no-intra-epoch-redo` turns the rewind off.

## Several GPUs and nodes

**One machine.** `gracemaker -m` (or `fit: strategy: mirrored`) fits on all visible GPUs in
parallel. `batch_size` and `test_batch_size` are per GPU: the global batch is `batch_size` times
the number of GPUs. It works with all three data modes.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 gracemaker input.yaml -m
```

**Several nodes.** Use [precomputed shards](#precomputed-shards) (at least as many as GPUs in
total) and start one `gracemaker` per node with `TF_CONFIG` set (TensorFlow
MultiWorkerMirroredStrategy). A complete SLURM script:
[examples/slurm_multinode/submit.sbatch](https://github.com/ICAMS/grace-tensorpotential/blob/master/examples/slurm_multinode/submit.sbatch).

!!! tip "A multi-GPU fit hangs at the start"
    On machines whose GPUs talk only over PCIe (no NVLink), NCCL can hang. Try
    `export NCCL_P2P_DISABLE=1`.

## Dense neighbour batching (GPU speed)

`potential: dense_nbr: True` sums neighbours with a batched matmul over a per-atom-uniform
`[n_atoms * max_neigh]` layout instead of the per-bond `segment_sum`. It is faster on GPU
(especially for deeper models and high coordination) but computes the padded neighbour slots,
so it pays off only when the padding is low. Results are bit-exact with the default.

In dense mode `auto_bucket_max_padding` is the single target for the net neighbour padding
(default 0.15). An adaptive search splits it between the `max_neigh` width buckets and the
per-atom buckets, with the fewest distinct buffer shapes.

```yaml
fit:
  # dense_max_shapes: 64         # hard cap on distinct (n_atoms, max_neigh) shapes (= XLA
  #                              # compilations); above it the padding rises, with a warning
  # dense_slot_budget: auto      # max n_atoms*max_neigh per batch; auto = batch_size*mean(nat)*mean(max_neigh)
  # dense_n_neigh_buckets: auto  # fixed number of max_neigh buckets (instead of the search)
  # dense_max_neigh_cap:         # drop structures with a larger per-atom max_neigh (off)
```

The log reports, per dataset: `dense bucketing: N batches, S/cap distinct shapes, net
neighbour padding X%`.

## Custom training

To fit something other than energy, forces and stress, or with your own model, combine these
keys. The example fits a second-rank tensor (the complete version:
[Fitting generic tensors](gen_tensor.md)):

```yaml
potential:
  preset: TENSOR_2L                   # a model with the extra output, or your own:
  # custom: model.custom_model        #   function custom_model in ./model.py
  kwargs: {tensor_components: [0, 2]} # passed to the preset or your function

data:
  extra_components: {                 # extra data builders: read more columns from the data
    ReferenceTensorDataBuilder: {tensor_rank: 2, per_structure: False},
  }

fit:
  train_function: ComputeBatchEFTensor     # what the model computes during training
  compute_function: ComputeBatchEFTensor   # ... and in the exported model
  compute_function_config: {tensor_components: [0, 2]}
  loss: {
    energy: { type: huber, weight: 1, delta: 0.01 },
    extra_components: {                    # extra loss terms
      WeightedTensorLoss: { weight: 10, type: huber, delta: 0.1 },
    },
  }
```

| Key | What | Where it is found |
|---|---|---|
| `potential: custom` | function that builds the model; called with `element_map`, `rcut`, `cutoff_dict`, `avg_n_neigh`, `constant_out_shift`, `constant_out_scale`, `atomic_shift_map` and `kwargs`, returns the instructions (like the presets in `presets.py`) | your file: `model.custom_model` = function `custom_model` in `./model.py` |
| `data: extra_components` | data builders, each called with its settings and `float_dtype` | `tensorpotential.extra.extra_data_builders` |
| `fit: loss: extra_components` | loss terms, each called with its `weight` and the other settings | `tensorpotential.extra.extra_losses` |
| `fit: train_function`, `compute_function` | built with `compute_function_config` | `tensorpotential.tpmodel`, then `tensorpotential.extra.model_computes` |

* Defaults: `train_function: ComputeBatchEnergyAndForces` (`ComputeBatchEnergyForcesVirials` when
  stress or virial is in the loss), `compute_function: ComputeStructureEnergyAndForcesAndVirial`.
* Only the model can come from your own file. A new data builder, loss term or compute function
  must be added to one of the modules in the table (for example, imported into
  `tensorpotential/extra/extra_losses.py`).

## Rarely needed keys

* `fit: loss_norm_by_batch_size: False`: normalize the total loss by the global batch size
  (backward compatibility).
* Legacy loss weights without `loss:`: `energy_loss_weight` (1.0), `forces_loss_weight` (100.0).
