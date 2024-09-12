# Gracemaker

```
usage: gracemaker [-h] [-l LOG] [-m] [-rl] [-r] [-rs RESTART_SUFFIX] [-p POTENTIAL] [-s] [-sf] [-e] [-nj] [--seed SEED] [-cm] [input]

Fitting utility for (graph) atomic cluster expansion potentials.

positional arguments:
  input                 input YAML file, default: input.yaml

optional arguments:
  -h, --help            show this help message and exit
  -l LOG, --log LOG     log filename, default: log.txt
  -m, --multigpu        Single host/multi GPU distributed fit
  -rl, --restart-latest
                        Restart from latest checkpoint
  -r, --restart-best-test
                        Restart from latest best test checkpoint
  -rs RESTART_SUFFIX, --restart-suffix RESTART_SUFFIX
                        Suffix of checkpoint to restart from, i.e. .epoch_10
  -p POTENTIAL, --potential POTENTIAL
                        Potential configuration to load
  -s, --save-model      Export model as TF saved model
  -sf, --save--fs       Export FS model as yaml to be loaded in CPP
  -e, --eager           Eager graph execution
  -nj, --no-jit         No JIT
  --seed SEED           Random seed (will overwrite value from input.yaml
  -cm, --check-model    Check model consistency, without performing fit

```

# Quick start

## Create `input.yaml` file

Example of `input.yaml` can be found in `examples/grace` folder

```yaml
seed: 42
cutoff: 6

data:
  filename: "/path/to/train.pckl.gzip"
  #  train_size: 100
  test_filename: "/path/to/test.pckl.gzip"
  #  test_size: 0.05 # 50
  # reference_energy: 0 
  # reference_energy: {Al: -1.23, Li: -3.56}
  # save_dataset: False # default is True
  # stress_units: eV/A3 # eV/A3 (default) or GPa or kbar or -kbar

potential:
  #  elements: ["C", "H", "O"] # If not provided - determined automatically from data
  preset: "MOD_ACE" # MOD_ACE, LINEAR, FS, MLP, GRACE, GRACE_MD17, GRACE_CE, GRACE_1LAYER, GRACE_N
#  custom: model.custom_model # custom model from model.py file, function custom_model
#  kwargs: {n_rad_max: 16}  # kw-arguments that will be passed to preset or custom function

#  filename: model.yaml # configuration (WITHOUT weights!) of the model
#  shift: True
#  scale: True
#  float_dtype: float64 # float64, float32, tfloat32

fit:
  #  strategy: mirrored # or -m flag

  loss: {
    energy: { weight: 1, }, # or { type: huber, weight: 1, delta: 0.01 } #   normalize_by_samples: True
    forces: { weight: 5. }, # or type: square and no delta #   normalize_by_samples: True
#    stress: { weight: 0.001},
    
#    switch: { after_iter: 1,
#              energy: { weight: 10.0 }, 
#              forces: { weight: 100.0 }, 
#              stress: {weight: 0.001},
#               learning_rate: 0.001}
    
   # l2_reg: 1.0e-10
  }
  
  # normalize_weights: True ## norm per-sample weights to sum-up to one
  # normalize_force_per_structure: True ## force-weights is divided by number of atoms
  
## Adaptive loss change, not efficient, will be removed
#  adaptive_loss_weights: {beta: 1.0,  freq: 5, factor: 1.0 }
  
## energy based weighting  
#  weighting: {type: energy_based}  # the simplest default
  
## OR more detailed config:
 
#  weighting: {type: energy_based, 
#    ## number of structures to randomly select from the initial dataset
#    nfit: 10000,
#    ## only the structures with energy up to E_min + DEup will be selected
#    DEup: 10.0,  ## eV, upper energy range (E_min + DElow, E_min + DEup)        
#    ## only the structures with maximal force on atom  up to DFup will be selected
#    DFup: 50.0, ## eV/A
#    ## lower energy range (E_min, E_min + DElow)
#    DElow: 1.0,  ## eV
#    ## delta_E  shift for weights, see paper
#    DE: 1.0,
#    ## delta_F  shift for weights, see paper
#    DF: 1.0,
#    ## 0<wlow<1 or None: if provided, the renormalization weights of the structures on lower energy range (see DElow)
#    wlow: 0.75,
#    ##  "convex_hull" or "cohesive" or "zero_formation_energy": method to compute the E_min
#    energy: convex_hull,
#    ## structures types: all (default), bulk or cluster
#    reftype: all,
#    ## random number seed
#    seed: 42}
  
  # OLD-STYLE
  #energy_loss_weight: 1
  #forces_loss_weight: 100.0

  maxiter: 1500 # Number of iterations

  # OLD-STYLE
  #  loss_weights_switch: 100
  #  energy_loss_weight_2: 100
  #  forces_loss_weight_2: 10.0

  optimizer: Adam
  opt_params: { learning_rate: 0.01, amsgrad: True, use_ema: True }
  
  # for learning-rate reduction
  learning_rate_reduction: { patience: 5, factor: 0.8, min: 1.0e-3, stop_at_min: False, resume_lr: True, } #  loss_explosion_threshold: 2

  #  optimizer: L-BFGS-B
  #  opt_params: { "maxcor": 100, "maxls": 20 }

  # compute_convex_hull: False
  batch_size: 10 # Important hyperparameter for Adam and irrelevant (but must be) for L-BFGS-B
  test_batch_size: 20 # test batch size (optional)
  jit_compile: True

  # To use jit_compile efficiently, data must be padded.
  # Bucket is group of batches padded to the same shape
  train_max_n_buckets: 5 # max number of buckets (group of batches of same shape) in train set  
  test_max_n_buckets: 1 # same for test

  checkpoint_freq: 10 # frequency for **REGULAR** checkpoints. 
# save_all_regular_checkpoints: True # to store ALL regular checkpoints
#  progressbar: True # show batch-evaluation progress bar
#  train_shuffle: True # shuffle train batches on every epoch

# loss_norm_by_batch_size: False # normalization of total loss by global batch size (for backward compat)

```

## (TODO) Model presets and configuration

## (TODO) Checkpointing

Use `checkpoint_freq` to specify how frequently save regular checkpoints (only last state will be saved into
checkpoint).
If you want to keep all regular checkpoints, then add flag `save_all_regular_checkpoints: True`

## (TODO) Single-GPU / Multi-GPU / Multi-worker modes