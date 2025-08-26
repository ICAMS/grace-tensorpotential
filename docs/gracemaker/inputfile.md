# Input file `input.yaml`

Example of `input.yaml` can be found in `examples/grace` folder

```{ .yaml }
seed: 42
cutoff: 6

# cutoff_dict: {Mo: 4, MoNb: 3, W: 5, Ta*: 7 } ## Defining cutoff for each bond type separately, used by certain models
## possible defaults: DEFAULT_CUTOFF_1L, DEFAULT_CUTOFF_2L

######################
##       DATA       ##
######################
data:
  filename: /path/to/train.pckl.gzip
  #  train_size: 100
  test_filename: /path/to/test.pckl.gzip
  #  test_size: 0.05 # 
  reference_energy: 0 
  # reference_energy: {Al: -1.23, Li: -3.56}
  # save_dataset: False # default is True
  # stress_units: eV/A3 # eV/A3 (default) or GPa or kbar or -kbar
  # max_workers: 6 # for parallel data builder

######################
##    POTENTIAL     ##
######################
potential:
  #  elements: [C, H, O] # If not provided - determined automatically from data


  ## Option 1. Presets
  preset: GRACE_1LAYER # FS, GRACE_1LAYER, GRACE_2LAYER
  # kwargs: {n_rad_max: 16}  # kw-arguments that will be passed to preset or custom model

  ## Option 2. Custom model in python file (advanced)
  # custom: model.custom_model # custom model from model.py file, function custom_model



  ## Option 3. Model in model.yaml
  # filename: model.yaml # configuration (WITHOUT weights!) of the model
  # checkpoint_name: /path/to/checkpoint.index  #  path to checkpoint index file  


  ## Option 4. Fine-tune foundation model
  # finetune_foundation_model: GRACE-1L-OAM
  # reduce_elements: True #  default - False, reduce elements to those provided in dataset
  
  ## Option 4a. LORA (experimental, not supported)
  # lora: {all: {rank: 16, alpha: 1}, Z: {rank: 8, alpha: 1}, I: {rank: 4, alpha: 1, keep_dims: 1} }
  ## reduce_lora: True # reduce LORA model
  
  ## Other parameters: 
  # shift: False # True/False - automatic shift by energy
  # scale: False # False/True or float  - automatic scale data by force RMSE
  # avg_n_neigh: 40 # Average number of neighbours. By default - automatically determined
  # float_dtype: float64 # float64, float32
  # custom ZBL core repulsion for model:  kwargs: {zbl_cutoff: {Mo: 1, MoNb: 2, W: 1, Ta*: 3 }}

######################
##       FIT        ##
######################
fit:
  # Explicit specification of the train and compute functions for the model with optional parameters
  #compute_function: ComputeStructureEnergyAndForcesAndVirial
  #train_function: ComputeBatchEnergyAndForces
  #compute_function_config: {}
  loss: {
    energy: { type: huber, weight: 16,  delta: 0.01}, # or { type: square, weight: 1} 
    forces: { type: huber, weight: 32., delta: 0.01}, 
  # stress: { type: huber, weight: 32., delta: 0.01},   #

  ## Change weights for energy/forces/stress loss components
  ## and learning_rate after "after_iter" epochs
  #    switch: { after_iter: 350,
  #              energy: { weight: 5.0 }, 
  #              forces: { weight: 2.0 }, 
  #              stress: {weight: 0.001},
  #               learning_rate: 0.001}    
  }

  
  maxiter: 500 # Max number of optimization epochs
  optimizer: Adam
    # Optimization with Adam: good for large number of parameters, first-order method
  opt_params: { learning_rate: 0.01, use_ema: True, ema_momentum: 0.99,  weight_decay: 1.e-20, clipnorm: 1.0}
  # reset_optimizer: True  # reset optimizer state, after being loaded from checkpoint
  # reset_epoch_and_step: False # reset epoch and step internal counters (stored in checkpoint)
  scheduler: cosine_decay # scheduler for learning-rate reduction during training 
    # available options are: reduce_on_plateau, cosine_decay, linear_decay, exponential_decay
  scheduler_params: {"warmup_epochs": 2, "cold_learning_rate": 0.1, "minimal_learning_rate": 0.05}
    # If :warmup_epochs: > 0, begin optimization with :cold_learning_rate: and reach :opt_params::learning_rate:
    # within :warmup_epochs: (can be < 1). Else, begin optimization with :opt_params::learning_rate: and decay down to
    # minimum_learning_rate within :maxiter: epochs
  # legacy format for reduce_on_plateau lr scheduler 
  # learning_rate_reduction: { patience: 5, factor: 0.98, min: 5.0e-4, stop_at_min: True, resume_lr: True, }
  
  
  ## Optimization with BFGS: good for SMALL number of parameters (up to 10k), "second"-order method.
  ## scipy optimizer on CPU will be used
  #  optimizer: L-BFGS-B # 'L-BFGS-B' for memory limited or 'BFGS' for full method
  #  opt_params: {maxcor: 100, maxls: 20 }  # options for L-BFGS-B

  
  batch_size: 32 # Important hyperparameter for Adam and irrelevant (but must be) for L-BFGS-B/BFGS
  test_batch_size: 200 # test batch size (optional)


  ## Uniform weighting is used if not specified
  ## Energy based weighting can be used as:  
  #  weighting: {type: energy_based}  # the simplest default

  # compute_convex_hull: False ## for train+test dataset compute convex hull and distance to it 
  # eval_init_stats: False  ## Compute train/test metrics before start fitting 

  jit_compile: True  # for XLA compilation, must be used in almost all cases
  ## To use jit_compile efficiently, data must be padded.
  ## Bucket is a group of batches padded to the same shape
  train_max_n_buckets: 10  ## max number of buckets in train set  
  test_max_n_buckets: 3  ## same for test

  checkpoint_freq: 10 # frequency for **REGULAR** checkpoints. 
  # save_all_regular_checkpoints: False # to store ALL regular checkpoints
  # progressbar: True # show batch-evaluation progress bar
  # train_shuffle: True # shuffle train batches on every epoch
  # strategy: mirrored # or -m flag # for parallel multi-GPU parameterization

  # trainable_variable_names: ["rho/reducing_", "Z/ChemicalEmbedding"] ### specify trainable variables name pattern
  
  ## technical parameters for normalization
  #  loss_norm_by_batch_size: False # normalization of total loss by global batch size (for backward compat)
  #  normalize_weights: True ## norm per-sample weights to sum-up to one
  #  normalize_force_per_structure: True ## force-weights is divided by number of atoms
```

This is complete list of parameters. For the most of practical purposes
it is sufficient to generate input file with `gracemaker -t` utility.

Detailed weighting option:

```{ .yaml }

potential:
  weighting: {type: energy_based, 
    ## number of structures to randomly select from the initial dataset
    nfit: 10000,
    ## only the structures with energy up to E_min + DEup will be selected
    DEup: 10.0,  ## eV, upper energy range (E_min + DElow, E_min + DEup)        
    ## only the structures with maximal force on atom  up to DFup will be selected
    DFup: 50.0, ## eV/A
    ## lower energy range (E_min, E_min + DElow)
    DElow: 1.0,  ## eV
    ## delta_E  shift for weights, see paper
    DE: 1.0,
    ## delta_F  shift for weights, see paper
    DF: 1.0,
    ## 0<wlow<1 or None: if provided, the renormalization weights of the structures on lower energy range (see DElow)
    wlow: 0.75,
    ##  "convex_hull" or "cohesive" or "zero_formation_energy": method to compute the E_min
    energy: convex_hull,
    ## structures types: all (default), bulk or cluster
    reftype: all,
    ## random number seed
    seed: 42}
```
