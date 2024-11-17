
# Input file `input.yaml`

Example of `input.yaml` can be found in `examples/grace` folder

```{ .yaml }
seed: 42
cutoff: 6

# cutoff_dict: {Mo: 4, MoNb: 3, W: 5, Ta*: 7 } ## Defining cutoff for each bond type separately, used by certain models
## possible defaults: DEFAULT_CUTOFF_1L, DEFAULT_CUTOFF_2L
# use_per_specie_n_nei: False

data:
  filename: "/path/to/train.pckl.gzip"
  #  train_size: 100
  test_filename: "/path/to/test.pckl.gzip"
  #  test_size: 0.05 # 
  reference_energy: 0 
  # reference_energy: {Al: -1.23, Li: -3.56}
  # save_dataset: False # default is True
  # stress_units: eV/A3 # eV/A3 (default) or GPa or kbar or -kbar
  # max_workers: 6 # for parallel data builder
  
potential:
  #  elements: ["C", "H", "O"] # If not provided - determined automatically from data
  preset: "GRACE_1LAYER" # FS, GRACE_1LAYER, GRACE_2LAYER
#  custom: model.custom_model # custom model from model.py file, function custom_model
#  kwargs: {n_rad_max: 16}  # kw-arguments that will be passed to preset or custom model

#  custom ZBL core repulsion for model:  kwargs: {zbl_cutoff: {Mo: 1, MoNb: 2, W: 1, Ta*: 3 }}

#  filename: model.yaml # configuration (WITHOUT weights!) of the model
#  shift: False # True/False
#  scale: False # False/True or float 
#  float_dtype: float64 # float64, float32

fit:
  #  strategy: mirrored # or -m flag

  loss: {
    energy: { type: square, weight: 1, }, # or { type: huber, weight: 1, delta: 0.01 } 
    forces: { type: huber, weight: 5., delta: 0.1 }, # or type: square and no delta 
#    stress: { type: square, weight: 1.},
    
#    switch: { after_iter: 10,
#              energy: { weight: 5.0 }, 
#              forces: { weight: 2.0 }, 
#              stress: {weight: 0.001},
#               learning_rate: 0.001}
    
   # l2_reg: 1.0e-10
  }
 
## Uniform weighting is used if not specified
## Energy based weighting can be used as:  
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

  maxiter: 500 # Max number of iterations


  optimizer: Adam
  opt_params: { learning_rate: 0.01, amsgrad: True, use_ema: True, ema_momentum: 0.99,  weight_decay: 1.e-20, clipvalue: 1.0}
  
  # for learning-rate reduction
  learning_rate_reduction: { patience: 5, factor: 0.98, min: 5.0e-4, stop_at_min: True, resume_lr: True, } 

  ## Other optimizer option
  #  optimizer: L-BFGS-B
  #  opt_params: { "maxcor": 100, "maxls": 20 }

  # compute_convex_hull: False
  batch_size: 32 # Important hyperparameter for Adam and irrelevant (but must be) for L-BFGS-B
  test_batch_size: 200 # test batch size (optional)
  jit_compile: True

  # eval_init_stats: False  ## Compute train metrics before fitting 
  
  ## To use jit_compile efficiently, data must be padded.
  ## Bucket is a group of batches padded to the same shape
  train_max_n_buckets: 10  ## max number of buckets in train set  
  test_max_n_buckets: 3  ## same for test

  checkpoint_freq: 10 # frequency for **REGULAR** checkpoints. 
#  save_all_regular_checkpoints: False # to store ALL regular checkpoints
#  progressbar: True # show batch-evaluation progress bar
#  train_shuffle: True # shuffle train batches on every epoch

## technical parameters for normalization
#  loss_norm_by_batch_size: False # normalization of total loss by global batch size (for backward compat)
#  normalize_weights: True ## norm per-sample weights to sum-up to one
#  normalize_force_per_structure: True ## force-weights is divided by number of atoms
```

This is complete list of parameters. For the most of practical purposes
it is sufficient to generate input file with `gracemaker -t` utility.   
