# Input file `input.yaml`

Example of `input.yaml` can be found in `examples/grace` folder

```{ .yaml }
seed: 42
cutoff: 6

# cutoff_dict: {Mo: 4, MoNb: 3, W: 5, Ta*: 7 } ## Defining cutoff for each bond type separately, used by certain models
## possible defaults: DEFAULT_CUTOFF_1L, DEFAULT_CUTOFF_2L, CUTOFF_2L

######################
##       DATA       ##
######################
data:
  filename: /path/to/train.pckl.gzip
  # filename: /path/to/train.extxyz
  #  train_size: 100
  test_filename: /path/to/test.pckl.gzip
  #  test_size: 0.05 # 
  reference_energy: 0 
  # reference_energy: auto          # auto: least-squares fit of per-element E0 from train set
  # reference_energy: {Al: -1.23, Li: -3.56}
  # save_dataset: False # default is True
  # stress_units: eV/A3 # eV/A3 (default) or GPa or kbar or -kbar
  # max_workers: 6 # for parallel data builder

  ## Data pipeline mode: "in_memory" (default) or "streaming"
  ## "streaming" computes neighbour lists on-the-fly, keeping only one batch in memory.
  ## Recommended for large datasets that do not fit in RAM.
  ## NOTE: weight normalization is not supported in streaming mode (normalize_weights must be False).
  # pipeline: streaming
  # streaming:
  #   target_metric: 3000          # total metric value per batch (e.g. total neighbours)
  #   metric_strategy: neighbours  # grouping metric: "neighbours", "atoms", or "structures"
  #   num_bins: 10                 # number of parallel accumulation bins
  #   sorting_buffer_size: 32      # sort structures every N arrivals for load balancing
  #   growth_fraction: 0.1         # bucket growth on overflow: <1 = relative (10%), >=1 = absolute (+N slots)
  #                                # per-axis dict also supported: {atom: 0.1, bond: 0.05, structure: 5}
  #   outlier_strategy: expand     # "expand" (grow bucket), "warn_skip" (drop), or "error" (raise)
  #   verbose: false               # log bucket discovery events
  #   prefetch_queue_size: 0       # set to >0 to thread prefetcing

  ## Extra input/reference DataBuilder/s required for model
  # extra_components: {
  #   MagMomDataBuilder: {},
  # }

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
  # shift: False # True/False/"auto" - automatic shift by energy
  #   True  — least-squares per-element shift on training energies (for training from scratch)
  #   "auto" — when finetuning a foundation model, computes per-element shifts by comparing
  #            FM predictions with reference DFT data and injects them into the model
  # scale: False # False/True or float  - automatic scale data by force RMSE
  # avg_n_neigh: 40 # Average number of neighbours. By default - automatically determined
  # float_dtype: float64 # float64, float32
  # custom ZBL core repulsion for model:  kwargs: {zbl_cutoff: {Mo: 1, MoNb: 2, W: 1, Ta*: 3 }}
  # dense_nbr: False # Dense (reshape) equivariant neighbour aggregation. When True the model runs
  #                  # a batched matmul over a per-atom-uniform [n_atoms*max_neigh] bond layout
  #                  # instead of the per-bond einsum + segment_sum scatter. It is compute-bound
  #                  # (faster on GPU, esp. deeper models / higher coordination) but materializes
  #                  # padded neighbour slots, so it pays off only when the per-batch padding is
  #                  # low -- tune the fit::dense_* batching knobs below. The data pipeline emits
  #                  # the matching layout automatically. Bit-exact with the default path.

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
  ## and learning_rate after "after_iter" epochs (or fraction: 0.75 = 75% of maxiter, or "auto" = 0.75)
  ## NOTE: switch requires `scheduler: reduce_on_plateau` (set automatically by `gracemaker -t`)
  #    switch: { after_iter: 0.75,  # or e.g. 350
  #              energy: { weight: 5.0 }, 
  #              forces: { weight: 2.0 }, 
  #              stress: {weight: 0.001},
  #               learning_rate: 0.001}    
  }

  
  maxiter: auto # Max number of optimization epochs.
    ## "auto" mode: ~50k total updates for scratch, ~10k for finetuning.
    ## For BFGS, "auto" is ~500 epochs (scratch) or ~100 (finetuning).
  # target_total_updates: 50000 
    ## Alternative to maxiter: specify total gradient updates. 
    ## maxiter will be auto-computed. Recommended ~100-500 for BFGS.
  optimizer: Adam
    # Optimization with Adam: good for large number of parameters, first-order method
  opt_params: { learning_rate: 0.008, use_ema: True, ema_momentum: 0.99,  weight_decay: 1.e-20, clipnorm: 1.0}
  # reset_optimizer: True  # reset optimizer state, after being loaded from checkpoint
  # reset_epoch_and_step: False # reset epoch and step internal counters (stored in checkpoint)
  scheduler: cosine_decay # scheduler for learning-rate reduction during training 
    # available options are: reduce_on_plateau, cosine_decay, linear_decay, exponential_decay
  scheduler_params: {"minimum_learning_rate": 0.0001}
  #scheduler_params: {"warmup_epochs": 2, "cold_learning_rate": 0.1, "minimum_learning_rate": 0.05}
    # If :warmup_epochs: > 0, begin optimization with :cold_learning_rate: and reach :opt_params::learning_rate:
    # within :warmup_epochs: (can be < 1). Else, begin optimization with :opt_params::learning_rate: and decay down to
    # minimum_learning_rate within :maxiter: epochs
    # NOTE: the correct key is `minimum_learning_rate`. Using `minimal_learning_rate` (common typo) is silently ignored
    #       and the default (1e-4) is used instead.
  ## reduce_on_plateau scheduler (new API)
  # scheduler: reduce_on_plateau
  # scheduler_params: { patience: 10, reduction_factor: 0.8, minimum_learning_rate: 5.0e-4,
  #                     stop_at_min: False, resume_lr: True, cooldown: 0, monitor: test_loss }
  # legacy format for reduce_on_plateau lr scheduler (DEPRECATED — uses keys `factor` and `min`)
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
  ## Bucket is a group of batches padded to the same shape for efficient JIT execution.
  ## max_n_buckets can be an integer or "auto".
  ## In "auto" mode, the number of buckets is estimated as ~sqrt(num_batches), clamped to [1, 32].
  ## `train_max_n_buckets`: "auto" (default) or integer. Max number of distinct buffer shapes (buckets) for training.
  ##   - "auto": dynamically determines the minimum number of buckets (1-32) that keeps padding overhead below `auto_bucket_max_padding`.
  ## `test_max_n_buckets`: "auto" (default) or integer. Same for test set.
  ## `auto_bucket_max_padding`: 0.3 (default). Target maximum padding overhead fraction for neighbours when using `"auto"` bucketing. 0.3 means 30%.
  ##   NOTE: in dense mode (potential::dense_nbr: True) this knob is REPURPOSED as the unified net
  ##   neighbour-padding target and defaults to 0.15 (see the dense batching block below).
  train_max_n_buckets: auto  ## max number of buckets in train set
  test_max_n_buckets: auto   ## same for test

  ## ---- Dense (reshape) batching (only used when potential::dense_nbr: True) ----
  ## In the dense layout the atom and neighbour axes are bound (a padding/"fake" atom occupies a
  ## full reshape `width` of neighbour slots), and neighbour padding is what drives the wasted
  ## compute. So a single target controls everything:
  ## `auto_bucket_max_padding`: in dense mode this is the UNIFIED target on the NET neighbour-padding
  ##   fraction  1 - (real neighbours) / (sum of n_atoms*max_neigh over batches).  Default 0.15.
  ##   An adaptive search splits this budget between (a) the max_neigh "width" bucketing -- structures
  ##   are sorted by per-atom max_neigh and cut into contiguous buckets, each padded to its OWN max
  ##   (a DP-optimal partition of the max_neigh histogram, not fixed tiers) -- and (b) the per-atom
  ##   (nat) bucketing (fake atoms), choosing the allocation that meets the target with the FEWEST
  ##   distinct buffer shapes. The segment_sum path keeps the 0.3 default and pads the two axes
  ##   independently.
  # dense_max_shapes: 64       ## HARD cap on the number of distinct (n_atoms, max_neigh) buffer shapes
  #                            ## == XLA recompiles (the compile budget). shapes <= this by construction;
  #                            ## if the padding target needs more, the cap wins (padding rises) and a
  #                            ## warning is logged. A LARGER batch_size hits the target with fewer shapes.
  # dense_slot_budget: auto    ## max n_atoms*max_neigh per batch (caps per-batch memory / isolates
  #                            ## high-coordination outliers). auto = batch_size*mean(nat)*mean(max_neigh).
  # dense_n_neigh_buckets: auto ## explicit number of max_neigh width buckets (overrides the adaptive search)
  # dense_max_neigh_cap:       ## drop structures whose per-atom max_neigh exceeds this (broken/degenerate
  #                            ## cells); default off. Never truncates neighbours.
  ## A run logs, per dataset: "dense bucketing: N batches, S/cap distinct shapes, net neighbour padding X%".

  checkpoint_freq: 10 # frequency for **REGULAR** checkpoints. 
  # save_all_regular_checkpoints: False # to store ALL regular checkpoints
  # checkpoint_freq_steps: 500 # opt-in mid-epoch checkpoint every N training steps (Adam only).
  #                            # A crash loses at most N steps. Overwrites the default `checkpoint`
  #                            # file; on `-rl` restart, train_adam rewinds to the start of the
  #                            # interrupted epoch and fast-forwards the data iterator past the
  #                            # already-trained batches (gradients are preserved in the restored
  #                            # optimizer state). Use `--no-intra-epoch-redo` to skip the rewind.
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
