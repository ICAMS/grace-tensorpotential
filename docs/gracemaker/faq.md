## How to perform multi-GPU fit ?



## How to restart/continue fit ?

Simply run `gracemaker -r` in the folder of the original fit 

## How to fit and export GRACE/FS model? 

## What is buckets  (`train_max_n_buckets` and `test_max_n_buckets`) ?

GRACE models are JIT compiled which requires all batches to have the same size. This is achieved by padding. However,
padding all batches to the identical dimensions might be inefficient. Instead, batches are
split into buckets which are then padded. Parameters `train_max_n_buckets` and `test_max_n_buckets` determines
maximum number of buckets for padding train and test data respectively. The more buckets there are the less padding
is required. The optimal number of buckets can be estimated by looking at `[TRAIN] dataset stats:` log line,
where the amount of padded neighbours is printed, i.e.
```
 [TRAIN] dataset stats: num. batches: 18 | num. real structures: 576 (+2.78%) | num. real atoms: 10942 (+5.25%) | num. real neighbours: 292102 (+1.74%) 
```
Here it is only +1.74% padded neighbours. It is recommended to keep this number within 15%.
Same applies for `[TEST] dataset stats`.

## How to reduce verbosity level of TensorFlow ?

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

or 

```bash
export TF_CPP_MIN_LOG_LEVEL=3
```

## How to run GRACE-1LAYER model in parallel within LAMMPS?
```bash
TF_CPP_MIN_LOG_LEVEL=1 mpirun -np 4 --bind-to none  bash -c 'CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_RANK % 4)) lmp -in in.lammps'
```


## How to evaluate uncertainty indication for GRACE models ?
TODO: Ensemble, with multiple models , provided to TPCalculator 

## (TODO) Checkpointing

Use `checkpoint_freq` to specify how frequently save regular checkpoints (only last state will be saved into
checkpoint).
If you want to keep all regular checkpoints, then add flag `save_all_regular_checkpoints: True`


## (TODO) Single-GPU / Multi-GPU / Multi-worker modes