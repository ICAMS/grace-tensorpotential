## How to perform multi-GPU fit ?

## How to restart/continue fit ?

## How to fit and export GRACE/FS model? 

## What is buckets  (`train_max_n_buckets` and `test_max_n_buckets`) ?

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


## (TODO) Checkpointing

Use `checkpoint_freq` to specify how frequently save regular checkpoints (only last state will be saved into
checkpoint).
If you want to keep all regular checkpoints, then add flag `save_all_regular_checkpoints: True`


## (TODO) Single-GPU / Multi-GPU / Multi-worker modes