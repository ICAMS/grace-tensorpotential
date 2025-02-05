import os
import random

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import glob
import tensorflow as tf
from tensorpotential import constants as tc

compression = "GZIP"
dataset_path = "tf_dataset"

# Define the pattern to match all of your datasets
dataset_pattern = os.path.join(dataset_path, "stage3/shard_*-of-*")

print(f"{dataset_pattern=}")
datasets_fnames = glob.glob(dataset_pattern)
random.shuffle(datasets_fnames)
print(f"{datasets_fnames=}")


def dataset_fn(context):
    print(f"context=", context)
    total = context.num_input_pipelines
    ind = context.input_pipeline_id
    datasets = [
        tf.data.Dataset.load(filepath, compression=compression)
        for filepath in datasets_fnames[ind::total]
    ]
    print(f"Dataset_fn: len(datasets) = {len(datasets)}")
    interleaved_dataset = tf.data.Dataset.from_tensor_slices(datasets).interleave(
        lambda x: x,  # cycle_length=len(datasets), num_parallel_calls=tf.data.AUTOTUNE
    )

    return interleaved_dataset


strategy = tf.distribute.get_strategy()
dds = strategy.distribute_datasets_from_function(
    dataset_fn,
)

b_count = 0
sum_sqr_forces = 0
tot_nat = 0
tot_nneigh = 0
tot_nstruct = 0
for b in dds:
    b_count += 1
    nat_real = b[tc.N_ATOMS_BATCH_REAL].numpy()
    n_neigh_real = b[tc.N_NEIGHBORS_REAL].numpy()
    n_struct = b[tc.N_STRUCTURES_BATCH_REAL].numpy()

    cur_forces = b[tc.DATA_REFERENCE_FORCES].numpy()
    cur_forces = cur_forces[:nat_real]

    sum_sqr_forces += np.sum(cur_forces**2)
    tot_nat += nat_real
    tot_nneigh += n_neigh_real
    tot_nstruct += n_struct

    # print(b["true_energy"])
    # break

print(f"{b_count=}")
print(f"{tot_nstruct=}")
print(f"{tot_nat=}")
print(f"{tot_nneigh=}")
print(f"Average n.neigh = {tot_nneigh/tot_nat}")
# currently avg per-component. Maybe must be per-vector ?, i.e. sqrt(3) times arger ?
rms_f = np.sqrt(sum_sqr_forces / tot_nat / 3)
print(f"{rms_f=}, {1/rms_f=}")
