#!/bin/bash

set -e


# INPUT PARAMETERS
#DATASET_NAME=$(ls *shard*.pkl.gz)
DATASET_NAME=../data/MoNbTaW_train50.pkl.gz


N_SHARDS=2
STRATEGY="neighbours" # "structures"(default), "atoms", "neighbours"
BATCH_SIZE=3000 # number of "strategy" items per batch

BUCKETS=4 # 100 - buckets for padding
CUTOFF=5
CUTOFF_DICT="CUTOFF_2L"
OUTPUT="tf_dataset"

N_WORKERS=1 # can be less than N_SHARDS
THREADS=2  # number of threads for OMP, MKL, NUMEXPR etc. To share resources on single machine

# SCRIPT
export CUDA_VISIBLE_DEVICES=-1
echo "Dataset name: ${DATASET_NAME}"

ARGS_ITEMS=(
  "--rerun"                  # Option to re-run if previous output exists
  "-o" "${OUTPUT}"           # Output directory for results
  "--total-task-num" "${N_SHARDS}" # Total number of tasks for sharding
  "-b" "${BATCH_SIZE}"       # Batch size for processing
  "-bu" "${BUCKETS}"         # Buckets configuration
  "-c" "${CUTOFF}"           # Cutoff value for calculations
  "-cd" "${CUTOFF_DICT}"     # Cutoff-dict value for calculations
  "--energy-col" "energy"    # Specifies the energy column
  "--is-fit-stress"          # Flag to enable stress fitting
  "--elements" "MP"          # Elements to consider (e.g., Materials Project related)
  "--strategy" "${STRATEGY}" # Optimization strategy to use
  "--remove_stage1"          # remove stage-1 data after being processed
  # "--sharded-input"        # This option is used if there are many shards in DATASET_NAME (for big datasets)
)

ARGS="${ARGS_ITEMS[*]}"

export TF_CPP_MIN_LOG_LEVEL=3


export MKL_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export OMP_NUM_THREADS=$THREADS
export TF_NUM_INTEROP_THREADS=$THREADS
export TF_NUM_INTRAOP_THREADS=$THREADS

# Function to kill all child processes when the script receives a termination signal
cleanup() {
  echo "Terminating all child processes..."
  pkill -P $$
  exit 1
}

# Set trap to catch signals and trigger the cleanup function
trap cleanup SIGINT SIGTERM


echo "Command args:"
echo "$DATASET_NAME ${ARGS}"

# STAGE 1
echo "*******************************"
echo "*         Stage 1             *"
echo "*******************************"
for ((i=0; i<N_SHARDS; i++)); do
  echo "Running shard $i"
  grace_preprocess $DATASET_NAME --task-id $i ${ARGS} --stage-1 &

    # Check if we need to wait for a chunk to complete
  if (( (i+1) % N_WORKERS == 0 || i == N_SHARDS -1 )); then
      wait # Wait for background processes to complete
  fi
done

wait

# STAGE 2
echo "*******************************"
echo "*         Stage 2             *"
echo "*******************************"
for ((i=0; i<N_SHARDS; i++)); do
  echo "Running shard $i"
  grace_preprocess $DATASET_NAME --task-id $i ${ARGS} --stage-2 &

    # Check if we need to wait for a chunk to complete
  if (( (i+1) % N_WORKERS == 0 || i == N_SHARDS -1 )); then
      wait # Wait for background processes to complete
  fi
done

# STAGE 3
echo "*******************************"
echo "*         Stage 3             *"
echo "*******************************"
for ((i=0; i<N_SHARDS; i++)); do
  echo "Running shard $i"
  grace_preprocess $DATASET_NAME --task-id $i ${ARGS} --stage-3 &

    # Check if we need to wait for a chunk to complete
  if (( (i+1) % N_WORKERS == 0 || i == N_SHARDS -1 )); then
      wait # Wait for background processes to complete
  fi
done

# STAGE 4
echo "*******************************"
echo "*         Stage 4             *"
echo "*******************************"
grace_preprocess $DATASET_NAME --task-id 0 ${ARGS} --stage-4


echo "All stages are done"
