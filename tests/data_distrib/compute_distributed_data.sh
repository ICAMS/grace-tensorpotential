#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

N_WORKERS=2
#ELEMENTS=""
BATCH_SIZE=256
STRATEGY="atoms" # default "structures"

BUCKETS=4

CUTOFF=5

DATASET_NAME=/home/users/lysogy36/acefit/GRACE/examples/HEA-25/HEA25-test.pkl.gz
#DATASET_NAME=$(ls dataset_shard*.pkl.gz)
echo "Dataset name: ${DATASET_NAME}"

ARGS="--total-task-num ${N_WORKERS} -b ${BATCH_SIZE} -bu ${BUCKETS} -c ${CUTOFF} --strategy ${STRATEGY}"
#ARGS="${ARGS} -o tf_dataset_nogzip --compression NONE"

# STAGE 1
echo "*******************************"
echo "*         Stage 1             *"
echo "*******************************"
for ((i=0; i<N_WORKERS; i++)); do
  echo "Running worker $i"
  grace_preprocess $DATASET_NAME --task-id $i ${ARGS} --stage-1 &
done

wait

# STAGE 2
echo "*******************************"
echo "*         Stage 2             *"
echo "*******************************"
for ((i=0; i<N_WORKERS; i++)); do
  echo "Running worker $i"
  grace_preprocess $DATASET_NAME --task-id $i ${ARGS} --stage-2 &
done
wait

# STAGE 3
echo "*******************************"
echo "*         Stage 3             *"
echo "*******************************"
for ((i=0; i<N_WORKERS; i++)); do
  echo "Running worker $i"
  grace_preprocess $DATASET_NAME --task-id $i ${ARGS} --stage-3 &
done
wait

# STAGE 4
echo "*******************************"
echo "*         Stage 4             *"
echo "*******************************"
grace_preprocess $DATASET_NAME --task-id 0 ${ARGS} --stage-4


echo "All stages are done"