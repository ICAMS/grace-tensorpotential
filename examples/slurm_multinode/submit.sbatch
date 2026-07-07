#!/bin/bash
# Multi-node GRACE fit on Paderborn Noctua2 (PC2) via MultiWorkerMirroredStrategy.
#
# One srun task per node. Each task sees all 4 local GPUs; TF's MWMS handles
# NCCL within a node and across nodes over InfiniBand. The dataset must be
# pre-sharded with at least num_nodes * gpus_per_node shards (see
# grace_preprocess with --total-task-num).
#
# Usage (all args after the script name are forwarded to gracemaker):
#   sbatch --nodes=4 submit.sbatch input.yaml
#   sbatch --nodes=8 submit.sbatch input.yaml
#
# Set your account / partition below, then submit. Self-contained: no
# companion launcher script needed.

#SBATCH -t 1-00:00:00
#SBATCH -J "grace-mwms"
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH -p gpu
# #SBATCH --account=YOUR_ACCOUNT
# Prefer single-leaf placement; give up and schedule anywhere after 30 min.
#SBATCH --switches=1@00:30:00
#SBATCH --output=logs/%x-%j/slurm-%N-node%n.out
#SBATCH --error=logs/%x-%j/slurm-%N-node%n.err

set -euo pipefail

mkdir -p "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"

# --- TF / dataset env ---
export TF_USE_LEGACY_KERAS=1
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

# --- NCCL ---
# Let NCCL auto-detect the IB bootstrap iface and HCAs. On PC2/Noctua2,
# explicitly setting NCCL_SOCKET_IFNAME=ib0 broke bootstrap ("no socket
# interface found"); NCCL picks ib1 (IPoIB) on its own and routes data
# over mlx5_0/mlx5_1 via IB verbs. Uncomment only to override for debugging.
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# --- MWMS port (same on all nodes) ---
export TF_PORT="${TF_PORT:-2222}"

# --- thread pinning (avoid oversubscription) ---
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

# --- Build the cluster.worker list once (same on every node) ---
HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
TF_CLUSTER_WORKERS=""
for h in "${HOSTS[@]}"; do
  [ -n "$TF_CLUSTER_WORKERS" ] && TF_CLUSTER_WORKERS="${TF_CLUSTER_WORKERS},"
  TF_CLUSTER_WORKERS="${TF_CLUSTER_WORKERS}\"${h}:${TF_PORT}\""
done
export TF_CLUSTER_WORKERS

echo "=== SLURM job ${SLURM_JOB_ID} on ${SLURM_JOB_NUM_NODES} nodes ==="
echo "Nodelist: ${SLURM_JOB_NODELIST}"
echo "TF_PORT=${TF_PORT} NCCL_DEBUG=${NCCL_DEBUG}"

# Each srun task injects its own SLURM_NODEID into TF_CONFIG and execs gracemaker.
srun --kill-on-bad-exit=1 bash -c '
  export TF_CONFIG="{\"cluster\":{\"worker\":[${TF_CLUSTER_WORKERS}]},\"task\":{\"type\":\"worker\",\"index\":${SLURM_NODEID}}}"
  echo "[node ${SLURM_NODEID}] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[node ${SLURM_NODEID}] TF_CONFIG=${TF_CONFIG}"
  exec gracemaker "$@"
' _ "$@"
