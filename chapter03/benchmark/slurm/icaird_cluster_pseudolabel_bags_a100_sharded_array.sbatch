#!/bin/bash
#SBATCH --job-name=icaird-cluster-bags
#SBATCH --partition=gpu.A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-35%12
#SBATCH --output=/sharedscratch/dm236/synthetic_bags/icaird_cervical_conditional_edm_mlp_kmeans24_pca42_clusters_a100_sharded/slurm/%x_%A_%a.out
#SBATCH --error=/sharedscratch/dm236/synthetic_bags/icaird_cervical_conditional_edm_mlp_kmeans24_pca42_clusters_a100_sharded/slurm/%x_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/dm236/repos/icaird-cervical-benchmark}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/icaird_cervical_cluster_pseudolabel_bags_conditional_edm_a100_sharded.yml}"
CHECKPOINT="${CHECKPOINT:-/sharedscratch/dm236/synthetic_embeddings/icaird100k_uni2h_edm_conditional_kmeans24_pca42_clusters/checkpoints/best.pt}"
NUM_SHARDS="${NUM_SHARDS:-12}"
MULTIPLIERS_STRING="${MULTIPLIERS:-0.5 1.0 2.0}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/sharedscratch/dm236/synthetic_bags/icaird_cervical_conditional_edm_mlp_kmeans24_pca42_clusters_a100_sharded/slurm}"

read -r -a MULTIPLIERS_ARRAY <<< "${MULTIPLIERS_STRING}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
MULTIPLIER_INDEX=$((TASK_ID / NUM_SHARDS))
SHARD_INDEX=$((TASK_ID % NUM_SHARDS))

if [[ "${MULTIPLIER_INDEX}" -ge "${#MULTIPLIERS_ARRAY[@]}" ]]; then
  echo "array task ${TASK_ID} exceeds multiplier list; exiting"
  exit 0
fi

MULTIPLIER="${MULTIPLIERS_ARRAY[${MULTIPLIER_INDEX}]}"

cd "${PROJECT_ROOT}"
mkdir -p "${SLURM_LOG_DIR}"

echo "host=$(hostname)"
echo "job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}} array_task=${SLURM_ARRAY_TASK_ID:-manual}"
echo "config=${CONFIG}"
echo "checkpoint=${CHECKPOINT}"
echo "multiplier=${MULTIPLIER}"
echo "num_shards=${NUM_SHARDS}"
echo "shard_index=${SHARD_INDEX}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader || true

uv run --no-sync icaird-cervical-benchmark synthetic-bags-generate-shard \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --multiplier "${MULTIPLIER}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}"
