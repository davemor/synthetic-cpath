#!/bin/bash
#SBATCH --job-name=icaird-synth-finalize
#SBATCH --partition=large-short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/sharedscratch/dm236/synthetic_bags/icaird_cervical_conditional_edm_mlp_a100_sharded/slurm/%x_%j.out
#SBATCH --error=/sharedscratch/dm236/synthetic_bags/icaird_cervical_conditional_edm_mlp_a100_sharded/slurm/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/dm236/repos/icaird-cervical-benchmark}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/icaird_cervical_synthetic_bags_conditional_edm_a100_sharded.yml}"
CHECKPOINT="${CHECKPOINT:-/sharedscratch/dm236/synthetic_embeddings/icaird100k_uni2h_edm_conditional/checkpoints/best.pt}"
NUM_SHARDS="${NUM_SHARDS:-12}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/sharedscratch/dm236/synthetic_bags/icaird_cervical_conditional_edm_mlp_a100_sharded/slurm}"

cd "${PROJECT_ROOT}"
mkdir -p "${SLURM_LOG_DIR}"

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID:-manual}"
echo "config=${CONFIG}"
echo "checkpoint=${CHECKPOINT}"
echo "num_shards=${NUM_SHARDS}"

uv run --no-sync icaird-cervical-benchmark synthetic-bags-finalize-shards \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --num-shards "${NUM_SHARDS}"

uv run --no-sync icaird-cervical-benchmark synthetic-bags-write-mil-configs --config "${CONFIG}"
