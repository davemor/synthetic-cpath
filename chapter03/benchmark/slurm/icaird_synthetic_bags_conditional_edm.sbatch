#!/bin/bash
#SBATCH --job-name=icaird-synth-bags
#SBATCH --partition=gpu.L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/sharedscratch/dm236/synthetic_bags/icaird_cervical_conditional_edm_mlp/slurm/%x_%j.out
#SBATCH --error=/sharedscratch/dm236/synthetic_bags/icaird_cervical_conditional_edm_mlp/slurm/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/dm236/repos/icaird-cervical-benchmark}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/icaird_cervical_synthetic_bags_conditional_edm.yml}"
CHECKPOINT="${CHECKPOINT:-/sharedscratch/dm236/synthetic_embeddings/icaird100k_uni2h_edm_conditional/checkpoints/best.pt}"
STAGE="${STAGE:-all}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/sharedscratch/dm236/synthetic_bags/icaird_cervical_conditional_edm_mlp/slurm}"

cd "${PROJECT_ROOT}"
mkdir -p "${SLURM_LOG_DIR}"

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID:-manual}"
echo "config=${CONFIG}"
echo "checkpoint=${CHECKPOINT}"
echo "stage=${STAGE}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader || true

if [[ "${STAGE}" == "fit" || "${STAGE}" == "all" ]]; then
  uv run --no-sync icaird-cervical-benchmark synthetic-bags-fit --config "${CONFIG}"
fi

if [[ "${STAGE}" == "generate" || "${STAGE}" == "all" ]]; then
  uv run --no-sync icaird-cervical-benchmark synthetic-bags-generate \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}"
fi

if [[ "${STAGE}" == "mil-configs" || "${STAGE}" == "all" ]]; then
  uv run --no-sync icaird-cervical-benchmark synthetic-bags-write-mil-configs --config "${CONFIG}"
fi
