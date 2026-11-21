#!/bin/bash
#SBATCH --job-name=icaird-synth-mil
#SBATCH --partition=gpu.L40S
#SBATCH --exclude=gpu02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/home/dm236/scratch/icaird_cervical_benchmark/slurm/icaird-synth-mil_%A_%a.out
#SBATCH --error=/home/dm236/scratch/icaird_cervical_benchmark/slurm/icaird-synth-mil_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/dm236/repos/icaird-cervical-benchmark}"
CONFIG="${CONFIG:?Set CONFIG to one generated synthetic MIL config.}"
MANIFEST="${MANIFEST:?Set MANIFEST to the matching generated synthetic MIL manifest.}"
RUN_NAME="${RUN_NAME:-icaird_mil_uni2h_synthetic_bags_$(date +%Y%m%d_%H%M%S)}"

cd "${PROJECT_ROOT}"

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID}"
echo "config=${CONFIG}"
echo "manifest=${MANIFEST}"
echo "run_name=${RUN_NAME}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader || true

uv run --no-sync icaird-cervical-benchmark run-task \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --run-name "${RUN_NAME}"
