#!/bin/bash
#SBATCH --job-name=icaird-images
#SBATCH --partition=gpu.L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-14%3
#SBATCH --output=/home/dm236/scratch/icaird_cervical_benchmark/slurm/icaird-images_%A_%a.out
#SBATCH --error=/home/dm236/scratch/icaird_cervical_benchmark/slurm/icaird-images_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/dm236/repos/icaird-cervical-benchmark}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/icaird_cervical_hypatia.yml}"
MANIFEST="${MANIFEST:-/home/dm236/scratch/icaird_cervical_benchmark/manifests/image_tasks.jsonl}"
RUN_NAME="${RUN_NAME:-icaird120k_20260612}"

cd "${PROJECT_ROOT}"

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader || true
uv run --no-sync python -c 'import torch; print(f"torch={torch.__version__} torch_cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}"); assert torch.cuda.is_available(), "CUDA is required for image jobs"'

uv run --no-sync icaird-cervical-benchmark run-task \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --run-name "${RUN_NAME}"
