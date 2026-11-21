#!/bin/bash
#SBATCH --job-name=hproj-fullcal-curve
#SBATCH --partition=gpu.A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-12:00:00
#SBATCH --array=0-71%8
#SBATCH --output=/home/dm236/scratch/hproj/output/slurm/hproj-fullcal-curve_%A_%a.out
#SBATCH --error=/home/dm236/scratch/hproj/output/slurm/hproj-fullcal-curve_%A_%a.err

set -euo pipefail

export EMBEDDINGS_ROOT=/home/dm236/scratch/hproj/embeddings
export DATA_ROOT=/home/dm236/scratch/hproj/output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RUN_ID="${RUN_ID:-hproj-2026-06-26-icaird-fullcal-lr10d}"
RUN_ROOT="${DATA_ROOT}/runs/${RUN_ID}"
CURVE_PROJECTION_CACHE="${CURVE_PROJECTION_CACHE:-${RUN_ROOT}/curve/projection_cache}"

cd /home/dm236/repos/hproj
source .venv/bin/activate

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID}"
echo "run_id=${RUN_ID}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader || true

hproj estimate-curve \
  --run-id "${RUN_ID}" \
  --num-shards 72 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --projection-cache-dir "${CURVE_PROJECTION_CACHE}" \
  --tasks-only
