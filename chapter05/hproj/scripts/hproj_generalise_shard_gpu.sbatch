#!/bin/bash
#SBATCH --job-name=hproj-gen-shard
#SBATCH --partition=gpu.A100,gpu.L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-119%6
#SBATCH --output=/home/dm236/scratch/hproj/output/slurm/hproj-gen-shard_%A_%a.out
#SBATCH --error=/home/dm236/scratch/hproj/output/slurm/hproj-gen-shard_%A_%a.err

set -euo pipefail

export EMBEDDINGS_ROOT=/home/dm236/scratch/hproj/embeddings
export DATA_ROOT=/home/dm236/scratch/hproj/output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RUN_ID=hproj-2026-05-25_16-41-27-bb2a5203

cd /home/dm236/repos/hproj
source .venv/bin/activate

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,pci.bus_id,ecc.errors.uncorrected.volatile.total --format=csv,noheader || true

hproj generalise \
  --run-id "${RUN_ID}" \
  --num-shards 120 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --tasks-only
