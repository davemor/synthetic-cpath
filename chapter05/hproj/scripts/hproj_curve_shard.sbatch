#!/bin/bash
#SBATCH --job-name=hproj-curve-shard
#SBATCH --partition=gpu.L40S
#SBATCH --nodelist=gpu01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-12:00:00
#SBATCH --array=0-59%8
#SBATCH --output=/home/dm236/scratch/hproj/output/slurm/hproj-curve-shard_%A_%a.out
#SBATCH --error=/home/dm236/scratch/hproj/output/slurm/hproj-curve-shard_%A_%a.err

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

hproj estimate-curve \
  --run-id "${RUN_ID}" \
  --num-shards 60 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --tasks-only
