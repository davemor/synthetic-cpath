#!/bin/bash
#SBATCH --job-name=hproj-icaird-curve
#SBATCH --partition=gpu.A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/home/dm236/scratch/hproj/output/slurm/hproj-icaird-curve_%A_%a.out
#SBATCH --error=/home/dm236/scratch/hproj/output/slurm/hproj-icaird-curve_%A_%a.err

set -euo pipefail

export EMBEDDINGS_ROOT=/home/dm236/scratch/hproj/embeddings
export DATA_ROOT=/home/dm236/scratch/hproj/output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RUN_ID=hproj-2026-05-25_16-41-27-bb2a5203-icaird100k

cd /home/dm236/repos/hproj
source .venv/bin/activate

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID}"
echo "run_id=${RUN_ID}"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader || true

hproj estimate-curve \
  --run-id "${RUN_ID}" \
  --dataset-name icaird100k_uni \
  --dataset-name icaird100k_uni2 \
  --dataset-name icaird100k_hibou \
  --dataset-name icaird100k_dinov2 \
  --num-shards 12 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --tasks-only
