#!/bin/bash
#SBATCH --job-name=hproj-gen-batch
#SBATCH --partition=gpu.A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/dm236/scratch/hproj/output/slurm/hproj-gen-batch_%A_%a.out
#SBATCH --error=/home/dm236/scratch/hproj/output/slurm/hproj-gen-batch_%A_%a.err

set -euo pipefail

export EMBEDDINGS_ROOT=/home/dm236/scratch/hproj/embeddings
export DATA_ROOT=/home/dm236/scratch/hproj/output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RUN_ID=hproj-2026-05-25_16-41-27-bb2a5203
RUN_ROOT="${DATA_ROOT}/runs/${RUN_ID}"
MANIFEST="${RUN_ROOT}/generalisation/task_manifest_principled_reduced.jsonl"
PROJECTION_CACHE="${RUN_ROOT}/generalisation/projection_cache_principled_reduced"
TASK_BATCH_SIZE="${TASK_BATCH_SIZE:-1}"

if [[ ! -s "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 2
fi

cd /home/dm236/repos/hproj
source .venv/bin/activate

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID}"
echo "manifest=${MANIFEST}"
echo "projection_cache=${PROJECTION_CACHE}"
echo "task_batch_size=${TASK_BATCH_SIZE}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,pci.bus_id,ecc.errors.uncorrected.volatile.total --format=csv,noheader || true

hproj generalise \
  --run-id "${RUN_ID}" \
  --task-manifest "${MANIFEST}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --task-batch-size "${TASK_BATCH_SIZE}" \
  --projection-cache-dir "${PROJECTION_CACHE}" \
  --tasks-only
