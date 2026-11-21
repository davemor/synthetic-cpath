#!/bin/bash
#SBATCH --job-name=hproj-fullcal-postcurve
#SBATCH --partition=gpu.A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/dm236/scratch/hproj/output/slurm/hproj-fullcal-postcurve_%j.out
#SBATCH --error=/home/dm236/scratch/hproj/output/slurm/hproj-fullcal-postcurve_%j.err

set -euo pipefail

export EMBEDDINGS_ROOT=/home/dm236/scratch/hproj/embeddings
export DATA_ROOT=/home/dm236/scratch/hproj/output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RUN_ID="${RUN_ID:-hproj-2026-06-26-icaird-fullcal-lr10d}"
RUN_ROOT="${DATA_ROOT}/runs/${RUN_ID}"
MANIFEST="${RUN_ROOT}/generalisation/task_manifest_fullcal_principled_reduced.jsonl"

cd /home/dm236/repos/hproj
source .venv/bin/activate

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID}"
echo "run_id=${RUN_ID}"
echo "manifest=${MANIFEST}"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader || true

hproj estimate-curve --run-id "${RUN_ID}"
hproj intrinsic-dims --run-id "${RUN_ID}"
hproj thresholds --run-id "${RUN_ID}"
hproj generalise \
  --run-id "${RUN_ID}" \
  --task-manifest "${MANIFEST}" \
  --manifest-only \
  --tasks-only
