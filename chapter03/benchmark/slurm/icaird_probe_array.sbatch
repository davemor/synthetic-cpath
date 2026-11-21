#!/bin/bash
#SBATCH --job-name=icaird-probes
#SBATCH --partition=small-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-44%8
#SBATCH --output=/home/dm236/scratch/icaird_cervical_benchmark/slurm/icaird-probes_%A_%a.out
#SBATCH --error=/home/dm236/scratch/icaird_cervical_benchmark/slurm/icaird-probes_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/dm236/repos/icaird-cervical-benchmark}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/icaird_cervical_hypatia.yml}"
MANIFEST="${MANIFEST:-/home/dm236/scratch/icaird_cervical_benchmark/manifests/probe_tasks.jsonl}"
RUN_NAME="${RUN_NAME:-icaird120k_20260612}"

cd "${PROJECT_ROOT}"

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID}"
echo "config=${CONFIG}"
echo "manifest=${MANIFEST}"

uv run --no-sync icaird-cervical-benchmark run-task \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --run-name "${RUN_NAME}"
