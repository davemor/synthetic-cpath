#!/bin/bash
#SBATCH --job-name=hproj-fullcal-proj-sum
#SBATCH --partition=gpu.A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/dm236/scratch/hproj/output/slurm/hproj-fullcal-proj-sum_%j.out
#SBATCH --error=/home/dm236/scratch/hproj/output/slurm/hproj-fullcal-proj-sum_%j.err

set -euo pipefail

export EMBEDDINGS_ROOT=/home/dm236/scratch/hproj/embeddings
export DATA_ROOT=/home/dm236/scratch/hproj/output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RUN_ID="${RUN_ID:-hproj-2026-06-26-icaird-fullcal-lr10d}"

cd /home/dm236/repos/hproj
source .venv/bin/activate

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID}"
echo "run_id=${RUN_ID}"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader || true

hproj calibrate-projector --run-id "${RUN_ID}"
