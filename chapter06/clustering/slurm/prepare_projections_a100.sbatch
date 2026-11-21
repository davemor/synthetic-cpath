#!/bin/bash
#SBATCH --job-name=cluster-proj-manifest
#SBATCH --partition=gpu.A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/dm236/scratch/histology_clustering/slurm/cluster-prep_%j.out
#SBATCH --error=/home/dm236/scratch/histology_clustering/slurm/cluster-prep_%j.err

set -euo pipefail

module load cuda/12.6.2 || module load nvidia-cuda/12.1.1 || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/dm236/repos/histology-clustering-experiment}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/hypatia.yml}"
PROJECTION_MANIFEST="${PROJECTION_MANIFEST:-/home/dm236/scratch/histology_clustering/manifests/projection_tasks.jsonl}"
PYTHON="${PYTHON:-/home/dm236/repos/hproj/.venv/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version --format=csv,noheader || true

"${PYTHON}" -m cluster_exp.cli validate-runtime
"${PYTHON}" -m cluster_exp.cli make-projection-manifest \
  --config "${CONFIG}" \
  --output "${PROJECTION_MANIFEST}"
