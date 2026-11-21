#!/bin/bash
#SBATCH --job-name=cluster-validate
#SBATCH --partition=gpu.A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/dm236/scratch/histology_clustering/slurm/cluster-validate_%j.out
#SBATCH --error=/home/dm236/scratch/histology_clustering/slurm/cluster-validate_%j.err

set -euo pipefail

module load cuda/12.6.2 || module load nvidia-cuda/12.1.1 || true

PROJECT_ROOT="${PROJECT_ROOT:-/home/dm236/repos/histology-clustering-experiment}"
PYTHON="${PYTHON:-/home/dm236/repos/hproj/.venv/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version --format=csv,noheader

"${PYTHON}" -m cluster_exp.cli validate-runtime
