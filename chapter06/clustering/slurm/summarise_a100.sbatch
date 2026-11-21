#!/bin/bash
#SBATCH --job-name=cluster-summary
#SBATCH --partition=gpu.A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/dm236/scratch/histology_clustering/slurm/cluster-summary_%j.out
#SBATCH --error=/home/dm236/scratch/histology_clustering/slurm/cluster-summary_%j.err

set -euo pipefail

module load cuda/12.6.2 || module load nvidia-cuda/12.1.1 || true

PROJECT_ROOT="${PROJECT_ROOT:-/home/dm236/repos/histology-clustering-experiment}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/hypatia.yml}"
PYTHON="${PYTHON:-/home/dm236/repos/hproj/.venv/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"

"${PYTHON}" -m cluster_exp.cli summarise --config "${CONFIG}"
