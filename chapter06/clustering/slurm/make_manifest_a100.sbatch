#!/bin/bash
#SBATCH --job-name=cluster-manifest
#SBATCH --partition=gpu.A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dm236/scratch/histology_clustering/slurm/cluster-manifest_%j.out
#SBATCH --error=/home/dm236/scratch/histology_clustering/slurm/cluster-manifest_%j.err

set -euo pipefail

module load cuda/12.6.2 || module load nvidia-cuda/12.1.1 || true

PROJECT_ROOT="${PROJECT_ROOT:-/home/dm236/repos/histology-clustering-experiment}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/hypatia.yml}"
MANIFEST="${MANIFEST:-/home/dm236/scratch/histology_clustering/manifests/clustering_tasks.jsonl}"
PYTHON="${PYTHON:-/home/dm236/repos/hproj/.venv/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"

"${PYTHON}" -m cluster_exp.cli validate-runtime
"${PYTHON}" -m cluster_exp.cli make-manifest --config "${CONFIG}" --output "${MANIFEST}"
