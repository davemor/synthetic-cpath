#!/bin/bash
#SBATCH --job-name=cluster-sd-index
#SBATCH --partition=gpu.A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/dm236/scratch/histology_clustering/slurm/cluster-sd-index_%j.out
#SBATCH --error=/home/dm236/scratch/histology_clustering/slurm/cluster-sd-index_%j.err

set -euo pipefail

module load cuda/12.6.2 || module load nvidia-cuda/12.1.1 || true

PROJECT_ROOT="${PROJECT_ROOT:-/home/dm236/repos/histology-clustering-experiment}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/hypatia_spherical_kmeans.yml}"
RUN_DIR="${RUN_DIR:-/home/dm236/scratch/histology_clustering/output/clustering-curve-targets}"
BASE_MANIFEST="${BASE_MANIFEST:-/home/dm236/scratch/histology_clustering/manifests/clustering_tasks.jsonl}"
SPHERICAL_MANIFEST="${SPHERICAL_MANIFEST:-/home/dm236/scratch/histology_clustering/manifests/spherical_kmeans_tasks.jsonl}"
PYTHON="${PYTHON:-/home/dm236/repos/hproj/.venv/bin/python}"
WORKERS="${WORKERS:-16}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"

"${PYTHON}" -m cluster_exp.cli add-sd-index \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --manifest "${BASE_MANIFEST}" \
  --manifest "${SPHERICAL_MANIFEST}" \
  --workers "${WORKERS}"
