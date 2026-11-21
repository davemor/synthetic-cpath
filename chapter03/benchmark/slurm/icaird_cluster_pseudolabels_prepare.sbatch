#!/bin/bash
#SBATCH --job-name=icaird-cluster-labels
#SBATCH --partition=large-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/sharedscratch/dm236/data/icaird_cervical/cluster_pseudolabels/icaird100k_uni2h_kmeans24_pca42/slurm/%x_%j.out
#SBATCH --error=/sharedscratch/dm236/data/icaird_cervical/cluster_pseudolabels/icaird100k_uni2h_kmeans24_pca42/slurm/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/dm236/repos/icaird-cervical-benchmark}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/icaird_cervical_cluster_pseudolabel_bags_conditional_edm_a100_sharded.yml}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/sharedscratch/dm236/data/icaird_cervical/cluster_pseudolabels/icaird100k_uni2h_kmeans24_pca42/slurm}"

cd "${PROJECT_ROOT}"
mkdir -p "${SLURM_LOG_DIR}"

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID:-manual}"
echo "config=${CONFIG}"

uv run --no-sync icaird-cervical-benchmark cluster-pseudolabels-prepare --config "${CONFIG}"
