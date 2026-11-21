#!/bin/bash
#SBATCH --job-name=icaird-report
#SBATCH --partition=small-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/dm236/scratch/icaird_cervical_benchmark/slurm/icaird-report_%j.out
#SBATCH --error=/home/dm236/scratch/icaird_cervical_benchmark/slurm/icaird-report_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/dm236/repos/icaird-cervical-benchmark}"
RUN_DIR="${RUN_DIR:-/home/dm236/scratch/icaird_cervical_benchmark/runs/icaird120k_20260612}"

cd "${PROJECT_ROOT}"

uv run --no-sync icaird-cervical-benchmark report "${RUN_DIR}"
