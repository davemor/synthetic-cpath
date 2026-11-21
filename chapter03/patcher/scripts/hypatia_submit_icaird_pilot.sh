#!/bin/bash
set -euo pipefail

DATASET_ROOT="${ICAIRD_DATASET_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
REPO_ROOT="${PATCHER_REPO_ROOT:-$HOME/repos/patcher}"

mkdir -p "$DATASET_ROOT/slurm_logs"
cd "$REPO_ROOT"
sbatch scripts/slurm/hypatia_icaird_pilot.sbatch
