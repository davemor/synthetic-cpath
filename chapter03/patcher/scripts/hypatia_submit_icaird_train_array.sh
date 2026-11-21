#!/bin/bash
set -euo pipefail

DATASET_ROOT="${ICAIRD_DATASET_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
REPO_ROOT="${PATCHER_REPO_ROOT:-$HOME/repos/patcher}"
chunks_dir="$DATASET_ROOT/chunks"
chunk_count="$(find "$chunks_dir" -maxdepth 1 -name 'train_chunk_*.csv' | wc -l | tr -d ' ')"

if [[ "$chunk_count" -eq 0 ]]; then
  echo "No chunk manifests found under $chunks_dir" >&2
  exit 1
fi

mkdir -p "$DATASET_ROOT/slurm_logs"
cd "$REPO_ROOT"
sbatch --array="0-$((chunk_count - 1))%4" scripts/slurm/hypatia_icaird_train_array.sbatch
