#!/bin/bash
set -euo pipefail

DATASET_ROOT="${ICAIRD_DATASET_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
SAMPLE_ROOT="${ICAIRD_STRATIFIED_SAMPLE_ROOT:-$DATASET_ROOT/stratified_120}"
REPO_ROOT="${PATCHER_REPO_ROOT:-$HOME/repos/patcher}"
UPSTREAM_JOB_ID="${UPSTREAM_JOB_ID-3303721}"
CACHE_ARRAY_CONCURRENCY="${CACHE_ARRAY_CONCURRENCY:-12}"

chunks_dir="$SAMPLE_ROOT/chunks"
chunk_count="$(find "$chunks_dir" -maxdepth 1 -name 'stratified_120_chunk_*.csv' | wc -l | tr -d ' ')"

if [[ "$chunk_count" -eq 0 ]]; then
  echo "No chunk manifests found under $chunks_dir" >&2
  exit 1
fi

mkdir -p "$DATASET_ROOT/slurm_logs"
cd "$REPO_ROOT"

dependency_args=()
if [[ -n "$UPSTREAM_JOB_ID" ]]; then
  dependency_args=(--dependency="afterok:$UPSTREAM_JOB_ID")
fi

sbatch \
  "${dependency_args[@]}" \
  --array="0-$((chunk_count - 1))%$CACHE_ARRAY_CONCURRENCY" \
  scripts/slurm/hypatia_icaird_stratified_cache_array.sbatch
