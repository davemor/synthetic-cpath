#!/bin/bash
set -euo pipefail

REPO_ROOT="${HISTOKIT_REPO_ROOT:-$HOME/repos/histokit}"
DATASET_ROOT="${ICAIRD_DATASET_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
SPLIT_PARTITION="${SPLIT_PARTITION:-large-short}"
SPLIT_TIME="${SPLIT_TIME:-04:00:00}"
SPLIT_CPUS="${SPLIT_CPUS:-2}"
SPLIT_MEM="${SPLIT_MEM:-8G}"

mkdir -p "$DATASET_ROOT/slurm_logs"
cd "$REPO_ROOT"

job_id="$(
  sbatch --parsable \
    --partition="$SPLIT_PARTITION" \
    --time="$SPLIT_TIME" \
    --cpus-per-task="$SPLIT_CPUS" \
    --mem="$SPLIT_MEM" \
    --export=ALL,HISTOKIT_REPO_ROOT="$REPO_ROOT" \
    scripts/slurm/hypatia_create_resnet_split_dataset.sbatch
)"

cat <<EOF
Submitted split dataset job $job_id.

Watch it with:
  squeue -j $job_id
  tail -f "$DATASET_ROOT/slurm_logs/split_patches_${job_id}.out"

Stop it with:
  scancel $job_id
EOF
