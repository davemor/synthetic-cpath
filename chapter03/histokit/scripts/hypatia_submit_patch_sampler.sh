#!/bin/bash
set -euo pipefail

REPO_ROOT="${HISTOKIT_REPO_ROOT:-$HOME/repos/histokit}"
DATASET_ROOT="${ICAIRD_DATASET_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
SAMPLER_PARTITION="${SAMPLER_PARTITION:-large-long}"
SAMPLER_TIME="${SAMPLER_TIME:-1-00:00:00}"
SAMPLER_CPUS="${SAMPLER_CPUS:-4}"
SAMPLER_MEM="${SAMPLER_MEM:-32G}"

mkdir -p "$DATASET_ROOT/slurm_logs"
cd "$REPO_ROOT"

job_id="$(
  sbatch --parsable \
    --partition="$SAMPLER_PARTITION" \
    --time="$SAMPLER_TIME" \
    --cpus-per-task="$SAMPLER_CPUS" \
    --mem="$SAMPLER_MEM" \
    --export=ALL,HISTOKIT_REPO_ROOT="$REPO_ROOT" \
    scripts/slurm/hypatia_sample_patcher_consensus_patches.sbatch
)"

cat <<EOF
Submitted patch sampler job $job_id.

Watch it with:
  squeue -j $job_id
  tail -f "$DATASET_ROOT/slurm_logs/sample_patches_${job_id}.out"

Default output:
  $DATASET_ROOT/resnet_patch_subset/stratified_120_consensus_100k

Stop it with:
  scancel $job_id
EOF
