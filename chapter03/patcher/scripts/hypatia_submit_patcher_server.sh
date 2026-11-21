#!/bin/bash
set -euo pipefail

REPO_ROOT="${PATCHER_REPO_ROOT:-$HOME/repos/patcher}"
DATASET_ROOT="${ICAIRD_DATASET_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
PATCHER_PARTITION="${PATCHER_PARTITION:-large-short}"
PATCHER_TIME="${PATCHER_TIME:-08:00:00}"
PATCHER_CPUS="${PATCHER_CPUS:-4}"
PATCHER_MEM="${PATCHER_MEM:-16G}"
PATCHER_PORT="${PATCHER_PORT:-8000}"
PATCHER_CONFIG_PATH="${PATCHER_CONFIG_PATH:-$REPO_ROOT/configs/icaird-cervical-stratified-120-hypatia.yml}"

mkdir -p "$DATASET_ROOT/slurm_logs"
cd "$REPO_ROOT"

job_id="$(
  sbatch --parsable \
    --partition="$PATCHER_PARTITION" \
    --time="$PATCHER_TIME" \
    --cpus-per-task="$PATCHER_CPUS" \
    --mem="$PATCHER_MEM" \
    --export=ALL,PATCHER_REPO_ROOT="$REPO_ROOT",PATCHER_PORT="$PATCHER_PORT",PATCHER_CONFIG_PATH="$PATCHER_CONFIG_PATH" \
    scripts/slurm/hypatia_patcher_server.sbatch
)"

cat <<EOF
Submitted patcher server job $job_id.

Watch it start with:
  squeue -j $job_id
  tail -f "$DATASET_ROOT/slurm_logs/patcher_server_${job_id}.out"

When the log prints the compute node, run the printed ssh -L command from
your local machine. The annotator URLs will be:
  http://localhost:${PATCHER_PORT}/?annotator=annotator_1
  http://localhost:${PATCHER_PORT}/?annotator=annotator_2
  http://localhost:${PATCHER_PORT}/?annotator=annotator_3

Stop the server with:
  scancel $job_id
EOF
