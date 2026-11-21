#!/bin/bash
#SBATCH --job-name=patcher-server
#SBATCH --output=/sharedscratch/dm236/data/icaird_cervical/slurm_logs/patcher_server_%j.out
#SBATCH --error=/sharedscratch/dm236/data/icaird_cervical/slurm_logs/patcher_server_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail

REPO_ROOT="${PATCHER_REPO_ROOT:-$HOME/repos/patcher}"
PATCHER_PORT="${PATCHER_PORT:-8000}"
NODE_NAME="$(hostname -f)"

cat <<EOF
Patcher server starting on compute node:
  $NODE_NAME

From your local machine, run:
  ssh -N -L ${PATCHER_PORT}:${NODE_NAME}:${PATCHER_PORT} hypatia

Then open:
  http://localhost:${PATCHER_PORT}/?annotator=annotator_1
  http://localhost:${PATCHER_PORT}/?annotator=annotator_2
  http://localhost:${PATCHER_PORT}/?annotator=annotator_3
EOF

cd "$REPO_ROOT"
PATCHER_PORT="$PATCHER_PORT" scripts/hypatia_run_patcher.sh
