#!/usr/bin/env bash
set -euo pipefail

PARTITION="${SLIDE_TISSUE_PARTITION:-small-short}"
TIME_LIMIT="${SLIDE_TISSUE_TIME_LIMIT:-02:00:00}"
CPUS="${SLIDE_TISSUE_CPUS:-8}"
MEMORY="${SLIDE_TISSUE_MEMORY:-32G}"
PYTHON_BIN="${SLIDE_TISSUE_PYTHON_BIN:-/sharedscratch/dm236/.venvs/histokit-resnet-train-cu12/bin/python}"
DATA_ROOT="${ICAIRD_DATA_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
LOG_ROOT="${DATA_ROOT}/slurm_logs"

mkdir -p "$LOG_ROOT"

cd "$(dirname "$0")/.."

sbatch --parsable \
  --partition="$PARTITION" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT" \
  --job-name=slide-tissue \
  --output="${LOG_ROOT}/slide_tissue_%j.out" \
  --error="${LOG_ROOT}/slide_tissue_%j.err" \
  --wrap="cd ~/repos/histokit && PYTHONPATH=\$PWD/src ${PYTHON_BIN} scripts/tune_slide_tissue_detector_threshold.py $*"
