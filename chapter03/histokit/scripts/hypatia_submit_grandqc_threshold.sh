#!/usr/bin/env bash
set -euo pipefail

PARTITION="${GRANDQC_PARTITION:-gpu.L40S}"
TIME_LIMIT="${GRANDQC_TIME_LIMIT:-1-00:00:00}"
CPUS="${GRANDQC_CPUS:-8}"
MEMORY="${GRANDQC_MEMORY:-64G}"
GRES="${GRANDQC_GRES:-gpu:1}"
PYTHON_BIN="${GRANDQC_PYTHON_BIN:-/sharedscratch/dm236/.venvs/histokit-resnet-train-cu12/bin/python}"
DATA_ROOT="${ICAIRD_DATA_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
LOG_ROOT="${DATA_ROOT}/slurm_logs"
OUTPUT_DIR="${GRANDQC_OUTPUT_DIR:-${DATA_ROOT}/segmentation_thresholds/grandqc_train}"
MODEL_PATH="${GRANDQC_MODEL_PATH:-${DATA_ROOT}/grandqc_weights/Tissue_Detection_MPP10.pth}"

mkdir -p "$LOG_ROOT"

cd "$(dirname "$0")/.."

sbatch --parsable \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT" \
  --job-name=grandqc-thresh \
  --output="${LOG_ROOT}/grandqc_threshold_%j.out" \
  --error="${LOG_ROOT}/grandqc_threshold_%j.err" \
  --wrap="cd ~/repos/histokit && PYTHONPATH=\$PWD/src HISTOKIT_DEVICE=cuda ${PYTHON_BIN} scripts/tune_grandqc_tissue_threshold.py --output-dir ${OUTPUT_DIR} --model-path ${MODEL_PATH} $*"
