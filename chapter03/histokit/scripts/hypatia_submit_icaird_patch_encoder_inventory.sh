#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PATCH_ENCODER_INVENTORY_PARTITION:-small-short}"
TIME_LIMIT="${PATCH_ENCODER_INVENTORY_TIME_LIMIT:-02:00:00}"
CPUS="${PATCH_ENCODER_INVENTORY_CPUS:-2}"
MEMORY="${PATCH_ENCODER_INVENTORY_MEMORY:-8G}"
PYTHON_BIN="${PATCH_ENCODER_PYTHON_BIN:-/sharedscratch/dm236/.venvs/histokit-resnet-train-cu12/bin/python}"
DATA_ROOT="${ICAIRD_DATA_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
OUTPUT_ROOT="${PATCH_ENCODER_OUTPUT_ROOT:-${DATA_ROOT}/bags/full_canny005_or_resnet_light_grandqc_artifact}"
LOG_ROOT="${DATA_ROOT}/slurm_logs"
HISTOKIT_REPO="${HISTOKIT_REPO:-${HOME}/repos/histokit}"

mkdir -p "$LOG_ROOT"

sbatch --parsable \
  --partition="$PARTITION" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT" \
  --job-name=icaird-pe-inv \
  --output="${LOG_ROOT}/icaird_patch_encoder_inventory_%j.out" \
  --error="${LOG_ROOT}/icaird_patch_encoder_inventory_%j.err" \
  --wrap="cd ${HISTOKIT_REPO} && PYTHONPATH=\$PWD/src ${PYTHON_BIN} scripts/build_icaird_cervical_patch_encoder_dataset.py --mode inventory --output-root ${OUTPUT_ROOT} $*"
