#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PATCH_ENCODER_SUMMARY_PARTITION:-small-short}"
TIME_LIMIT="${PATCH_ENCODER_SUMMARY_TIME_LIMIT:-02:00:00}"
CPUS="${PATCH_ENCODER_SUMMARY_CPUS:-4}"
MEMORY="${PATCH_ENCODER_SUMMARY_MEMORY:-32G}"
PYTHON_BIN="${PATCH_ENCODER_PYTHON_BIN:-/sharedscratch/dm236/.venvs/histokit-resnet-train-cu12/bin/python}"
DATA_ROOT="${ICAIRD_DATA_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
OUTPUT_ROOT="${PATCH_ENCODER_OUTPUT_ROOT:-${DATA_ROOT}/bags/full_canny005_or_resnet_light_grandqc_artifact}"
LOG_ROOT="${DATA_ROOT}/slurm_logs"
HISTOKIT_REPO="${HISTOKIT_REPO:-${HOME}/repos/histokit}"

mkdir -p "$LOG_ROOT"

dependency_args=()
if [[ -n "${PATCH_ENCODER_BUILD_JOB_ID:-}" ]]; then
  dependency_args=(--dependency="afterok:${PATCH_ENCODER_BUILD_JOB_ID}")
fi

sbatch --parsable \
  "${dependency_args[@]}" \
  --partition="$PARTITION" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT" \
  --job-name=icaird-pe-sum \
  --output="${LOG_ROOT}/icaird_patch_encoder_summarize_%j.out" \
  --error="${LOG_ROOT}/icaird_patch_encoder_summarize_%j.err" \
  --wrap="cd ${HISTOKIT_REPO} && PYTHONPATH=\$PWD/src ${PYTHON_BIN} scripts/build_icaird_cervical_patch_encoder_dataset.py --mode summarize --output-root ${OUTPUT_ROOT} $*"
