#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PATCH_ENCODER_BUILD_PARTITION:-small-long}"
TIME_LIMIT="${PATCH_ENCODER_BUILD_TIME_LIMIT:-1-06:00:00}"
CPUS="${PATCH_ENCODER_BUILD_CPUS:-8}"
MEMORY="${PATCH_ENCODER_BUILD_MEMORY:-64G}"
CONCURRENCY="${PATCH_ENCODER_BUILD_CONCURRENCY:-24}"
PYTHON_BIN="${PATCH_ENCODER_PYTHON_BIN:-/sharedscratch/dm236/.venvs/histokit-resnet-train-cu12/bin/python}"
DATA_ROOT="${ICAIRD_DATA_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
OUTPUT_ROOT="${PATCH_ENCODER_OUTPUT_ROOT:-${DATA_ROOT}/bags/full_canny005_or_resnet_light_grandqc_artifact}"
LOG_ROOT="${DATA_ROOT}/slurm_logs"
HISTOKIT_REPO="${HISTOKIT_REPO:-${HOME}/repos/histokit}"
CHUNK_DIR="${PATCH_ENCODER_BUILD_CHUNK_DIR:-${OUTPUT_ROOT}/chunks/build}"

mkdir -p "$LOG_ROOT"

chunk_count="$(find "$CHUNK_DIR" -maxdepth 1 -name 'chunk_*.csv' | wc -l)"
if [[ "$chunk_count" -lt 1 ]]; then
  echo "No build chunks found in $CHUNK_DIR. Run inventory first." >&2
  exit 1
fi
last_task=$((chunk_count - 1))

dependency_args=()
if [[ -n "${PATCH_ENCODER_DEPENDENCY_JOB_ID:-}" ]]; then
  dependency_args=(--dependency="afterok:${PATCH_ENCODER_DEPENDENCY_JOB_ID}")
fi

sbatch --parsable \
  "${dependency_args[@]}" \
  --partition="$PARTITION" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT" \
  --array="0-${last_task}%${CONCURRENCY}" \
  --job-name=icaird-pe-build \
  --output="${LOG_ROOT}/icaird_patch_encoder_build_%A_%a.out" \
  --error="${LOG_ROOT}/icaird_patch_encoder_build_%A_%a.err" \
  --wrap="chunk=\$(printf '${CHUNK_DIR}/chunk_%04d.csv' \${SLURM_ARRAY_TASK_ID}); cd ${HISTOKIT_REPO} && PYTHONPATH=\$PWD/src ${PYTHON_BIN} scripts/build_icaird_cervical_patch_encoder_dataset.py --mode build-slide --output-root ${OUTPUT_ROOT} --chunk-csv \${chunk} $*"
