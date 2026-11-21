#!/usr/bin/env bash
set -euo pipefail

PARTITION="${ICAIRD_BAGS_ARTIFACT_SCORE_PARTITION:-small-long}"
TIME_LIMIT="${ICAIRD_BAGS_ARTIFACT_SCORE_TIME_LIMIT:-12:00:00}"
CPUS="${ICAIRD_BAGS_ARTIFACT_SCORE_CPUS:-4}"
MEMORY="${ICAIRD_BAGS_ARTIFACT_SCORE_MEMORY:-32G}"
CONCURRENCY="${ICAIRD_BAGS_ARTIFACT_SCORE_CONCURRENCY:-24}"
PYTHON_BIN="${PATCH_ENCODER_PYTHON_BIN:-/sharedscratch/dm236/.venvs/histokit-resnet-train-cu12/bin/python}"
DATA_ROOT="${ICAIRD_DATA_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
OUTPUT_ROOT="${PATCH_ENCODER_OUTPUT_ROOT:-${DATA_ROOT}/bags/full_canny005_or_resnet_light_grandqc_artifact}"
LOG_ROOT="${DATA_ROOT}/slurm_logs"
HISTOKIT_REPO="${HISTOKIT_REPO:-${HOME}/repos/histokit}"
CHUNK_DIR="${ICAIRD_BAGS_ARTIFACT_SCORE_CHUNK_DIR:-${OUTPUT_ROOT}/chunks/grandqc_artifact}"
OFFICIAL_GRANDQC_OUTPUT_ROOT="${ICAIRD_BAGS_OFFICIAL_GRANDQC_OUTPUT_ROOT:-${OUTPUT_ROOT}/grandqc_artifact_masks}"

mkdir -p "$LOG_ROOT"

chunk_count="$(find "$CHUNK_DIR" -maxdepth 1 -name 'chunk_*.csv' | wc -l)"
if [[ "$chunk_count" -lt 1 ]]; then
  echo "No official GrandQC artifact chunks found in $CHUNK_DIR. Run inventory first." >&2
  exit 1
fi
last_task=$((chunk_count - 1))

dependency_args=()
if [[ -n "${ICAIRD_BAGS_ARTIFACT_SCORE_DEPENDENCY_JOB_ID:-}" ]]; then
  dependency_args=(--dependency="afterok:${ICAIRD_BAGS_ARTIFACT_SCORE_DEPENDENCY_JOB_ID}")
fi

sbatch --parsable \
  "${dependency_args[@]}" \
  --partition="$PARTITION" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT" \
  --array="0-${last_task}%${CONCURRENCY}" \
  --job-name=icaird-bag-as \
  --output="${LOG_ROOT}/icaird_bags_artifact_score_%A_%a.out" \
  --error="${LOG_ROOT}/icaird_bags_artifact_score_%A_%a.err" \
  --wrap="chunk=\$(printf '${CHUNK_DIR}/chunk_%04d.csv' \${SLURM_ARRAY_TASK_ID}); cd ${HISTOKIT_REPO} && PYTHONPATH=\$PWD/src ${PYTHON_BIN} scripts/build_icaird_cervical_patch_encoder_dataset.py --mode official-artifact-score --output-root ${OUTPUT_ROOT} --chunk-csv \${chunk} --official-grandqc-output-root ${OFFICIAL_GRANDQC_OUTPUT_ROOT} $*"
