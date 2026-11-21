#!/usr/bin/env bash
set -euo pipefail

PARTITION="${ICAIRD_BAGS_RESNET_PARTITION:-gpu.L40S}"
TIME_LIMIT="${ICAIRD_BAGS_RESNET_TIME_LIMIT:-1-00:00:00}"
CPUS="${ICAIRD_BAGS_RESNET_CPUS:-8}"
MEMORY="${ICAIRD_BAGS_RESNET_MEMORY:-64G}"
GRES="${ICAIRD_BAGS_RESNET_GRES:-gpu:1}"
CONCURRENCY="${ICAIRD_BAGS_RESNET_CONCURRENCY:-6}"
EXCLUDE="${ICAIRD_BAGS_RESNET_EXCLUDE:-}"
PYTHON_BIN="${PATCH_ENCODER_PYTHON_BIN:-/sharedscratch/dm236/.venvs/histokit-resnet-train-cu12/bin/python}"
DATA_ROOT="${ICAIRD_DATA_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
OUTPUT_ROOT="${PATCH_ENCODER_OUTPUT_ROOT:-${DATA_ROOT}/bags/full_canny005_or_resnet_light_grandqc_artifact}"
LOG_ROOT="${DATA_ROOT}/slurm_logs"
HISTOKIT_REPO="${HISTOKIT_REPO:-${HOME}/repos/histokit}"
CHUNK_DIR="${ICAIRD_BAGS_RESNET_CHUNK_DIR:-${OUTPUT_ROOT}/chunks/resnet}"
RESNET_DIR="${ICAIRD_BAGS_RESNET_DIR:-${DATA_ROOT}/resnet_patch_subset/stratified_120_consensus_100k/resnet19_patch_classifier}"

mkdir -p "$LOG_ROOT"

chunk_count="$(find "$CHUNK_DIR" -maxdepth 1 -name 'chunk_*.csv' | wc -l)"
if [[ "$chunk_count" -lt 1 ]]; then
  echo "No ResNet chunks found in $CHUNK_DIR. Run inventory first." >&2
  exit 1
fi
last_task=$((chunk_count - 1))

dependency_args=()
if [[ -n "${ICAIRD_BAGS_RESNET_DEPENDENCY_JOB_ID:-}" ]]; then
  dependency_args=(--dependency="afterok:${ICAIRD_BAGS_RESNET_DEPENDENCY_JOB_ID}")
fi
exclude_args=()
if [[ -n "$EXCLUDE" ]]; then
  exclude_args=(--exclude="$EXCLUDE")
fi

sbatch --parsable \
  "${dependency_args[@]}" \
  "${exclude_args[@]}" \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT" \
  --array="0-${last_task}%${CONCURRENCY}" \
  --job-name=icaird-bag-rn \
  --output="${LOG_ROOT}/icaird_bags_resnet_%A_%a.out" \
  --error="${LOG_ROOT}/icaird_bags_resnet_%A_%a.err" \
  --wrap="chunk=\$(printf '${CHUNK_DIR}/chunk_%04d.csv' \${SLURM_ARRAY_TASK_ID}); cd ${HISTOKIT_REPO} && PYTHONPATH=\$PWD/src ${PYTHON_BIN} scripts/build_icaird_cervical_patch_encoder_dataset.py --mode resnet-score --output-root ${OUTPUT_ROOT} --chunk-csv \${chunk} --resnet-dir ${RESNET_DIR} --device cuda $*"
