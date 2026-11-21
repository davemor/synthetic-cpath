#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PATCH_ENCODER_GRANDQC_PARTITION:-gpu.A100}"
TIME_LIMIT="${PATCH_ENCODER_GRANDQC_TIME_LIMIT:-1-00:00:00}"
CPUS="${PATCH_ENCODER_GRANDQC_CPUS:-8}"
MEMORY="${PATCH_ENCODER_GRANDQC_MEMORY:-64G}"
GRES="${PATCH_ENCODER_GRANDQC_GRES:-gpu:1}"
CONCURRENCY="${PATCH_ENCODER_GRANDQC_CONCURRENCY:-6}"
PYTHON_BIN="${PATCH_ENCODER_PYTHON_BIN:-/sharedscratch/dm236/.venvs/histokit-resnet-train-cu12/bin/python}"
DATA_ROOT="${ICAIRD_DATA_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
OUTPUT_ROOT="${PATCH_ENCODER_OUTPUT_ROOT:-${DATA_ROOT}/bags/full_canny005_or_resnet_light_grandqc_artifact}"
LOG_ROOT="${DATA_ROOT}/slurm_logs"
HISTOKIT_REPO="${HISTOKIT_REPO:-${HOME}/repos/histokit}"
CHUNK_DIR="${OUTPUT_ROOT}/chunks/grandqc"
MODEL_PATH="${PATCH_ENCODER_TISSUE_MODEL_PATH:-${DATA_ROOT}/grandqc_weights/Tissue_Detection_MPP10.pth}"
ARTIFACT_MODEL_PATH="${PATCH_ENCODER_ARTIFACT_MODEL_PATH:-${DATA_ROOT}/grandqc_weights/GrandQC_MPP15.pth}"

mkdir -p "$LOG_ROOT"

chunk_count="$(find "$CHUNK_DIR" -maxdepth 1 -name 'chunk_*.csv' | wc -l)"
if [[ "$chunk_count" -lt 1 ]]; then
  echo "No GrandQC chunks found in $CHUNK_DIR. Run inventory first." >&2
  exit 1
fi
last_task=$((chunk_count - 1))
download_flag=""
if [[ "${PATCH_ENCODER_DOWNLOAD_ARTIFACT_MODEL:-0}" == "1" ]]; then
  download_flag="--download-artifact-model-if-missing"
fi

sbatch --parsable \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT" \
  --array="0-${last_task}%${CONCURRENCY}" \
  --job-name=icaird-pe-gqc \
  --output="${LOG_ROOT}/icaird_patch_encoder_grandqc_%A_%a.out" \
  --error="${LOG_ROOT}/icaird_patch_encoder_grandqc_%A_%a.err" \
  --wrap="chunk=\$(printf '${CHUNK_DIR}/chunk_%04d.csv' \${SLURM_ARRAY_TASK_ID}); cd ${HISTOKIT_REPO} && PYTHONPATH=\$PWD/src HISTOKIT_DEVICE=cuda HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ${PYTHON_BIN} scripts/build_icaird_cervical_patch_encoder_dataset.py --mode grandqc-score --output-root ${OUTPUT_ROOT} --chunk-csv \${chunk} --model-path ${MODEL_PATH} --artifact-model-path ${ARTIFACT_MODEL_PATH} ${download_flag} $*"
