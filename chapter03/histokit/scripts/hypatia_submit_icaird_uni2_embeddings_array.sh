#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
BAGS_ROOT="${BAGS_ROOT:-/sharedscratch/dm236/data/icaird_cervical/bags/full_canny005_or_resnet_light_grandqc_artifact}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BAGS_ROOT}/embeddings/uni2_h}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-8}"
PARTITION="${PARTITION:-gpu.L40S}"
TIME="${TIME:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
PYTHON_BIN="${PYTHON_BIN:-/sharedscratch/dm236/.venvs/histokit-resnet-train-cu12/bin/python}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"
HF_HOME="${HF_HOME:-/sharedscratch/dm236/.cache/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
if [[ -n "${EXTRA_PYTHONPATH}" ]]; then
  RUN_PYTHONPATH="${REPO_ROOT}/src:${EXTRA_PYTHONPATH}"
else
  RUN_PYTHONPATH="${REPO_ROOT}/src"
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

PYTHONPATH="${RUN_PYTHONPATH}" "${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/embed_icaird_cervical_uni2.py" \
  --mode chunks \
  --bags-root "${BAGS_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --chunk-size "${CHUNK_SIZE}"

CHUNKS_DIR="${OUTPUT_ROOT}/chunks"
NUM_CHUNKS="$(find "${CHUNKS_DIR}" -name 'chunk_*.csv' | wc -l | tr -d ' ')"
if [[ "${NUM_CHUNKS}" -lt 1 ]]; then
  echo "No chunk CSVs found in ${CHUNKS_DIR}" >&2
  exit 1
fi

ARRAY_END=$((NUM_CHUNKS - 1))
SBATCH_NODE_ARGS=()
if [[ -n "${EXCLUDE_NODES}" ]]; then
  SBATCH_NODE_ARGS+=(--exclude="${EXCLUDE_NODES}")
fi

sbatch \
  "${SBATCH_NODE_ARGS[@]}" \
  --job-name=icaird-uni2 \
  --partition="${PARTITION}" \
  --gres=gpu:1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --array="0-${ARRAY_END}%${ARRAY_CONCURRENCY}" \
  --output="${LOG_DIR}/%x_%A_%a.out" \
  --error="${LOG_DIR}/%x_%A_%a.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",BAGS_ROOT="${BAGS_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",CHUNKS_DIR="${CHUNKS_DIR}",BATCH_SIZE="${BATCH_SIZE}",DEVICE="${DEVICE}",ENV_FILE="${ENV_FILE}",PYTHON_BIN="${PYTHON_BIN}",RUN_PYTHONPATH="${RUN_PYTHONPATH}",HF_HOME="${HF_HOME}",HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}",TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  --wrap='set -euo pipefail
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
cd "${REPO_ROOT}"
CHUNK_CSV="$(printf "%s/chunk_%05d.csv" "${CHUNKS_DIR}" "${SLURM_ARRAY_TASK_ID}")"
PYTHONPATH="${RUN_PYTHONPATH}" "${PYTHON_BIN}" scripts/embed_icaird_cervical_uni2.py \
  --mode embed \
  --bags-root "${BAGS_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --chunk-csv "${CHUNK_CSV}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --env-file "${ENV_FILE}"'
