#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
DATASET_ROOT="${DATASET_ROOT:-/home/dm236/scratch/data/icaird_cervical/icaird_cervical_120k_trainonly_slide_split_uni2_labelbalanced_kcenter}"
MODELS="${MODELS:-uni uni2_h dinov2-large hibou-l manual_features}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-8}"
GPU_PARTITION="${GPU_PARTITION:-gpu.L40S}"
CPU_PARTITION="${CPU_PARTITION:-large-long}"
GPU_TIME="${GPU_TIME:-24:00:00}"
CPU_TIME="${CPU_TIME:-12:00:00}"
MERGE_TIME="${MERGE_TIME:-02:00:00}"
GPU_CPUS_PER_TASK="${GPU_CPUS_PER_TASK:-8}"
CPU_CPUS_PER_TASK="${CPU_CPUS_PER_TASK:-4}"
GPU_MEM="${GPU_MEM:-64G}"
CPU_MEM="${CPU_MEM:-32G}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
GPU_EXCLUDE_NODES="${GPU_EXCLUDE_NODES:-}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
LOG_DIR="${LOG_DIR:-${DATASET_ROOT}/embeddings/logs}"
HF_HOME="${HF_HOME:-/sharedscratch/dm236/.cache/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
UV_RUN_PREFIX="${UV_RUN_PREFIX:-uv run --extra embed}"
PYTHON_BIN="${PYTHON_BIN:-}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"

if [[ -n "${EXTRA_PYTHONPATH}" ]]; then
  RUN_PYTHONPATH="${REPO_ROOT}/src:${EXTRA_PYTHONPATH}"
else
  RUN_PYTHONPATH="${REPO_ROOT}/src"
fi

run_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    PYTHONPATH="${RUN_PYTHONPATH}" "${PYTHON_BIN}" "$@"
  else
    ${UV_RUN_PREFIX} python "$@"
  fi
}

mkdir -p "${LOG_DIR}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

model_output_name() {
  case "$1" in
    uni) echo "uni" ;;
    uni2|uni2_h) echo "uni2_h" ;;
    dinov2|dinov2-large|dinov2_large) echo "dinov2_large" ;;
    hibou|hibou-l|hibou_l) echo "hibou_l" ;;
    manual_features) echo "manual_features" ;;
    *) echo "$1" | tr '-' '_' ;;
  esac
}

is_manual_model() {
  [[ "$1" == "manual_features" ]]
}

cd "${REPO_ROOT}"

for MODEL in ${MODELS}; do
  OUTPUT_NAME="$(model_output_name "${MODEL}")"
  OUTPUT_ROOT="${DATASET_ROOT}/embeddings/${OUTPUT_NAME}"

  run_python scripts/embed_icaird_120k_manifest.py \
    --mode chunks \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --model "${MODEL}" \
    --chunk-size "${CHUNK_SIZE}"

  NUM_CHUNKS="$(($(wc -l < "${OUTPUT_ROOT}/chunk_index.csv") - 1))"
  if [[ "${NUM_CHUNKS}" -lt 1 ]]; then
    echo "No chunks found for ${MODEL}" >&2
    exit 1
  fi
  ARRAY_END=$((NUM_CHUNKS - 1))

  if is_manual_model "${MODEL}"; then
    PARTITION="${CPU_PARTITION}"
    TIME="${CPU_TIME}"
    CPUS_PER_TASK="${CPU_CPUS_PER_TASK}"
    MEM="${CPU_MEM}"
    GRES_ARGS=()
    EXCLUDE_ARGS=()
    RUN_DEVICE="cpu"
    JOB_NAME="icaird120k-manual"
  else
    PARTITION="${GPU_PARTITION}"
    TIME="${GPU_TIME}"
    CPUS_PER_TASK="${GPU_CPUS_PER_TASK}"
    MEM="${GPU_MEM}"
    GRES_ARGS=(--gres=gpu:1)
    if [[ -n "${GPU_EXCLUDE_NODES}" ]]; then
      EXCLUDE_ARGS=(--exclude="${GPU_EXCLUDE_NODES}")
    else
      EXCLUDE_ARGS=()
    fi
    RUN_DEVICE="${DEVICE}"
    JOB_NAME="icaird120k-${OUTPUT_NAME}"
  fi

  ARRAY_JOB_ID="$(
    sbatch \
      --parsable \
      --job-name="${JOB_NAME}" \
      --partition="${PARTITION}" \
      "${GRES_ARGS[@]}" \
      "${EXCLUDE_ARGS[@]}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --mem="${MEM}" \
      --time="${TIME}" \
      --array="0-${ARRAY_END}%${ARRAY_CONCURRENCY}" \
      --output="${LOG_DIR}/%x_%A_%a.out" \
      --error="${LOG_DIR}/%x_%A_%a.err" \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",DATASET_ROOT="${DATASET_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",MODEL="${MODEL}",BATCH_SIZE="${BATCH_SIZE}",RUN_DEVICE="${RUN_DEVICE}",ENV_FILE="${ENV_FILE}",HF_HOME="${HF_HOME}",HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}",TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}",UV_RUN_PREFIX="${UV_RUN_PREFIX}",PYTHON_BIN="${PYTHON_BIN}",RUN_PYTHONPATH="${RUN_PYTHONPATH}" \
      --wrap='set -euo pipefail
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
cd "${REPO_ROOT}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  export PYTHONPATH="${RUN_PYTHONPATH}"
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=(uv run --extra embed python)
fi
"${PYTHON_CMD[@]}" scripts/embed_icaird_120k_manifest.py \
  --mode embed \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --model "${MODEL}" \
  --chunk-id "${SLURM_ARRAY_TASK_ID}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${RUN_DEVICE}" \
  --env-file "${ENV_FILE}"'
  )"

  MERGE_JOB_ID="$(
    sbatch \
      --parsable \
      --dependency="afterok:${ARRAY_JOB_ID}" \
      --job-name="merge-${OUTPUT_NAME}" \
      --partition="${CPU_PARTITION}" \
      --cpus-per-task=2 \
      --mem="${CPU_MEM}" \
      --time="${MERGE_TIME}" \
      --output="${LOG_DIR}/%x_%j.out" \
      --error="${LOG_DIR}/%x_%j.err" \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",DATASET_ROOT="${DATASET_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",MODEL="${MODEL}",UV_RUN_PREFIX="${UV_RUN_PREFIX}",PYTHON_BIN="${PYTHON_BIN}",RUN_PYTHONPATH="${RUN_PYTHONPATH}" \
      --wrap='set -euo pipefail
cd "${REPO_ROOT}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  export PYTHONPATH="${RUN_PYTHONPATH}"
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=(uv run --extra embed python)
fi
"${PYTHON_CMD[@]}" scripts/embed_icaird_120k_manifest.py \
  --mode summarize \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --model "${MODEL}"'
  )"

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${MODEL}" "${OUTPUT_ROOT}" "${ARRAY_JOB_ID}" "${MERGE_JOB_ID}" "${LOG_DIR}"
done
