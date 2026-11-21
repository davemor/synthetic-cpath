#!/usr/bin/env bash
set -euo pipefail

IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
RUN_ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
DATA_MOUNT="${SYNPATH_DGX_DATA_MOUNT:-/raid/dm236/data}"
ENV_FILE="${SYNPATH_DGX_ENV_FILE:-${RUN_ROOT}/dgx.env}"
EXPERIMENT_CONFIG="${SYNPATH_EXPERIMENT_CONFIG:-configs/experiments/main_pcam_budget.yaml}"
GPUS="${SYNPATH_EVALUATION_GPUS:-8}"
CPU_TOTAL="$(nproc)"
CPU_WORKERS="$((CPU_TOTAL / GPUS))"

if [[ "${CPU_WORKERS}" -lt 2 ]]; then
    CPU_WORKERS=2
fi

mkdir -p "${RUN_ROOT}/logs"

pids=()
for gpu in $(seq 0 "$((GPUS - 1))"); do
    name="synpath-evaluation-additions-gpu${gpu}"
    log="${RUN_ROOT}/logs/evaluation_additions_gpu${gpu}.log"
    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run \
        --name "${name}" \
        --gpus all \
        --ipc=host \
        --env-file "${ENV_FILE}" \
        -e CUDA_VISIBLE_DEVICES="${gpu}" \
        -e OMP_NUM_THREADS=2 \
        -e MKL_NUM_THREADS=2 \
        -e SYNPATH_INCEPTION_NUM_WORKERS="${CPU_WORKERS}" \
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        -v "${DATA_MOUNT}:${DATA_MOUNT}:ro" \
        -w "${RUN_ROOT}/repo" \
        "${IMAGE}" \
        uv run --extra gpu \
        python scripts/run/recompute_evaluation_additions.py \
        --config "${EXPERIMENT_CONFIG}" \
        --num-shards "${GPUS}" \
        --shard-index "${gpu}" >"${log}" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

exit "${status}"
