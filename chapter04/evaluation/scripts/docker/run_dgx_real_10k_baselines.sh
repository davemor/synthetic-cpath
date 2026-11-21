#!/usr/bin/env bash
set -euo pipefail

IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
RUN_ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
ENV_FILE="${SYNPATH_DGX_ENV_FILE:-${RUN_ROOT}/dgx.env}"
GPU_COUNT="${SYNPATH_UTILITY_GPUS:-8}"
CPU_WORKERS="${SYNPATH_UTILITY_NUM_WORKERS:-$(( $(nproc) / GPU_COUNT ))}"
LOG_ROOT="${RUN_ROOT}/logs/real_only_10k"

DATASETS=(
    "pathmnist224:configs/experiments/main_pathmnist224_budget.yaml"
    "icaird_cervical100k:configs/experiments/main_icaird_cervical100k_budget.yaml"
    "pcam:configs/experiments/main_pcam_budget.yaml"
)
SEEDS=(0 1 2)

mkdir -p "${LOG_ROOT}"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing env file: ${ENV_FILE}" >&2
    exit 1
fi

run_job() {
    local gpu="$1"
    local dataset="$2"
    local config="$3"
    local seed="$4"
    local name="synpath-real10k-${dataset}-seed${seed}"
    local log="${LOG_ROOT}/${dataset}_seed${seed}.log"

    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run --rm \
        --name "${name}" \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --env-file "${ENV_FILE}" \
        -e SYNPATH_RUN_ROOT="${RUN_ROOT}" \
        -e SYNPATH_DATA_ROOT="${RUN_ROOT}/data" \
        -e SYNPATH_CACHE_ROOT="${RUN_ROOT}/cache" \
        -e SYNPATH_OUTPUTS_ROOT="${RUN_ROOT}/outputs" \
        -e SYNPATH_REPORTS_ROOT="${RUN_ROOT}/reports" \
        -e CUDA_VISIBLE_DEVICES="${gpu}" \
        -e SYNPATH_UTILITY_NUM_WORKERS="${CPU_WORKERS}" \
        -e OMP_NUM_THREADS=2 \
        -e MKL_NUM_THREADS=2 \
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        -v /raid/dm236/data:/raid/dm236/data:ro \
        -v "${PWD}:/workspace/synpath" \
        -w /workspace/synpath \
        "${IMAGE}" \
        bash scripts/docker/dgx_entrypoint.sh \
        uv run --extra gpu synpath train-utility \
        "${dataset}" real_only_10k --seed "${seed}" \
        --config "${config}" --resume >"${log}" 2>&1
}

jobs=()
for dataset_config in "${DATASETS[@]}"; do
    dataset="${dataset_config%%:*}"
    config="${dataset_config#*:}"
    for seed in "${SEEDS[@]}"; do
        jobs+=("${dataset}:${config}:${seed}")
    done
done

pids=()
for gpu in $(seq 0 "$((GPU_COUNT - 1))"); do
    (
        for index in "${!jobs[@]}"; do
            if (( index % GPU_COUNT != gpu )); then
                continue
            fi
            IFS=: read -r dataset config seed <<<"${jobs[index]}"
            echo "Starting ${dataset} seed ${seed} on GPU ${gpu}"
            run_job "${gpu}" "${dataset}" "${config}" "${seed}"
            echo "Completed ${dataset} seed ${seed} on GPU ${gpu}"
        done
    ) &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

exit "${status}"
