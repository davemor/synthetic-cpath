#!/usr/bin/env bash
set -euo pipefail

IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
RUN_ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
SOURCE_ROOT="${SYNPATH_UTILITY_SOURCE_ROOT:-}"
ENV_FILE="${SYNPATH_DGX_ENV_FILE:-${RUN_ROOT}/dgx.env}"
EXPERIMENT_CONFIG="${SYNPATH_EXPERIMENT_CONFIG:-configs/experiments/main_pathmnist224_budget.yaml}"
DATASET="${SYNPATH_UTILITY_DATASET:-pathmnist224}"
GPUS="${SYNPATH_UTILITY_GPUS:-8}"
CPU_WORKERS="${SYNPATH_UTILITY_NUM_WORKERS:-}"
SYNTHETIC_MODELS="${SYNPATH_UTILITY_SYNTHETIC_MODELS:-stylegan2ada sd_lora}"

DEFAULT_SOURCES="real_only synthetic_only_true_label real_plus_synthetic_true_label synthetic_only_label_global_cluster real_plus_synthetic_label_global_cluster synthetic_only_label_local_cluster real_plus_synthetic_label_local_cluster"
read -r -a SOURCES <<<"${SYNPATH_UTILITY_SOURCES:-${DEFAULT_SOURCES}}"
SEEDS=(0 1 2)

mkdir -p "${RUN_ROOT}/logs"

SOURCE_MOUNT_ARGS=()
if [[ -n "${SOURCE_ROOT}" && "${SOURCE_ROOT}" != "${RUN_ROOT}" ]]; then
    SOURCE_MOUNT_ARGS=(-v "${SOURCE_ROOT}:${SOURCE_ROOT}:ro")
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing env file: ${ENV_FILE}" >&2
    exit 1
fi

if [[ -z "${CPU_WORKERS}" ]]; then
    CPU_TOTAL="$(nproc)"
    CPU_WORKERS="$((CPU_TOTAL / GPUS))"
    if [[ "${CPU_WORKERS}" -lt 2 ]]; then
        CPU_WORKERS=2
    fi
fi

run_worker() {
    local gpu="$1"
    local name="synpath-utility-gpu${gpu}"
    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run \
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
        "${SOURCE_MOUNT_ARGS[@]}" \
        -v "${PWD}:/workspace/synpath" \
        -w /workspace/synpath \
        "${IMAGE}" \
        bash scripts/docker/dgx_entrypoint.sh bash -lc "$(worker_command "${gpu}")"
}

worker_command() {
    local gpu="$1"
    cat <<EOF
set -euo pipefail
sources=(${SOURCES[*]})
seeds=(${SEEDS[*]})
synthetic_models=(${SYNTHETIC_MODELS})
idx=0
for source in "\${sources[@]}"; do
    if [[ "\${source}" == real_only* ]]; then
        for seed in "\${seeds[@]}"; do
            if (( idx % ${GPUS} == ${gpu} )); then
                echo "running utility gpu=${gpu} source=\${source} seed=\${seed}"
                uv run --extra gpu synpath train-utility "${DATASET}" "\${source}" --seed "\${seed}" --config "${EXPERIMENT_CONFIG}" --resume
                echo "completed utility gpu=${gpu} source=\${source} seed=\${seed}"
            fi
            idx=\$((idx + 1))
        done
    else
        for synthetic_model in "\${synthetic_models[@]}"; do
            for seed in "\${seeds[@]}"; do
                if (( idx % ${GPUS} == ${gpu} )); then
                    echo "running utility gpu=${gpu} model=\${synthetic_model} source=\${source} seed=\${seed}"
                    uv run --extra gpu synpath train-utility "${DATASET}" "\${source}" --synthetic-model "\${synthetic_model}" --seed "\${seed}" --config "${EXPERIMENT_CONFIG}" --resume
                    echo "completed utility gpu=${gpu} model=\${synthetic_model} source=\${source} seed=\${seed}"
                fi
                idx=\$((idx + 1))
            done
        done
    fi
done
EOF
}

pids=()
for gpu in $(seq 0 "$((GPUS - 1))"); do
    run_worker "${gpu}" >"${RUN_ROOT}/logs/utility_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if [[ "${status}" -eq 0 ]]; then
    docker rm -f synpath-utility-report >/dev/null 2>&1 || true
    docker run \
        --name synpath-utility-report \
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
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        "${SOURCE_MOUNT_ARGS[@]}" \
        -v "${PWD}:/workspace/synpath" \
        -w /workspace/synpath \
        "${IMAGE}" \
        bash scripts/docker/dgx_entrypoint.sh uv run --extra gpu synpath report --config "${EXPERIMENT_CONFIG}"
fi

exit "${status}"
