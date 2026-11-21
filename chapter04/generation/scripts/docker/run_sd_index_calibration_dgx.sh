#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
RAPIDS_IMAGE="${SYNPATH_RAPIDS_IMAGE:-rapidsai/base:26.06-cuda12-py3.12}"
RUN_ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
DATA_MOUNT="${SYNPATH_DGX_DATA_MOUNT:-/raid/dm236/data}"
BALANCED_ROOT="${SYNPATH_BALANCED_CLUSTER_ROOT:-/raid/dm236/synpath_balanced_global_clusters}"
ENV_FILE="${SYNPATH_DGX_ENV_FILE:-${RUN_ROOT}/dgx.env}"
CONFIG="${SYNPATH_SD_INDEX_CONFIG:-configs/experiments/sd_index_calibration.yaml}"
LOG_ROOT="${RUN_ROOT}/logs/cluster_sd_index_calibration"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing env file: ${ENV_FILE}" >&2
    exit 1
fi
if [[ ! "${MODE}" =~ ^(all|prepare|embed|fit|rescore|report)$ ]]; then
    echo "Mode must be one of: all, prepare, embed, fit, rescore, report" >&2
    exit 1
fi

mkdir -p "${LOG_ROOT}"
available_kb="$(df --output=avail "${RUN_ROOT}" | tail -1)"
if (( available_kb < 20 * 1024 * 1024 )); then
    echo "At least 20 GiB free is required under ${RUN_ROOT}" >&2
    exit 1
fi

run_prepare_pool() {
    local dataset="$1"
    local experiment_config="$2"
    local name="synpath-sd-index-prepare-${dataset}"
    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run --rm \
        --name "${name}" \
        --user 0:0 \
        --env-file "${ENV_FILE}" \
        -e OMP_NUM_THREADS=4 \
        -e MKL_NUM_THREADS=4 \
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        -v "${DATA_MOUNT}:${DATA_MOUNT}:ro" \
        -v "${PWD}:/workspace/synpath" \
        -w /workspace/synpath \
        "${IMAGE}" \
        bash scripts/docker/dgx_entrypoint.sh \
        uv run --no-sync synpath prepare-data "${dataset}" \
        --config "${experiment_config}"
}

run_preparation() {
    local tracks=(
        "pathmnist224 configs/experiments/main_pathmnist224_budget.yaml"
        "icaird_cervical100k configs/experiments/main_icaird_cervical100k_budget.yaml"
        "pcam configs/experiments/main_pcam_budget.yaml"
    )
    local pids=()
    for track in "${tracks[@]}"; do
        read -r dataset experiment_config <<<"${track}"
        run_prepare_pool "${dataset}" "${experiment_config}" \
            >"${LOG_ROOT}/prepare_${dataset}.log" 2>&1 &
        pids+=("$!")
    done
    wait_for_workers "${pids[@]}"
}

run_synpath_track() {
    local gpu="$1"
    local dataset="$2"
    local encoder="$3"
    local data_parallel=0
    if [[ "${gpu}" == *,* ]]; then
        data_parallel=1
    fi
    local name="synpath-sd-index-embed-${dataset}-${encoder}"
    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run --rm \
        --name "${name}" \
        --user 0:0 \
        --gpus all \
        --ipc=host \
        --env-file "${ENV_FILE}" \
        -e CUDA_VISIBLE_DEVICES="${gpu}" \
        -e SYNPATH_EMBED_DATA_PARALLEL="${data_parallel}" \
        -e OMP_NUM_THREADS=2 \
        -e MKL_NUM_THREADS=2 \
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        -v "${DATA_MOUNT}:${DATA_MOUNT}:ro" \
        -v "${BALANCED_ROOT}:${BALANCED_ROOT}:ro" \
        -v "${PWD}:/workspace/synpath" \
        -w /workspace/synpath \
        "${IMAGE}" \
        bash scripts/docker/dgx_entrypoint.sh \
        uv run --no-sync synpath calibrate-sd-index \
        --config "${CONFIG}" \
        --stage embed \
        --dataset "${dataset}" \
        --encoder "${encoder}" \
        --resume
}

run_embeddings() {
    local tracks=(
        "0 pathmnist224 dino_vitb8"
        "1 icaird_cervical100k dino_vitb8"
        "2 pcam dino_vitb8"
        "3 pathmnist224 uni2"
        "4 icaird_cervical100k uni2"
        "3,4,5,6,7 pcam uni2"
    )
    local pids=()
    for track in "${tracks[@]}"; do
        read -r gpu dataset encoder <<<"${track}"
        run_synpath_track "${gpu}" "${dataset}" "${encoder}" \
            >"${LOG_ROOT}/embed_${dataset}_${encoder}.log" 2>&1 &
        pids+=("$!")
    done
    wait_for_workers "${pids[@]}"
}

run_rapids_worker() {
    local gpu="$1"
    local dataset="$2"
    local encoder="$3"
    local k_values="$4"
    local worker_mode="${5:-fit}"
    local extra_args=()
    if [[ "${worker_mode}" == "rescore" ]]; then
        extra_args+=(--rescore-only)
    fi
    local name="synpath-sd-index-fit-gpu${gpu}"
    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run --rm \
        --name "${name}" \
        --user 0:0 \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --env-file "${ENV_FILE}" \
        -e CUDA_VISIBLE_DEVICES="${gpu}" \
        -e PYTHONPATH=/workspace/synpath/src \
        -e PYTHONDONTWRITEBYTECODE=1 \
        -e OMP_NUM_THREADS=8 \
        -e MKL_NUM_THREADS=8 \
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        -v "${DATA_MOUNT}:${DATA_MOUNT}:ro" \
        -v "${BALANCED_ROOT}:${BALANCED_ROOT}:ro" \
        -v "${PWD}:/workspace/synpath:ro" \
        -w /workspace/synpath \
        "${RAPIDS_IMAGE}" \
        python -m synpath.calibration.worker \
        --config "${CONFIG}" \
        --dataset "${dataset}" \
        --encoder "${encoder}" \
        --k-values "${k_values}" \
        --resume \
        "${extra_args[@]}"
}

run_fits() {
    local worker_mode="${1:-fit}"
    local all_k even_k odd_k
    all_k="$(seq -s, 2 50)"
    even_k="$(seq -s, 2 2 50)"
    odd_k="$(seq -s, 3 2 49)"
    local tracks=(
        "0 pathmnist224 dino_vitb8 ${all_k}"
        "1 icaird_cervical100k dino_vitb8 ${all_k}"
        "2 pcam dino_vitb8 ${all_k}"
        "3 pcam uni2 ${all_k}"
        "4 pathmnist224 uni2 ${even_k}"
        "5 pathmnist224 uni2 ${odd_k}"
        "6 icaird_cervical100k uni2 ${even_k}"
        "7 icaird_cervical100k uni2 ${odd_k}"
    )
    local pids=()
    for track in "${tracks[@]}"; do
        read -r gpu dataset encoder k_values <<<"${track}"
        run_rapids_worker \
            "${gpu}" "${dataset}" "${encoder}" "${k_values}" "${worker_mode}" \
            >"${LOG_ROOT}/${worker_mode}_gpu${gpu}_${dataset}_${encoder}.log" 2>&1 &
        pids+=("$!")
    done
    wait_for_workers "${pids[@]}"
}

run_report() {
    local name="synpath-sd-index-report"
    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run --rm \
        --name "${name}" \
        --env-file "${ENV_FILE}" \
        -e MPLCONFIGDIR="${RUN_ROOT}/cache/matplotlib" \
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        -v "${PWD}:/workspace/synpath" \
        -w /workspace/synpath \
        "${IMAGE}" \
        bash scripts/docker/dgx_entrypoint.sh \
        uv run --no-sync synpath calibrate-sd-index \
        --config "${CONFIG}" \
        --stage report >"${LOG_ROOT}/report.log" 2>&1
}

wait_for_workers() {
    local status=0
    for pid in "$@"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    if (( status != 0 )); then
        echo "One or more calibration workers failed. Check ${LOG_ROOT}" >&2
        exit "${status}"
    fi
}

if [[ "${MODE}" == "all" || "${MODE}" == "prepare" ]]; then
    run_preparation
fi
if [[ "${MODE}" == "all" || "${MODE}" == "embed" ]]; then
    run_embeddings
fi
if [[ "${MODE}" == "all" || "${MODE}" == "fit" ]]; then
    run_fits fit
fi
if [[ "${MODE}" == "rescore" ]]; then
    run_fits rescore
    run_report
fi
if [[ "${MODE}" == "all" || "${MODE}" == "report" ]]; then
    run_report
fi

echo "Calibration complete"
echo "Logs: ${LOG_ROOT}"
echo "Report: ${RUN_ROOT}/reports/cluster_sd_index_calibration"
