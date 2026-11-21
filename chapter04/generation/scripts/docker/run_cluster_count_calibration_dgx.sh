#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
RAPIDS_IMAGE="${SYNPATH_RAPIDS_IMAGE:-synpath-rapids-sdbw:26.06}"
RUN_ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
DATA_MOUNT="${SYNPATH_DGX_DATA_MOUNT:-/raid/dm236/data}"
BALANCED_ROOT="${SYNPATH_BALANCED_CLUSTER_ROOT:-/raid/dm236/synpath_balanced_global_clusters}"
ENV_FILE="${SYNPATH_DGX_ENV_FILE:-${RUN_ROOT}/dgx.env}"
CONFIG="${SYNPATH_CLUSTER_COUNT_CONFIG:-configs/experiments/cluster_count_calibration_2_50.yaml}"
RUN_NAME="cluster_k_calibration_2_50"
LOG_ROOT="${RUN_ROOT}/logs/${RUN_NAME}"

if [[ ! "${MODE}" =~ ^(all|curate|embed|fit|report)$ ]]; then
    echo "Mode must be one of: all, curate, embed, fit, report" >&2
    exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing env file: ${ENV_FILE}" >&2
    exit 1
fi

mkdir -p "${LOG_ROOT}"
available_kb="$(df --output=avail "${RUN_ROOT}" | tail -1)"
if (( available_kb < 20 * 1024 * 1024 )); then
    echo "At least 20 GiB free is required under ${RUN_ROOT}" >&2
    exit 1
fi

for manifest in \
    pathmnist224_full.csv \
    icaird_cervical100k_full.csv \
    pcam_full.csv \
    pcam_kaggle_full.csv; do
    path="${RUN_ROOT}/data/manifests/${manifest}"
    if [[ ! -f "${path}" ]]; then
        echo "Missing required training manifest: ${path}" >&2
        exit 1
    fi
done

run_synpath_stage() {
    local stage="$1"
    local name="synpath-cluster-count-${stage}"
    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run --rm \
        --name "${name}" \
        --user 0:0 \
        --env-file "${ENV_FILE}" \
        -e MPLCONFIGDIR="${RUN_ROOT}/cache/matplotlib" \
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        -v "${DATA_MOUNT}:${DATA_MOUNT}:ro" \
        -v "${BALANCED_ROOT}:${BALANCED_ROOT}:ro" \
        -v "${PWD}:/workspace/synpath:ro" \
        -w /workspace/synpath \
        "${IMAGE}" \
        bash scripts/docker/dgx_entrypoint.sh \
        uv run --no-sync synpath calibrate-cluster-count \
        --config "${CONFIG}" --stage "${stage}" --resume
}

run_embed_worker() {
    local gpu="$1"
    local dataset="$2"
    local encoder="$3"
    local name="synpath-cluster-embed-${gpu}-${dataset}-${encoder}"
    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run --rm \
        --name "${name}" \
        --user 0:0 \
        --gpus all \
        --ipc=host \
        --env-file "${ENV_FILE}" \
        -e CUDA_VISIBLE_DEVICES="${gpu}" \
        -e OMP_NUM_THREADS=8 \
        -e MKL_NUM_THREADS=8 \
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        -v "${DATA_MOUNT}:${DATA_MOUNT}:ro" \
        -v "${BALANCED_ROOT}:${BALANCED_ROOT}:ro" \
        -v "${PWD}:/workspace/synpath:ro" \
        -w /workspace/synpath \
        "${IMAGE}" \
        bash scripts/docker/dgx_entrypoint.sh \
        uv run --no-sync synpath calibrate-cluster-count \
        --config "${CONFIG}" --stage embed \
        --dataset "${dataset}" --encoder "${encoder}" --resume
}

run_fit_worker() {
    local gpu="$1"
    local dataset="$2"
    local encoder="$3"
    local name="synpath-cluster-fit-${gpu}-${dataset}-${encoder}"
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
        --config "${CONFIG}" --dataset "${dataset}" \
        --encoder "${encoder}" --k-values "$(seq -s, 2 50)" --resume
}

wait_for_workers() {
    local status=0
    for pid in "$@"; do
        wait "${pid}" || status=1
    done
    if (( status != 0 )); then
        echo "One or more workers failed; check ${LOG_ROOT}" >&2
        exit "${status}"
    fi
}

tracks=(
    "0 pathmnist224 dino_vitb8"
    "1 pathmnist224 uni2"
    "2 icaird_cervical100k dino_vitb8"
    "3 icaird_cervical100k uni2"
    "4 pcam dino_vitb8"
    "5 pcam uni2"
    "6 pcam_osorio dino_vitb8"
    "7 pcam_osorio uni2"
)

if [[ "${MODE}" == "all" || "${MODE}" == "curate" ]]; then
    run_synpath_stage curate >"${LOG_ROOT}/curate.log" 2>&1
fi

if [[ "${MODE}" == "all" ]]; then
    pids=()
    for track in "${tracks[@]}"; do
        read -r gpu dataset encoder <<<"${track}"
        (
            run_embed_worker "${gpu}" "${dataset}" "${encoder}" \
                >"${LOG_ROOT}/embed_${dataset}_${encoder}.log" 2>&1
            run_fit_worker "${gpu}" "${dataset}" "${encoder}" \
                >"${LOG_ROOT}/fit_${dataset}_${encoder}.log" 2>&1
        ) &
        pids+=("$!")
    done
    wait_for_workers "${pids[@]}"
    run_synpath_stage report >"${LOG_ROOT}/report.log" 2>&1
    echo "Calibration complete"
    echo "Logs: ${LOG_ROOT}"
    echo "Report: ${RUN_ROOT}/reports/${RUN_NAME}"
    exit 0
fi

if [[ "${MODE}" == "embed" ]]; then
    pids=()
    for track in "${tracks[@]}"; do
        read -r gpu dataset encoder <<<"${track}"
        run_embed_worker "${gpu}" "${dataset}" "${encoder}" \
            >"${LOG_ROOT}/embed_${dataset}_${encoder}.log" 2>&1 &
        pids+=("$!")
    done
    wait_for_workers "${pids[@]}"
fi

if [[ "${MODE}" == "fit" ]]; then
    pids=()
    for track in "${tracks[@]}"; do
        read -r gpu dataset encoder <<<"${track}"
        run_fit_worker "${gpu}" "${dataset}" "${encoder}" \
            >"${LOG_ROOT}/fit_${dataset}_${encoder}.log" 2>&1 &
        pids+=("$!")
    done
    wait_for_workers "${pids[@]}"
fi

if [[ "${MODE}" == "report" ]]; then
    run_synpath_stage report >"${LOG_ROOT}/report.log" 2>&1
fi

echo "Calibration complete"
echo "Logs: ${LOG_ROOT}"
echo "Report: ${RUN_ROOT}/reports/${RUN_NAME}"
