#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
RAPIDS_IMAGE="${SYNPATH_RAPIDS_IMAGE:-rapidsai/base:26.06-cuda12-py3.12}"
RUN_ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
DATA_MOUNT="${SYNPATH_DGX_DATA_MOUNT:-/raid/dm236/data}"
ENV_FILE="${SYNPATH_DGX_ENV_FILE:-${RUN_ROOT}/dgx.env}"
CONFIG="${SYNPATH_SD_INDEX_CONFIG:-configs/experiments/sd_index_calibration_pcam_osorio.yaml}"
RUN_NAME="${SYNPATH_SD_INDEX_RUN_NAME:-cluster_sd_index_calibration_pcam_osorio_official}"
LOG_ROOT="${RUN_ROOT}/logs/${RUN_NAME}"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing env file: ${ENV_FILE}" >&2
    exit 1
fi
if [[ ! "${MODE}" =~ ^(all|curate|embed|fit|report)$ ]]; then
    echo "Mode must be one of: all, curate, embed, fit, report" >&2
    exit 1
fi

mkdir -p "${LOG_ROOT}"

run_synpath_stage() {
    local stage="$1"
    local name="synpath-pcam-calibration-${stage}"
    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run --rm \
        --name "${name}" \
        --user 0:0 \
        --env-file "${ENV_FILE}" \
        -e OMP_NUM_THREADS=8 \
        -e MKL_NUM_THREADS=8 \
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        -v "${DATA_MOUNT}:${DATA_MOUNT}:ro" \
        -v "${PWD}/src:/workspace/synpath/src:ro" \
        -v "${PWD}/configs:/workspace/synpath/configs:ro" \
        -w /workspace/synpath \
        "${IMAGE}" \
        bash scripts/docker/dgx_entrypoint.sh \
        uv run --no-sync synpath calibrate-sd-index \
        --config "${CONFIG}" \
        --stage "${stage}" \
        --resume
}

run_embed_track() {
    local gpu="$1"
    local encoder="$2"
    local name="synpath-pcam-calibration-embed-${encoder}"
    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run --rm \
        --name "${name}" \
        --user 0:0 \
        --gpus all \
        --ipc=host \
        --env-file "${ENV_FILE}" \
        -e CUDA_VISIBLE_DEVICES="${gpu}" \
        -e SYNPATH_EMBED_DATA_PARALLEL=1 \
        -e OMP_NUM_THREADS=12 \
        -e MKL_NUM_THREADS=12 \
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        -v "${DATA_MOUNT}:${DATA_MOUNT}:ro" \
        -v "${PWD}/src:/workspace/synpath/src:ro" \
        -v "${PWD}/configs:/workspace/synpath/configs:ro" \
        -w /workspace/synpath \
        "${IMAGE}" \
        bash scripts/docker/dgx_entrypoint.sh \
        uv run --no-sync synpath calibrate-sd-index \
        --config "${CONFIG}" \
        --stage embed \
        --dataset pcam \
        --encoder "${encoder}" \
        --resume
}

run_fit_worker() {
    local gpu="$1"
    local encoder="$2"
    local k_values="$3"
    local name="synpath-pcam-calibration-fit-gpu${gpu}"
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
        -v "${PWD}:/workspace/synpath:ro" \
        -w /workspace/synpath \
        "${RAPIDS_IMAGE}" \
        python -m synpath.calibration.worker \
        --config "${CONFIG}" \
        --dataset pcam \
        --encoder "${encoder}" \
        --k-values "${k_values}" \
        --resume
}

wait_for_workers() {
    local status=0
    for pid in "$@"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    if (( status != 0 )); then
        echo "One or more workers failed. Check ${LOG_ROOT}" >&2
        exit "${status}"
    fi
}

run_embeddings() {
    local pids=()
    run_embed_track "0,1,2,3" dino_vitb8 \
        >"${LOG_ROOT}/embed_dino_vitb8.log" 2>&1 &
    pids+=("$!")
    run_embed_track "4,5,6,7" uni2 \
        >"${LOG_ROOT}/embed_uni2.log" 2>&1 &
    pids+=("$!")
    wait_for_workers "${pids[@]}"
}

run_fits() {
    local pids=()
    local gpu encoder k_values
    local tracks=(
        "0 dino_vitb8 $(seq -s, 2 4 50)"
        "1 dino_vitb8 $(seq -s, 3 4 50)"
        "2 dino_vitb8 $(seq -s, 4 4 50)"
        "3 dino_vitb8 $(seq -s, 5 4 49)"
        "4 uni2 $(seq -s, 2 4 50)"
        "5 uni2 $(seq -s, 3 4 50)"
        "6 uni2 $(seq -s, 4 4 50)"
        "7 uni2 $(seq -s, 5 4 49)"
    )
    for track in "${tracks[@]}"; do
        read -r gpu encoder k_values <<<"${track}"
        run_fit_worker "${gpu}" "${encoder}" "${k_values}" \
            >"${LOG_ROOT}/fit_gpu${gpu}_${encoder}.log" 2>&1 &
        pids+=("$!")
    done
    wait_for_workers "${pids[@]}"
}

if [[ "${MODE}" == "all" || "${MODE}" == "curate" ]]; then
    run_synpath_stage curate >"${LOG_ROOT}/curate.log" 2>&1
fi
if [[ "${MODE}" == "all" || "${MODE}" == "embed" ]]; then
    run_embeddings
fi
if [[ "${MODE}" == "all" || "${MODE}" == "fit" ]]; then
    run_fits
fi
if [[ "${MODE}" == "all" || "${MODE}" == "report" ]]; then
    run_synpath_stage report >"${LOG_ROOT}/report.log" 2>&1
fi

echo "Calibration complete"
echo "Logs: ${LOG_ROOT}"
echo "Report: ${RUN_ROOT}/reports/${RUN_NAME}"
