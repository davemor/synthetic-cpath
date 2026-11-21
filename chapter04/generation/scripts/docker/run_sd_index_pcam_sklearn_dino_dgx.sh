#!/usr/bin/env bash
set -euo pipefail

IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
WORKER_IMAGE="${SYNPATH_SKLEARN_IMAGE:-synpath-rapids-sdbw:26.06}"
RUN_ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
DATA_MOUNT="${SYNPATH_DGX_DATA_MOUNT:-/raid/dm236/data}"
ENV_FILE="${SYNPATH_DGX_ENV_FILE:-${RUN_ROOT}/dgx.env}"
CONFIG="configs/experiments/sd_index_calibration_pcam_kaggle_osorio_sklearn_dino.yaml"
RUN_NAME="cluster_sd_index_calibration_pcam_kaggle_osorio_sklearn_dino"
LOG_ROOT="${RUN_ROOT}/logs/${RUN_NAME}"

mkdir -p "${LOG_ROOT}"

run_stage() {
    local stage="$1"
    docker run --rm \
        --name "synpath-pcam-sklearn-${stage}" \
        --user 0:0 \
        --env-file "${ENV_FILE}" \
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        -v "${DATA_MOUNT}:${DATA_MOUNT}:ro" \
        -v "${PWD}/src:/workspace/synpath/src:ro" \
        -v "${PWD}/configs:/workspace/synpath/configs:ro" \
        -w /workspace/synpath \
        "${IMAGE}" \
        bash scripts/docker/dgx_entrypoint.sh \
        uv run --no-sync synpath calibrate-sd-index \
        --config "${CONFIG}" --stage "${stage}" --resume
}

run_worker() {
    local worker="$1"
    local cpus="$2"
    local k_values="$3"
    docker run --rm \
        --name "synpath-pcam-sklearn-worker-${worker}" \
        --user 0:0 \
        --cpuset-cpus "${cpus}" \
        --env-file "${ENV_FILE}" \
        -e PYTHONPATH=/workspace/synpath/src \
        -e PYTHONDONTWRITEBYTECODE=1 \
        -e OMP_NUM_THREADS=20 \
        -e MKL_NUM_THREADS=20 \
        -e OPENBLAS_NUM_THREADS=20 \
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        -v "${PWD}:/workspace/synpath:ro" \
        -w /workspace/synpath \
        "${WORKER_IMAGE}" \
        python -m synpath.calibration.worker \
        --config "${CONFIG}" --dataset pcam --encoder dino_vitb8 \
        --k-values "${k_values}" --resume
}

wait_for_workers() {
    local status=0
    for pid in "$@"; do
        wait "${pid}" || status=1
    done
    if (( status != 0 )); then
        echo "One or more scikit-learn workers failed; see ${LOG_ROOT}" >&2
        exit "${status}"
    fi
}

run_stage curate >"${LOG_ROOT}/curate.log" 2>&1
run_stage embed >"${LOG_ROOT}/embed.log" 2>&1

pids=()
workers=(
    "0 0-19 $(seq -s, 2 4 50)"
    "1 20-39 $(seq -s, 3 4 50)"
    "2 40-59 $(seq -s, 4 4 50)"
    "3 60-79 $(seq -s, 5 4 49)"
)
for spec in "${workers[@]}"; do
    read -r worker cpus k_values <<<"${spec}"
    run_worker "${worker}" "${cpus}" "${k_values}" \
        >"${LOG_ROOT}/worker_${worker}.log" 2>&1 &
    pids+=("$!")
done
wait_for_workers "${pids[@]}"

run_stage report >"${LOG_ROOT}/report.log" 2>&1
echo "Calibration complete: ${RUN_ROOT}/reports/${RUN_NAME}"
