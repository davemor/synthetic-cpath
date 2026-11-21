#!/usr/bin/env bash
set -euo pipefail

EMBED_IMAGE="${SYNPATH_OSORIO_DINO_IMAGE:-synpath-osorio-dino:2023}"
WORKER_IMAGE="${SYNPATH_SKLEARN_IMAGE:-synpath-rapids-sdbw:26.06}"
REPORT_IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
RUN_ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
DATA_MOUNT="${SYNPATH_DGX_DATA_MOUNT:-/raid/dm236/data}"
ENV_FILE="${SYNPATH_DGX_ENV_FILE:-${RUN_ROOT}/dgx.env}"
CONFIG="configs/experiments/sd_index_calibration_pcam_kaggle_osorio_released_dino.yaml"
RUN_NAME="cluster_sd_index_calibration_pcam_kaggle_osorio_released_dino"
LOG_ROOT="${RUN_ROOT}/logs/${RUN_NAME}"

mkdir -p "${LOG_ROOT}"

if ! docker image inspect "${EMBED_IMAGE}" >/dev/null 2>&1; then
    docker build -f Dockerfile.osorio-dino -t "${EMBED_IMAGE}" .
fi

run_pinned_stage() {
    local stage="$1"
    local gpu_args=()
    local parallel=0
    if [[ "${stage}" == "embed" ]]; then
        gpu_args=(--gpus all)
        parallel=1
    fi
    docker run --rm \
        --name "synpath-pcam-osorio-released-${stage}" \
        --user 0:0 \
        --ipc=host \
        "${gpu_args[@]}" \
        --env-file "${ENV_FILE}" \
        -e PYTHONPATH=/workspace/synpath/src \
        -e SYNPATH_EMBED_DATA_PARALLEL="${parallel}" \
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        -v "${DATA_MOUNT}:${DATA_MOUNT}:ro" \
        -v "${PWD}:/workspace/synpath:ro" \
        -w /workspace/synpath \
        "${EMBED_IMAGE}" \
        python -c 'from synpath.cli.main import main; main()' \
        calibrate-sd-index --config "${CONFIG}" \
        --stage "${stage}" --resume
}

run_worker() {
    local worker="$1"
    local cpus="$2"
    local k_values="$3"
    docker run --rm \
        --name "synpath-pcam-osorio-released-worker-${worker}" \
        --user 0:0 \
        --cpuset-cpus "${cpus}" \
        --env-file "${ENV_FILE}" \
        -e PYTHONPATH=/workspace/synpath/src \
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

run_report() {
    docker run --rm \
        --name synpath-pcam-osorio-released-report \
        --user 0:0 \
        --env-file "${ENV_FILE}" \
        -e PYTHONPATH=/workspace/synpath/src \
        -v "${RUN_ROOT}:${RUN_ROOT}" \
        -v "${PWD}:/workspace/synpath:ro" \
        -w /workspace/synpath \
        "${REPORT_IMAGE}" \
        bash scripts/docker/dgx_entrypoint.sh \
        uv run --no-sync python -c \
        'from pathlib import Path; from synpath.config import load_resolved_yaml; from synpath.paths import SynpathPaths; from synpath.calibration.report import build_report; cfg = load_resolved_yaml(Path("configs/experiments/sd_index_calibration_pcam_kaggle_osorio_released_dino.yaml")); print(build_report(cfg, SynpathPaths.from_env()))'
}

run_pinned_stage curate >"${LOG_ROOT}/curate.log" 2>&1
run_pinned_stage embed >"${LOG_ROOT}/embed.log" 2>&1

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

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
if (( status != 0 )); then
    echo "One or more workers failed; see ${LOG_ROOT}" >&2
    exit "${status}"
fi

run_report >"${LOG_ROOT}/report.log" 2>&1
echo "Calibration complete: ${RUN_ROOT}/reports/${RUN_NAME}"
