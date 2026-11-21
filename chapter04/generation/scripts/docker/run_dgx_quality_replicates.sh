#!/usr/bin/env bash
set -euo pipefail

IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
RUN_ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
DATA_MOUNT="${SYNPATH_DGX_DATA_MOUNT:-/raid/dm236/data}"
ENV_FILE="${SYNPATH_DGX_ENV_FILE:-${RUN_ROOT}/dgx.env}"
CONTAINER_NAME="${SYNPATH_DGX_CONTAINER:-synpath-quality-replicates}"
LOG_ROOT="${RUN_ROOT}/logs/quality_replicates"

mkdir -p "${LOG_ROOT}"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing env file: ${ENV_FILE}" >&2
    exit 1
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d \
    --name "${CONTAINER_NAME}" \
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
    -e SYNPATH_REPLICATE_DATASETS="${SYNPATH_REPLICATE_DATASETS:-pathmnist224 icaird_cervical100k pcam}" \
    -e SYNPATH_REPLICATE_MODELS="${SYNPATH_REPLICATE_MODELS:-stylegan2ada sd_lora dcgan}" \
    -e SYNPATH_REPLICATE_CONDITIONING="${SYNPATH_REPLICATE_CONDITIONING:-true_label unconditioned pseudo_label label_global_cluster label_local_cluster}" \
    -e SYNPATH_REPLICATE_SEEDS="${SYNPATH_REPLICATE_SEEDS:-0 1 2}" \
    -v "${RUN_ROOT}:${RUN_ROOT}" \
    -v "${DATA_MOUNT}:${DATA_MOUNT}:ro" \
    -v "${PWD}:/workspace/synpath" \
    -w /workspace/synpath \
    "${IMAGE}" \
    bash scripts/docker/dgx_entrypoint.sh \
    bash scripts/run/run_quality_replicates.sh \
    >"${LOG_ROOT}/launcher.log"

echo "container=${CONTAINER_NAME}"
echo "progress=docker logs -f ${CONTAINER_NAME}"
echo "launcher_log=${LOG_ROOT}/launcher.log"
