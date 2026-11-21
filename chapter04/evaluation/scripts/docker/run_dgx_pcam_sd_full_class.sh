#!/usr/bin/env bash
set -euo pipefail

IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
RUN_ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
DATA_MOUNT="${SYNPATH_DGX_DATA_MOUNT:-/raid/dm236/data}"
ENV_FILE="${SYNPATH_DGX_ENV_FILE:-${RUN_ROOT}/dgx.env}"
CONTAINER_NAME="${SYNPATH_DGX_CONTAINER:-synpath-pcam-sd-full-class}"
LOG_ROOT="${RUN_ROOT}/logs/pcam_sd_full_class_label"
OUTPUT_ROOT="${RUN_ROOT}/outputs/audits/pcam_sd_full_class_label"
CONFIG="${SYNPATH_PCAM_SD_FULL_CONFIG:-configs/experiments/pcam_sd_full_class.yaml}"

mkdir -p "${LOG_ROOT}"
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
    -e SYNPATH_PCAM_SD_FULL_CONFIG="${CONFIG}" \
    -v "${RUN_ROOT}:${RUN_ROOT}" \
    -v "${DATA_MOUNT}:${DATA_MOUNT}:ro" \
    -v "${PWD}/src/synpath/audits:/workspace/synpath/src/synpath/audits:ro" \
    -v "${PWD}/src/synpath/cli/main.py:/workspace/synpath/src/synpath/cli/main.py:ro" \
    -v "${PWD}/src/synpath/generation/wrappers.py:/workspace/synpath/src/synpath/generation/wrappers.py:ro" \
    -v "${PWD}/src/synpath/generation/sd_full_sample.py:/workspace/synpath/src/synpath/generation/sd_full_sample.py:ro" \
    -v "${PWD}/configs/experiments:/workspace/synpath/configs/experiments:ro" \
    -v "${PWD}/scripts/run/run_pcam_sd_full_class.sh:/workspace/synpath/scripts/run/run_pcam_sd_full_class.sh:ro" \
    -w /workspace/synpath \
    "${IMAGE}" \
    bash scripts/docker/dgx_entrypoint.sh \
    bash scripts/run/run_pcam_sd_full_class.sh \
    >"${LOG_ROOT}/launcher.log"

echo "container=${CONTAINER_NAME}"
echo "progress=docker logs -f ${CONTAINER_NAME}"
echo "output_root=${OUTPUT_ROOT}"
