#!/usr/bin/env bash
set -euo pipefail

IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
RUN_ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
DATA_MOUNT="${SYNPATH_DGX_DATA_MOUNT:-/raid/dm236/data}"
ENV_FILE="${SYNPATH_DGX_ENV_FILE:-${RUN_ROOT}/dgx.env}"
CONTAINER_NAME="${SYNPATH_DGX_CONTAINER:-synpath-sd-lora-convergence}"
LOG_ROOT="${RUN_ROOT}/logs/sd_lora_convergence"
AUDIT_ROOT="${RUN_ROOT}/outputs/generators/pathmnist224/sd_lora/true_label/seed101"

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
    -e SYNPATH_SD_SAMPLE_GPUS=8 \
    -v "${RUN_ROOT}:${RUN_ROOT}" \
    -v "${DATA_MOUNT}:${DATA_MOUNT}:ro" \
    -v "${PWD}:/workspace/synpath" \
    -w /workspace/synpath \
    "${IMAGE}" \
    bash scripts/docker/dgx_entrypoint.sh \
    bash scripts/run/run_sd_lora_convergence_audit.sh \
    >"${LOG_ROOT}/launcher.log"

echo "container=${CONTAINER_NAME}"
echo "progress=docker logs -f ${CONTAINER_NAME}"
echo "training_log=${AUDIT_ROOT}/sd_lora_train.log"
echo "report=${AUDIT_ROOT}/convergence_audit/report.md"
