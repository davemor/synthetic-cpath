#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-main}"
IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
RUN_ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
DATA_MOUNT="${SYNPATH_DGX_DATA_MOUNT:-/raid/dm236/data}"
ENV_FILE="${SYNPATH_DGX_ENV_FILE:-${RUN_ROOT}/dgx.env}"
CONTAINER_NAME="${SYNPATH_DGX_CONTAINER:-synpath-${MODE}}"
EXPERIMENT_CONFIG="${SYNPATH_EXPERIMENT_CONFIG:-configs/experiments/main.yaml}"

mkdir -p "${RUN_ROOT}/logs"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing env file: ${ENV_FILE}" >&2
    echo "Create it from configs/docker/dgx.env.example and add private tokens there." >&2
    exit 1
fi

case "${MODE}" in
    main)
        COMMAND=(uv run --extra gpu synpath run --config "${EXPERIMENT_CONFIG}" --resume)
        ;;
    smoke)
        COMMAND=(uv run --extra gpu synpath run --config configs/experiments/smoke.yaml --resume)
        ;;
    plan)
        COMMAND=(uv run --extra gpu synpath plan --config "${EXPERIMENT_CONFIG}")
        ;;
    shell)
        COMMAND=(sleep infinity)
        ;;
    *)
        echo "Unknown DGX mode: ${MODE}" >&2
        echo "Expected one of: main, smoke, plan, shell" >&2
        exit 1
        ;;
esac

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --env-file "${ENV_FILE}" \
    -v "${RUN_ROOT}:${RUN_ROOT}" \
    -v "${DATA_MOUNT}:${DATA_MOUNT}:ro" \
    -v "${PWD}:/workspace/synpath" \
    -w /workspace/synpath \
    "${IMAGE}" \
    bash scripts/docker/dgx_entrypoint.sh "${COMMAND[@]}"

echo "container=${CONTAINER_NAME}"
echo "log=docker logs -f ${CONTAINER_NAME}"
