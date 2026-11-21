#!/usr/bin/env bash
set -euo pipefail

IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
RUN_ROOT="${SYNPATH_BALANCED_GLOBAL_ROOT:-/raid/dm236/synpath_balanced_global_clusters}"
BASE_ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
ENV_FILE="${SYNPATH_DGX_ENV_FILE:-${BASE_ROOT}/dgx.env}"
CONFIG="configs/experiments/main_pathmnist224_balanced_global_cluster.yaml"
LOG_ROOT="${RUN_ROOT}/logs/main_pathmnist224_balanced_global_cluster"

mkdir -p "${LOG_ROOT}"

docker rm -f synpath-pathmnist-balanced-global >/dev/null 2>&1 || true
docker run --rm \
    --name synpath-pathmnist-balanced-global \
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
    -v "${BASE_ROOT}:${BASE_ROOT}" \
    -v "${RUN_ROOT}:${RUN_ROOT}" \
    -v "${PWD}:/workspace/synpath" \
    -w /workspace/synpath \
    "${IMAGE}" \
    bash scripts/docker/dgx_entrypoint.sh \
    uv run --extra gpu synpath run --config "${CONFIG}" --resume \
    >"${LOG_ROOT}/generation_and_quality.log" 2>&1

SYNPATH_DGX_ROOT="${RUN_ROOT}" \
SYNPATH_DGX_ENV_FILE="${ENV_FILE}" \
SYNPATH_EXPERIMENT_CONFIG="${CONFIG}" \
SYNPATH_UTILITY_DATASET=pathmnist224 \
SYNPATH_UTILITY_GPUS=8 \
SYNPATH_UTILITY_SOURCE_ROOT="${BASE_ROOT}" \
SYNPATH_UTILITY_SYNTHETIC_MODELS="stylegan2ada sd_lora dcgan" \
SYNPATH_UTILITY_SOURCES="synthetic_only_label_global_cluster real_plus_synthetic_label_global_cluster" \
bash scripts/docker/run_dgx_utility_parallel.sh \
    >"${LOG_ROOT}/utility_parallel.log" 2>&1

echo "Balanced-global Kather experiment complete: ${RUN_ROOT}/reports"
