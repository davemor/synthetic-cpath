#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${LOWGEN_DGX_IMAGE:-lowgen-dgx:latest}"
LOWGEN_DGX_TMP="${LOWGEN_DGX_TMP:-/raid/experiments/lowgen_v2/container_tmp}"
LOWGEN_DGX_HOME="${LOWGEN_DGX_HOME:-/raid/experiments/lowgen_v2/container_home}"

mkdir -p "${LOWGEN_DGX_TMP}" "${LOWGEN_DGX_HOME}/cupy"

TTY_ARGS=()
if [ -t 0 ]; then
  TTY_ARGS=(-it)
fi

exec docker run --rm "${TTY_ARGS[@]}" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e TMPDIR="${LOWGEN_DGX_TMP}" \
  -e HOME="${LOWGEN_DGX_HOME}" \
  -e CUPY_CACHE_DIR="${LOWGEN_DGX_HOME}/cupy" \
  -v /raid:/raid \
  -v /home:/home \
  -w /workspace/lowgen \
  "${IMAGE_TAG}" \
  "$@"
