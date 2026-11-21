#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${SYNPATH_RUN_ROOT:-/raid/dm236/synpath}"

export SYNPATH_DATA_ROOT="${SYNPATH_DATA_ROOT:-${RUN_ROOT}/data}"
export SYNPATH_CACHE_ROOT="${SYNPATH_CACHE_ROOT:-${RUN_ROOT}/cache}"
export SYNPATH_OUTPUTS_ROOT="${SYNPATH_OUTPUTS_ROOT:-${RUN_ROOT}/outputs}"
export SYNPATH_REPORTS_ROOT="${SYNPATH_REPORTS_ROOT:-${RUN_ROOT}/reports}"
export SYNPATH_EXTERNAL_ROOT="${SYNPATH_EXTERNAL_ROOT:-${RUN_ROOT}/external}"
export SYNPATH_STYLEGAN_BACKEND="${SYNPATH_STYLEGAN_BACKEND:-stylegan3}"
export SYNPATH_STYLEGAN_PYTHON="${SYNPATH_STYLEGAN_PYTHON:-${SYNPATH_EXTERNAL_ROOT}/stylegan2ada-env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${RUN_ROOT}/torch_extensions}"
export HF_HOME="${HF_HOME:-${RUN_ROOT}/huggingface}"

mkdir -p \
    "${SYNPATH_DATA_ROOT}" \
    "${SYNPATH_CACHE_ROOT}" \
    "${SYNPATH_OUTPUTS_ROOT}" \
    "${SYNPATH_REPORTS_ROOT}" \
    "${SYNPATH_EXTERNAL_ROOT}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${HF_HOME}" \
    "${RUN_ROOT}/logs"

echo "run_root=${RUN_ROOT}"
echo "data_root=${SYNPATH_DATA_ROOT}"
echo "cache_root=${SYNPATH_CACHE_ROOT}"
echo "outputs_root=${SYNPATH_OUTPUTS_ROOT}"
echo "reports_root=${SYNPATH_REPORTS_ROOT}"
echo "external_root=${SYNPATH_EXTERNAL_ROOT}"
echo "stylegan_backend=${SYNPATH_STYLEGAN_BACKEND}"
echo "stylegan_python=${SYNPATH_STYLEGAN_PYTHON}"
echo "torch_extensions_dir=${TORCH_EXTENSIONS_DIR}"
echo "hf_home=${HF_HOME}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

exec "$@"
