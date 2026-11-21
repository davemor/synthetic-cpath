#!/usr/bin/env bash
set -euo pipefail

CONFIG="${SYNPATH_PCAM_SD_FULL_CONFIG:-configs/experiments/pcam_sd_full_class.yaml}"

uv run --extra gpu synpath train-pcam-sd-full-class \
    --config "${CONFIG}" \
    --stage "${SYNPATH_PCAM_SD_FULL_STAGE:-all}" \
    --resume
