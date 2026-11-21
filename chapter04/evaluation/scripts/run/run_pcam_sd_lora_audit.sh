#!/usr/bin/env bash
set -euo pipefail

CONFIG="${SYNPATH_PCAM_AUDIT_CONFIG:-configs/experiments/audit_pcam_sd_lora.yaml}"

uv run --extra gpu synpath audit-pcam-sd-lora \
    --config "${CONFIG}" \
    --stage "${SYNPATH_PCAM_AUDIT_STAGE:-all}" \
    --resume
