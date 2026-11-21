#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${SYNPATH_RUN_ROOT:-/raid/dm236/synpath}"
CONFIG="configs/experiments/convergence_sd_lora_kather_class_label.yaml"
SOURCE_IMAGES="${RUN_ROOT}/outputs/generators/pathmnist224/sd_lora/true_label/seed0/training_images"
AUDIT_ROOT="${RUN_ROOT}/outputs/generators/pathmnist224/sd_lora/true_label/seed101"

if [[ ! -d "${SOURCE_IMAGES}" ]]; then
    echo "Missing reusable seed-0 training images: ${SOURCE_IMAGES}" >&2
    exit 1
fi

mkdir -p "${AUDIT_ROOT}"
if [[ ! -e "${AUDIT_ROOT}/training_images" ]]; then
    ln -s "${SOURCE_IMAGES}" "${AUDIT_ROOT}/training_images"
fi

uv run --extra gpu synpath audit-sd-lora-convergence \
    --config "${CONFIG}" --stage all --resume
