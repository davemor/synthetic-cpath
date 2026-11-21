#!/usr/bin/env bash
set -euo pipefail

uv sync --extra gpu

EXTERNAL_ROOT="${SYNPATH_EXTERNAL_ROOT:-external}"
mkdir -p "${EXTERNAL_ROOT}"
if [[ -d "${EXTERNAL_ROOT}/diffusers/.git" ]]; then
    echo "diffusers already exists at ${EXTERNAL_ROOT}/diffusers"
else
    git clone https://github.com/huggingface/diffusers.git "${EXTERNAL_ROOT}/diffusers"
fi

echo "Stable Diffusion LoRA dependencies installed through uv."
echo "Run 'accelerate config' on the DGX before full training."
