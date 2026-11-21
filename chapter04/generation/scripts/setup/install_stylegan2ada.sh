#!/usr/bin/env bash
set -euo pipefail

EXTERNAL_ROOT="${SYNPATH_EXTERNAL_ROOT:-external}"
REPO_DIR="${EXTERNAL_ROOT}/stylegan2-ada-pytorch"
STYLEGAN3_REPO_DIR="${EXTERNAL_ROOT}/stylegan3"
ENV_DIR="${SYNPATH_STYLEGAN_ENV:-${EXTERNAL_ROOT}/stylegan2ada-env}"
PYTHON_VERSION="${SYNPATH_STYLEGAN_PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${SYNPATH_STYLEGAN_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu117}"

mkdir -p "${EXTERNAL_ROOT}"

if [[ -d "${REPO_DIR}/.git" ]]; then
    echo "stylegan2-ada-pytorch already exists at ${REPO_DIR}"
else
    git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git "${REPO_DIR}"
fi

if [[ -d "${STYLEGAN3_REPO_DIR}/.git" ]]; then
    echo "stylegan3 already exists at ${STYLEGAN3_REPO_DIR}"
else
    git clone https://github.com/NVlabs/stylegan3.git "${STYLEGAN3_REPO_DIR}"
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    uv venv --python "${PYTHON_VERSION}" "${ENV_DIR}"
fi

uv pip install --python "${ENV_DIR}/bin/python" \
    --index-url "${TORCH_INDEX_URL}" \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117
uv pip install --python "${ENV_DIR}/bin/python" \
    click==8.0.4 \
    "imageio-ffmpeg>=0.4.3" \
    "ninja>=1.10" \
    "numpy<2" \
    "pillow>=9" \
    "psutil>=5.9" \
    "pyspng>=0.1.0" \
    "packaging>=23" \
    "requests>=2.28" \
    "setuptools>=68,<81" \
    "scipy>=1.8" \
    "tqdm>=4.64" \
    "wheel>=0.40"

echo "StyleGAN2-ADA external repository: ${REPO_DIR}"
echo "StyleGAN3 external repository: ${STYLEGAN3_REPO_DIR}"
echo "StyleGAN2-ADA Python: ${ENV_DIR}/bin/python"
echo "Set SYNPATH_STYLEGAN_PYTHON=${ENV_DIR}/bin/python if needed."
