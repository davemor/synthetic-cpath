#!/bin/bash
set -euo pipefail

REPO_ROOT="${PATCHER_REPO_ROOT:-$HOME/repos/patcher}"
VENV_PATH="${PATCHER_VENV_PATH:-/sharedscratch/dm236/.venvs/patcher-3.14}"

module load python/3.14.2
export UV_LINK_MODE=copy
mkdir -p "$(dirname "$VENV_PATH")"

cd "$REPO_ROOT"
uv venv --python 3.14 "$VENV_PATH"
source "$VENV_PATH/bin/activate"
uv sync --active
