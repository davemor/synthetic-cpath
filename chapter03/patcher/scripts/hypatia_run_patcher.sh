#!/bin/bash
set -euo pipefail

REPO_ROOT="${PATCHER_REPO_ROOT:-$HOME/repos/patcher}"
VENV_PATH="${PATCHER_VENV_PATH:-/sharedscratch/dm236/.venvs/patcher-3.14}"
HOSTNAME_SHORT="$(hostname -s)"

if [[ "${ALLOW_LOGIN_NODE:-0}" != "1" && "$HOSTNAME_SHORT" =~ ^login[0-9]*$ ]]; then
  cat >&2 <<EOF
Refusing to start patcher on login node '$HOSTNAME_SHORT'.

Start it through Slurm instead:
  cd "$REPO_ROOT"
  scripts/hypatia_submit_patcher_server.sh

If you are debugging and really mean to run here, set ALLOW_LOGIN_NODE=1.
EOF
  exit 1
fi

module load python/3.14.2
source "$VENV_PATH/bin/activate"

export PATCHER_DATA_ROOT="${PATCHER_DATA_ROOT:-/sharedscratch/dm236/data}"
export PATCHER_CACHE_ROOT="${PATCHER_CACHE_ROOT:-/sharedscratch/dm236/data}"
export PATCHER_CONFIG_PATH="${PATCHER_CONFIG_PATH:-$REPO_ROOT/configs/icaird-cervical-stratified-120-hypatia.yml}"
PATCHER_PORT="${PATCHER_PORT:-8000}"

cd "$REPO_ROOT"
uv run fastapi run --entrypoint backend.main:app --host 0.0.0.0 --port "$PATCHER_PORT"
