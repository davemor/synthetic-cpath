#!/bin/bash
set -euo pipefail

PATCHER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HISTOKIT_ROOT="$(cd "$PATCHER_ROOT/../histokit" && pwd)"
REMOTE_HOST="${HYPATIA_HOST:-hypatia}"
REMOTE_REPOS_DIR="${HYPATIA_REPOS_DIR:-~/repos}"

ssh "$REMOTE_HOST" "mkdir -p $REMOTE_REPOS_DIR"
rsync -az --delete \
  --exclude=.git \
  --exclude=.venv \
  --exclude=.mypy_cache \
  --exclude=__pycache__ \
  --exclude=.DS_Store \
  --exclude=tmp_patch_compare.png \
  --exclude=cache \
  "$PATCHER_ROOT/" \
  "$REMOTE_HOST:$REMOTE_REPOS_DIR/patcher/"

rsync -az --delete \
  --exclude=.git \
  --exclude=.venv \
  --exclude=.mypy_cache \
  --exclude=.pytest_cache \
  --exclude=.ruff_cache \
  --exclude=.DS_Store \
  --exclude=data \
  --exclude=output \
  --exclude=dist \
  --exclude=notebooks \
  "$HISTOKIT_ROOT/" \
  "$REMOTE_HOST:$REMOTE_REPOS_DIR/histokit/"
