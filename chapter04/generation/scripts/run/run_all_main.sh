#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiments/main.yaml}"

uv run synpath plan --config "${CONFIG}"
uv run synpath run --config "${CONFIG}" --resume
