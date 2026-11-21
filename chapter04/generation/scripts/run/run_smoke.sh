#!/usr/bin/env bash
set -euo pipefail

uv run synpath plan --config configs/experiments/smoke.yaml
