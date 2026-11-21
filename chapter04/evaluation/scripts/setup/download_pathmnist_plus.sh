#!/usr/bin/env bash
set -euo pipefail

ROOT="${SYNPATH_DATA_ROOT:-data}/raw/medmnist"
uv run python scripts/setup/download_pathmnist_plus.py --root "${ROOT}" --size 224
