#!/usr/bin/env bash
set -euo pipefail

ROOT="${SYNPATH_DATA_ROOT:-data}/raw/spider"

for ORGAN in skin colorectal thorax breast; do
    uv run python scripts/setup/download_spider.py \
        --organ "${ORGAN}" \
        --root "${ROOT}"
done
