#!/usr/bin/env bash
set -euo pipefail

uv pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cuml-cu12==25.12.*"

echo "Installed optional RAPIDS dependencies."
