#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-lowgen-dgx:latest}"

cd "$(dirname "$0")/.."
docker build \
  -f docker/Dockerfile.dgx \
  -t "${IMAGE_TAG}" \
  .
