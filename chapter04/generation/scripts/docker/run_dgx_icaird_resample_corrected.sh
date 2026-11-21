#!/usr/bin/env bash
set -euo pipefail

IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
CONFIG="configs/experiments/main_icaird_cervical100k_budget.yaml"
DATASET="icaird_cervical100k"
CONDITIONS=(true_label pseudo_label label_global_cluster label_local_cluster)
MODELS=(stylegan2ada sd_lora dcgan)

cd "${ROOT}/repo"
mkdir -p "${ROOT}/logs/icaird_resample_corrected"

cleanup_paths=()
for model in "${MODELS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        arm="${ROOT}/outputs/generators/${DATASET}/${model}/${condition}/seed0"
        cleanup_paths+=(
            "${arm}/samples"
            "${arm}/metrics"
            "${arm}/synthetic_manifest.csv"
        )
    done
done
docker run --rm -v "${ROOT}:${ROOT}" "${IMAGE}" rm -rf "${cleanup_paths[@]}"

docker rm -f synpath-icaird-resample-corrected >/dev/null 2>&1 || true
docker run -d \
    --name synpath-icaird-resample-corrected \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --env-file "${ROOT}/dgx.env" \
    -e SYNPATH_SD_SAMPLE_GPUS=8 \
    -v "${ROOT}:${ROOT}" \
    -v /raid/dm236/data:/raid/dm236/data:ro \
    -v "${ROOT}/repo:/workspace/synpath" \
    -w /workspace/synpath \
    "${IMAGE}" \
    bash scripts/docker/dgx_entrypoint.sh bash -lc "
set -euo pipefail
for model in ${MODELS[*]}; do
  for condition in ${CONDITIONS[*]}; do
    echo \"START sample model=\${model} condition=\${condition} \$(date -Is)\"
    uv run --extra gpu synpath sample '${DATASET}' \"\${model}\" \"\${condition}\" --seed 0 --config '${CONFIG}'
    echo \"DONE sample model=\${model} condition=\${condition} \$(date -Is)\"
  done
done
" >"${ROOT}/logs/icaird_resample_corrected/launcher.log"

echo "container=synpath-icaird-resample-corrected"
echo "log=docker logs -f synpath-icaird-resample-corrected"
