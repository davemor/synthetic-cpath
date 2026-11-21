#!/usr/bin/env bash
set -euo pipefail

IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
CONFIG="configs/experiments/main_icaird_cervical100k_budget.yaml"
DATASET="icaird_cervical100k"
GENERATION_CONTAINER="synpath-icaird-resample-corrected"
LOG_ROOT="${ROOT}/logs/icaird_resample_corrected"
MODELS=(stylegan2ada sd_lora dcgan)
CONDITIONS=(true_label pseudo_label label_global_cluster label_local_cluster)

mkdir -p "${LOG_ROOT}"
while docker ps --format '{{.Names}}' | grep -qx "${GENERATION_CONTAINER}"; do
    sleep 60
done
status="$(docker inspect -f '{{.State.ExitCode}}' "${GENERATION_CONTAINER}")"
if [[ "${status}" != 0 ]]; then
    echo "Generation failed with exit code ${status}" >&2
    exit "${status}"
fi

jobs=()
for model in "${MODELS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        jobs+=("${model}:${condition}")
    done
done

pids=()
for gpu in $(seq 0 7); do
    name="synpath-icaird-quality-${gpu}"
    docker rm -f "${name}" >/dev/null 2>&1 || true
    commands="set -euo pipefail; jobs=(${jobs[*]}); i=0; for job in \"\${jobs[@]}\"; do if (( i % 8 == ${gpu} )); then model=\${job%%:*}; condition=\${job#*:}; echo START quality \$model \$condition \$(date -Is); uv run --extra gpu synpath evaluate-quality ${DATASET} \$model \$condition --seed 0 --config ${CONFIG}; fi; i=\$((i+1)); done"
    docker run --name "${name}" --gpus all --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --env-file "${ROOT}/dgx.env" -e CUDA_VISIBLE_DEVICES="${gpu}" \
        -v "${ROOT}:${ROOT}" -v /raid/dm236/data:/raid/dm236/data:ro \
        -v "${ROOT}/repo:/workspace/synpath" -w /workspace/synpath \
        "${IMAGE}" bash scripts/docker/dgx_entrypoint.sh bash -lc "${commands}" \
        >"${LOG_ROOT}/quality_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

docker run --rm -v "${ROOT}:${ROOT}" "${IMAGE}" rm -rf \
    "${ROOT}/outputs/utility/${DATASET}/resnet18/stylegan2ada" \
    "${ROOT}/outputs/utility/${DATASET}/resnet18/sd_lora" \
    "${ROOT}/outputs/utility/${DATASET}/resnet18/dcgan"

cd "${ROOT}/repo"
SYNPATH_UTILITY_DATASET="${DATASET}" \
SYNPATH_EXPERIMENT_CONFIG="${CONFIG}" \
SYNPATH_UTILITY_SYNTHETIC_MODELS="stylegan2ada sd_lora dcgan" \
SYNPATH_UTILITY_GPUS=8 \
bash scripts/docker/run_dgx_utility_parallel.sh \
    >"${LOG_ROOT}/utility_launcher.log" 2>&1

echo "Correction pipeline complete at $(date -Is)"
