#!/usr/bin/env bash
set -euo pipefail

IMAGE="${SYNPATH_DOCKER_IMAGE:-synpath:latest}"
RUN_ROOT="${SYNPATH_DGX_ROOT:-/raid/dm236/synpath}"
ENV_FILE="${SYNPATH_DGX_ENV_FILE:-${RUN_ROOT}/dgx.env}"
CONFIG="${SYNPATH_EXPERIMENT_CONFIG:-configs/experiments/main_pathmnist224_budget.yaml}"
MODEL="stylegan2ada_10k"
SOURCES=(synthetic_only_true_label real_plus_synthetic_true_label)
SEEDS=(0 1 2)
WORKERS="${SYNPATH_UTILITY_NUM_WORKERS:-10}"

mkdir -p "${RUN_ROOT}/logs/kather_stylegan2ada_10k"

gpu=0
for source in "${SOURCES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        name="synpath-kather-sg2-10k-${source}-seed${seed}"
        log="${RUN_ROOT}/logs/kather_stylegan2ada_10k/${source}-seed${seed}.log"
        docker rm -f "${name}" >/dev/null 2>&1 || true
        docker run -d \
            --name "${name}" \
            --gpus all \
            --ipc=host \
            --ulimit memlock=-1 \
            --ulimit stack=67108864 \
            --env-file "${ENV_FILE}" \
            -e CUDA_VISIBLE_DEVICES="${gpu}" \
            -e SYNPATH_UTILITY_NUM_WORKERS="${WORKERS}" \
            -e OMP_NUM_THREADS=2 \
            -e MKL_NUM_THREADS=2 \
            -v "${RUN_ROOT}:${RUN_ROOT}" \
            -v "${PWD}:/workspace/synpath" \
            -w /workspace/synpath \
            "${IMAGE}" \
            bash scripts/docker/dgx_entrypoint.sh bash -lc \
            "uv run --extra gpu synpath train-utility pathmnist224 ${source} --synthetic-model ${MODEL} --seed ${seed} --config ${CONFIG}" \
            >"${log}" 2>&1
        echo "gpu=${gpu} container=${name} log=${log}"
        gpu=$((gpu + 1))
    done
done
