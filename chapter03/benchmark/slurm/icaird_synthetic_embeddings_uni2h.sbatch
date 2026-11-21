#!/bin/bash
#SBATCH --job-name=icaird-synth-emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/nvme/synthetic_embeddings/icaird100k_uni2h_vitdae_like/slurm/%x_%j.out
#SBATCH --error=/nvme/synthetic_embeddings/icaird100k_uni2h_vitdae_like/slurm/%x_%j.err

set -euo pipefail

CONFIG=${CONFIG:-configs/icaird_synthetic_uni2h_vitdae_like.yml}
STAGE=${STAGE:-all}
SYNTHETIC_DIR=${SYNTHETIC_DIR:-}

cd /home/dm236/icaird-cervical-benchmark
mkdir -p /nvme/synthetic_embeddings/icaird100k_uni2h_vitdae_like/slurm

if [[ "${STAGE}" == "prepare" || "${STAGE}" == "all" ]]; then
  uv run icaird-cervical-benchmark synthetic-prepare --config "${CONFIG}"
fi

if [[ "${STAGE}" == "baselines" || "${STAGE}" == "all" ]]; then
  uv run icaird-cervical-benchmark synthetic-baselines --config "${CONFIG}"
fi

if [[ "${STAGE}" == "train" || "${STAGE}" == "all" ]]; then
  uv run icaird-cervical-benchmark synthetic-train --config "${CONFIG}"
fi

if [[ "${STAGE}" == "sample" || "${STAGE}" == "all" ]]; then
  uv run icaird-cervical-benchmark synthetic-sample --config "${CONFIG}"
fi

if [[ "${STAGE}" == "evaluate" ]]; then
  if [[ -z "${SYNTHETIC_DIR}" ]]; then
    echo "Set SYNTHETIC_DIR=/path/to/generated/sample_dir for STAGE=evaluate" >&2
    exit 2
  fi
  uv run icaird-cervical-benchmark synthetic-evaluate \
    --config "${CONFIG}" \
    --synthetic-dir "${SYNTHETIC_DIR}"
fi
