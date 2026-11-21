#!/bin/bash
set -euo pipefail

REPO_ROOT="${PATCHER_REPO_ROOT:-$HOME/repos/patcher}"
VENV_PATH="${PATCHER_VENV_PATH:-/sharedscratch/dm236/.venvs/patcher-3.14}"
DATASET_ROOT="${ICAIRD_DATASET_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
SLIDE_ROOT="${ICAIRD_SLIDE_ROOT:-/sharedscratch/cggm1/data/icaird_cervical/ometiff}"
ANNOTATION_ROOT="${ICAIRD_ANNOTATION_ROOT:-/sharedscratch/dm236/data/icaird_cervical/annotations}"

module load python/3.14.2
source "$VENV_PATH/bin/activate"

mkdir -p "$DATASET_ROOT/chunks" "$DATASET_ROOT/slurm_logs"

cd "$REPO_ROOT"
python scripts/prepare_hypatia_icaird_dataset.py \
  --index "$DATASET_ROOT/index.csv" \
  --labels "$DATASET_ROOT/labels.json" \
  --slide-root "$SLIDE_ROOT" \
  --annotation-root "$ANNOTATION_ROOT" \
  --output-root "$DATASET_ROOT"
