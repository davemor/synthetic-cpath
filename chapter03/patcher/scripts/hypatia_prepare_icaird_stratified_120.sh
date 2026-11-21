#!/bin/bash
set -euo pipefail

REPO_ROOT="${PATCHER_REPO_ROOT:-$HOME/repos/patcher}"
VENV_PATH="${PATCHER_VENV_PATH:-/sharedscratch/dm236/.venvs/patcher-3.14}"
SOURCE_DATASET_ROOT="${ICAIRD_SOURCE_DATASET_ROOT:-/sharedscratch/dcchb-group/datasets/icaird_cervical}"
DATASET_ROOT="${ICAIRD_DATASET_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
SAMPLE_ROOT="${ICAIRD_STRATIFIED_SAMPLE_ROOT:-$DATASET_ROOT/stratified_120}"
SAMPLE_CSV="${ICAIRD_STRATIFIED_SAMPLE_CSV:-$DATASET_ROOT/sampled_stratified_120_from_train.csv}"
CHUNK_SIZE="${ICAIRD_STRATIFIED_CHUNK_SIZE:-2}"

module load python/3.14.2
source "$VENV_PATH/bin/activate"

mkdir -p "$SAMPLE_ROOT/chunks" "$DATASET_ROOT/slurm_logs"

cd "$REPO_ROOT"
python scripts/prepare_hypatia_icaird_stratified_sample.py \
  --sample-csv "$SAMPLE_CSV" \
  --dataset-index "$SOURCE_DATASET_ROOT/index.csv" \
  --labels "$SOURCE_DATASET_ROOT/labels.json" \
  --output-root "$SAMPLE_ROOT" \
  --chunk-size "$CHUNK_SIZE"
