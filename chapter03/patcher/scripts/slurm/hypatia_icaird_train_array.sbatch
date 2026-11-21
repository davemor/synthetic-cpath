#!/bin/bash
#SBATCH --job-name=icaird-train-cache
#SBATCH --partition=small-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/sharedscratch/dm236/data/icaird_cervical/slurm_logs/%x_%A_%a.out
#SBATCH --error=/sharedscratch/dm236/data/icaird_cervical/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${PATCHER_REPO_ROOT:-$HOME/repos/patcher}"
VENV_PATH="${PATCHER_VENV_PATH:-/sharedscratch/dm236/.venvs/patcher-3.14}"
DATASET_ROOT="${ICAIRD_DATASET_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
SLIDE_ROOT="${ICAIRD_SLIDE_ROOT:-/sharedscratch/cggm1/data/icaird_cervical/ometiff}"
ANNOTATION_ROOT="${ICAIRD_ANNOTATION_ROOT:-/sharedscratch/dm236/data/icaird_cervical/annotations}"
OUTPUT_ROOT="${PATCHER_DATA_ROOT:-/sharedscratch/dm236/data}"
CHUNK_PATH="$(printf "%s/chunks/train_chunk_%03d.csv" "$DATASET_ROOT" "${SLURM_ARRAY_TASK_ID:?}")"

if [[ ! -f "$CHUNK_PATH" ]]; then
  echo "Chunk manifest not found: $CHUNK_PATH" >&2
  exit 1
fi

module load python/3.14.2
source "$VENV_PATH/bin/activate"

cd "$REPO_ROOT"
python scripts/build_hypatia_icaird_cache.py \
  --index "$CHUNK_PATH" \
  --labels "$DATASET_ROOT/labels_tiffslide.json" \
  --slides-root "$SLIDE_ROOT" \
  --annotations-root "$ANNOTATION_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --pixels-per-patch 16
