#!/bin/bash
#SBATCH --job-name=sample-patches
#SBATCH --output=/sharedscratch/dm236/data/icaird_cervical/slurm_logs/sample_patches_%j.out
#SBATCH --error=/sharedscratch/dm236/data/icaird_cervical/slurm_logs/sample_patches_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

set -euo pipefail

REPO_ROOT="${HISTOKIT_REPO_ROOT:-$HOME/repos/histokit}"
VENV_PATH="${HISTOKIT_VENV_PATH:-/sharedscratch/dm236/.venvs/patcher-3.14}"

CONSENSUS_PATH="${CONSENSUS_PATH:-/sharedscratch/dm236/data/icaird_cervical/patcher_annotations/stratified_120/consensus/patch_consensus.parquet}"
METADATA_PATH="${METADATA_PATH:-/sharedscratch/dm236/data/icaird_cervical/patcher_annotations/stratified_120/consensus/annotation_metadata.parquet}"
PATCHES_ROOT="${PATCHES_ROOT:-/sharedscratch/dm236/data/icaird_cervical/patcher_cache/stratified_120/pipeline_research/patches}"
OUTPUT_DIR="${OUTPUT_DIR:-/sharedscratch/dm236/data/icaird_cervical/resnet_patch_subset/stratified_120_consensus_100k}"
TARGET_PER_CLASS="${TARGET_PER_CLASS:-50000}"
SEED="${SEED:-236}"
MAX_CANDIDATES_PER_SLIDE_CLASS="${MAX_CANDIDATES_PER_SLIDE_CLASS:-}"
CLUSTERS_PER_SLIDE_CLASS="${CLUSTERS_PER_SLIDE_CLASS:-}"
REQUIRE_UNANIMOUS="${REQUIRE_UNANIMOUS:-0}"
COPY_IMAGES="${COPY_IMAGES:-1}"
OVERWRITE="${OVERWRITE:-0}"

module load python/3.14.2
source "$VENV_PATH/bin/activate"

args=(
  --consensus "$CONSENSUS_PATH"
  --metadata "$METADATA_PATH"
  --patches-root "$PATCHES_ROOT"
  --output-dir "$OUTPUT_DIR"
  --target-per-class "$TARGET_PER_CLASS"
  --seed "$SEED"
)

if [[ -n "$MAX_CANDIDATES_PER_SLIDE_CLASS" ]]; then
  args+=(--max-candidates-per-slide-class "$MAX_CANDIDATES_PER_SLIDE_CLASS")
fi

if [[ -n "$CLUSTERS_PER_SLIDE_CLASS" ]]; then
  args+=(--clusters-per-slide-class "$CLUSTERS_PER_SLIDE_CLASS")
fi

if [[ "$REQUIRE_UNANIMOUS" == "1" ]]; then
  args+=(--require-unanimous)
fi

if [[ "$COPY_IMAGES" == "1" ]]; then
  args+=(--copy-images)
else
  args+=(--no-copy-images)
fi

if [[ "$OVERWRITE" == "1" ]]; then
  args+=(--overwrite)
fi

cd "$REPO_ROOT"
uv run python scripts/sample_patcher_consensus_patches.py "${args[@]}"
