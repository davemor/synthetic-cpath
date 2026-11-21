#!/bin/bash
#SBATCH --job-name=split-patches
#SBATCH --output=/sharedscratch/dm236/data/icaird_cervical/slurm_logs/split_patches_%j.out
#SBATCH --error=/sharedscratch/dm236/data/icaird_cervical/slurm_logs/split_patches_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="${HISTOKIT_REPO_ROOT:-$HOME/repos/histokit}"
VENV_PATH="${HISTOKIT_VENV_PATH:-/sharedscratch/dm236/.venvs/patcher-3.14}"
SUBSET_DIR="${SUBSET_DIR:-/sharedscratch/dm236/data/icaird_cervical/resnet_patch_subset/stratified_120_consensus_100k}"
SAMPLE_INDEX="${SAMPLE_INDEX:-/sharedscratch/dm236/data/icaird_cervical/stratified_120/index_sampled_stratified_120_ometiff.csv}"
OVERWRITE="${OVERWRITE:-1}"

module load python/3.14.2
source "$VENV_PATH/bin/activate"

args=(
  --subset-dir "$SUBSET_DIR"
  --sample-index "$SAMPLE_INDEX"
)

if [[ "$OVERWRITE" == "1" ]]; then
  args+=(--overwrite)
fi

cd "$REPO_ROOT"
uv run python scripts/create_resnet_slide_split_dataset.py "${args[@]}"
