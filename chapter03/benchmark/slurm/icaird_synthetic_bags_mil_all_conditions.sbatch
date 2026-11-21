#!/bin/bash
#SBATCH --job-name=icaird-synth-mil-all
#SBATCH --partition=gpu.A100,gpu.L40S
#SBATCH --exclude=gpu02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-17%18
#SBATCH --output=/home/dm236/scratch/icaird_cervical_benchmark/slurm/icaird-synth-mil-all_%A_%a.out
#SBATCH --error=/home/dm236/scratch/icaird_cervical_benchmark/slurm/icaird-synth-mil-all_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/dm236/repos/icaird-cervical-benchmark}"
SYNTHETIC_ROOT="${SYNTHETIC_ROOT:-/sharedscratch/dm236/synthetic_bags/icaird_cervical_conditional_edm_mlp_a100_sharded}"
RUN_GROUP="${RUN_GROUP:-icaird_mil_uni2h_synthetic_bags_a100_sharded_$(date +%Y%m%d_%H%M%S)}"

CONDITIONS=(
  "synthetic_only_0p5x"
  "synthetic_only_1x"
  "synthetic_only_2x"
  "real_plus_synthetic_0p5x"
  "real_plus_synthetic_1x"
  "real_plus_synthetic_2x"
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
CONDITION_INDEX=$((TASK_ID / 3))
MIL_TASK_INDEX=$((TASK_ID % 3))

if [[ "${CONDITION_INDEX}" -ge "${#CONDITIONS[@]}" ]]; then
  echo "array task ${TASK_ID} exceeds condition list; exiting"
  exit 0
fi

CONDITION="${CONDITIONS[${CONDITION_INDEX}]}"
CONFIG="${SYNTHETIC_ROOT}/mil_configs/icaird_cervical_mil_uni2h_${CONDITION}.yml"
MANIFEST="${SYNTHETIC_ROOT}/mil_configs/manifests/${CONDITION}.jsonl"
RUN_NAME="${RUN_GROUP}_${CONDITION}"

cd "${PROJECT_ROOT}"
mkdir -p /home/dm236/scratch/icaird_cervical_benchmark/slurm

echo "host=$(hostname)"
echo "job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}} array_task=${SLURM_ARRAY_TASK_ID:-manual}"
echo "condition=${CONDITION}"
echo "mil_task_index=${MIL_TASK_INDEX}"
echo "config=${CONFIG}"
echo "manifest=${MANIFEST}"
echo "run_name=${RUN_NAME}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader || true

uv run --no-sync icaird-cervical-benchmark run-task \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --task-index "${MIL_TASK_INDEX}" \
  --run-name "${RUN_NAME}"
