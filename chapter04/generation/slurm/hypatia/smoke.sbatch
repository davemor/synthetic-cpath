#!/usr/bin/env bash
#SBATCH --job-name=synpath-smoke
#SBATCH --partition=gpu.L40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

export UV_PYTHON="${UV_PYTHON:-3.12}"
export SYNPATH_SCRATCH_ROOT="${SYNPATH_SCRATCH_ROOT:-${HOME}/scratch/synpath}"
export SYNPATH_DATA_ROOT="${SYNPATH_DATA_ROOT:-${SYNPATH_SCRATCH_ROOT}/data}"
export SYNPATH_CACHE_ROOT="${SYNPATH_CACHE_ROOT:-${SYNPATH_SCRATCH_ROOT}/cache}"
export SYNPATH_OUTPUTS_ROOT="${SYNPATH_OUTPUTS_ROOT:-${SYNPATH_SCRATCH_ROOT}/outputs}"
export SYNPATH_REPORTS_ROOT="${SYNPATH_REPORTS_ROOT:-${SYNPATH_SCRATCH_ROOT}/reports}"
export SYNPATH_EXTERNAL_ROOT="${SYNPATH_EXTERNAL_ROOT:-${SYNPATH_SCRATCH_ROOT}/external}"
export SYNPATH_STYLEGAN_BACKEND="${SYNPATH_STYLEGAN_BACKEND:-stylegan3}"
export SYNPATH_STYLEGAN_PYTHON="${SYNPATH_STYLEGAN_PYTHON:-${SYNPATH_EXTERNAL_ROOT}/stylegan2ada-env/bin/python}"
export SYNPATH_CUDA_MODULE="${SYNPATH_CUDA_MODULE:-nvidia-cuda/12.1.1}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${SYNPATH_SCRATCH_ROOT}/torch_extensions}"
export HF_HOME="${HF_HOME:-${SYNPATH_SCRATCH_ROOT}/huggingface}"
export SYNPATH_EMBED_DEVICE="${SYNPATH_EMBED_DEVICE:-cpu}"
if [[ -n "${SYNPATH_CUDA_MODULE}" ]]; then
    module load "${SYNPATH_CUDA_MODULE}"
fi
mkdir -p \
    "${SYNPATH_DATA_ROOT}" \
    "${SYNPATH_CACHE_ROOT}" \
    "${SYNPATH_OUTPUTS_ROOT}" \
    "${SYNPATH_REPORTS_ROOT}" \
    "${SYNPATH_EXTERNAL_ROOT}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${HF_HOME}"

echo "job=${SLURM_JOB_NAME}"
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "submit_dir=${SLURM_SUBMIT_DIR}"
echo "partition=${SLURM_JOB_PARTITION}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "synpath_scratch_root=${SYNPATH_SCRATCH_ROOT}"
echo "synpath_data_root=${SYNPATH_DATA_ROOT}"
echo "synpath_cache_root=${SYNPATH_CACHE_ROOT}"
echo "synpath_outputs_root=${SYNPATH_OUTPUTS_ROOT}"
echo "synpath_reports_root=${SYNPATH_REPORTS_ROOT}"
echo "synpath_external_root=${SYNPATH_EXTERNAL_ROOT}"
echo "synpath_stylegan_backend=${SYNPATH_STYLEGAN_BACKEND}"
echo "synpath_stylegan_python=${SYNPATH_STYLEGAN_PYTHON}"
echo "synpath_cuda_module=${SYNPATH_CUDA_MODULE}"
echo "torch_extensions_dir=${TORCH_EXTENSIONS_DIR}"

uv run synpath run --config configs/experiments/smoke.yaml --resume
