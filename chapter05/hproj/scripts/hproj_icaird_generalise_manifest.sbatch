#!/bin/bash
#SBATCH --job-name=hproj-icaird-gen-manifest
#SBATCH --partition=gpu.A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/home/dm236/scratch/hproj/output/slurm/hproj-icaird-gen-manifest_%j.out
#SBATCH --error=/home/dm236/scratch/hproj/output/slurm/hproj-icaird-gen-manifest_%j.err

set -euo pipefail

export EMBEDDINGS_ROOT=/home/dm236/scratch/hproj/embeddings
export DATA_ROOT=/home/dm236/scratch/hproj/output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RUN_ID=hproj-2026-05-25_16-41-27-bb2a5203-icaird100k
RUN_ROOT="${DATA_ROOT}/runs/${RUN_ID}"
FULL_MANIFEST="${RUN_ROOT}/generalisation/task_manifest_principled_reduced_icaird_full.jsonl"
MISSING_MANIFEST="${RUN_ROOT}/generalisation/task_manifest_principled_reduced_icaird_missing.jsonl"

cd /home/dm236/repos/hproj
source .venv/bin/activate

echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID}"
echo "run_id=${RUN_ID}"
echo "full_manifest=${FULL_MANIFEST}"
echo "missing_manifest=${MISSING_MANIFEST}"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader || true

hproj generalise \
  --run-id "${RUN_ID}" \
  --task-manifest "${FULL_MANIFEST}" \
  --manifest-only \
  --tasks-only \
  --force

python - <<'PY'
import json
from pathlib import Path

full = Path("/home/dm236/scratch/hproj/output/runs/hproj-2026-05-25_16-41-27-bb2a5203-icaird100k/generalisation/task_manifest_principled_reduced_icaird_full.jsonl")
missing = Path("/home/dm236/scratch/hproj/output/runs/hproj-2026-05-25_16-41-27-bb2a5203-icaird100k/generalisation/task_manifest_principled_reduced_icaird_missing.jsonl")
records = []
for line in full.read_text().splitlines():
    if not line.strip():
        continue
    record = json.loads(line)
    task_path = Path(record["task_path"])
    if record["dataset"] == "icaird100k" and not task_path.exists():
        records.append(record)
with missing.open("w") as f:
    for record in records:
        f.write(json.dumps(record))
        f.write("\n")
print(f"missing_icaird_tasks={len(records)}")
PY
