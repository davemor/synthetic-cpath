#!/usr/bin/env bash
set -euo pipefail

PARTITION="${ICAIRD_BAGS_OFFICIAL_GRANDQC_PARTITION:-gpu.L40S}"
TIME_LIMIT="${ICAIRD_BAGS_OFFICIAL_GRANDQC_TIME_LIMIT:-1-00:00:00}"
CPUS="${ICAIRD_BAGS_OFFICIAL_GRANDQC_CPUS:-8}"
MEMORY="${ICAIRD_BAGS_OFFICIAL_GRANDQC_MEMORY:-64G}"
GRES="${ICAIRD_BAGS_OFFICIAL_GRANDQC_GRES:-gpu:1}"
CONCURRENCY="${ICAIRD_BAGS_OFFICIAL_GRANDQC_CONCURRENCY:-4}"
EXCLUDE="${ICAIRD_BAGS_OFFICIAL_GRANDQC_EXCLUDE:-}"
DATA_ROOT="${ICAIRD_DATA_ROOT:-/sharedscratch/dm236/data/icaird_cervical}"
OUTPUT_ROOT="${PATCH_ENCODER_OUTPUT_ROOT:-${DATA_ROOT}/bags/full_canny005_or_resnet_light_grandqc_artifact}"
LOG_ROOT="${DATA_ROOT}/slurm_logs"
GRANDQC_REPO="${GRANDQC_OFFICIAL_REPO:-/sharedscratch/dm236/src/grandqc/02_WSI_inference_OME_TIFF_QC}"
GRANDQC_PYTHON="${GRANDQC_OFFICIAL_PYTHON:-/sharedscratch/dm236/.venvs/grandqc-official-py310/bin/python}"
SLIDES_ROOT="${ICAIRD_SLIDES_ROOT:-/sharedscratch/dcchb-group/datasets/icaird_cervical/ometiff}"
CHUNK_DIR="${ICAIRD_BAGS_OFFICIAL_GRANDQC_CHUNK_DIR:-${OUTPUT_ROOT}/chunks/grandqc_artifact}"
MASK_ROOT="${OUTPUT_ROOT}/grandqc_artifact_masks"
MPP_MODEL="${ICAIRD_BAGS_OFFICIAL_GRANDQC_MPP:-1.5}"
CREATE_GEOJSON="${ICAIRD_BAGS_OFFICIAL_GRANDQC_CREATE_GEOJSON:-N}"
OVERWRITE="${ICAIRD_BAGS_OFFICIAL_GRANDQC_OVERWRITE:-0}"

mkdir -p "$LOG_ROOT" "$MASK_ROOT"

chunk_count="$(find "$CHUNK_DIR" -maxdepth 1 -name 'chunk_*.csv' | wc -l)"
if [[ "$chunk_count" -lt 1 ]]; then
  echo "No official GrandQC chunks found in $CHUNK_DIR. Run inventory first." >&2
  exit 1
fi
last_task=$((chunk_count - 1))

dependency_args=()
if [[ -n "${ICAIRD_BAGS_OFFICIAL_GRANDQC_DEPENDENCY_JOB_ID:-}" ]]; then
  dependency_args=(--dependency="afterok:${ICAIRD_BAGS_OFFICIAL_GRANDQC_DEPENDENCY_JOB_ID}")
fi
exclude_args=()
if [[ -n "$EXCLUDE" ]]; then
  exclude_args=(--exclude="$EXCLUDE")
fi

sbatch --parsable \
  "${dependency_args[@]}" \
  "${exclude_args[@]}" \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --cpus-per-task="$CPUS" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT" \
  --array="0-${last_task}%${CONCURRENCY}" \
  --job-name=icaird-bag-gqc \
  --output="${LOG_ROOT}/icaird_bags_official_grandqc_%A_%a.out" \
  --error="${LOG_ROOT}/icaird_bags_official_grandqc_%A_%a.err" \
  --wrap="chunk=\$(printf '${CHUNK_DIR}/chunk_%04d.csv' \${SLURM_ARRAY_TASK_ID}); while IFS=, read -r slide_id; do [[ \"\$slide_id\" == slide_id || -z \"\$slide_id\" ]] && continue; slide_name=\"\${slide_id}.ome.tiff\"; slide_path=\"${SLIDES_ROOT}/\${slide_name}\"; out_dir=\"${MASK_ROOT}/\${slide_id}\"; mask_path=\"\${out_dir}/mask_qc/\${slide_name}_mask.png\"; if [[ -f \"\$mask_path\" && \"${OVERWRITE}\" != 1 ]]; then echo \"Skipping existing official GrandQC mask for \$slide_id\"; continue; fi; input_dir=\"\${out_dir}/slides\"; rm -rf \"\$input_dir\"; mkdir -p \"\$input_dir\" \"\$out_dir\"; ln -s \"\$slide_path\" \"\${input_dir}/\${slide_name}\"; cd ${GRANDQC_REPO}; ${GRANDQC_PYTHON} wsi_tis_detect.py --slide_folder \"\$input_dir\" --output_dir \"\${out_dir}/\"; ${GRANDQC_PYTHON} main.py --slide_folder \"\$input_dir\" --output_dir \"\${out_dir}/\" --create_geojson \"${CREATE_GEOJSON}\" --mpp_model \"${MPP_MODEL}\"; done < \"\$chunk\""
