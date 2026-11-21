#!/usr/bin/env bash
set -euo pipefail

IMAGE="${OSORIO_IMAGE:-synpath-osorio-pcam:0.15.1}"
REPO="${SYNPATH_REPO:-/home/dm236/synpath}"
RUN_ROOT="${OSORIO_RUN_ROOT:-/raid/dm236/synpath/outputs/reproductions/osorio_pcam_diffusers_legacy}"
CACHE_ROOT="/raid/dm236/synpath/cache/cluster_sd_index_calibration_pcam_kaggle_osorio_released_dino/pcam"
HF_HOME="/raid/dm236/synpath/huggingface"

mkdir -p "${RUN_ROOT}/logs"

docker run --rm \
  -v /raid/dm236:/raid/dm236 \
  -v "${REPO}:/workspace/synpath:ro" \
  -w /workspace/synpath \
  synpath:latest uv run python scripts/audits/prepare_osorio_pcam.py \
    --curated-manifest "${CACHE_ROOT}/curation/curated_manifest.csv" \
    --cluster-labels "${CACHE_ROOT}/dino_vitb8/fits/k_33_labels.npy" \
    --output-root "${RUN_ROOT}/data"

run_arm() {
  local arm="$1" gpus="$2" port="$3"
  docker rm -f "osorio-pcam-${arm}" >/dev/null 2>&1 || true
  docker run -d \
    --name "osorio-pcam-${arm}" \
    --gpus all \
    --ipc=host \
    -e CUDA_VISIBLE_DEVICES="${gpus}" \
    -e HF_HOME="${HF_HOME}" \
    -e TRANSFORMERS_CACHE="${HF_HOME}" \
    -v /raid/dm236:/raid/dm236 \
    "${IMAGE}" \
    python -m accelerate.commands.launch \
      --num_processes 4 --multi_gpu --mixed_precision fp16 \
      --main_process_port "${port}" \
      /opt/osorio/diffusers/train_text_to_image.py \
      --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
      --train_data_dir "${RUN_ROOT}/data/${arm}/train" \
      --caption_column text --resolution 512 --center_crop \
      --train_batch_size 8 --gradient_accumulation_steps 2 \
      --max_train_steps 12500 --learning_rate 1e-5 \
      --mixed_precision fp16 --gradient_checkpointing \
      --enable_xformers_memory_efficient_attention \
      --dataloader_num_workers 8 --seed 18021999 \
      --checkpointing_steps 2500 --checkpoints_total_limit 1 \
      --output_dir "${RUN_ROOT}/${arm}/model"
}

run_arm class_only 0,1,2,3 29601
run_arm mep 4,5,6,7 29602

cat >"${RUN_ROOT}/launch_metadata.txt" <<EOF
image=${IMAGE}
diffusers=0.15.1
torch=1.13.1
steps=12500
resolution=512
per_gpu_batch=8
gradient_accumulation=2
gpus_per_arm=4
effective_batch=64
learning_rate=1e-5
precision=fp16
seed=18021999
EOF

echo "class_only: docker logs -f osorio-pcam-class_only"
echo "mep:        docker logs -f osorio-pcam-mep"
echo "outputs:    ${RUN_ROOT}"
