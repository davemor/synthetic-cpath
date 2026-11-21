#!/usr/bin/env bash
set -euo pipefail

ROOT=/raid/dm236/synpath/outputs/reproductions/osorio_pcam_diffusers_legacy
docker run -d \
  --name osorio-pcam-preflight \
  --gpus all \
  --ipc=host \
  --env-file /raid/dm236/synpath/dgx.env \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HOME=/raid/dm236/synpath/huggingface \
  -e TRANSFORMERS_CACHE=/raid/dm236/synpath/huggingface \
  -v /raid/dm236:/raid/dm236 \
  synpath-osorio-pcam:0.15.1 \
  python -m accelerate.commands.launch \
    --num_processes 4 --multi_gpu --mixed_precision fp16 \
    --main_process_port 29600 \
    /opt/osorio/diffusers/train_text_to_image.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --train_data_dir "${ROOT}/data/class_only/train" \
    --caption_column text --resolution 512 --center_crop \
    --train_batch_size 8 --gradient_accumulation_steps 2 \
    --max_train_steps 20 --learning_rate 1e-5 --mixed_precision fp16 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --dataloader_num_workers 8 --seed 18021999 \
    --output_dir "${ROOT}/preflight"
