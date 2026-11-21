export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="/nvme/cervical_processed2/patches_dataset_hf"
export OUTPUT_DIR="/nvme/cervical_processed2/trained-sd3-lora-patches"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --caption_column=additional_feature \
  --checkpoints_total_limit=3
