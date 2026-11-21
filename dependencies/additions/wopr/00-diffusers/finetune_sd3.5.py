import os
import torch
import argparse
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel
from diffusers.utils import load_image
from tqdm import tqdm

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="./dataset/", help="Path to dataset folder")
parser.add_argument("--output_dir", type=str, default="./output/", help="Where to save fine-tuned model")
parser.add_argument("--pretrained_model", type=str, default="stabilityai/stable-diffusion-3.5-medium", help="Base model")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--use_ema", action="store_true", help="Use EMA for stable training")
parser.add_argument("--max_train_steps", type=int, default=20000, help="Max training steps")
parser.add_argument("--save_every", type=int, default=500, help="Save model every N steps")
parser.add_argument("--resolution", type=int, default=256, help="Image resolution")
parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank for efficient finetuning")
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Load dataset from ImageFolder
dataset = load_dataset("imagefolder", data_dir=args.dataset_dir)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((args.resolution, args.resolution)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Apply transformations to dataset
def preprocess(example):
    image = Image.open(example["file_path"]).convert("RGB")
    example["pixel_values"] = transform(image)
    example["text"] = example["label"]  # Use class label as prompt
    return example

dataset = dataset.map(preprocess, remove_columns=["file_path"])

# Create DataLoader
dataloader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True)

# Load Model Components
pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

# Apply LoRA
lora_config = LoraConfig(
    r=args.lora_rank, lora_alpha=16, target_modules=["attn1.to_q", "attn1.to_v"],
    lora_dropout=0.1, bias="none"
)
pipe.unet = get_peft_model(pipe.unet, lora_config)

# Enable GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

# Use EMA (if enabled)
if args.use_ema:
    ema_model = EMAModel(pipe.unet)

# Optimizer
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=args.lr)

# Training Loop
accelerator = Accelerator()

pipe.unet.train()
for epoch in range(args.num_epochs):
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")):
        pixel_values = batch["pixel_values"].to(device)
        text_inputs = pipe.tokenizer(batch["text"], padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)

        loss = pipe.unet(pixel_values, encoder_hidden_states=pipe.text_encoder(text_inputs)).loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # EMA Update
        if args.use_ema:
            ema_model.step(pipe.unet)

        # Save model
        if step % args.save_every == 0:
            pipe.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{epoch+1}-{step}"))

# Save final model
pipe.save_pretrained(args.output_dir)

# Generate images for each class
os.makedirs(f"{args.output_dir}/generated_images", exist_ok=True)

for label in dataset["train"].features["label"].names:
    prompt = label
    images = pipe(prompt, num_inference_steps=50).images
    for i, img in enumerate(images):
        img.save(f"{args.output_dir}/generated_images/{label}_{i}.png")
