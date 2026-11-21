#!/usr/bin/env python3
import argparse
import contextlib
import os
import warnings
from io import BytesIO

warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import requests
import torch
from PIL import Image
from transformers.utils import logging as transformers_logging

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


DEFAULT_PROMPT = (
    "Describe this histopathology patch in neutral morphologic terms. "
    "Mention tissue architecture, cell types, staining pattern, atypia, necrosis, "
    "inflammation, and visible artifacts. Do not give a definitive diagnosis unless "
    "the image is sufficient."
)


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        response = requests.get(path_or_url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def pick_conv_mode(model_name: str) -> str:
    lowered = model_name.lower()
    if "llama-2" in lowered:
        return "llava_llama_2"
    if "v1" in lowered:
        return "llava_v1"
    if "mpt" in lowered:
        return "mpt"
    return "llava_v0"


def main() -> None:
    transformers_logging.set_verbosity_error()

    parser = argparse.ArgumentParser(description="Describe a histopathology patch with Quilt-LLaVA.")
    parser.add_argument("image", help="Path or URL to a histopathology patch image")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model-path", default="wisdomik/Quilt-Llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conv-mode", default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--image-aspect-ratio", default="pad")
    parser.add_argument("--verbose", action="store_true", help="Show model-loading warnings and progress bars")
    args = parser.parse_args()

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    with contextlib.ExitStack() as stack:
        if not args.verbose:
            stack.enter_context(contextlib.redirect_stderr(open(os.devnull, "w")))
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path,
            args.model_base,
            model_name,
            args.load_8bit,
            args.load_4bit,
            device=args.device,
        )

    conv_mode = args.conv_mode or pick_conv_mode(model_name)
    conv = conv_templates[conv_mode].copy()

    image = load_image(args.image)
    image_tensor = process_images([image], image_processor, args)
    if isinstance(image_tensor, list):
        image_tensor = [item.to(model.device, dtype=torch.float16) for item in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    prompt = args.prompt
    if model.config.mm_use_im_start_end:
        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
    else:
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        full_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    output = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
    if output.endswith(stop_str):
        output = output[: -len(stop_str)].strip()
    print(output)


if __name__ == "__main__":
    main()
