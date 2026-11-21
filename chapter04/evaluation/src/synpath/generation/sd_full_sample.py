from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypeVar

import torch

from synpath.manifests import read_manifest

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a full SD pipeline.")
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--condition_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    from diffusers import StableDiffusionPipeline

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.condition_manifest)
    indexed = [
        (index, row)
        for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_index
    ]
    for batch in batched(indexed, args.batch_size):
        pending = [
            (index, row)
            for index, row in batch
            if not (args.output_dir / f"{index:08d}.png").exists()
        ]
        if not pending:
            continue
        generators = [
            torch.Generator(device=device).manual_seed(args.seed + index)
            for index, _ in pending
        ]
        images = pipe(
            [row["prompt"] for _, row in pending],
            generator=generators,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
        ).images
        for (index, _), image in zip(pending, images, strict=True):
            image.save(args.output_dir / f"{index:08d}.png")


def batched(values: list[T], size: int) -> list[list[T]]:
    if size < 1:
        raise ValueError("Batch size must be at least 1")
    return [values[start : start + size] for start in range(0, len(values), size)]


if __name__ == "__main__":
    main()
