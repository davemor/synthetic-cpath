#!/usr/bin/env python3
"""Create a labelled grid of randomly sampled Kather100K image patches."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


CLASS_NAMES = {
    "ADI": "Adipose tissue",
    "BACK": "Background",
    "DEB": "Debris",
    "LYM": "Lymphocytes",
    "MUC": "Mucus",
    "MUS": "Smooth muscle",
    "NORM": "Normal colon mucosa",
    "STR": "Cancer-associated stroma",
    "TUM": "Colorectal adenocarcinoma",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def wrap_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    label_font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap a class label to fit the label panel."""
    lines: list[str] = []
    for word in label.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and draw.textlength(candidate, font=label_font) > max_width:
            lines.append(word)
        elif lines:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    split_root = args.dataset_root / args.split
    classes = [code for code in CLASS_NAMES if (split_root / code).is_dir()]
    if not classes:
        raise RuntimeError(f"No Kather100K classes found under {split_root}")

    patch_size, gap, label_width, top = 224, 10, 560, 100
    margin = 22
    width = margin * 2 + label_width + args.samples * patch_size + (args.samples - 1) * gap
    height = top + margin + len(classes) * patch_size + (len(classes) - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, name_font = font(46, True), font(48, True)
    draw.text((margin, 18), "Kather100K tissue classes", fill="#171717", font=title_font)

    for row, code in enumerate(classes):
        files = sorted(
            p for p in (split_root / code).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(files) < args.samples:
            raise RuntimeError(f"{code} contains only {len(files)} images")
        selected = rng.sample(files, args.samples)
        y = top + margin + row * (patch_size + gap)
        draw.rounded_rectangle(
            (margin, y, margin + label_width - gap, y + patch_size),
            radius=10,
            fill="#f1f3f5",
        )
        lines = wrap_label(draw, CLASS_NAMES[code], name_font, label_width - 58)
        line_height = 58
        label_y = y + (patch_size - len(lines) * line_height) // 2
        for line_number, line in enumerate(lines):
            draw.text(
                (margin + 24, label_y + line_number * line_height),
                line,
                fill="#1f2937",
                font=name_font,
            )
        for col, image_path in enumerate(selected):
            x = margin + label_width + col * (patch_size + gap)
            with Image.open(image_path) as image:
                tile = ImageOps.fit(image.convert("RGB"), (patch_size, patch_size))
            canvas.paste(tile, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95, dpi=(300, 300))
    print(args.output)


if __name__ == "__main__":
    main()
