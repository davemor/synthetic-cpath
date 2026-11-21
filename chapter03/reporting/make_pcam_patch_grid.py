#!/usr/bin/env python3
"""Create side-by-side colour-sorted PCam class grids."""

from __future__ import annotations

import argparse
import colorsys
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    raise RuntimeError("Could not locate Arial or DejaVu Sans")


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def colour_key(path: Path) -> tuple[float, float, float]:
    """Sort by mean lightness, then hue and saturation for a smooth colour progression."""
    with Image.open(path) as image:
        red, green, blue = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
    hue, saturation, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
    lightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return lightness, hue, saturation * value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    columns, rows_per_class = 8, 8
    samples_per_class = columns * rows_per_class
    tile, gap, panel_gap, margin, header = 96, 3, 28, 18, 74
    grid_width = columns * tile + (columns - 1) * gap
    grid_height = rows_per_class * tile + (rows_per_class - 1) * gap
    canvas = Image.new(
        "RGB",
        (margin * 2 + 2 * grid_width + panel_gap, margin * 2 + header + grid_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    label_font = font(38, True)
    classes = [
        ("normal", "Normal", "#E8F5E9"),
        ("tumor", "Malignant", "#FCE8EC"),
    ]

    for class_index, (folder, label, colour) in enumerate(classes):
        files = image_files(args.samples_root / folder)
        if len(files) < samples_per_class:
            raise RuntimeError(
                f"Expected at least {samples_per_class} {folder} patches; found {len(files)}"
            )
        # Descending lightness places pale patches at the top and dark patches at the bottom.
        selected = sorted(
            rng.sample(files, samples_per_class),
            key=colour_key,
            reverse=True,
        )
        section_x = margin + class_index * (grid_width + panel_gap)
        draw.rounded_rectangle(
            (section_x, margin, section_x + grid_width, margin + header - gap),
            radius=10,
            fill=colour,
        )
        bounds = draw.textbbox((0, 0), label, font=label_font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        draw.text(
            (
                section_x + (grid_width - text_width) / 2,
                margin + (header - gap - text_height) / 2 - bounds[1],
            ),
            label,
            fill="#1f2937",
            font=label_font,
        )
        for index, path in enumerate(selected):
            row, column = divmod(index, columns)
            x = section_x + column * (tile + gap)
            y = margin + header + row * (tile + gap)
            with Image.open(path) as image:
                patch = ImageOps.fit(image.convert("RGB"), (tile, tile))
            canvas.paste(patch, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95, dpi=(300, 300))
    print(f"Saved {args.output} ({canvas.width} x {canvas.height})")


if __name__ == "__main__":
    main()
