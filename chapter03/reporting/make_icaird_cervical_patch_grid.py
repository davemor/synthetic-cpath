#!/usr/bin/env python3
"""Create a four-quadrant iCAIRD Cervical 100K category grid."""

from __future__ import annotations

import argparse
import colorsys
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


CATEGORIES = [
    ("Normal", ["normal_inflammation"], "#E8F5E9"),
    ("Low grade", ["hpv", "cin1"], "#FFF8D6"),
    ("High grade", ["cin2", "cin3"], "#FFF0DC"),
    (
        "Malignant",
        ["adenocarcinoma", "cgin", "other", "squamous_carcinoma"],
        "#FCE8EC",
    ),
]
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


def colour_key(path: Path) -> tuple[float, float, float]:
    with Image.open(path) as image:
        red, green, blue = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
    hue, saturation, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
    lightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return lightness, hue, saturation * value


def category_files(root: Path, subclasses: list[str]) -> list[Path]:
    return sorted(
        path
        for subclass in subclasses
        for path in (root / "train" / subclass).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    columns = rows = 4
    samples_per_category = columns * rows
    tile, gap, panel_gap, margin, header = 160, 5, 30, 20, 82
    grid_width = columns * tile + (columns - 1) * gap
    grid_height = rows * tile + (rows - 1) * gap
    panel_width = grid_width
    panel_height = header + grid_height
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + panel_width * 2 + panel_gap,
            margin * 2 + panel_height * 2 + panel_gap,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    label_font = font(40, True)
    selections: list[tuple[str, str, list[Path]]] = []

    for category_index, (label, subclasses, colour) in enumerate(CATEGORIES):
        files = category_files(args.samples_root, subclasses)
        if len(files) < samples_per_category:
            raise RuntimeError(
                f"Expected {samples_per_category} patches for {label}; found {len(files)}"
            )
        selected = sorted(
            rng.sample(files, samples_per_category),
            key=colour_key,
            reverse=True,
        )
        selections.append((label, colour, selected))
        panel_row, panel_column = divmod(category_index, 2)
        panel_x = margin + panel_column * (panel_width + panel_gap)
        panel_y = margin + panel_row * (panel_height + panel_gap)
        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + panel_width, panel_y + header - gap),
            radius=10,
            fill=colour,
        )
        bounds = draw.textbbox((0, 0), label, font=label_font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        draw.text(
            (
                panel_x + (panel_width - text_width) / 2,
                panel_y + (header - gap - text_height) / 2 - bounds[1],
            ),
            label,
            fill="#1f2937",
            font=label_font,
        )
        for patch_index, path in enumerate(selected):
            row, column = divmod(patch_index, columns)
            x = panel_x + column * (tile + gap)
            y = panel_y + header + row * (tile + gap)
            with Image.open(path) as image:
                patch = ImageOps.fit(image.convert("RGB"), (tile, tile))
            canvas.paste(patch, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95, dpi=(300, 300))
    print(f"Saved {args.output} ({canvas.width} x {canvas.height})")

    # Also render all four categories as one 8x8 grid, with two rows per label.
    stacked_columns, stacked_rows = 8, 2
    stacked_grid_width = stacked_columns * tile + (stacked_columns - 1) * gap
    stacked_grid_height = stacked_rows * tile + (stacked_rows - 1) * gap
    section_gap = 22
    stacked_width = margin * 2 + stacked_grid_width
    stacked_height = (
        margin * 2
        + len(selections) * (header + stacked_grid_height)
        + (len(selections) - 1) * section_gap
    )
    stacked = Image.new("RGB", (stacked_width, stacked_height), "white")
    stacked_draw = ImageDraw.Draw(stacked)
    for section_index, (label, colour, selected) in enumerate(selections):
        section_y = margin + section_index * (header + stacked_grid_height + section_gap)
        stacked_draw.rounded_rectangle(
            (margin, section_y, margin + stacked_grid_width, section_y + header - gap),
            radius=10,
            fill=colour,
        )
        bounds = stacked_draw.textbbox((0, 0), label, font=label_font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        stacked_draw.text(
            (
                margin + (stacked_grid_width - text_width) / 2,
                section_y + (header - gap - text_height) / 2 - bounds[1],
            ),
            label,
            fill="#1f2937",
            font=label_font,
        )
        for patch_index, path in enumerate(selected):
            row, column = divmod(patch_index, stacked_columns)
            x = margin + column * (tile + gap)
            y = section_y + header + row * (tile + gap)
            with Image.open(path) as image:
                patch = ImageOps.fit(image.convert("RGB"), (tile, tile))
            stacked.paste(patch, (x, y))
    stacked_output = args.output.parent / "icaird-cervical-100k-8x8-grid.png"
    stacked.save(stacked_output, quality=95, dpi=(300, 300))
    print(f"Saved {stacked_output} ({stacked.width} x {stacked.height})")


if __name__ == "__main__":
    main()
