#!/usr/bin/env python3
"""Create a class-balanced iCAIRD real-versus-synthetic patch grid."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

CLASS_ORDER = ("normal", "low_grade", "high_grade", "malignant")
CLASS_DISPLAY = {
    "normal": "Normal",
    "low_grade": "Low grade",
    "high_grade": "High grade",
    "malignant": "Malignant",
}
CLASS_COLOURS = {
    "normal": "#E8F5E9",
    "low_grade": "#FFF7D6",
    "high_grade": "#FFF0D9",
    "malignant": "#FCE8EC",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    from PIL import ImageFont

    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
        if bold
        else Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    raise RuntimeError("Arial or DejaVu Sans is required")


def discover_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_path(value: str, image_root: Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    candidates = [image_root / value, image_root / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(image_root.rglob(path.name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not uniquely resolve {value!r} under {image_root}")


def sample_manifest(
    manifest_path: Path,
    image_root: Path,
    *,
    per_class: int,
    seed: int,
    source: str,
    split: str | None = None,
) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        frame = list(csv.DictReader(handle))
    if not frame:
        raise ValueError(f"{manifest_path} is empty")
    required = {"image_path", "label"}
    if not required.issubset(frame[0]):
        raise ValueError(f"{manifest_path} must contain {sorted(required)}")
    if split is not None:
        if "split" not in frame[0]:
            raise ValueError(f"{manifest_path} does not contain a split column")
        frame = [row for row in frame if row["split"] == split]
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for label in CLASS_ORDER:
        candidates = [row for row in frame if row["label"] == label]
        if len(candidates) < per_class:
            raise ValueError(
                f"{source}/{label} has {len(candidates)} images; need {per_class}"
            )
        selected = [candidates[index] for index in sorted(
            rng.sample(range(len(candidates)), per_class)
        )]
        for row in selected:
            rows.append(
                {
                    "source": source,
                    "label": label,
                    "image_path": str(resolve_path(row["image_path"], image_root)),
                }
            )
    return rows


def render(
    rows: list[dict[str, str]],
    output: Path,
    *,
    per_class: int,
    columns: int = 4,
    source_titles: dict[str, str] | None = None,
) -> None:
    from PIL import Image, ImageDraw, ImageOps, JpegImagePlugin

    _ = JpegImagePlugin

    if per_class % columns:
        raise ValueError("per_class must be divisible by columns")
    patch_rows = per_class // columns
    tile, gap = 224, 5
    margin, source_header, class_header = 24, 84, 62
    panel_gap, row_gap = 32, 22
    panel_width = columns * tile + (columns - 1) * gap
    grid_height = patch_rows * tile + (patch_rows - 1) * gap
    row_height = class_header + grid_height
    sources = list(dict.fromkeys(row["source"] for row in rows))
    if not sources:
        raise ValueError("No image rows were provided")
    source_titles = source_titles or {source: source for source in sources}
    width = margin * 2 + panel_width * len(sources) + panel_gap * (len(sources) - 1)
    height = (
        margin * 2
        + source_header
        + len(CLASS_ORDER) * row_height
        + (len(CLASS_ORDER) - 1) * row_gap
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    source_font = font(38, bold=True)
    class_font = font(32, bold=True)

    for source_index, source in enumerate(sources):
        x0 = margin + source_index * (panel_width + panel_gap)
        title = source_titles.get(source, source)
        bounds = draw.textbbox((0, 0), title, font=source_font)
        text_width = bounds[2] - bounds[0]
        draw.text(
            (x0 + (panel_width - text_width) / 2, margin - bounds[1]),
            title,
            fill="#111827",
            font=source_font,
        )
        for row_index, label in enumerate(CLASS_ORDER):
            y0 = margin + source_header + row_index * (row_height + row_gap)
            draw.rounded_rectangle(
                (x0, y0, x0 + panel_width, y0 + class_header - gap),
                radius=9,
                fill=CLASS_COLOURS[label],
            )
            display = CLASS_DISPLAY[label]
            bounds = draw.textbbox((0, 0), display, font=class_font)
            draw.text(
                (
                    x0 + (panel_width - (bounds[2] - bounds[0])) / 2,
                    y0 + (class_header - gap - (bounds[3] - bounds[1])) / 2 - bounds[1],
                ),
                display,
                fill="#1F2937",
                font=class_font,
            )
            selected = [
                row for row in rows
                if row["source"] == source and row["label"] == label
            ]
            if len(selected) != per_class:
                raise ValueError(f"Expected {per_class} rows for {source}/{label}")
            for index, row in enumerate(selected):
                patch_row, patch_column = divmod(index, columns)
                x = x0 + patch_column * (tile + gap)
                y = y0 + class_header + patch_row * (tile + gap)
                with Image.open(row["image_path"]) as image:
                    patch = ImageOps.fit(image.convert("RGB"), (tile, tile))
                canvas.paste(patch, (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300))
    canvas.save(output.with_suffix(".pdf"), resolution=300)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--comparison-manifest", type=Path)
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=16)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    rows = sample_manifest(
        args.real_manifest,
        args.real_root,
        per_class=args.per_class,
        seed=args.seed,
        source="Real",
        split="train",
    )
    rows.extend(
        sample_manifest(
            args.synthetic_manifest,
            args.synthetic_root,
            per_class=args.per_class,
            seed=args.seed,
            source="StyleGAN2-ADA",
        )
    )
    if bool(args.comparison_manifest) != bool(args.comparison_root):
        parser.error("--comparison-manifest and --comparison-root must be used together")
    if args.comparison_manifest:
        rows.extend(
            sample_manifest(
                args.comparison_manifest,
                args.comparison_root,
                per_class=args.per_class,
                seed=args.seed,
                source="DCGAN",
            )
        )
    sample_path = args.output.with_name(f"{args.output.stem}-samples.csv")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "label", "image_path"])
        writer.writeheader()
        writer.writerows(rows)
    if args.select_only:
        return
    source_titles = {
        "Real": "Real",
        "StyleGAN2-ADA": "StyleGAN2-ADA (FID: 19.66)",
        "DCGAN": "DCGAN (FID: 182.57)",
    }
    render(
        rows,
        args.output,
        per_class=args.per_class,
        columns=args.columns,
        source_titles=source_titles,
    )
    args.output.with_name(f"{args.output.stem}-metadata.json").write_text(
        json.dumps(
            {
                "dataset": "iCAIRD Cervical 100K",
                "synthetic_generator": "StyleGAN2-ADA",
                "synthetic_conditioning": "class + global cluster",
                "synthetic_fid": 19.6557,
                "comparison_generator": "DCGAN",
                "comparison_conditioning": "class label",
                "comparison_fid": 182.5679,
                "seed": args.seed,
                "patches_per_class_per_source": args.per_class,
                "classes": list(CLASS_ORDER),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
