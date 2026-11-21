#!/usr/bin/env python3
"""Create a contact sheet of the strongest GrandQC-background patches."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


Image.MAX_IMAGE_PIXELS = None
BACKGROUND_CLASS_ID = 7
OFFICIAL_GRANDQC_MPP = 0.5
DEFAULT_DATASET_ROOT = Path(
    "/home/dm236/scratch/data/icaird_cervical/icaird_cervical_100k_uni2_kcenter"
)
DEFAULT_INDEX = Path("/home/dm236/scratch/data/icaird_cervical/index_train_ometiff.csv")
DEFAULT_SLIDES_ROOT = Path("/mnt/histology/outer_hebrides/cervical/20240426_ometiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--slides-root", type=Path, default=DEFAULT_SLIDES_ROOT)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--patch-level", type=int, default=1)
    parser.add_argument("--official-grandqc-mpp", type=float, default=OFFICIAL_GRANDQC_MPP)
    parser.add_argument("--official-grandqc-model-patch-size", type=int, default=256)
    parser.add_argument("--thumb-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stop-at-perfect", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_manifest(dataset_root: Path) -> Path:
    for path in [
        dataset_root / "manifests" / "sample_manifest.parquet",
        dataset_root / "sample_manifest.parquet",
    ]:
        if path.exists():
            return path
    raise FileNotFoundError(f"sample_manifest.parquet not found under {dataset_root}")


def slide_id_from_name(name: str) -> str:
    return Path(name).name.removesuffix(".ome.tiff").removesuffix(".tiff").removesuffix(".svs")


def load_index(path: Path, slides_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["slide"] = frame["slide"].astype(str)
    frame["slide_id"] = frame["slide"].map(slide_id_from_name)
    frame["slide_path"] = frame["slide"].map(lambda name: str(slides_root / name))
    return frame


def slide_mpp(slide: object) -> float:
    props = slide.properties
    for prefix in ("tiffslide", "openslide"):
        try:
            mpp_x = float(props[f"{prefix}.mpp-x"])
            mpp_y = float(props[f"{prefix}.mpp-y"])
            return (mpp_x + mpp_y) / 2.0
        except (KeyError, TypeError, ValueError):
            continue
    raise ValueError("Slide lacks MPP metadata")


def mask_mapping(
    *,
    slide_path: str,
    patch_size: int,
    patch_level: int,
    official_grandqc_mpp: float,
    official_grandqc_model_patch_size: int,
) -> tuple[float, float]:
    from tiffslide import TiffSlide

    with TiffSlide(slide_path) as slide:
        downsample = float(slide.level_downsamples[patch_level])
        mpp = slide_mpp(slide)
    p_s = int(official_grandqc_mpp / mpp * official_grandqc_model_patch_size)
    scale = official_grandqc_model_patch_size / p_s
    return downsample, scale


def background_fraction(mask: np.ndarray, y_mask: int, x_mask: int, window: int) -> float:
    half = max(1, window // 2)
    y0 = max(0, int(y_mask) - half)
    x0 = max(0, int(x_mask) - half)
    y1 = min(mask.shape[0], int(y_mask) + half)
    x1 = min(mask.shape[1], int(x_mask) + half)
    crop = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return float("nan")
    return float(np.mean(crop == BACKGROUND_CLASS_ID))


def background_fractions(mask: np.ndarray, y_mask: np.ndarray, x_mask: np.ndarray, window: int) -> np.ndarray:
    half = max(1, window // 2)
    y0 = np.maximum(0, y_mask.astype(int) - half)
    x0 = np.maximum(0, x_mask.astype(int) - half)
    y1 = np.minimum(mask.shape[0], y_mask.astype(int) + half)
    x1 = np.minimum(mask.shape[1], x_mask.astype(int) + half)
    binary = (mask == BACKGROUND_CLASS_ID).astype(np.uint8)
    integral = np.pad(binary, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    counts = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    areas = np.maximum(1, (y1 - y0) * (x1 - x0))
    return counts / areas


def score_background_rows(
    frame: pd.DataFrame,
    *,
    patch_size: int,
    patch_level: int,
    official_grandqc_mpp: float,
    official_grandqc_model_patch_size: int,
    stop_at_perfect: bool,
    top_k: int,
) -> pd.DataFrame:
    scored: list[pd.DataFrame] = []
    grouped = list(frame.groupby(["slide_id", "official_grandqc_artifact_mask_path"], sort=False))
    grouped.sort(key=lambda item: len(item[1]), reverse=True)
    for slide_index, ((slide_id, mask_path), group) in enumerate(grouped, start=1):
        print(f"Scoring {slide_index}/{len(grouped)} {slide_id} ({len(group)} background rows)")
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        downsample, scale = mask_mapping(
            slide_path=str(group["slide_path"].iloc[0]),
            patch_size=patch_size,
            patch_level=patch_level,
            official_grandqc_mpp=official_grandqc_mpp,
            official_grandqc_model_patch_size=official_grandqc_model_patch_size,
        )
        x_mask = np.clip(
            np.floor((group["column"].to_numpy(dtype=float) + 0.5) * patch_size * downsample * scale),
            0,
            mask.shape[1] - 1,
        ).astype(int)
        y_mask = np.clip(
            np.floor((group["row"].to_numpy(dtype=float) + 0.5) * patch_size * downsample * scale),
            0,
            mask.shape[0] - 1,
        ).astype(int)
        window = max(1, int(round(patch_size * downsample * scale)))
        group = group.copy()
        group["grandqc_background_fraction"] = background_fractions(mask, y_mask, x_mask, window)
        group["grandqc_mask_y"] = y_mask
        group["grandqc_mask_x"] = x_mask
        group["grandqc_mask_window"] = window
        scored.append(group)
        if stop_at_perfect:
            current = pd.concat(scored, ignore_index=True)
            if int((current["grandqc_background_fraction"] >= 1.0).sum()) >= top_k:
                print(f"Stopping early after finding at least {top_k} perfect background scores")
                break
    return pd.concat(scored, ignore_index=True)


def contact_sheet(frame: pd.DataFrame, output_path: Path, thumb_size: int) -> None:
    cols = 6
    label_h = 38
    rows = math.ceil(len(frame) / cols)
    sheet = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(frame.itertuples(index=False)):
        thumb = Image.open(row.patch_png_path).convert("RGB")
        thumb.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        x = (index % cols) * thumb_size
        y = (index // cols) * (thumb_size + label_h)
        tile = Image.new("RGB", (thumb_size, thumb_size), "white")
        tile.paste(thumb, ((thumb_size - thumb.width) // 2, (thumb_size - thumb.height) // 2))
        sheet.paste(tile, (x, y))
        label = f"{index + 1:02d} bg={row.grandqc_background_fraction:.3f}"
        draw.rectangle((x, y + thumb_size, x + thumb_size, y + thumb_size + label_h), fill="white")
        draw.text((x + 6, y + thumb_size + 5), label, fill="black")
        draw.text((x + 6, y + thumb_size + 20), f"{row.slide_id} r{row.row} c{row.column}", fill="black")
    sheet.save(output_path, quality=95)


def main() -> None:
    args = parse_args()
    manifest_path = resolve_manifest(args.dataset_root)
    out_dir = args.output_dir or (args.dataset_root / "audit_grandqc_examples")
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "slide_id",
        "row",
        "column",
        "patch_png_path",
        "official_grandqc_artifact_class_id",
        "official_grandqc_artifact_class",
        "official_grandqc_artifact_mask_path",
        "slide_path",
    ]
    frame = pd.read_parquet(manifest_path, columns=columns)
    background = frame.loc[frame["official_grandqc_artifact_class_id"] == BACKGROUND_CLASS_ID].copy()
    scored = score_background_rows(
        background,
        patch_size=args.patch_size,
        patch_level=args.patch_level,
        official_grandqc_mpp=args.official_grandqc_mpp,
        official_grandqc_model_patch_size=args.official_grandqc_model_patch_size,
        stop_at_perfect=args.stop_at_perfect,
        top_k=args.top_k,
    )
    top = scored.sort_values(
        ["grandqc_background_fraction", "slide_id", "row", "column"],
        ascending=[False, True, True, True],
    ).head(args.top_k)
    csv_path = out_dir / f"top_{args.top_k}_background_by_grandqc_fraction.csv"
    cache_path = out_dir / "background_grandqc_fraction_cache.parquet"
    sheet_path = out_dir / f"top_{args.top_k}_background_by_grandqc_fraction.jpg"
    scored.to_parquet(cache_path, index=False)
    top.to_csv(csv_path, index=False)
    contact_sheet(top, sheet_path, args.thumb_size)
    print(f"Wrote {cache_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {sheet_path}")
    print(
        top[
            ["slide_id", "row", "column", "grandqc_background_fraction", "patch_png_path"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
