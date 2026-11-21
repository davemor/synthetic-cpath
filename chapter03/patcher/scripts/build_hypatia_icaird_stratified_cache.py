#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import histokit.io.annotations  # noqa: F401 - register annotation loaders
import histokit.io.slides  # noqa: F401 - register slide backends
import pandas as pd
from PIL import Image

from histokit.dataset.dataset import Dataset
from histokit.io.slides.slide import Region
from histokit.patchset.patchset import PatchSet


PATCH_DISPLAY_SIZE = 16
CACHE_PREFIX = Path("icaird_cervical/patcher_cache/stratified_120/pipeline_research")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build patcher cache artifacts for the stratified iCAIRD sample "
            "from existing histokit patchsets."
        ),
    )
    parser.add_argument("--index", type=Path, required=True, help="Prepared chunk index.")
    parser.add_argument("--labels", type=Path, required=True, help="Prepared labels_tiffslide.json path.")
    parser.add_argument(
        "--slides-root",
        type=Path,
        required=True,
        help="Root directory containing .ome.tiff slides.",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        required=True,
        help="Root directory containing GeoJSON annotations.",
    )
    parser.add_argument(
        "--source-patchsets-root",
        type=Path,
        required=True,
        help="Root containing research_default shard outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Shared patcher data/cache root (e.g. /sharedscratch/dm236/data).",
    )
    parser.add_argument(
        "--pixels-per-patch",
        type=int,
        default=PATCH_DISPLAY_SIZE,
        help="Pixels per patch to use for generated patchmap thumbnails.",
    )
    return parser.parse_args()


def label_name_for_value(sample, annotation_value: object) -> str:
    if pd.isna(annotation_value):
        return "unlabelled"

    if sample.annotation_schema is None:
        return str(int(annotation_value))

    value_to_name = {
        value: name for name, value in sample.annotation_schema.label_map.items()
    }
    return value_to_name.get(int(annotation_value), str(int(annotation_value)))


def patchset_dir(cache_root: Path, sample_id: str) -> Path:
    return cache_root / "patchsets" / f"{sample_id}.patchset"


def thumbnail_path(
    cache_root: Path,
    slide_name: str,
    patch_level: int,
    patch_size: int,
    pixels_per_patch: int,
) -> Path:
    stem = Path(slide_name).stem
    return (
        cache_root
        / "thumbnails"
        / f"{pixels_per_patch}pix_per_patch"
        / f"{stem}_patchmap_L{patch_level}_P{patch_size}_S{pixels_per_patch}.png"
    )


def status_path(cache_root: Path, sample_id: str) -> Path:
    return cache_root / "status" / f"{sample_id}.json"


def is_status_valid(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, KeyError):
        return False

    required_paths = [
        Path(payload["patchset_dir"]) / "frame.parquet",
        Path(payload["patchset_dir"]) / "manifest.json",
        Path(payload["thumbnail_path"]),
    ]
    return all(candidate.exists() for candidate in required_paths)


def find_source_patchset(source_root: Path, sample_id: str) -> Path:
    direct = source_root / f"{sample_id}.patchset"
    candidates = [direct] if direct.exists() else []
    candidates.extend(sorted(source_root.glob(f"*/{sample_id}.patchset")))

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique_candidates.append(candidate)
            seen.add(resolved)

    if not unique_candidates:
        raise FileNotFoundError(
            f"No source patchset found for {sample_id} under {source_root}"
        )
    if len(unique_candidates) > 1:
        joined = ", ".join(str(path) for path in unique_candidates[:10])
        raise RuntimeError(
            f"Multiple source patchsets found for {sample_id}: {joined}"
        )
    return unique_candidates[0]


def export_patch_images_for_patcher(patchset: PatchSet, patches_root: Path) -> None:
    frame = patchset.frame

    for context_id, group in frame.groupby("context_id"):
        context = patchset.contexts[int(context_id)]
        sample = context.sample
        with sample.open_slide() as slide:
            downsample = float(slide.level_downsamples()[context.level])
            for _, row in group.iterrows():
                row_idx = int(row["row"])
                col_idx = int(row["column"])
                x0 = int(col_idx * context.patch_size * downsample)
                y0 = int(row_idx * context.patch_size * downsample)
                region = Region.patch(x0, y0, context.patch_size, context.level)
                image = slide.read_region(region).convert("RGB")

                label_name = label_name_for_value(sample, row.get("annotation_label"))
                output_dir = patches_root / label_name
                output_dir.mkdir(parents=True, exist_ok=True)
                image.save(output_dir / f"{sample.id}_{row_idx}_{col_idx}.png")


def save_patchmap(
    sample,
    output_path: Path,
    patch_size: int,
    patch_level: int,
    pixels_per_patch: int,
) -> None:
    with sample.open_slide() as slide:
        patch_grid = slide.size_in_patches(patch_size, patch_level)
        width = patch_grid.width * pixels_per_patch
        height = patch_grid.height * pixels_per_patch
        thumb = slide.get_thumbnail_for_size(width=width, height=height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(thumb).save(output_path)


def process_sample(
    dataset: Dataset,
    sample,
    source_root: Path,
    cache_root: Path,
    pixels_per_patch: int,
) -> None:
    sample_status_path = status_path(cache_root, sample.id)
    if is_status_valid(sample_status_path):
        print(f"Skipping {sample.id}: valid status marker already exists")
        return

    source_patchset_dir = find_source_patchset(source_root, sample.id)
    patchset = PatchSet.load(source_patchset_dir, dataset)
    if len(patchset.contexts) != 1:
        raise RuntimeError(
            f"Expected one context in {source_patchset_dir}, "
            f"found {len(patchset.contexts)}"
        )

    context = patchset.contexts[0]
    sample_patchset_dir = patchset_dir(cache_root, sample.id)
    sample_patchset_dir.parent.mkdir(parents=True, exist_ok=True)
    patchset.save(sample_patchset_dir)

    patches_root = cache_root / "patches"
    patches_root.mkdir(parents=True, exist_ok=True)
    export_patch_images_for_patcher(patchset, patches_root)

    sample_thumbnail_path = thumbnail_path(
        cache_root,
        sample.slide_path.name,
        context.level,
        context.patch_size,
        pixels_per_patch,
    )
    save_patchmap(
        sample,
        sample_thumbnail_path,
        context.patch_size,
        context.level,
        pixels_per_patch,
    )

    frame = patchset.frame
    keep_count = int(frame["keep"].sum()) if "keep" in frame.columns else len(frame)
    sample_status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_status_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "sample_id": sample.id,
                "patch_count": int(len(frame)),
                "keep_count": keep_count,
                "patchset_dir": str(sample_patchset_dir),
                "thumbnail_path": str(sample_thumbnail_path),
                "source_patchset_dir": str(source_patchset_dir),
                "patch_size": int(context.patch_size),
                "patch_level": int(context.level),
                "pixels_per_patch": int(pixels_per_patch),
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    print(f"Completed {sample.id}: {len(frame)} patches from {source_patchset_dir}")


def main() -> None:
    args = parse_args()

    dataset = Dataset.from_index(
        args.index,
        args.labels,
        slides_dir=args.slides_root,
        annotations_dir=args.annotations_root,
    )

    source_root = args.source_patchsets_root.expanduser().resolve()
    cache_root = args.output_root.expanduser().resolve() / CACHE_PREFIX
    (cache_root / "patchsets").mkdir(parents=True, exist_ok=True)
    (cache_root / "patches").mkdir(parents=True, exist_ok=True)
    (cache_root / "thumbnails" / f"{args.pixels_per_patch}pix_per_patch").mkdir(
        parents=True,
        exist_ok=True,
    )
    (cache_root / "status").mkdir(parents=True, exist_ok=True)

    for sample in dataset.samples():
        process_sample(
            dataset=dataset,
            sample=sample,
            source_root=source_root,
            cache_root=cache_root,
            pixels_per_patch=args.pixels_per_patch,
        )


if __name__ == "__main__":
    main()
