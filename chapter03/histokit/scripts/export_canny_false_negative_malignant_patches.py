#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from build_icaird_cervical_patch_encoder_dataset import (
    open_ome_pyramid,
    read_ome_pyramid_patch,
)


DATA_ROOT = Path("/sharedscratch/dm236/data/icaird_cervical")
DEFAULT_FALSE_NEGATIVES = (
    DATA_ROOT
    / "bags_filter_rule_evaluation/test20_cached_full_bags/"
    "canny_005_false_negative_annotation_labels/"
    "canny_false_negatives_with_patch_annotations.parquet"
)
DEFAULT_BAGS_ROOT = (
    DATA_ROOT / "bags/full_canny005_or_resnet_light_grandqc_artifact"
)
DEFAULT_OUTPUT_DIR = (
    DATA_ROOT
    / "bags_filter_rule_evaluation/test20_cached_full_bags/"
    "canny_005_false_negative_malignant_patches"
)
PATCH_PATH_COLUMNS = [
    "split_patch_path",
    "sample_patch_path",
    "patch_path",
    "patch_png_path",
    "export_patch_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export malignant diagnostic annotation patches among Canny false "
            "negatives: majority foreground patches with canny_tissue_score <= threshold."
        )
    )
    parser.add_argument("--false-negatives", type=Path, default=DEFAULT_FALSE_NEGATIVES)
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_BAGS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--annotation-label-name", default="malignant")
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--patch-level", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"False-negative table not found: {path}")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {
        "slide_id",
        "row",
        "column",
        "majority_label",
        "canny_tissue_score",
        "annotation_label_name",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame["slide_id"] = frame["slide_id"].astype(str)
    frame["row"] = frame["row"].astype(int)
    frame["column"] = frame["column"].astype(int)
    return frame


def load_slide_paths(bags_root: Path) -> dict[str, Path]:
    path = bags_root / "dataset_index_ometiff.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset index not found: {path}")
    index = pd.read_csv(path)
    return {
        str(row.id): Path(str(row.slide_path))
        for row in index[["id", "slide_path"]].itertuples(index=False)
    }


def source_patch_path(row: pd.Series) -> Path | None:
    for column in PATCH_PATH_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value:
            path = Path(value)
            if path.exists():
                return path
    return None


def export_patch(
    *,
    row: pd.Series,
    target: Path,
    slide_paths: dict[str, Path],
    patch_level: int,
    patch_size: int,
    overwrite: bool,
) -> None:
    if target.exists() and not overwrite:
        return
    source = source_patch_path(row)
    if source is not None:
        shutil.copy2(source, target)
        return
    slide_id = str(row["slide_id"])
    slide_path = slide_paths.get(slide_id)
    if slide_path is None:
        raise FileNotFoundError(f"Slide {slide_id} missing from dataset index")
    pyramid = open_ome_pyramid(slide_path)
    patch = read_ome_pyramid_patch(
        pyramid=pyramid,
        patch_level=patch_level,
        row=int(row["row"]),
        column=int(row["column"]),
        patch_size=patch_size,
    )
    Image.fromarray(patch).convert("RGB").save(target)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    frame = load_frame(args.false_negatives)
    selected = frame.loc[
        frame["annotation_label_name"].astype(str).str.lower()
        == args.annotation_label_name.lower()
    ].copy()
    selected = selected.sort_values(["slide_id", "row", "column"]).reset_index(drop=True)
    slide_paths = load_slide_paths(args.bags_root)

    export_paths: list[str] = []
    for _, row in selected.iterrows():
        filename = (
            f"{row['slide_id']}__r{int(row['row']):05d}"
            f"_c{int(row['column']):05d}"
            f"__{args.annotation_label_name}.png"
        )
        target = image_dir / filename
        export_patch(
            row=row,
            target=target,
            slide_paths=slide_paths,
            patch_level=args.patch_level,
            patch_size=args.patch_size,
            overwrite=args.overwrite,
        )
        export_paths.append(str(target))

    selected["export_patch_path"] = export_paths
    selected.to_parquet(args.output_dir / "malignant_false_negative_manifest.parquet", index=False)
    selected.to_csv(args.output_dir / "malignant_false_negative_manifest.csv", index=False)
    summary = {
        "false_negatives": str(args.false_negatives),
        "bags_root": str(args.bags_root),
        "output_dir": str(args.output_dir),
        "annotation_label_name": args.annotation_label_name,
        "exported_patch_count": int(len(selected)),
        "slide_count": int(selected["slide_id"].nunique()) if len(selected) else 0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
