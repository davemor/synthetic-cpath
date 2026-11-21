#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from build_icaird_cervical_patch_encoder_dataset import (
    ARTIFACT_CLASSES,
    open_ome_pyramid,
    read_ome_pyramid_patch,
)


DATA_ROOT = Path("/sharedscratch/dm236/data/icaird_cervical")
DEFAULT_JOINED = (
    DATA_ROOT
    / "bags_filter_rule_evaluation/test20_cached_full_bags/"
    "joined_cached_test_scores.parquet"
)
DEFAULT_BAGS_ROOT = (
    DATA_ROOT / "bags/full_canny005_or_resnet_light_grandqc_artifact"
)
DEFAULT_OUTPUT_DIR = (
    DATA_ROOT
    / "bags_filter_rule_evaluation/test20_cached_full_bags/"
    "canny_005_grandqc_severe_rejected_foreground_patches"
)
PATCH_PATH_COLUMNS = [
    "split_patch_path",
    "sample_patch_path",
    "patch_path",
    "patch_png_path",
]
ANNOTATOR_COLUMNS = [
    "annotator_1_human_tissue_annotation",
    "annotator_2_human_tissue_annotation",
    "annotator_3_human_tissue_annotation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export foreground test patches kept by Canny but rejected by "
            "GrandQC artifact QC."
        )
    )
    parser.add_argument("--joined-scores", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_BAGS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--canny-threshold", type=float, default=0.05)
    parser.add_argument(
        "--reject-class-ids",
        default="3,4,5",
        help="Comma-separated official GrandQC artifact class IDs to reject.",
    )
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--patch-level", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_reject_class_ids(value: str) -> set[int]:
    ids = {int(part.strip()) for part in value.split(",") if part.strip()}
    unknown = ids - set(ARTIFACT_CLASSES)
    if unknown:
        raise ValueError(f"Unknown artifact class ids: {sorted(unknown)}")
    return ids


def load_joined(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Joined cached scores not found: {path}")
    frame = pd.read_parquet(path)
    required = {
        "slide_id",
        "row",
        "column",
        "majority_label",
        "canny_tissue_score",
        "official_grandqc_artifact_class_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame["slide_id"] = frame["slide_id"].astype(str)
    frame["row"] = frame["row"].astype(int)
    frame["column"] = frame["column"].astype(int)
    frame["official_grandqc_artifact_class_id"] = frame[
        "official_grandqc_artifact_class_id"
    ].astype(int)
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
    slide_path = slide_paths.get(str(row["slide_id"]))
    if slide_path is None:
        raise FileNotFoundError(f"Slide {row['slide_id']} missing from dataset index")
    pyramid = open_ome_pyramid(slide_path)
    patch = read_ome_pyramid_patch(
        pyramid=pyramid,
        patch_level=patch_level,
        row=int(row["row"]),
        column=int(row["column"]),
        patch_size=patch_size,
    )
    Image.fromarray(patch).convert("RGB").save(target)


def add_unanimity(frame: pd.DataFrame) -> pd.Series:
    if "unanimous" in frame.columns:
        return frame["unanimous"].astype(bool)
    missing = set(ANNOTATOR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            "Cannot compute unanimity; missing annotator columns: "
            f"{sorted(missing)}"
        )
    annotations = frame[ANNOTATOR_COLUMNS]
    complete = ~annotations.isna().any(axis=1)
    return complete & annotations.nunique(axis=1, dropna=False).eq(1)


def main() -> None:
    args = parse_args()
    reject_ids = parse_reject_class_ids(args.reject_class_ids)
    reject_names = [ARTIFACT_CLASSES[class_id] for class_id in sorted(reject_ids)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    joined = load_joined(args.joined_scores)
    joined["recomputed_unanimous"] = add_unanimity(joined)
    selected = joined.loc[
        (joined["majority_label"].astype(str) == "foreground")
        & (joined["canny_tissue_score"].astype(float) > args.canny_threshold)
        & (joined["official_grandqc_artifact_class_id"].isin(reject_ids))
    ].copy()
    selected = selected.sort_values(["slide_id", "row", "column"]).reset_index(drop=True)
    slide_paths = load_slide_paths(args.bags_root)

    export_paths: list[str] = []
    for _, row in selected.iterrows():
        filename = (
            f"{row['slide_id']}__r{int(row['row']):05d}"
            f"_c{int(row['column']):05d}"
            f"__gqc{int(row['official_grandqc_artifact_class_id'])}.png"
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
    selected["canny_threshold"] = args.canny_threshold
    selected["grandqc_reject_class_ids"] = ",".join(str(i) for i in sorted(reject_ids))
    selected["grandqc_reject_class_names"] = ",".join(reject_names)
    selected.to_parquet(args.output_dir / "rejected_foreground_manifest.parquet", index=False)
    selected.to_csv(args.output_dir / "rejected_foreground_manifest.csv", index=False)

    full_unanimous = add_unanimity(joined)
    summary = {
        "joined_scores": str(args.joined_scores),
        "bags_root": str(args.bags_root),
        "output_dir": str(args.output_dir),
        "canny_threshold": args.canny_threshold,
        "reject_class_ids": sorted(reject_ids),
        "reject_class_names": reject_names,
        "rejected_foreground_count": int(len(selected)),
        "rejected_foreground_unanimous_count": int(selected["recomputed_unanimous"].sum()),
        "rejected_foreground_unanimous_fraction": float(
            selected["recomputed_unanimous"].mean()
        )
        if len(selected)
        else None,
        "full_test_patch_count": int(len(joined)),
        "full_test_unanimous_count": int(full_unanimous.sum()),
        "full_test_unanimous_fraction": float(full_unanimous.mean()),
    }
    by_artifact = (
        selected.groupby(
            [
                "official_grandqc_artifact_class_id",
                "official_grandqc_artifact_class",
            ],
            dropna=False,
        )
        .agg(
            patch_count=("majority_label", "size"),
            unanimous_count=("recomputed_unanimous", "sum"),
        )
        .reset_index()
    )
    if not by_artifact.empty:
        by_artifact["unanimous_fraction"] = (
            by_artifact["unanimous_count"] / by_artifact["patch_count"]
        )
    by_artifact.to_csv(args.output_dir / "rejected_by_artifact_class.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not by_artifact.empty:
        print("\nBy artifact class:")
        print(by_artifact.to_string(index=False))


if __name__ == "__main__":
    main()
