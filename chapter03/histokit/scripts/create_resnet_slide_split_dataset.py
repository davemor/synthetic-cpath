#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DEFAULT_SUBSET_DIR = Path(
    "/sharedscratch/dm236/data/icaird_cervical/resnet_patch_subset/"
    "stratified_120_consensus_100k"
)
DEFAULT_SAMPLE_INDEX = Path(
    "/sharedscratch/dm236/data/icaird_cervical/stratified_120/"
    "index_sampled_stratified_120_ometiff.csv"
)
CLASS_LABELS = ["background", "foreground"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join slide-level train/test splits onto a sampled patch manifest "
            "and create split-specific copied image folders."
        )
    )
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--sample-index", type=Path, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to <subset-dir>/sample_manifest.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <subset-dir>/split_dataset.",
    )
    parser.add_argument(
        "--copy-images",
        dest="copy_images",
        action="store_true",
        default=True,
        help="Copy split images into split_images/{train,test}/{class}.",
    )
    parser.add_argument(
        "--no-copy-images",
        dest="copy_images",
        action="store_false",
        help="Only write split manifests; do not copy images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace any existing split image directories.",
    )
    return parser.parse_args()


def load_split_index(path: Path) -> pd.DataFrame:
    index = pd.read_csv(path, index_col=0)
    required = {"id", "category", "subcategory", "sample_split"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"Sample index missing columns: {sorted(missing)}")
    if set(index["sample_split"].dropna()) != {"train", "test"}:
        raise ValueError("sample_split must contain exactly train/test labels")
    return index[["id", "category", "subcategory", "sample_split"]].copy()


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_parquet(path)
    required = {"slide_id", "row", "column", "majority_label", "sample_patch_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return manifest.copy()


def attach_split(manifest: pd.DataFrame, split_index: pd.DataFrame) -> pd.DataFrame:
    merged = manifest.merge(
        split_index,
        left_on="slide_id",
        right_on="id",
        how="left",
        validate="many_to_one",
    ).drop(columns=["id"])
    missing = merged["sample_split"].isna()
    if missing.any():
        examples = ", ".join(merged.loc[missing, "slide_id"].drop_duplicates().head(10))
        raise ValueError(f"Missing sample_split for slides: {examples}")
    return merged


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> None:
    if overwrite:
        split_images = output_dir / "split_images"
        if split_images.exists():
            shutil.rmtree(split_images)
    output_dir.mkdir(parents=True, exist_ok=True)


def destination_path(output_dir: Path, row: pd.Series) -> Path:
    label = str(row["majority_label"])
    split = str(row["sample_split"])
    filename = f"{row['slide_id']}_{int(row['row'])}_{int(row['column'])}.png"
    return output_dir / "split_images" / split / label / filename


def copy_split_images(manifest: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    manifest = manifest.copy()
    destinations = []
    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Copying split images",
    ):
        source = Path(row["sample_patch_path"])
        if not source.exists():
            raise FileNotFoundError(f"Missing sampled patch image: {source}")
        dest = destination_path(output_dir, row)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        destinations.append(str(dest))
    manifest["split_patch_path"] = destinations
    return manifest


def write_outputs(manifest: pd.DataFrame, output_dir: Path, copy_images: bool) -> None:
    manifest.to_parquet(output_dir / "sample_manifest_with_split.parquet", index=False)
    manifest.to_csv(output_dir / "sample_manifest_with_split.csv", index=False)

    for split in ["train", "test"]:
        split_manifest = manifest.loc[manifest["sample_split"] == split].copy()
        split_manifest.to_parquet(
            output_dir / f"sample_manifest_{split}.parquet", index=False
        )
        split_manifest.to_csv(output_dir / f"sample_manifest_{split}.csv", index=False)

    summary = {
        "total_patches": int(len(manifest)),
        "copy_images": bool(copy_images),
        "slides_by_split": {
            split: int(count)
            for split, count in manifest.groupby("sample_split")["slide_id"]
            .nunique()
            .items()
        },
        "patches_by_split_and_class": {
            f"{split}/{label}": int(count)
            for (split, label), count in manifest.groupby(
                ["sample_split", "majority_label"]
            )
            .size()
            .items()
        },
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or args.subset_dir / "sample_manifest.parquet"
    output_dir = args.output_dir or args.subset_dir / "split_dataset"

    prepare_output_dirs(output_dir, args.overwrite)
    manifest = load_manifest(manifest_path)
    split_index = load_split_index(args.sample_index)
    manifest = attach_split(manifest, split_index)

    if args.copy_images:
        manifest = copy_split_images(manifest, output_dir)
    else:
        manifest["split_patch_path"] = ""

    write_outputs(manifest, output_dir, args.copy_images)
    print(f"Wrote split dataset to {output_dir}")
    print(manifest.groupby(["sample_split", "majority_label"]).size().to_string())


if __name__ == "__main__":
    main()
