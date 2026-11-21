#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train-only Hypatia manifests for the ICAIRD cervical dataset.",
    )
    parser.add_argument("--index", type=Path, required=True, help="Source index.csv path.")
    parser.add_argument("--labels", type=Path, required=True, help="Source labels.json path.")
    parser.add_argument(
        "--pilot-ids",
        nargs=5,
        required=True,
        metavar="SLIDE_ID",
        help="The five pilot slide IDs from your private study configuration.",
    )
    parser.add_argument(
        "--slide-root",
        type=Path,
        required=True,
        help="Directory containing .ome.tiff slides on Hypatia.",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        required=True,
        help="Directory containing copied GeoJSON annotations on Hypatia.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory where the generated manifests should be written.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Number of slides per generated chunk manifest.",
    )
    return parser.parse_args()


def sample_id_from_slide_name(slide_name: str) -> str:
    return Path(slide_name).name.split(".", 1)[0]


def rewrite_slide_name(slide_name: str) -> str:
    if slide_name.endswith(".isyntax"):
        return f"{slide_name[:-len('.isyntax')]}.ome.tiff"
    return slide_name


def write_index_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True)


def main() -> None:
    args = parse_args()

    output_root = args.output_root.expanduser().resolve()
    chunks_dir = output_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(args.index, index_col=0)
    train_df = index_df[index_df["split"] == "train"].copy()
    train_df["slide"] = train_df["slide"].map(rewrite_slide_name)
    train_df["id"] = train_df["slide"].map(sample_id_from_slide_name)

    train_df["slide_exists"] = train_df["slide"].map(
        lambda name: (args.slide_root / name).exists()
    )
    train_df["annotation_exists"] = train_df["annotation"].map(
        lambda name: (args.annotation_root / name).exists()
    )

    missing_annotations = train_df[~train_df["annotation_exists"]].copy()
    if not missing_annotations.empty:
        missing_list = ", ".join(missing_annotations["id"].tolist()[:10])
        raise RuntimeError(
            "Missing annotation files for train rows. "
            f"Examples: {missing_list}"
        )

    missing_wsi_df = train_df[~train_df["slide_exists"]].copy()
    missing_wsi_df["expected_slide_path"] = missing_wsi_df["slide"].map(
        lambda name: str(args.slide_root / name)
    )

    available_train_df = train_df[train_df["slide_exists"]].copy()
    available_train_df = available_train_df.drop(
        columns=["slide_exists", "annotation_exists"]
    )

    missing_wsi_path = output_root / "missing_train_wsi.csv"
    missing_wsi_export = missing_wsi_df.drop(
        columns=["slide_exists", "annotation_exists"]
    )
    missing_wsi_export.to_csv(missing_wsi_path, index=True)

    train_index_path = output_root / "index_train_ometiff.csv"
    write_index_csv(available_train_df, train_index_path)

    pilot_df = available_train_df.set_index("id", drop=False).loc[args.pilot_ids].copy()
    pilot_index_path = output_root / "index_train_pilot_5_ometiff.csv"
    write_index_csv(pilot_df, pilot_index_path)

    for chunk_path in chunks_dir.glob("train_chunk_*.csv"):
        chunk_path.unlink()

    for chunk_idx, start in enumerate(range(0, len(available_train_df), args.chunk_size)):
        chunk_df = available_train_df.iloc[start : start + args.chunk_size].copy()
        write_index_csv(chunk_df, chunks_dir / f"train_chunk_{chunk_idx:03d}.csv")

    with open(args.labels, "r", encoding="utf-8") as handle:
        labels_payload = json.load(handle)
    labels_payload["slides"]["type"] = "tiffslide"
    labels_path = output_root / "labels_tiffslide.json"
    with open(labels_path, "w", encoding="utf-8") as handle:
        json.dump(labels_payload, handle, indent=4)
        handle.write("\n")

    print(f"Wrote {train_index_path}")
    print(f"Wrote {pilot_index_path}")
    print(f"Wrote {labels_path}")
    print(f"Wrote {missing_wsi_path}")
    print(f"Wrote {len(list(chunks_dir.glob('train_chunk_*.csv')))} chunk manifests to {chunks_dir}")
    print(f"Available train slides: {len(available_train_df)}")
    print(f"Missing train WSIs: {len(missing_wsi_export)}")


if __name__ == "__main__":
    main()
