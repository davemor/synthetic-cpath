#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_CHUNK_SIZE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare patcher manifests for the stratified iCAIRD cervical sample."
        ),
    )
    parser.add_argument(
        "--sample-csv",
        type=Path,
        required=True,
        help="Stratified sample CSV containing a Filename column.",
    )
    parser.add_argument(
        "--dataset-index",
        type=Path,
        required=True,
        help="Canonical iCAIRD cervical histokit index.csv.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Canonical iCAIRD cervical labels.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory for the sampled index, labels, and chunk manifests.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of slides per generated chunk manifest.",
    )
    return parser.parse_args()


def sample_id_from_slide_name(slide_name: str) -> str:
    return Path(str(slide_name)).name.split(".", 1)[0]


def rewrite_slide_name(slide_name: str) -> str:
    slide_name = str(slide_name)
    if slide_name.endswith(".isyntax"):
        return f"{slide_name.removesuffix('.isyntax')}.ome.tiff"
    return slide_name


def write_index_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True)


def load_labels_tiffslide(labels_path: Path) -> dict:
    with open(labels_path, "r", encoding="utf-8") as handle:
        labels_payload = json.load(handle)
    labels_payload["slides"]["type"] = "tiffslide"
    return labels_payload


def main() -> None:
    args = parse_args()

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")

    output_root = args.output_root.expanduser().resolve()
    chunks_dir = output_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    sample_df = pd.read_csv(args.sample_csv)
    if "Filename" not in sample_df.columns:
        raise ValueError("Sample CSV must contain a Filename column")

    sample_df = sample_df.copy()
    sample_df["id"] = sample_df["Filename"].map(sample_id_from_slide_name)
    duplicated_sample_ids = sample_df[sample_df["id"].duplicated()]["id"].tolist()
    if duplicated_sample_ids:
        examples = ", ".join(duplicated_sample_ids[:10])
        raise ValueError(f"Sample CSV contains duplicated slide ids: {examples}")

    dataset_index = pd.read_csv(args.dataset_index, index_col=0)
    dataset_index = dataset_index.copy()
    dataset_index["slide"] = dataset_index["slide"].map(rewrite_slide_name)
    dataset_index["id"] = dataset_index["slide"].map(sample_id_from_slide_name)

    duplicated_index_ids = dataset_index[dataset_index["id"].duplicated()]["id"].tolist()
    if duplicated_index_ids:
        examples = ", ".join(duplicated_index_ids[:10])
        raise ValueError(f"Dataset index contains duplicated slide ids: {examples}")

    indexed_dataset = dataset_index.set_index("id", drop=False)
    missing_ids = [
        sample_id
        for sample_id in sample_df["id"].tolist()
        if sample_id not in indexed_dataset.index
    ]
    if missing_ids:
        examples = ", ".join(missing_ids[:10])
        raise RuntimeError(
            "Sample CSV contains slides missing from the dataset index. "
            f"Examples: {examples}"
        )

    sampled_index = indexed_dataset.loc[sample_df["id"].tolist()].copy()
    sampled_index = sampled_index.reset_index(drop=True)
    if "sample_split" in sample_df.columns:
        sampled_index["sample_split"] = sample_df["sample_split"].to_numpy()

    sampled_index_path = output_root / "index_sampled_stratified_120_ometiff.csv"
    write_index_csv(sampled_index, sampled_index_path)

    for chunk_path in chunks_dir.glob("stratified_120_chunk_*.csv"):
        chunk_path.unlink()

    for chunk_idx, start in enumerate(range(0, len(sampled_index), args.chunk_size)):
        chunk_df = sampled_index.iloc[start : start + args.chunk_size].copy()
        write_index_csv(chunk_df, chunks_dir / f"stratified_120_chunk_{chunk_idx:03d}.csv")

    labels_path = output_root / "labels_tiffslide.json"
    with open(labels_path, "w", encoding="utf-8") as handle:
        json.dump(load_labels_tiffslide(args.labels), handle, indent=4)
        handle.write("\n")

    print(f"Wrote {sampled_index_path}")
    print(f"Wrote {labels_path}")
    print(
        f"Wrote {len(list(chunks_dir.glob('stratified_120_chunk_*.csv')))} "
        f"chunk manifests to {chunks_dir}"
    )
    print(f"Sampled slides: {len(sampled_index)}")


if __name__ == "__main__":
    main()
