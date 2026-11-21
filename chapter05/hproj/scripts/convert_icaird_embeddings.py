#!/usr/bin/env python
"""Convert merged iCAIRD embedding exports to hproj cache directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ENCODERS = {
    "uni": "uni",
    "uni2_h": "uni2",
    "hibou_l": "hibou",
    "dinov2_large": "dinov2",
}
LABEL_ORDER = ["normal", "low_grade", "high_grade", "malignant"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}
SOURCE_TO_HPROJ_SPLIT = {
    "train": "train",
    "valid": "train",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root of the iCAIRD export containing embeddings/{encoder}.",
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        required=True,
        help="hproj EMBEDDINGS_ROOT where cache-{dataset}-{encoder} dirs are written.",
    )
    parser.add_argument("--dataset", default="icaird100k")
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target cache files for this dataset.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate already-converted target caches without writing.",
    )
    return parser.parse_args()


def expected_output_dirs(embeddings_root: Path, dataset: str) -> dict[str, Path]:
    return {
        target_encoder: embeddings_root / f"cache-{dataset}-{target_encoder}"
        for target_encoder in ENCODERS.values()
    }


def load_source(
    source_root: Path, source_encoder: str
) -> tuple[pd.DataFrame, np.ndarray]:
    encoder_root = source_root / "embeddings" / source_encoder
    index_path = encoder_root / "index.parquet"
    features_path = encoder_root / "features.npz"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    if not features_path.exists():
        raise FileNotFoundError(features_path)

    index = pd.read_parquet(index_path)
    features = np.load(features_path)["features"]
    if len(index) != len(features):
        raise ValueError(
            f"{source_encoder}: index rows ({len(index)}) != features ({len(features)})"
        )
    return index, features


def split_frame(index: pd.DataFrame) -> pd.DataFrame:
    missing_splits = sorted(set(index["split"]) - set(SOURCE_TO_HPROJ_SPLIT))
    if missing_splits:
        raise ValueError(f"Unexpected source splits: {missing_splits}")

    labels = sorted(set(index["annotation_label_name"]))
    missing_labels = sorted(set(labels) - set(LABEL_TO_ID))
    if missing_labels:
        raise ValueError(f"Unexpected labels: {missing_labels}")

    frame = index.copy()
    frame["hproj_split"] = frame["split"].map(SOURCE_TO_HPROJ_SPLIT)
    frame["hproj_label"] = frame["annotation_label_name"].map(LABEL_TO_ID)
    return frame


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def clear_split(split_dir: Path) -> None:
    for pattern in ("emb_*.pt", "labels_*.pt", "paths_*.json"):
        for path in split_dir.glob(pattern):
            path.unlink()


def write_split(
    target_root: Path,
    split_name: str,
    frame: pd.DataFrame,
    features: np.ndarray,
    source_encoder: str,
    target_encoder: str,
    dataset: str,
    shard_size: int,
) -> None:
    split_dir = target_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    clear_split(split_dir)

    row_indices = frame.index.to_numpy()
    labels = frame["hproj_label"].to_numpy(dtype=np.int64)
    paths = frame["patch_png_path"].astype(str).tolist()
    shard_count = int(np.ceil(len(row_indices) / shard_size))

    for shard in range(shard_count):
        start = shard * shard_size
        stop = min(start + shard_size, len(row_indices))
        shard_features = torch.from_numpy(features[row_indices[start:stop]].copy())
        shard_labels = torch.from_numpy(labels[start:stop].copy())
        torch.save(shard_features, split_dir / f"emb_{shard:04d}.pt")
        torch.save(shard_labels, split_dir / f"labels_{shard:04d}.pt")
        write_json(split_dir / f"paths_{shard:04d}.json", paths[start:stop])

    torch.save(
        {
            "mean": torch.from_numpy(features[row_indices].mean(axis=0)),
            "std": torch.from_numpy(features[row_indices].std(axis=0)),
        },
        split_dir / "emb_stats.pt",
    )
    write_json(
        split_dir / "meta.json",
        {
            "dataset": dataset,
            "model": source_encoder,
            "encoder": target_encoder,
            "split": split_name,
            "emb_dim": int(features.shape[1]),
            "num_shards": shard_count,
            "shard_size": shard_size,
            "num_samples": int(len(row_indices)),
            "source_splits": sorted(frame["split"].unique().tolist()),
        },
    )


def convert_encoder(
    source_root: Path,
    embeddings_root: Path,
    dataset: str,
    source_encoder: str,
    target_encoder: str,
    shard_size: int,
    overwrite: bool,
) -> None:
    target_root = embeddings_root / f"cache-{dataset}-{target_encoder}"
    if target_root.exists() and not overwrite:
        raise FileExistsError(
            f"{target_root} exists; pass --overwrite to replace shards"
        )

    index, features = load_source(source_root, source_encoder)
    frame = split_frame(index)
    target_root.mkdir(parents=True, exist_ok=True)

    label_map = {str(idx): label for label, idx in LABEL_TO_ID.items()}
    write_json(target_root / "label_map.json", label_map)
    write_json(
        target_root / "source_manifest.json",
        {
            "source_root": str(source_root),
            "source_encoder": source_encoder,
            "target_encoder": target_encoder,
            "source_format": "npz+parquet",
            "label_source": "annotation_label_name",
            "label_order": LABEL_ORDER,
            "split_policy": {"train": ["train", "valid"], "test": ["test"]},
        },
    )

    for split_name in ("train", "test"):
        split_rows = frame[frame["hproj_split"] == split_name]
        write_split(
            target_root,
            split_name,
            split_rows,
            features,
            source_encoder,
            target_encoder,
            dataset,
            shard_size,
        )


def validate_cache(target_root: Path) -> dict[str, int]:
    label_map_path = target_root / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(label_map_path)
    with label_map_path.open() as f:
        label_map = json.load(f)
    if label_map != {str(idx): label for label, idx in LABEL_TO_ID.items()}:
        raise ValueError(f"Unexpected label_map in {target_root}: {label_map}")

    counts = {}
    dims = set()
    for split in ("train", "test"):
        split_dir = target_root / split
        emb_paths = sorted(split_dir.glob("emb_[0-9][0-9][0-9][0-9].pt"))
        label_paths = sorted(split_dir.glob("labels_[0-9][0-9][0-9][0-9].pt"))
        path_jsons = sorted(split_dir.glob("paths_[0-9][0-9][0-9][0-9].json"))
        if (
            not emb_paths
            or len(emb_paths) != len(label_paths)
            or len(emb_paths) != len(path_jsons)
        ):
            raise ValueError(f"Shard mismatch in {split_dir}")

        split_count = 0
        for emb_path, label_path, paths_path in zip(emb_paths, label_paths, path_jsons):
            emb = torch.load(emb_path, map_location="cpu")
            labels = torch.load(label_path, map_location="cpu")
            paths = json.loads(paths_path.read_text())
            if emb.shape[0] != labels.shape[0] or emb.shape[0] != len(paths):
                raise ValueError(f"Row mismatch in {emb_path}")
            dims.add(int(emb.shape[1]))
            split_count += int(emb.shape[0])
        counts[split] = split_count
    if len(dims) != 1:
        raise ValueError(f"Mixed embedding dimensions in {target_root}: {sorted(dims)}")
    counts["dim"] = dims.pop()
    counts["total"] = counts["train"] + counts["test"]
    return counts


def main() -> None:
    args = parse_args()
    summaries = {}

    if not args.validate_only:
        for source_encoder, target_encoder in ENCODERS.items():
            convert_encoder(
                args.source_root,
                args.embeddings_root,
                args.dataset,
                source_encoder,
                target_encoder,
                args.shard_size,
                args.overwrite,
            )

    for target_encoder, target_root in expected_output_dirs(
        args.embeddings_root, args.dataset
    ).items():
        summaries[target_encoder] = validate_cache(target_root)

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
