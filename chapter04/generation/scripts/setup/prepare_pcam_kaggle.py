#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from synpath.data.prepare import build_pcam_kaggle_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SynPath manifest for Kaggle's deduplicated PCam train set."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=220_025)
    return parser.parse_args()


def build_kaggle_manifest(
    root: Path,
    output: Path,
    expected: int = 220_025,
) -> Path:
    labels_path = root / "train_labels.csv"
    image_root = root / "train"
    rows = build_pcam_kaggle_manifest(
        "pcam_kaggle",
        {"root": str(root), "expected_train": expected},
    )
    manifest = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "kind": "kaggle_deduplicated_pcam_train",
                "labels_path": str(labels_path),
                "image_root": str(image_root),
                "manifest_path": str(output),
                "n_samples": len(manifest),
                "label_counts": manifest["label"].value_counts().to_dict(),
                "row_order": "train_labels_csv",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return output


def main() -> None:
    args = parse_args()
    output = build_kaggle_manifest(args.root, args.output, args.expected)
    print(output)


if __name__ == "__main__":
    main()
