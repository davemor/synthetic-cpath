#!/usr/bin/env python3
"""Render hproj intrinsic dimensionality estimates as a thesis-ready CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DATASET_LABELS = {
    "kather100k": "Kather100K",
    "spider-colorectal": "SPIDER colorectal",
    "spider-breast": "SPIDER breast",
    "spider-skin": "SPIDER skin",
    "spider-thorax": "SPIDER thorax",
    "icaird100k": "iCAIRD100K",
}

ENCODER_LABELS = {
    "uni": "UNI",
    "uni2": "UNI2",
    "hibou": "Hibou",
    "dinov2": "DINOv2",
}

DATASET_ORDER = [
    "kather100k",
    "spider-colorectal",
    "spider-breast",
    "spider-skin",
    "spider-thorax",
    "icaird100k",
]

ENCODER_ORDER = ["uni", "uni2", "hibou", "dinov2"]


def parse_args() -> argparse.Namespace:
    run_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=run_root / "intrinsic_dims.csv",
        help="Input intrinsic_dims.csv path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=run_root / "tables" / "intrinsic_dims_table.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Decimal places for intrinsic dimensionality estimates.",
    )
    return parser.parse_args()


def format_number(value: str, digits: int) -> str:
    if value is None or value == "":
        return ""
    number = float(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def sort_key(row: dict[str, str]) -> tuple[int, int]:
    dataset = row["dataset"]
    encoder = row["encoder"]
    dataset_index = DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else 999
    encoder_index = ENCODER_ORDER.index(encoder) if encoder in ENCODER_ORDER else 999
    return dataset_index, encoder_index


def main() -> None:
    args = parse_args()
    with args.input.open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=sort_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Dataset", "Embedder", "D", "n", "MLE", "TwoNN"])
        for row in rows:
            writer.writerow(
                [
                    DATASET_LABELS.get(row["dataset"], row["dataset"]),
                    ENCODER_LABELS.get(row["encoder"], row["encoder"]),
                    int(float(row["num_dimensions"])),
                    int(float(row["n_samples"])),
                    format_number(row["intrinsic_dim_mle"], args.digits),
                    format_number(row["intrinsic_dim_twonn"], args.digits),
                ]
            )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
