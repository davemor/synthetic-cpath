from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

DATASET_COLUMNS = [
    "image_path",
    "label",
    "label_id",
    "split",
    "dataset",
    "slide_id",
    "patient_id",
]

CONDITION_COLUMNS = [
    "image_path",
    "label",
    "label_id",
    "cluster_id",
    "condition",
    "condition_id",
    "prompt",
]

SYNTHETIC_COLUMNS = [
    "image_path",
    "dataset",
    "model",
    "conditioning",
    "seed",
    "label",
    "label_id",
    "cluster_id",
    "condition",
    "condition_id",
    "prompt",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(
    path: Path,
    rows: Iterable[dict[str, object]],
    columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def validate_columns(path: Path, required: list[str]) -> None:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc
    missing = [col for col in required if col not in header]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
