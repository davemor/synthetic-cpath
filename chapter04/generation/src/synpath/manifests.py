from __future__ import annotations

import csv
import random
from collections import defaultdict
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


def proportional_condition_schedule(
    rows: list[dict[str, str]],
    sample_count: int,
    seed: int = 0,
) -> list[dict[str, str]]:
    """Select a deterministic, proportionally stratified condition schedule."""
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if sample_count >= len(rows):
        return list(rows)
    if not rows or sample_count == 0:
        return []

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    exact = {
        condition: sample_count * len(group) / len(rows)
        for condition, group in grouped.items()
    }
    allocation = {condition: int(value) for condition, value in exact.items()}
    remainder = sample_count - sum(allocation.values())
    remainder_order = sorted(
        grouped,
        key=lambda condition: (
            -(exact[condition] - allocation[condition]),
            condition,
        ),
    )
    for condition in remainder_order[:remainder]:
        allocation[condition] += 1

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for condition in sorted(grouped):
        group = list(grouped[condition])
        rng.shuffle(group)
        selected.extend(group[: allocation[condition]])
    rng.shuffle(selected)
    return selected


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
