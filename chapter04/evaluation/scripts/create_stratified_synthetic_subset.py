from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path


def stable_key(row: dict[str, str], seed: int) -> str:
    return hashlib.sha256(f"{seed}|{row['image_path']}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with args.source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not rows or not fieldnames:
        raise ValueError(f"Empty source manifest: {args.source}")
    if args.count > len(rows):
        raise ValueError(f"Requested {args.count} rows from only {len(rows)} rows")

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["label_id"]].append(row)

    exact = {label: args.count * len(group) / len(rows) for label, group in groups.items()}
    quotas = {label: math.floor(value) for label, value in exact.items()}
    remainder_order = sorted(
        groups,
        key=lambda label: (-(exact[label] - quotas[label]), label),
    )
    for label in remainder_order[: args.count - sum(quotas.values())]:
        quotas[label] += 1

    selected = []
    for label, group in sorted(groups.items()):
        selected.extend(sorted(group, key=lambda row: stable_key(row, args.seed))[: quotas[label]])
    selected.sort(key=lambda row: stable_key(row, args.seed))

    if len(selected) != args.count:
        raise RuntimeError(f"Selected {len(selected)} rows, expected {args.count}")
    paths = [row["image_path"] for row in selected]
    if len(set(paths)) != len(paths):
        raise RuntimeError("Selected manifest contains duplicate image paths")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"source_count={len(rows)}")
    print(f"selected_count={len(selected)}")
    print(f"class_quotas={dict(sorted(quotas.items()))}")
    print(f"destination={args.destination}")


if __name__ == "__main__":
    main()
