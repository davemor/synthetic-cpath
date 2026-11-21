#!/usr/bin/env python3
"""Prepare the two matched PCam prompt populations from Osorio et al."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_WORD = {"non_metastatic": "healthy", "metastatic": "cancer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-manifest", type=Path, required=True)
    parser.add_argument("--cluster-labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=18021999)
    return parser.parse_args()


def balanced_counts(total: int, groups: int) -> list[int]:
    base, remainder = divmod(total, groups)
    return [base + int(index < remainder) for index in range(groups)]


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    os.link(source, destination)


def write_population(frame: pd.DataFrame, root: Path, morphology: bool) -> None:
    records = []
    for index, row in frame.reset_index(drop=True).iterrows():
        label = LABEL_WORD[row["label"]]
        relative = Path(label) / f"{index:08d}{Path(row['image_path']).suffix}"
        hardlink(Path(row["image_path"]), root / relative)
        prompt = f"Histology image of {label} tissue"
        if morphology:
            prompt += f", morphology type {int(row['cluster_id'])}"
        records.append({"file_name": str(relative), "text": prompt})
    with (root / "metadata.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.curated_manifest)
    labels = np.load(args.cluster_labels)
    if len(manifest) != 216886 or len(labels) != len(manifest):
        raise RuntimeError("Expected 216,886 aligned curated PCam rows")
    if set(manifest["label"]) != set(LABEL_WORD):
        raise RuntimeError("Unexpected PCam labels")

    frame = manifest.copy()
    frame["cluster_id"] = labels.astype(int)
    rng = np.random.default_rng(args.seed)
    selected = []
    selection = {}
    per_prompt = balanced_counts(51000, 42)
    cursor = 0
    for label in ("non_metastatic", "metastatic"):
        group = frame[frame["label"] == label]
        support = group["cluster_id"].value_counts().sort_values(ascending=False)
        retained = [int(cluster) for cluster in support.index[:21]]
        selection[label] = {"retained_clusters": retained, "source_support": {}}
        for cluster in retained:
            count = per_prompt[cursor]
            cursor += 1
            candidates = group[group["cluster_id"] == cluster]
            selection[label]["source_support"][str(cluster)] = int(len(candidates))
            if len(candidates) < count:
                raise RuntimeError(f"Insufficient support for {label}, cluster {cluster}")
            take = rng.choice(candidates.index.to_numpy(), count, replace=False)
            selected.append(frame.loc[take])

    population = pd.concat(selected).sample(frac=1, random_state=args.seed)
    strata = []
    for _, group in population.groupby(["label", "cluster_id"], sort=True):
        shuffled = group.sample(frac=1, random_state=args.seed)
        val_count = 24 if len(shuffled) == 1215 else 23
        strata.append(shuffled.assign(split=["val"] * val_count + ["train"] * (len(shuffled) - val_count)))
    population = pd.concat(strata).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    # Move the small residual needed to make the holdout exactly 1,000.
    current_val = int((population["split"] == "val").sum())
    if current_val < 1000:
        candidates = population.index[population["split"] == "train"].to_numpy()
        population.loc[rng.choice(candidates, 1000 - current_val, replace=False), "split"] = "val"
    if Counter(population["split"]) != Counter({"train": 50000, "val": 1000}):
        raise RuntimeError("Failed to construct the 50K/1K split")

    args.output_root.mkdir(parents=True, exist_ok=True)
    population.to_csv(args.output_root / "population_51000.csv", index=False)
    train = population[population["split"] == "train"].copy()
    write_population(train, args.output_root / "class_only" / "train", False)
    write_population(train, args.output_root / "mep" / "train", True)
    metadata = {
        "paper": "Osorio et al., Diagnostics 2024, 14, 1442",
        "curated_population": len(manifest),
        "balanced_population": len(population),
        "train_samples": len(train),
        "validation_samples": int((population["split"] == "val").sum()),
        "clusters": 33,
        "prompts_per_class": 21,
        "joint_prompts": 42,
        "seed": args.seed,
        "selection": selection,
        "manifest_sha256": hashlib.sha256(args.curated_manifest.read_bytes()).hexdigest(),
        "cluster_labels_sha256": hashlib.sha256(args.cluster_labels.read_bytes()).hexdigest(),
    }
    (args.output_root / "preparation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

