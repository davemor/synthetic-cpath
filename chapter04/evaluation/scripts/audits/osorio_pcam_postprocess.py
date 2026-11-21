#!/usr/bin/env python3
"""Prepare sampling manifests and evaluate the Osorio PCam reproduction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

from synpath.audits.pcam_sd_full import run_evaluation, write_report
from synpath.paths import SynpathPaths


ROOT = Path(
    "/raid/dm236/synpath/outputs/reproductions/"
    "osorio_pcam_diffusers_legacy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "finalize", "evaluate"))
    parser.add_argument("--arm", choices=("class_only", "mep"), required=True)
    return parser.parse_args()


def source_rows(arm: str) -> list[dict[str, str]]:
    metadata = ROOT / "data" / arm / "train" / "metadata.jsonl"
    return [json.loads(line) for line in metadata.read_text().splitlines()]


def condition_rows(arm: str) -> list[dict[str, str]]:
    prompts = [row["text"] for row in source_rows(arm)]
    condition_ids = {prompt: index for index, prompt in enumerate(sorted(set(prompts)))}
    rows = []
    for prompt in prompts:
        label = "metastatic" if "cancer tissue" in prompt else "non_metastatic"
        match = re.search(r"morphology type (\d+)$", prompt)
        cluster = match.group(1) if match else ""
        rows.append(
            {
                "image_path": "",
                "label": label,
                "label_id": "1" if label == "metastatic" else "0",
                "cluster_id": cluster,
                "condition": prompt,
                "condition_id": str(condition_ids[prompt]),
                "prompt": prompt,
            }
        )
    if len(rows) != 50000:
        raise RuntimeError(f"Expected 50,000 conditions, got {len(rows)}")
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(arm: str) -> None:
    write_csv(ROOT / arm / "sample_conditions.csv", condition_rows(arm))


def finalize(arm: str) -> None:
    conditions = condition_rows(arm)
    samples = ROOT / arm / "samples"
    rows = []
    for index, row in enumerate(conditions):
        image = samples / f"{index:08d}.png"
        if not image.exists():
            raise RuntimeError(f"Missing generated image {image}")
        rows.append(
            {
                "image_path": str(image),
                "dataset": "pcam",
                "model": "sd_full_osorio_reproduction",
                "conditioning": arm,
                "seed": "18021999",
                "label": row["label"],
                "label_id": row["label_id"],
                "cluster_id": row["cluster_id"],
                "condition": row["condition"],
                "condition_id": row["condition_id"],
                "prompt": row["prompt"],
            }
        )
    write_csv(ROOT / arm / "synthetic_manifest.csv", rows)


def evaluate(arm: str) -> None:
    audit = {
        "output_subdir": (
            "reproductions/osorio_pcam_diffusers_legacy/" + arm
        ),
        "real_test_manifest": (
            "/raid/dm236/synpath/outputs/audits/"
            "pcam_sd_lora_resolution_balance/real_test_manifest.csv"
        ),
        "test_embeddings": (
            "/raid/dm236/synpath/cache/embeddings/pcam/"
            "test_uni_224.parquet"
        ),
        "precision_recall_k": 3,
        "precision_recall_batch_size": 512,
        "synthetic_samples": 50000,
        "seed": 18021999,
    }
    paths = SynpathPaths.from_env()
    run_evaluation(audit, paths, resume=True)
    write_report(audit, paths)


def main() -> None:
    args = parse_args()
    {"prepare": prepare, "finalize": finalize, "evaluate": evaluate}[
        args.stage
    ](args.arm)


if __name__ == "__main__":
    main()

