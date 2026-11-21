#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATA_ROOT = Path("/sharedscratch/dm236/data/icaird_cervical")
DEFAULT_TEST_MANIFEST = (
    DATA_ROOT
    / "resnet_patch_subset/stratified_120_consensus_100k/split_dataset/"
    "sample_manifest_test.parquet"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "resnet_patch_subset/stratified_120_consensus_100k/reports"
ANNOTATOR_COLUMNS = [
    "annotator_1_human_tissue_annotation",
    "annotator_2_human_tissue_annotation",
    "annotator_3_human_tissue_annotation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize test patch annotation unanimity."
    )
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Test manifest not found: {path}")
    frame = pd.read_parquet(path)
    required = {*ANNOTATOR_COLUMNS, "majority_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame


def summarize(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    annotations = frame[ANNOTATOR_COLUMNS]
    complete = ~annotations.isna().any(axis=1)
    recomputed_unanimous = complete & annotations.nunique(axis=1, dropna=False).eq(1)
    stored_unanimous_matches = None
    if "unanimous" in frame.columns:
        stored_unanimous_matches = bool(
            recomputed_unanimous.eq(frame["unanimous"].astype(bool)).all()
        )

    summary = {
        "total_test_patches": int(len(frame)),
        "complete_annotation_patches": int(complete.sum()),
        "missing_annotation_patches": int((~complete).sum()),
        "unanimous_patches": int(recomputed_unanimous.sum()),
        "non_unanimous_patches": int((complete & ~recomputed_unanimous).sum()),
        "unanimous_fraction_of_all": float(recomputed_unanimous.mean()),
        "unanimous_fraction_of_complete": float(
            recomputed_unanimous.sum() / complete.sum()
        )
        if int(complete.sum())
        else None,
        "stored_unanimous_matches_recomputed": stored_unanimous_matches,
    }

    by_label = (
        frame.assign(recomputed_unanimous=recomputed_unanimous, complete_annotation=complete)
        .groupby("majority_label", dropna=False)
        .agg(
            total_patches=("majority_label", "size"),
            complete_annotation_patches=("complete_annotation", "sum"),
            unanimous_patches=("recomputed_unanimous", "sum"),
        )
        .reset_index()
    )
    by_label["non_unanimous_complete_patches"] = (
        by_label["complete_annotation_patches"] - by_label["unanimous_patches"]
    )
    by_label["unanimous_fraction_of_label"] = (
        by_label["unanimous_patches"] / by_label["total_patches"]
    )
    return summary, by_label


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_manifest(args.test_manifest)
    summary, by_label = summarize(frame)
    (args.output_dir / "test_patch_unanimity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    by_label.to_csv(args.output_dir / "test_patch_unanimity_by_label.csv", index=False)
    print(json.dumps(summary, indent=2))
    print("\nBy majority label:")
    print(by_label.to_string(index=False))


if __name__ == "__main__":
    main()
