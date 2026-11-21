#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATA_ROOT = Path("/sharedscratch/dm236/data/icaird_cervical")
DEFAULT_JOINED_SCORES = (
    DATA_ROOT
    / "bags_filter_rule_evaluation/test20_cached_full_bags/"
    "joined_cached_test_scores.parquet"
)
DEFAULT_PATCHSETS_ROOT = (
    DATA_ROOT
    / "bags/full_canny005_or_resnet_light_grandqc_artifact/patchsets"
)
DEFAULT_OUTPUT_DIR = (
    DATA_ROOT
    / "bags_filter_rule_evaluation/test20_cached_full_bags/"
    "canny_005_false_negative_annotation_labels"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count rendered diagnostic annotation labels for Canny false negatives: "
            "majority foreground patches with canny_tissue_score <= threshold."
        )
    )
    parser.add_argument("--joined-scores", type=Path, default=DEFAULT_JOINED_SCORES)
    parser.add_argument("--patchsets-root", type=Path, default=DEFAULT_PATCHSETS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--canny-threshold", type=float, default=0.05)
    return parser.parse_args()


def load_patchset_annotations(patchsets_root: Path, slide_ids: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = [
        "slide_id",
        "row",
        "column",
        "annotation_path",
        "has_annotation",
        "annotation_label",
        "annotation_label_name",
    ]
    for slide_id in slide_ids:
        frame_path = patchsets_root / f"{slide_id}.patchset" / "frame.parquet"
        if not frame_path.exists():
            raise FileNotFoundError(f"Patchset frame not found for {slide_id}: {frame_path}")
        frame = pd.read_parquet(frame_path)
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{frame_path} missing annotation columns: {sorted(missing)}")
        frames.append(frame[columns].copy())
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    joined = pd.read_parquet(args.joined_scores)
    required = {"slide_id", "row", "column", "majority_label", "canny_tissue_score"}
    missing = required - set(joined.columns)
    if missing:
        raise ValueError(f"{args.joined_scores} missing columns: {sorted(missing)}")

    false_negatives = joined.loc[
        (joined["majority_label"].astype(str) == "foreground")
        & (joined["canny_tissue_score"].astype(float) <= args.canny_threshold)
    ].copy()
    false_negatives["slide_id"] = false_negatives["slide_id"].astype(str)
    false_negatives["row"] = false_negatives["row"].astype(int)
    false_negatives["column"] = false_negatives["column"].astype(int)
    false_negatives = false_negatives.sort_values(
        ["slide_id", "row", "column"]
    ).reset_index(drop=True)

    annotations = load_patchset_annotations(
        args.patchsets_root,
        sorted(false_negatives["slide_id"].unique()),
    )
    annotations["slide_id"] = annotations["slide_id"].astype(str)
    annotations["row"] = annotations["row"].astype(int)
    annotations["column"] = annotations["column"].astype(int)

    out = false_negatives.merge(
        annotations,
        on=["slide_id", "row", "column"],
        how="left",
        validate="one_to_one",
    )
    missing_labels = out["annotation_label_name"].isna()
    if missing_labels.any():
        examples = out.loc[missing_labels, ["slide_id", "row", "column"]].head(10)
        raise ValueError(
            "Missing rendered annotation labels for "
            f"{int(missing_labels.sum())} Canny false negatives. Examples:\n{examples}"
        )

    label_counts = (
        out.groupby(["annotation_label", "annotation_label_name"], dropna=False)
        .size()
        .reset_index(name="patch_count")
        .sort_values(["patch_count", "annotation_label_name"], ascending=[False, True])
    )
    label_counts["fraction"] = label_counts["patch_count"] / len(out) if len(out) else 0.0

    by_slide = (
        out.groupby(["slide_id", "annotation_label", "annotation_label_name"], dropna=False)
        .size()
        .reset_index(name="patch_count")
        .sort_values(["slide_id", "patch_count"], ascending=[True, False])
    )

    out.to_parquet(
        args.output_dir / "canny_false_negatives_with_patch_annotations.parquet",
        index=False,
    )
    out.to_csv(
        args.output_dir / "canny_false_negatives_with_patch_annotations.csv",
        index=False,
    )
    label_counts.to_csv(args.output_dir / "annotation_label_counts.csv", index=False)
    by_slide.to_csv(args.output_dir / "annotation_label_counts_by_slide.csv", index=False)

    summary = {
        "joined_scores": str(args.joined_scores),
        "patchsets_root": str(args.patchsets_root),
        "output_dir": str(args.output_dir),
        "definition": (
            "False negative = majority_label == foreground and "
            "canny_tissue_score <= canny_threshold"
        ),
        "canny_threshold": args.canny_threshold,
        "false_negative_count": int(len(out)),
        "slide_count": int(out["slide_id"].nunique()) if len(out) else 0,
        "annotation_label_counts": label_counts.to_dict(orient="records"),
    }
    (args.output_dir / "annotation_label_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2))
    print("\nAnnotation label counts:")
    print(label_counts.to_string(index=False))


if __name__ == "__main__":
    main()
