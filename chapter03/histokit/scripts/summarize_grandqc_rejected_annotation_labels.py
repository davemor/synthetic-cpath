#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATA_ROOT = Path("/sharedscratch/dm236/data/icaird_cervical")
DEFAULT_REJECTED_MANIFEST = (
    DATA_ROOT
    / "bags_filter_rule_evaluation/test20_cached_full_bags/"
    "canny_005_grandqc_severe_rejected_foreground_patches/"
    "rejected_foreground_manifest.parquet"
)
DEFAULT_PATCHSETS_ROOT = (
    DATA_ROOT
    / "bags/full_canny005_or_resnet_light_grandqc_artifact/patchsets"
)
DEFAULT_OUTPUT_DIR = (
    DATA_ROOT
    / "bags_filter_rule_evaluation/test20_cached_full_bags/"
    "canny_005_grandqc_severe_rejected_foreground_patches"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize rendered polygon annotation labels for foreground test "
            "patches kept by Canny but rejected by severe GrandQC artifact QC."
        )
    )
    parser.add_argument("--rejected-manifest", type=Path, default=DEFAULT_REJECTED_MANIFEST)
    parser.add_argument("--patchsets-root", type=Path, default=DEFAULT_PATCHSETS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_patchset_annotations(patchsets_root: Path, slide_ids: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required = {
        "slide_id",
        "row",
        "column",
        "annotation_path",
        "has_annotation",
        "annotation_label",
        "annotation_label_name",
    }
    for slide_id in slide_ids:
        frame_path = patchsets_root / f"{slide_id}.patchset" / "frame.parquet"
        if not frame_path.exists():
            raise FileNotFoundError(f"Patchset frame not found for {slide_id}: {frame_path}")
        frame = pd.read_parquet(frame_path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{frame_path} missing annotation columns: {sorted(missing)}")
        frames.append(frame[list(required)].copy())
    if not frames:
        return pd.DataFrame(columns=sorted(required))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rejected = read_table(args.rejected_manifest)
    required = {"slide_id", "row", "column"}
    missing = required - set(rejected.columns)
    if missing:
        raise ValueError(f"{args.rejected_manifest} missing columns: {sorted(missing)}")

    rejected = rejected.copy()
    rejected["slide_id"] = rejected["slide_id"].astype(str)
    rejected["row"] = rejected["row"].astype(int)
    rejected["column"] = rejected["column"].astype(int)

    annotations = load_patchset_annotations(
        args.patchsets_root,
        sorted(rejected["slide_id"].unique()),
    )
    annotations["slide_id"] = annotations["slide_id"].astype(str)
    annotations["row"] = annotations["row"].astype(int)
    annotations["column"] = annotations["column"].astype(int)

    joined = rejected.merge(
        annotations,
        on=["slide_id", "row", "column"],
        how="left",
        validate="one_to_one",
    )
    missing_labels = joined["annotation_label_name"].isna()
    if missing_labels.any():
        examples = joined.loc[missing_labels, ["slide_id", "row", "column"]].head(10)
        raise ValueError(
            "Missing rendered annotation labels for "
            f"{int(missing_labels.sum())} rejected patches. Examples:\n{examples}"
        )

    label_counts = (
        joined.groupby(["annotation_label", "annotation_label_name"], dropna=False)
        .size()
        .reset_index(name="patch_count")
        .sort_values(["patch_count", "annotation_label_name"], ascending=[False, True])
    )
    label_counts["fraction"] = label_counts["patch_count"] / len(joined)

    by_artifact_and_label = (
        joined.groupby(
            [
                "official_grandqc_artifact_class_id",
                "official_grandqc_artifact_class",
                "annotation_label",
                "annotation_label_name",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="patch_count")
        .sort_values(
            [
                "official_grandqc_artifact_class_id",
                "patch_count",
                "annotation_label_name",
            ],
            ascending=[True, False, True],
        )
    )
    if not by_artifact_and_label.empty:
        totals = by_artifact_and_label.groupby(
            "official_grandqc_artifact_class_id"
        )["patch_count"].transform("sum")
        by_artifact_and_label["within_artifact_fraction"] = (
            by_artifact_and_label["patch_count"] / totals
        )

    joined.to_parquet(
        args.output_dir / "rejected_foreground_with_patch_annotations.parquet",
        index=False,
    )
    joined.to_csv(
        args.output_dir / "rejected_foreground_with_patch_annotations.csv",
        index=False,
    )
    label_counts.to_csv(args.output_dir / "annotation_label_counts.csv", index=False)
    by_artifact_and_label.to_csv(
        args.output_dir / "annotation_label_counts_by_artifact_class.csv",
        index=False,
    )
    summary = {
        "rejected_manifest": str(args.rejected_manifest),
        "patchsets_root": str(args.patchsets_root),
        "output_dir": str(args.output_dir),
        "patch_count": int(len(joined)),
        "slide_count": int(joined["slide_id"].nunique()),
        "annotation_label_counts": label_counts.to_dict(orient="records"),
    }
    (args.output_dir / "annotation_label_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2))
    print("\nAnnotation label counts:")
    print(label_counts.to_string(index=False))
    if not by_artifact_and_label.empty:
        print("\nAnnotation label counts by GrandQC artifact class:")
        print(by_artifact_and_label.to_string(index=False))


if __name__ == "__main__":
    main()
