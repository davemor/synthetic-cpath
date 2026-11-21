#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_ROOT = Path("/sharedscratch/dm236/data/icaird_cervical")
DEFAULT_BAGS_ROOT = (
    DATA_ROOT / "bags/full_canny005_or_resnet_light_grandqc_artifact"
)
DEFAULT_TEST_MANIFEST = (
    DATA_ROOT
    / "resnet_patch_subset/stratified_120_consensus_100k/split_dataset/"
    "sample_manifest_test.parquet"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "bags_filter_rule_evaluation/test20_cached"
PATCH_ID_COLUMNS = ["slide_id", "row", "column"]
CLASS_LABELS = ["background", "foreground"]
ARTIFACT_CLASSES = {
    1: "normal_tissue",
    2: "fold",
    3: "darkspot_foreign_object",
    4: "pen_marking",
    5: "air_bubble_slide_edge",
    6: "out_of_focus",
    7: "background",
}


@dataclass(frozen=True)
class RuleSpec:
    name: str
    description: str
    base: str
    reject_classes: frozenset[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate defensible final patch-export rules on the held-out "
            "20-slide test set using cached Canny, ResNet, and official "
            "GrandQC artifact scores only."
        )
    )
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_BAGS_ROOT)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--canny-threshold", type=float, default=0.05)
    parser.add_argument("--resnet-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Test manifest not found: {path}")
    frame = pd.read_parquet(path)
    required = {*PATCH_ID_COLUMNS, "majority_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.loc[frame["majority_label"].isin(CLASS_LABELS)].copy()
    frame["slide_id"] = frame["slide_id"].astype(str)
    frame["row"] = frame["row"].astype(int)
    frame["column"] = frame["column"].astype(int)
    frame["majority_label"] = frame["majority_label"].astype(str)
    return frame


def load_dataset_index(bags_root: Path) -> pd.DataFrame:
    path = bags_root / "dataset_index_ometiff.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset index not found: {path}")
    index = pd.read_csv(path)
    if "id" not in index.columns:
        raise ValueError(f"{path} missing id column")
    index["id"] = index["id"].astype(str)
    return index


def load_canny_scores(bags_root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    index = load_dataset_index(bags_root).set_index("id", drop=False)
    rows: list[pd.DataFrame] = []
    missing: list[str] = []
    for slide_id in sorted(manifest["slide_id"].unique()):
        source = ""
        if slide_id in index.index:
            value = index.loc[slide_id].get("source_canny_patchset_path", "")
            source = "" if pd.isna(value) else str(value)
        candidates = []
        if source:
            candidates.append(Path(source) / "frame.parquet")
        candidates.append(
            bags_root
            / "source_patchsets/missing_canny_patchsets"
            / f"{slide_id}.patchset/frame.parquet"
        )
        frame_path = next((path for path in candidates if path.exists()), None)
        if frame_path is None:
            missing.append(slide_id)
            continue
        frame = pd.read_parquet(frame_path)
        required = {"row", "column", "tissue_score"}
        missing_cols = required - set(frame.columns)
        if missing_cols:
            raise ValueError(f"{frame_path} missing columns: {sorted(missing_cols)}")
        frame = frame[["row", "column", "tissue_score"]].copy()
        frame["slide_id"] = slide_id
        rows.append(frame)
    if missing:
        raise FileNotFoundError(f"Missing Canny scores for slides: {missing[:10]}")
    out = pd.concat(rows, ignore_index=True).rename(
        columns={"tissue_score": "canny_tissue_score"}
    )
    out["row"] = out["row"].astype(int)
    out["column"] = out["column"].astype(int)
    return out


def load_score_dir(
    root: Path,
    manifest: pd.DataFrame,
    *,
    suffix: str,
    columns: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    missing: list[str] = []
    for slide_id in sorted(manifest["slide_id"].unique()):
        path = root / f"{slide_id}.{suffix}.parquet"
        if not path.exists():
            missing.append(slide_id)
            continue
        frame = pd.read_parquet(path)
        required = {"row", "column", *columns}
        missing_cols = required - set(frame.columns)
        if missing_cols:
            raise ValueError(f"{path} missing columns: {sorted(missing_cols)}")
        frame = frame[["row", "column", *columns]].copy()
        frame["slide_id"] = slide_id
        rows.append(frame)
    if missing:
        raise FileNotFoundError(f"Missing {suffix} files for slides: {missing[:10]}")
    out = pd.concat(rows, ignore_index=True)
    out["row"] = out["row"].astype(int)
    out["column"] = out["column"].astype(int)
    return out


def join_cached_scores(args: argparse.Namespace) -> pd.DataFrame:
    manifest = load_manifest(args.test_manifest)
    joined = manifest.merge(
        load_canny_scores(args.bags_root, manifest),
        on=PATCH_ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    joined = joined.merge(
        load_score_dir(
            args.bags_root / "resnet_scores",
            manifest,
            suffix="resnet_scores",
            columns=[
                "resnet19_background_probability",
                "resnet19_foreground_probability",
                "resnet19_prediction",
            ],
        ),
        on=PATCH_ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    joined = joined.merge(
        load_score_dir(
            args.bags_root / "grandqc_artifact_scores",
            manifest,
            suffix="grandqc_artifact_scores",
            columns=[
                "official_grandqc_artifact_class_id",
                "official_grandqc_artifact_class",
            ],
        ),
        on=PATCH_ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    required = [
        "canny_tissue_score",
        "resnet19_foreground_probability",
        "official_grandqc_artifact_class_id",
    ]
    missing = joined[required].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"Joined cached scores missing for {int(missing.sum())} rows")
    joined["canny_keep"] = joined["canny_tissue_score"].astype(float) > args.canny_threshold
    joined["resnet_keep"] = (
        joined["resnet19_foreground_probability"].astype(float) > args.resnet_threshold
    )
    joined["canny_or_resnet_keep"] = joined["canny_keep"] | joined["resnet_keep"]
    joined["canny_and_resnet_keep"] = joined["canny_keep"] & joined["resnet_keep"]
    joined["official_grandqc_artifact_class_id"] = joined[
        "official_grandqc_artifact_class_id"
    ].astype(int)
    return joined


def rule_specs() -> list[RuleSpec]:
    return [
        RuleSpec("canny_005", "Canny score > 0.05", "canny", frozenset()),
        RuleSpec("resnet_050", "ResNet foreground probability > 0.5", "resnet", frozenset()),
        RuleSpec("canny_or_resnet", "Canny OR ResNet tissue candidate", "or", frozenset()),
        RuleSpec("canny_and_resnet", "Canny AND ResNet conservative tissue candidate", "and", frozenset()),
        RuleSpec(
            "canny_or_resnet_reject_severe_3_4_5",
            "Canny OR ResNet, reject darkspot/foreign object, pen, air bubble/edge",
            "or",
            frozenset({3, 4, 5}),
        ),
        RuleSpec(
            "canny_or_resnet_reject_3_4_5_6",
            "Canny OR ResNet, reject severe artifacts plus out-of-focus",
            "or",
            frozenset({3, 4, 5, 6}),
        ),
        RuleSpec(
            "canny_or_resnet_reject_2_3_4_5",
            "Canny OR ResNet, reject fold plus severe artifacts",
            "or",
            frozenset({2, 3, 4, 5}),
        ),
        RuleSpec(
            "canny_or_resnet_reject_2_3_4_5_6",
            "Canny OR ResNet, reject fold/severe/out-of-focus artifacts",
            "or",
            frozenset({2, 3, 4, 5, 6}),
        ),
        RuleSpec(
            "canny_or_resnet_reject_3_4_5_7",
            "Canny OR ResNet, reject severe artifacts and GrandQC background",
            "or",
            frozenset({3, 4, 5, 7}),
        ),
        RuleSpec(
            "canny_or_resnet_artifact_normal_only",
            "Canny OR ResNet, keep only GrandQC normal tissue",
            "or",
            frozenset({2, 3, 4, 5, 6, 7}),
        ),
        RuleSpec(
            "canny_reject_severe_3_4_5",
            "Canny tissue candidate with severe artifact rejection",
            "canny",
            frozenset({3, 4, 5}),
        ),
        RuleSpec(
            "canny_reject_3_4_5_6",
            "Canny tissue candidate, reject severe artifacts plus out-of-focus",
            "canny",
            frozenset({3, 4, 5, 6}),
        ),
        RuleSpec(
            "canny_reject_2_3_4_5",
            "Canny tissue candidate, reject fold plus severe artifacts",
            "canny",
            frozenset({2, 3, 4, 5}),
        ),
        RuleSpec(
            "canny_reject_2_3_4_5_6",
            "Canny tissue candidate, reject fold/severe/out-of-focus artifacts",
            "canny",
            frozenset({2, 3, 4, 5, 6}),
        ),
        RuleSpec(
            "canny_reject_3_4_5_7",
            "Canny tissue candidate, reject severe artifacts and GrandQC background",
            "canny",
            frozenset({3, 4, 5, 7}),
        ),
        RuleSpec(
            "canny_artifact_normal_only",
            "Canny tissue candidate, keep only GrandQC normal tissue",
            "canny",
            frozenset({2, 3, 4, 5, 6, 7}),
        ),
        RuleSpec(
            "resnet_reject_severe_3_4_5",
            "ResNet tissue candidate with severe artifact rejection",
            "resnet",
            frozenset({3, 4, 5}),
        ),
    ]


def base_prediction(frame: pd.DataFrame, base: str) -> np.ndarray:
    if base == "canny":
        return frame["canny_keep"].to_numpy(dtype=bool)
    if base == "resnet":
        return frame["resnet_keep"].to_numpy(dtype=bool)
    if base == "or":
        return frame["canny_or_resnet_keep"].to_numpy(dtype=bool)
    if base == "and":
        return frame["canny_and_resnet_keep"].to_numpy(dtype=bool)
    raise ValueError(f"Unknown rule base: {base}")


def evaluate_rule(frame: pd.DataFrame, spec: RuleSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

    base = base_prediction(frame, spec.base)
    artifact_keep = ~frame["official_grandqc_artifact_class_id"].isin(spec.reject_classes).to_numpy()
    keep = base & artifact_keep
    y_true = frame["majority_label"].astype(str).to_numpy()
    y_pred = np.where(keep, "foreground", "background")
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_LABELS,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    report["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    rejected_by_artifact = int((base & ~artifact_keep).sum())
    row = {
        "rule": spec.name,
        "description": spec.description,
        "base": spec.base,
        "reject_class_ids": ",".join(str(value) for value in sorted(spec.reject_classes)),
        "reject_class_names": ",".join(
            ARTIFACT_CLASSES.get(value, f"unknown_{value}") for value in sorted(spec.reject_classes)
        ),
        "predicted_foreground": int((y_pred == "foreground").sum()),
        "predicted_background": int((y_pred == "background").sum()),
        "rejected_by_artifact": rejected_by_artifact,
        "accuracy": float(report["accuracy"]),
        "balanced_accuracy": float(report["balanced_accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "background_precision": float(report["background"]["precision"]),
        "background_recall": float(report["background"]["recall"]),
        "background_f1": float(report["background"]["f1-score"]),
        "foreground_precision": float(report["foreground"]["precision"]),
        "foreground_recall": float(report["foreground"]["recall"]),
        "foreground_f1": float(report["foreground"]["f1-score"]),
        "true_background_pred_background": int(matrix[0, 0]),
        "true_background_pred_foreground": int(matrix[0, 1]),
        "true_foreground_pred_background": int(matrix[1, 0]),
        "true_foreground_pred_foreground": int(matrix[1, 1]),
    }
    payload = {
        "classification_report": report,
        "classification_report_text": report_text,
        "confusion_matrix": matrix.tolist(),
    }
    return row, payload


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# Cached Bag Filter Rule Evaluation", ""]
    lines.append(
        "| Rule | Macro F1 | FG recall | FG precision | BG recall | FP | FN | Pred FG | Artifact rejects |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.rule} | {row.macro_f1:.4f} | {row.foreground_recall:.4f} | "
            f"{row.foreground_precision:.4f} | {row.background_recall:.4f} | "
            f"{row.true_background_pred_foreground} | "
            f"{row.true_foreground_pred_background} | {row.predicted_foreground} | "
            f"{row.rejected_by_artifact} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined = join_cached_scores(args)
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for spec in rule_specs():
        row, payload = evaluate_rule(joined, spec)
        rows.append(row)
        reports[spec.name] = payload
    summary = pd.DataFrame(rows).sort_values(
        ["macro_f1", "foreground_recall", "background_recall"],
        ascending=[False, False, False],
    )
    joined.to_parquet(args.output_dir / "joined_cached_test_scores.parquet", index=False)
    joined.to_csv(args.output_dir / "joined_cached_test_scores.csv", index=False)
    summary.to_csv(args.output_dir / "rule_metrics.csv", index=False)
    (args.output_dir / "classification_reports.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    write_report(summary, args.output_dir)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
