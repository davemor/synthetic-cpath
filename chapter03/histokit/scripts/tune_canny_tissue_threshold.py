#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TEST_MANIFEST = Path(
    "/sharedscratch/dm236/data/icaird_cervical/resnet_patch_subset/"
    "stratified_120_consensus_100k/split_dataset/sample_manifest_test.parquet"
)
DEFAULT_PATCHSETS_ROOT = Path(
    "/sharedscratch/dm236/data/icaird_cervical/patcher_cache/stratified_120/"
    "pipeline_research/patchsets"
)
DEFAULT_OUTPUT_DIR = Path(
    "/sharedscratch/dm236/data/icaird_cervical/segmentation_thresholds/"
    "canny_cached_test"
)
CLASS_LABELS = ["background", "foreground"]
PATCH_ID_COLUMNS = ["slide_id", "row", "column"]


@dataclass(frozen=True)
class Selection:
    threshold: float
    metric_value: float
    macro_f1: float
    foreground_recall: float
    threshold_metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune cached histokit Canny tissue_score thresholds against the "
            "iCAIRD cervical 3-annotator test-set consensus."
        )
    )
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--patchsets-root", type=Path, default=DEFAULT_PATCHSETS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--selection-metric",
        choices=["macro_f1", "foreground_f1", "balanced_accuracy", "accuracy"],
        default="macro_f1",
    )
    parser.add_argument("--min-threshold", type=float, default=0.0)
    parser.add_argument("--max-threshold", type=float, default=0.1)
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Optional comma-separated explicit thresholds, overriding min/max/step.",
    )
    parser.add_argument(
        "--require-complete",
        dest="require_complete",
        action="store_true",
        default=True,
        help="Fail if any test patch is missing a cached tissue_score.",
    )
    parser.add_argument(
        "--allow-incomplete",
        dest="require_complete",
        action="store_false",
        help="Drop test patches missing cached tissue_score instead of failing.",
    )
    return parser.parse_args()


def require_metrics() -> tuple[Any, Any, Any]:
    try:
        from sklearn.metrics import (
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
        )
    except ImportError as exc:
        raise RuntimeError(
            "This script requires scikit-learn. Install the training extra or "
            "run in the Hypatia training environment with scikit-learn present."
        ) from exc
    return balanced_accuracy_score, classification_report, confusion_matrix


def parse_thresholds(args: argparse.Namespace) -> list[float]:
    if args.thresholds:
        values = [float(value.strip()) for value in args.thresholds.split(",")]
        thresholds = sorted(set(values))
    else:
        if args.step <= 0:
            raise ValueError("--step must be positive")
        if args.max_threshold < args.min_threshold:
            raise ValueError("--max-threshold must be >= --min-threshold")
        count = int(np.floor((args.max_threshold - args.min_threshold) / args.step))
        thresholds = [
            round(args.min_threshold + i * args.step, 12)
            for i in range(count + 1)
        ]
        if thresholds[-1] < args.max_threshold - (args.step * 1e-6):
            thresholds.append(round(args.max_threshold, 12))
    if not thresholds:
        raise ValueError("No thresholds were provided")
    return thresholds


def load_test_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Test manifest not found: {path}")
    frame = pd.read_parquet(path)
    required = {*PATCH_ID_COLUMNS, "majority_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Test manifest missing columns: {sorted(missing)}")
    frame = frame.loc[frame["majority_label"].isin(CLASS_LABELS)].copy()
    frame["slide_id"] = frame["slide_id"].astype(str)
    frame["row"] = frame["row"].astype(int)
    frame["column"] = frame["column"].astype(int)
    return frame


def load_patchset_scores(patchsets_root: Path, slide_ids: list[str]) -> pd.DataFrame:
    if not patchsets_root.exists():
        raise FileNotFoundError(f"Patchsets root not found: {patchsets_root}")

    frames: list[pd.DataFrame] = []
    missing_patchsets: list[str] = []
    for slide_id in sorted(set(slide_ids)):
        frame_path = patchsets_root / f"{slide_id}.patchset" / "frame.parquet"
        if not frame_path.exists():
            missing_patchsets.append(slide_id)
            continue
        scores = pd.read_parquet(frame_path)
        required = {"row", "column", "tissue_score"}
        missing = required - set(scores.columns)
        if missing:
            raise ValueError(f"{frame_path} missing columns: {sorted(missing)}")
        scores = scores[["row", "column", "tissue_score"]].copy()
        scores["slide_id"] = slide_id
        scores["row"] = scores["row"].astype(int)
        scores["column"] = scores["column"].astype(int)
        frames.append(scores)

    if missing_patchsets:
        examples = ", ".join(missing_patchsets[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_patchsets)} patchset(s) under {patchsets_root}. "
            f"Examples: {examples}"
        )
    if not frames:
        raise RuntimeError(f"No patchset scores loaded from {patchsets_root}")
    return pd.concat(frames, ignore_index=True)


def join_scores(
    test_manifest: pd.DataFrame,
    patchset_scores: pd.DataFrame,
    *,
    require_complete: bool,
) -> pd.DataFrame:
    joined = test_manifest.merge(
        patchset_scores,
        on=PATCH_ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    missing = joined["tissue_score"].isna()
    if missing.any() and require_complete:
        examples = joined.loc[missing, PATCH_ID_COLUMNS].head(10).to_dict("records")
        raise ValueError(
            f"Missing cached tissue_score for {int(missing.sum())} test patch(es). "
            f"Examples: {examples}"
        )
    if missing.any():
        joined = joined.loc[~missing].copy()
    return joined


def report_to_row(
    threshold: float,
    report: dict[str, Any],
    matrix: np.ndarray,
    balanced_accuracy: float,
) -> dict[str, Any]:
    return {
        "threshold": threshold,
        "accuracy": float(report["accuracy"]),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
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


def evaluate_thresholds(
    frame: pd.DataFrame,
    thresholds: list[float],
    *,
    balanced_accuracy_score: Any,
    classification_report: Any,
    confusion_matrix: Any,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[float, np.ndarray]]:
    y_true = frame["majority_label"].astype(str).to_numpy()
    tissue_scores = frame["tissue_score"].astype(float).to_numpy()

    rows: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    matrices: dict[float, np.ndarray] = {}

    for threshold in thresholds:
        y_pred = np.where(tissue_scores > threshold, "foreground", "background")
        report = classification_report(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            target_names=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        rows.append(report_to_row(threshold, report, matrix, balanced_accuracy))
        reports[f"{threshold:.12g}"] = report
        matrices[threshold] = matrix

    return pd.DataFrame(rows), reports, matrices


def select_best(metrics: pd.DataFrame, selection_metric: str) -> Selection:
    metric_column = {
        "macro_f1": "macro_f1",
        "foreground_f1": "foreground_f1",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
    }[selection_metric]
    ordered = metrics.sort_values(
        [metric_column, "foreground_recall", "threshold"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    best = ordered.iloc[0].to_dict()
    return Selection(
        threshold=float(best["threshold"]),
        metric_value=float(best[metric_column]),
        macro_f1=float(best["macro_f1"]),
        foreground_recall=float(best["foreground_recall"]),
        threshold_metrics=best,
    )


def write_outputs(
    *,
    output_dir: Path,
    joined: pd.DataFrame,
    metrics: pd.DataFrame,
    reports: dict[str, dict[str, Any]],
    matrices: dict[float, np.ndarray],
    selection: Selection,
    selection_metric: str,
    test_manifest: Path,
    patchsets_root: Path,
    thresholds: list[float],
    classification_report: Any,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    best_predictions = np.where(
        joined["tissue_score"].astype(float).to_numpy() > selection.threshold,
        "foreground",
        "background",
    )
    joined = joined.copy()
    joined["canny_threshold"] = selection.threshold
    joined["canny_prediction"] = best_predictions

    metrics.to_csv(output_dir / "threshold_metrics.csv", index=False)
    (output_dir / "classification_reports.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    joined.to_parquet(output_dir / "joined_test_patch_scores.parquet", index=False)

    matrix = matrices[selection.threshold]
    pd.DataFrame(matrix, index=CLASS_LABELS, columns=CLASS_LABELS).to_csv(
        output_dir / "best_confusion_matrix.csv"
    )

    report_text = classification_report(
        joined["majority_label"],
        joined["canny_prediction"],
        labels=CLASS_LABELS,
        target_names=CLASS_LABELS,
        zero_division=0,
    )
    (output_dir / "best_classification_report.txt").write_text(report_text)

    summary = {
        "selection_metric": selection_metric,
        "selection_rule": (
            f"maximize {selection_metric}; tie-break by higher foreground recall, "
            "then lower threshold"
        ),
        "best": asdict(selection),
        "test_manifest": str(test_manifest),
        "patchsets_root": str(patchsets_root),
        "patch_count": int(len(joined)),
        "slide_count": int(joined["slide_id"].nunique()),
        "threshold_count": int(len(thresholds)),
        "min_threshold": float(min(thresholds)),
        "max_threshold": float(max(thresholds)),
    }
    (output_dir / "best_threshold.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    ) = require_metrics()
    thresholds = parse_thresholds(args)

    test_manifest = load_test_manifest(args.test_manifest)
    patchset_scores = load_patchset_scores(
        args.patchsets_root, test_manifest["slide_id"].tolist()
    )
    joined = join_scores(
        test_manifest,
        patchset_scores,
        require_complete=args.require_complete,
    )
    metrics, reports, matrices = evaluate_thresholds(
        joined,
        thresholds,
        balanced_accuracy_score=balanced_accuracy_score,
        classification_report=classification_report,
        confusion_matrix=confusion_matrix,
    )
    selection = select_best(metrics, args.selection_metric)
    write_outputs(
        output_dir=args.output_dir,
        joined=joined,
        metrics=metrics,
        reports=reports,
        matrices=matrices,
        selection=selection,
        selection_metric=args.selection_metric,
        test_manifest=args.test_manifest,
        patchsets_root=args.patchsets_root,
        thresholds=thresholds,
        classification_report=classification_report,
    )

    print(f"Evaluated {len(joined)} patches across {joined['slide_id'].nunique()} slides")
    print(f"Best threshold: {selection.threshold:.6g}")
    print(f"{args.selection_metric}: {selection.metric_value:.6f}")
    print(f"macro_f1: {selection.macro_f1:.6f}")
    print(f"foreground_recall: {selection.foreground_recall:.6f}")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
