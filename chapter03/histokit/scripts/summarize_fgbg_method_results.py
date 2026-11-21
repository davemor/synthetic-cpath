#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_resnet19_patch_classifier import (
    INDEX_TO_CLASS,
    PatchManifestDataset,
    make_resnet19,
    require_training_dependencies,
    resolve_device,
)


DATA_ROOT = Path("/sharedscratch/dm236/data/icaird_cervical")
DEFAULT_SPLIT_DIR = (
    DATA_ROOT
    / "resnet_patch_subset/stratified_120_consensus_100k/split_dataset"
)
DEFAULT_PATCHSETS_ROOT = (
    DATA_ROOT / "patcher_cache/stratified_120/pipeline_research/patchsets"
)
DEFAULT_GRANDQC_TRAIN_DIR = DATA_ROOT / "segmentation_thresholds/grandqc_train"
DEFAULT_GRANDQC_TEST_DIR = (
    DATA_ROOT / "segmentation_thresholds/grandqc_test_threshold_0_372"
)
DEFAULT_SEGMENTATION_THRESHOLDS_ROOT = DATA_ROOT / "segmentation_thresholds"
DEFAULT_RESNET_DIR = (
    DATA_ROOT
    / "resnet_patch_subset/stratified_120_consensus_100k/"
    "resnet19_patch_classifier"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "fgbg_method_comparison"

CLASS_LABELS = ["background", "foreground"]
PATCH_ID_COLUMNS = ["slide_id", "row", "column"]
SLIDE_DETECTORS = ["clam_segmentation_otsu", "otsu_hs_segmentation"]


@dataclass(frozen=True)
class MethodResult:
    method: str
    split: str
    threshold: float | None
    threshold_selection_split: str | None
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    background_precision: float
    background_recall: float
    background_f1: float
    foreground_precision: float
    foreground_recall: float
    foreground_f1: float
    support: int
    true_background_pred_background: int
    true_background_pred_foreground: int
    true_foreground_pred_background: int
    true_foreground_pred_foreground: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise foreground/background performance for cached Canny, "
            "ResNet-19, and GrandQC on the train and test patch splits."
        )
    )
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--patchsets-root", type=Path, default=DEFAULT_PATCHSETS_ROOT)
    parser.add_argument("--grandqc-train-dir", type=Path, default=DEFAULT_GRANDQC_TRAIN_DIR)
    parser.add_argument("--grandqc-test-dir", type=Path, default=DEFAULT_GRANDQC_TEST_DIR)
    parser.add_argument(
        "--segmentation-thresholds-root",
        type=Path,
        default=DEFAULT_SEGMENTATION_THRESHOLDS_ROOT,
    )
    parser.add_argument("--resnet-dir", type=Path, default=DEFAULT_RESNET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--canny-min-threshold", type=float, default=0.0)
    parser.add_argument("--canny-max-threshold", type=float, default=0.1)
    parser.add_argument("--canny-step", type=float, default=0.001)
    parser.add_argument("--grandqc-min-threshold", type=float, default=0.0)
    parser.add_argument("--grandqc-max-threshold", type=float, default=1.0)
    parser.add_argument("--grandqc-step", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for ResNet evaluation.",
    )
    parser.add_argument(
        "--skip-resnet",
        action="store_true",
        help="Only summarise threshold methods.",
    )
    return parser.parse_args()


def parse_threshold_grid(
    min_threshold: float,
    max_threshold: float,
    step: float,
) -> list[float]:
    if step <= 0:
        raise ValueError("Threshold step must be positive")
    if max_threshold < min_threshold:
        raise ValueError("Maximum threshold must be >= minimum threshold")
    count = int(np.floor((max_threshold - min_threshold) / step))
    values = [round(min_threshold + i * step, 12) for i in range(count + 1)]
    if values[-1] < max_threshold - (step * 1e-6):
        values.append(round(max_threshold, 12))
    return values


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
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


def load_canny_scores(patchsets_root: Path, manifests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    slide_ids = sorted(
        set(pd.concat([frame["slide_id"] for frame in manifests.values()]).astype(str))
    )
    frames: list[pd.DataFrame] = []
    missing_patchsets: list[str] = []
    for slide_id in slide_ids:
        path = patchsets_root / f"{slide_id}.patchset" / "frame.parquet"
        if not path.exists():
            missing_patchsets.append(slide_id)
            continue
        frame = pd.read_parquet(path)
        required = {"row", "column", "tissue_score"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frame = frame[["row", "column", "tissue_score"]].copy()
        frame["slide_id"] = slide_id
        frame["row"] = frame["row"].astype(int)
        frame["column"] = frame["column"].astype(int)
        frames.append(frame)
    if missing_patchsets:
        examples = ", ".join(missing_patchsets[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_patchsets)} Canny patchset(s). Examples: {examples}"
        )
    return pd.concat(frames, ignore_index=True)


def join_score_column(
    manifest: pd.DataFrame,
    scores: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    joined = manifest.merge(
        scores[[*PATCH_ID_COLUMNS, score_column]],
        on=PATCH_ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    missing = joined[score_column].isna()
    if missing.any():
        examples = joined.loc[missing, PATCH_ID_COLUMNS].head(10).to_dict("records")
        raise ValueError(
            f"Missing {score_column} for {int(missing.sum())} patch(es). "
            f"Examples: {examples}"
        )
    return joined


def load_grandqc_joined(train_dir: Path, test_dir: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "train": train_dir / "joined_train_patch_scores.parquet",
        "test": test_dir / "joined_test_patch_scores.parquet",
    }
    frames: dict[str, pd.DataFrame] = {}
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"GrandQC joined score file not found: {path}")
        frame = pd.read_parquet(path)
        required = {*PATCH_ID_COLUMNS, "majority_label", "grandqc_tissue_score"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frames[split] = frame.loc[frame["majority_label"].isin(CLASS_LABELS)].copy()
    return frames


def score_column(detector: str) -> str:
    return f"{detector}_score"


def format_threshold_for_path(threshold: float) -> str:
    return f"{threshold:.3f}".replace(".", "_")


def read_selected_threshold(path: Path) -> float:
    data = json.loads(path.read_text())
    return float(data["best"]["threshold"])


def find_slide_detector_test_dir(
    root: Path,
    detector: str,
    threshold: float,
) -> Path:
    expected = root / f"{detector}_test_threshold_{format_threshold_for_path(threshold)}"
    if expected.exists():
        return expected

    for candidate in sorted(root.glob(f"{detector}_test_threshold_*")):
        threshold_path = candidate / "best_threshold.json"
        if not threshold_path.exists():
            continue
        data = json.loads(threshold_path.read_text())
        fixed = data.get("fixed_threshold")
        if fixed is not None and np.isclose(float(fixed), threshold):
            return candidate
    raise FileNotFoundError(
        f"No test output found for {detector} threshold {threshold} under {root}"
    )


def load_slide_detector_joined(root: Path, detector: str) -> dict[str, pd.DataFrame]:
    column = score_column(detector)
    train_dir = root / f"{detector}_train"
    threshold_path = train_dir / "best_threshold.json"
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"{detector} training threshold file not found: {threshold_path}"
        )
    threshold = read_selected_threshold(threshold_path)
    test_dir = find_slide_detector_test_dir(root, detector, threshold)
    paths = {
        "train": train_dir / "joined_train_patch_scores.parquet",
        "test": test_dir / "joined_test_patch_scores.parquet",
    }
    frames: dict[str, pd.DataFrame] = {}
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{detector} joined score file not found: {path}")
        frame = pd.read_parquet(path)
        required = {*PATCH_ID_COLUMNS, "majority_label", column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frames[split] = frame.loc[frame["majority_label"].isin(CLASS_LABELS)].copy()
    return frames


def metric_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    classification_report: Any,
    confusion_matrix: Any,
    balanced_accuracy_score: Any,
) -> tuple[dict[str, Any], np.ndarray, str]:
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
    return report, matrix, report_text


def select_threshold(
    frame: pd.DataFrame,
    score_column: str,
    thresholds: list[float],
    *,
    classification_report: Any,
    confusion_matrix: Any,
    balanced_accuracy_score: Any,
) -> tuple[float, pd.DataFrame]:
    del classification_report, confusion_matrix, balanced_accuracy_score
    y_true = frame["majority_label"].astype(str).to_numpy()
    scores = frame[score_column].astype(float).to_numpy()
    is_background = y_true == "background"
    is_foreground = y_true == "foreground"
    rows = []
    for threshold in thresholds:
        pred_foreground = scores > threshold
        true_background_pred_background = int((is_background & ~pred_foreground).sum())
        true_background_pred_foreground = int((is_background & pred_foreground).sum())
        true_foreground_pred_background = int((is_foreground & ~pred_foreground).sum())
        true_foreground_pred_foreground = int((is_foreground & pred_foreground).sum())
        rows.append(
            threshold_metric_row(
                threshold,
                true_background_pred_background,
                true_background_pred_foreground,
                true_foreground_pred_background,
                true_foreground_pred_foreground,
            )
        )
    metrics = pd.DataFrame(rows)
    ordered = metrics.sort_values(
        ["macro_f1", "foreground_recall", "threshold"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return float(ordered.iloc[0]["threshold"]), metrics


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def threshold_metric_row(
    threshold: float,
    true_background_pred_background: int,
    true_background_pred_foreground: int,
    true_foreground_pred_background: int,
    true_foreground_pred_foreground: int,
) -> dict[str, Any]:
    background_support = (
        true_background_pred_background + true_background_pred_foreground
    )
    foreground_support = (
        true_foreground_pred_background + true_foreground_pred_foreground
    )
    support = background_support + foreground_support
    background_precision = safe_divide(
        true_background_pred_background,
        true_background_pred_background + true_foreground_pred_background,
    )
    background_recall = safe_divide(
        true_background_pred_background,
        background_support,
    )
    background_f1 = safe_divide(
        2 * background_precision * background_recall,
        background_precision + background_recall,
    )
    foreground_precision = safe_divide(
        true_foreground_pred_foreground,
        true_foreground_pred_foreground + true_background_pred_foreground,
    )
    foreground_recall = safe_divide(
        true_foreground_pred_foreground,
        foreground_support,
    )
    foreground_f1 = safe_divide(
        2 * foreground_precision * foreground_recall,
        foreground_precision + foreground_recall,
    )
    accuracy = safe_divide(
        true_background_pred_background + true_foreground_pred_foreground,
        support,
    )
    balanced_accuracy = (background_recall + foreground_recall) / 2
    macro_f1 = (background_f1 + foreground_f1) / 2
    weighted_f1 = safe_divide(
        background_f1 * background_support + foreground_f1 * foreground_support,
        support,
    )
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "background_precision": background_precision,
        "background_recall": background_recall,
        "background_f1": background_f1,
        "foreground_precision": foreground_precision,
        "foreground_recall": foreground_recall,
        "foreground_f1": foreground_f1,
        "true_background_pred_background": true_background_pred_background,
        "true_background_pred_foreground": true_background_pred_foreground,
        "true_foreground_pred_background": true_foreground_pred_background,
        "true_foreground_pred_foreground": true_foreground_pred_foreground,
    }


def result_row(
    method: str,
    split: str,
    threshold: float | None,
    threshold_selection_split: str | None,
    report: dict[str, Any],
    matrix: np.ndarray,
) -> MethodResult:
    return MethodResult(
        method=method,
        split=split,
        threshold=threshold,
        threshold_selection_split=threshold_selection_split,
        accuracy=float(report["accuracy"]),
        balanced_accuracy=float(report["balanced_accuracy"]),
        macro_f1=float(report["macro avg"]["f1-score"]),
        weighted_f1=float(report["weighted avg"]["f1-score"]),
        background_precision=float(report["background"]["precision"]),
        background_recall=float(report["background"]["recall"]),
        background_f1=float(report["background"]["f1-score"]),
        foreground_precision=float(report["foreground"]["precision"]),
        foreground_recall=float(report["foreground"]["recall"]),
        foreground_f1=float(report["foreground"]["f1-score"]),
        support=int(report["macro avg"]["support"]),
        true_background_pred_background=int(matrix[0, 0]),
        true_background_pred_foreground=int(matrix[0, 1]),
        true_foreground_pred_background=int(matrix[1, 0]),
        true_foreground_pred_foreground=int(matrix[1, 1]),
    )


def write_metric_artifacts(
    output_dir: Path,
    method: str,
    split: str,
    report: dict[str, Any],
    report_text: str,
    matrix: np.ndarray,
) -> None:
    method_dir = output_dir / method / split
    method_dir.mkdir(parents=True, exist_ok=True)
    (method_dir / "classification_report.txt").write_text(report_text)
    (method_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    pd.DataFrame(matrix, index=CLASS_LABELS, columns=CLASS_LABELS).to_csv(
        method_dir / "confusion_matrix.csv"
    )


def evaluate_threshold_method(
    method: str,
    split_frames: dict[str, pd.DataFrame],
    score_column: str,
    thresholds: list[float],
    output_dir: Path,
    *,
    classification_report: Any,
    confusion_matrix: Any,
    balanced_accuracy_score: Any,
) -> tuple[list[MethodResult], float, pd.DataFrame]:
    threshold, threshold_metrics = select_threshold(
        split_frames["train"],
        score_column,
        thresholds,
        classification_report=classification_report,
        confusion_matrix=confusion_matrix,
        balanced_accuracy_score=balanced_accuracy_score,
    )
    results: list[MethodResult] = []
    for split, frame in split_frames.items():
        y_true = frame["majority_label"].astype(str).to_numpy()
        scores = frame[score_column].astype(float).to_numpy()
        y_pred = np.where(scores > threshold, "foreground", "background")
        report, matrix, report_text = metric_bundle(
            y_true,
            y_pred,
            classification_report=classification_report,
            confusion_matrix=confusion_matrix,
            balanced_accuracy_score=balanced_accuracy_score,
        )
        write_metric_artifacts(output_dir, method, split, report, report_text, matrix)
        results.append(result_row(method, split, threshold, "train", report, matrix))
    return results, threshold, threshold_metrics


def evaluate_resnet(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    classification_report: Any,
    confusion_matrix: Any,
    balanced_accuracy_score: Any,
) -> list[MethodResult]:
    torch, nn, _, _ = require_training_dependencies()
    device = resolve_device(args.device, torch)
    checkpoint = args.resnet_dir / "resnet19_final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"ResNet checkpoint not found: {checkpoint}")

    model = make_resnet19(nn).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    results = []
    for split in ["train", "test"]:
        manifest = args.split_dir / f"sample_manifest_{split}.parquet"
        dataset = PatchManifestDataset(manifest, image_size=224, train=False)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        y_true_int: list[int] = []
        y_pred_int: list[int] = []
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                y_true_int.extend(int(label) for label in labels.tolist())
                y_pred_int.extend(int(pred) for pred in logits.argmax(dim=1).cpu().tolist())
        y_true = np.array([INDEX_TO_CLASS[index] for index in y_true_int])
        y_pred = np.array([INDEX_TO_CLASS[index] for index in y_pred_int])
        report, matrix, report_text = metric_bundle(
            y_true,
            y_pred,
            classification_report=classification_report,
            confusion_matrix=confusion_matrix,
            balanced_accuracy_score=balanced_accuracy_score,
        )
        write_metric_artifacts(output_dir, "resnet19", split, report, report_text, matrix)
        results.append(result_row("resnet19", split, None, None, report, matrix))
    return results


def make_markdown_report(results: list[MethodResult], thresholds: pd.DataFrame) -> str:
    summary = pd.DataFrame([asdict(row) for row in results])
    threshold_table = thresholds.copy()
    for column in threshold_table.select_dtypes(include=["float"]).columns:
        threshold_table[column] = threshold_table[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.6f}"
        )
    summary_table = summary[
        [
            "method",
            "split",
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "foreground_recall",
            "background_recall",
            "support",
        ]
    ].copy()
    for column in summary_table.select_dtypes(include=["float"]).columns:
        summary_table[column] = summary_table[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    lines = [
        "# Foreground/Background Method Comparison",
        "",
        "Thresholds for Canny, GrandQC, CLAM-Otsu, and Otsu-HS were selected "
        "on the training split by macro F1, with foreground recall and then "
        "lower threshold as tie-breakers. ResNet-19 uses argmax over the two "
        "output logits, so there is no scalar tissue threshold.",
        "",
        "## Selected Thresholds",
        "",
        dataframe_to_markdown(threshold_table),
        "",
        "## Summary Metrics",
        "",
        dataframe_to_markdown(summary_table),
        "",
        "## Confusion Matrices",
        "",
    ]
    for row in results:
        lines.extend(
            [
                f"### {row.method} {row.split}",
                "",
                "| true \\\\ predicted | background | foreground |",
                "|---|---:|---:|",
                (
                    "| background | "
                    f"{row.true_background_pred_background} | "
                    f"{row.true_background_pred_foreground} |"
                ),
                (
                    "| foreground | "
                    f"{row.true_foreground_pred_background} | "
                    f"{row.true_foreground_pred_foreground} |"
                ),
                "",
            ]
        )
    return "\n".join(lines)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [columns]
    rows.extend(
        [
            ["" if pd.isna(value) else str(value) for value in row]
            for row in frame.to_numpy()
        ]
    )
    widths = [
        max(len(str(row[index])) for row in rows)
        for index in range(len(columns))
    ]
    rendered = []
    for row_index, row in enumerate(rows):
        rendered.append(
            "| "
            + " | ".join(
                str(value).ljust(widths[index])
                for index, value in enumerate(row)
            )
            + " |"
        )
        if row_index == 0:
            rendered.append(
                "| "
                + " | ".join("-" * widths[index] for index in range(len(columns)))
                + " |"
            )
    return "\n".join(rendered)


def main() -> None:
    args = parse_args()
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = {
        "train": load_manifest(args.split_dir / "sample_manifest_train.parquet"),
        "test": load_manifest(args.split_dir / "sample_manifest_test.parquet"),
    }

    all_results: list[MethodResult] = []
    threshold_rows: list[dict[str, Any]] = []

    canny_scores = load_canny_scores(args.patchsets_root, manifests)
    canny_frames = {
        split: join_score_column(frame, canny_scores, "tissue_score")
        for split, frame in manifests.items()
    }
    canny_results, canny_threshold, canny_threshold_metrics = evaluate_threshold_method(
        "canny",
        canny_frames,
        "tissue_score",
        parse_threshold_grid(
            args.canny_min_threshold,
            args.canny_max_threshold,
            args.canny_step,
        ),
        output_dir,
        classification_report=classification_report,
        confusion_matrix=confusion_matrix,
        balanced_accuracy_score=balanced_accuracy_score,
    )
    canny_threshold_metrics.to_csv(
        output_dir / "canny_train_threshold_metrics.csv", index=False
    )
    all_results.extend(canny_results)
    threshold_rows.append(
        {
            "method": "canny",
            "selected_threshold": canny_threshold,
            "selection_split": "train",
            "selection_metric": "macro_f1",
        }
    )

    grandqc_frames = load_grandqc_joined(args.grandqc_train_dir, args.grandqc_test_dir)
    grandqc_results, grandqc_threshold, grandqc_threshold_metrics = (
        evaluate_threshold_method(
            "grandqc",
            grandqc_frames,
            "grandqc_tissue_score",
            parse_threshold_grid(
                args.grandqc_min_threshold,
                args.grandqc_max_threshold,
                args.grandqc_step,
            ),
            output_dir,
            classification_report=classification_report,
            confusion_matrix=confusion_matrix,
            balanced_accuracy_score=balanced_accuracy_score,
        )
    )
    grandqc_threshold_metrics.to_csv(
        output_dir / "grandqc_train_threshold_metrics.csv", index=False
    )
    all_results.extend(grandqc_results)
    threshold_rows.append(
        {
            "method": "grandqc",
            "selected_threshold": grandqc_threshold,
            "selection_split": "train",
            "selection_metric": "macro_f1",
        }
    )

    for detector in SLIDE_DETECTORS:
        detector_frames = load_slide_detector_joined(
            args.segmentation_thresholds_root,
            detector,
        )
        detector_results, detector_threshold, detector_threshold_metrics = (
            evaluate_threshold_method(
                detector,
                detector_frames,
                score_column(detector),
                parse_threshold_grid(0.0, 1.0, 0.001),
                output_dir,
                classification_report=classification_report,
                confusion_matrix=confusion_matrix,
                balanced_accuracy_score=balanced_accuracy_score,
            )
        )
        detector_threshold_metrics.to_csv(
            output_dir / f"{detector}_train_threshold_metrics.csv", index=False
        )
        all_results.extend(detector_results)
        threshold_rows.append(
            {
                "method": detector,
                "selected_threshold": detector_threshold,
                "selection_split": "train",
                "selection_metric": "macro_f1",
            }
        )

    if not args.skip_resnet:
        all_results.extend(
            evaluate_resnet(
                args,
                output_dir,
                classification_report=classification_report,
                confusion_matrix=confusion_matrix,
                balanced_accuracy_score=balanced_accuracy_score,
            )
        )
    threshold_rows.append(
        {
            "method": "resnet19",
            "selected_threshold": None,
            "selection_split": None,
            "selection_metric": "argmax logits",
        }
    )

    summary = pd.DataFrame([asdict(row) for row in all_results])
    thresholds = pd.DataFrame(threshold_rows)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    thresholds.to_csv(output_dir / "selected_thresholds.csv", index=False)
    (output_dir / "summary_metrics.json").write_text(
        json.dumps([asdict(row) for row in all_results], indent=2) + "\n"
    )
    (output_dir / "report.md").write_text(make_markdown_report(all_results, thresholds))
    print(f"Wrote report artifacts to {output_dir}")
    print(summary[["method", "split", "accuracy", "macro_f1", "foreground_recall"]])


if __name__ == "__main__":
    main()
