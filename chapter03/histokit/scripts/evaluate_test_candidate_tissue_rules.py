#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
DEFAULT_TEST_MANIFEST = (
    DATA_ROOT
    / "resnet_patch_subset/stratified_120_consensus_100k/split_dataset/"
    "sample_manifest_test.parquet"
)
DEFAULT_PATCHSETS_ROOT = (
    DATA_ROOT / "patcher_cache/stratified_120/pipeline_research/patchsets"
)
DEFAULT_GRANDQC_TEST_SCORES = (
    DATA_ROOT
    / "segmentation_thresholds/grandqc_test_threshold_0_372/"
    "joined_test_patch_scores.parquet"
)
DEFAULT_PATCH_ENCODER_ROOT = (
    DATA_ROOT / "patch_encoder_datasets/full_canny005_grandqc0372_artifact7x"
)
DEFAULT_RESNET_DIR = (
    DATA_ROOT
    / "resnet_patch_subset/stratified_120_consensus_100k/"
    "resnet19_patch_classifier"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "fgbg_test_candidate_rule_comparison"
CLASS_LABELS = ["background", "foreground"]
PATCH_ID_COLUMNS = ["slide_id", "row", "column"]
ARTIFACT_CLASSES = {
    0: "unlabelled",
    1: "normal_tissue",
    2: "fold",
    3: "darkspot_foreign_object",
    4: "pen_marking",
    5: "air_bubble_slide_edge",
    6: "out_of_focus",
    7: "background",
}
ARTIFACT_PROB_COLUMNS = {
    class_id: f"grandqc_artifact_probability_{name}"
    for class_id, name in ARTIFACT_CLASSES.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate test-only candidate tissue rules using cached Canny, "
            "existing GrandQC tissue scores, ResNet-19, and GrandQC artifact "
            "scores."
        )
    )
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--patchsets-root", type=Path, default=DEFAULT_PATCHSETS_ROOT)
    parser.add_argument(
        "--grandqc-test-scores",
        type=Path,
        default=DEFAULT_GRANDQC_TEST_SCORES,
    )
    parser.add_argument(
        "--patch-encoder-root",
        type=Path,
        default=DEFAULT_PATCH_ENCODER_ROOT,
    )
    parser.add_argument("--resnet-dir", type=Path, default=DEFAULT_RESNET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--canny-threshold", type=float, default=0.05)
    parser.add_argument("--grandqc-threshold", type=float, default=0.372)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--require-artifact-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_test_manifest(path: Path) -> pd.DataFrame:
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


def load_canny_scores(patchsets_root: Path, slide_ids: list[str]) -> pd.DataFrame:
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
        raise FileNotFoundError(
            "Missing Canny patchsets for "
            f"{len(missing_patchsets)} test slide(s): {missing_patchsets[:10]}"
        )
    return pd.concat(frames, ignore_index=True).rename(
        columns={"tissue_score": "canny_tissue_score"}
    )


def load_grandqc_tissue_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"GrandQC test score file not found: {path}")
    frame = pd.read_parquet(path)
    required = {*PATCH_ID_COLUMNS, "grandqc_tissue_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame[[*PATCH_ID_COLUMNS, "grandqc_tissue_score"]].copy()
    frame["slide_id"] = frame["slide_id"].astype(str)
    frame["row"] = frame["row"].astype(int)
    frame["column"] = frame["column"].astype(int)
    return frame


def load_grandqc_artifact_scores(
    patch_encoder_root: Path,
    slide_ids: list[str],
    *,
    require_complete: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing_scores: list[str] = []
    score_root = patch_encoder_root / "grandqc_scores"
    artifact_columns = list(ARTIFACT_PROB_COLUMNS.values())
    for slide_id in slide_ids:
        path = score_root / f"{slide_id}.grandqc_scores.parquet"
        if not path.exists():
            missing_scores.append(slide_id)
            continue
        frame = pd.read_parquet(path)
        required = {"row", "column", *artifact_columns}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        keep_columns = ["row", "column", *artifact_columns]
        frame = frame[keep_columns].copy()
        frame["slide_id"] = slide_id
        frame["row"] = frame["row"].astype(int)
        frame["column"] = frame["column"].astype(int)
        frames.append(frame)
    if missing_scores and require_complete:
        raise FileNotFoundError(
            "Missing GrandQC artifact scores for "
            f"{len(missing_scores)} test slide(s): {missing_scores[:10]}"
        )
    if not frames:
        return pd.DataFrame(columns=[*PATCH_ID_COLUMNS, *artifact_columns])
    return pd.concat(frames, ignore_index=True)


def predict_resnet(args: argparse.Namespace, manifest: pd.DataFrame) -> pd.DataFrame:
    torch, nn, functional, _ = require_training_dependencies()
    device = resolve_device(args.device, torch)
    checkpoint = args.resnet_dir / "resnet19_final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"ResNet checkpoint not found: {checkpoint}")

    model = make_resnet19(nn).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset = PatchManifestDataset(args.test_manifest, image_size=224, train=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions: list[str] = []
    foreground_probs: list[float] = []
    background_probs: list[float] = []
    with torch.inference_mode():
        for images, _labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = functional.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(axis=1)
            predictions.extend(INDEX_TO_CLASS[int(index)] for index in pred)
            background_probs.extend(float(value) for value in probs[:, 0])
            foreground_probs.extend(float(value) for value in probs[:, 1])
    if len(predictions) != len(manifest):
        raise ValueError(
            "ResNet prediction count does not match manifest rows: "
            f"{len(predictions)} != {len(manifest)}"
        )
    return pd.DataFrame(
        {
            **{column: manifest[column].to_numpy() for column in PATCH_ID_COLUMNS},
            "resnet19_prediction": predictions,
            "resnet19_background_probability": background_probs,
            "resnet19_foreground_probability": foreground_probs,
        }
    )


def join_inputs(args: argparse.Namespace) -> pd.DataFrame:
    manifest = load_test_manifest(args.test_manifest)
    slide_ids = sorted(manifest["slide_id"].unique().tolist())
    joined = manifest.merge(
        load_canny_scores(args.patchsets_root, slide_ids),
        on=PATCH_ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    joined = joined.merge(
        load_grandqc_tissue_scores(args.grandqc_test_scores),
        on=PATCH_ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    joined = joined.merge(
        predict_resnet(args, manifest),
        on=PATCH_ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    artifact = load_grandqc_artifact_scores(
        args.patch_encoder_root,
        slide_ids,
        require_complete=args.require_artifact_scores,
    )
    if not artifact.empty:
        joined = joined.merge(
            artifact,
            on=PATCH_ID_COLUMNS,
            how="left",
            validate="one_to_one",
        )
    required_score_columns = [
        "canny_tissue_score",
        "grandqc_tissue_score",
        "resnet19_prediction",
    ]
    for column in required_score_columns:
        missing = joined[column].isna()
        if missing.any():
            raise ValueError(f"Missing {column} for {int(missing.sum())} patch(es)")
    add_rule_columns(joined, args)
    return joined


def add_rule_columns(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    frame["canny_keep"] = frame["canny_tissue_score"].astype(float) > args.canny_threshold
    frame["resnet19_keep"] = frame["resnet19_prediction"].astype(str) == "foreground"
    frame["grandqc_tissue_keep"] = (
        frame["grandqc_tissue_score"].astype(float) > args.grandqc_threshold
    )
    artifact_columns = list(ARTIFACT_PROB_COLUMNS.values())
    if set(artifact_columns).issubset(frame.columns):
        artifact_probs = frame[artifact_columns].to_numpy(dtype=float)
        missing_artifact = np.isnan(artifact_probs).any(axis=1)
        artifact_idx = artifact_probs.argmax(axis=1)
        artifact_class = np.array([ARTIFACT_CLASSES[int(value)] for value in artifact_idx])
        artifact_class[missing_artifact] = ""
        frame["grandqc_artifact_class_id"] = np.where(
            missing_artifact,
            np.nan,
            artifact_idx,
        )
        frame["grandqc_artifact_class"] = artifact_class
        frame["grandqc_artifact_normal_tissue_keep"] = artifact_idx == 1
        frame.loc[missing_artifact, "grandqc_artifact_normal_tissue_keep"] = False
    else:
        frame["grandqc_artifact_class_id"] = np.nan
        frame["grandqc_artifact_class"] = ""
        frame["grandqc_artifact_normal_tissue_keep"] = False

    frame["canny_or_resnet_keep"] = frame["canny_keep"] | frame["resnet19_keep"]
    frame["canny_or_resnet_and_grandqc_tissue_keep"] = (
        frame["canny_or_resnet_keep"] & frame["grandqc_tissue_keep"]
    )
    frame["canny_or_resnet_and_grandqc_tissue_and_artifact_keep"] = (
        frame["canny_or_resnet_and_grandqc_tissue_keep"]
        & frame["grandqc_artifact_normal_tissue_keep"]
    )


def evaluate_rules(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )

    rules = {
        "canny": "canny_keep",
        "resnet19": "resnet19_keep",
        "canny_or_resnet": "canny_or_resnet_keep",
        "canny_or_resnet_and_grandqc_tissue": (
            "canny_or_resnet_and_grandqc_tissue_keep"
        ),
        "canny_or_resnet_and_grandqc_tissue_and_grandqc_artifact_normal": (
            "canny_or_resnet_and_grandqc_tissue_and_artifact_keep"
        ),
    }
    y_true = frame["majority_label"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for name, column in rules.items():
        y_pred = np.where(frame[column].to_numpy(dtype=bool), "foreground", "background")
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
        reports[name] = {
            "classification_report": report,
            "classification_report_text": report_text,
            "confusion_matrix": matrix.tolist(),
        }
        rows.append(
            {
                "method": name,
                "split": "test",
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
                "support": int(report["macro avg"]["support"]),
                "true_background_pred_background": int(matrix[0, 0]),
                "true_background_pred_foreground": int(matrix[0, 1]),
                "true_foreground_pred_background": int(matrix[1, 0]),
                "true_foreground_pred_foreground": int(matrix[1, 1]),
            }
        )
    return pd.DataFrame(rows), reports


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [columns]
    rows.extend(
        [
            ["" if pd.isna(value) else str(value) for value in row]
            for row in frame.to_numpy()
        ]
    )
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(columns))]
    lines = [
        "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(rows[0])) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |",
    ]
    for row in rows[1:]:
        lines.append(
            "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"
        )
    return "\n".join(lines)


def make_report(summary: pd.DataFrame, reports: dict[str, Any]) -> str:
    table = summary[
        [
            "method",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "foreground_precision",
            "foreground_recall",
            "background_recall",
            "support",
        ]
    ].copy()
    for column in table.select_dtypes(include=["float"]).columns:
        table[column] = table[column].map(lambda value: f"{value:.4f}")
    lines = [
        "# Test Candidate Tissue Rule Comparison",
        "",
        "Evaluation is restricted to the 20 held-out test slides.",
        "",
        dataframe_to_markdown(table),
        "",
        "## Confusion Matrices",
        "",
    ]
    for _, row in summary.iterrows():
        matrix = reports[str(row["method"])]["confusion_matrix"]
        lines.extend(
            [
                f"### {row['method']}",
                "",
                "| true \\\\ predicted | background | foreground |",
                "|---|---:|---:|",
                f"| background | {matrix[0][0]} | {matrix[0][1]} |",
                f"| foreground | {matrix[1][0]} | {matrix[1][1]} |",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    joined: pd.DataFrame,
    summary: pd.DataFrame,
    reports: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(output_dir / "joined_test_candidate_rule_scores.parquet", index=False)
    joined.to_csv(output_dir / "joined_test_candidate_rule_scores.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2) + "\n"
    )
    reports_for_json = {
        method: {
            "classification_report": data["classification_report"],
            "confusion_matrix": data["confusion_matrix"],
        }
        for method, data in reports.items()
    }
    (output_dir / "classification_reports.json").write_text(
        json.dumps(reports_for_json, indent=2) + "\n"
    )
    for method, data in reports.items():
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "classification_report.txt").write_text(
            data["classification_report_text"]
        )
        pd.DataFrame(
            data["confusion_matrix"],
            index=CLASS_LABELS,
            columns=CLASS_LABELS,
        ).to_csv(method_dir / "confusion_matrix.csv")
    (output_dir / "report.md").write_text(make_report(summary, reports) + "\n")


def main() -> None:
    args = parse_args()
    joined = join_inputs(args)
    summary, reports = evaluate_rules(joined)
    write_outputs(args.output_dir, joined, summary, reports)
    print(make_report(summary, reports))


if __name__ == "__main__":
    main()
