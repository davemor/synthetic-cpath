#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from evaluate_test_candidate_tissue_rules import (
    CLASS_LABELS,
    DEFAULT_GRANDQC_TEST_SCORES,
    DEFAULT_PATCHSETS_ROOT,
    DEFAULT_RESNET_DIR,
    DEFAULT_TEST_MANIFEST,
    PATCH_ID_COLUMNS,
    add_rule_columns,
    join_inputs,
    write_outputs,
)


DATA_ROOT = Path("/sharedscratch/dm236/data/icaird_cervical")
DEFAULT_OFFICIAL_OUTPUT = DATA_ROOT / "grandqc_official_test20_axisfix/output"
DEFAULT_SLIDE_INDEX = DATA_ROOT / "patch_encoder_datasets/full_canny005_grandqc0372_artifact7x/dataset_index_ometiff.csv"
DEFAULT_OUTPUT = DATA_ROOT / "fgbg_test_candidate_rule_comparison_official_grandqc"
OFFICIAL_CLASSES = {
    1: "normal_tissue",
    2: "fold",
    3: "darkspot_foreign_object",
    4: "pen_marking",
    5: "air_bubble_slide_edge",
    6: "out_of_focus",
    7: "background",
}
Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate test-only candidate tissue rules using official fixed "
            "GrandQC artifact masks mapped back to annotated patches."
        )
    )
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--patchsets-root", type=Path, default=DEFAULT_PATCHSETS_ROOT)
    parser.add_argument(
        "--grandqc-test-scores",
        type=Path,
        default=DEFAULT_GRANDQC_TEST_SCORES,
    )
    parser.add_argument("--resnet-dir", type=Path, default=DEFAULT_RESNET_DIR)
    parser.add_argument("--slide-index", type=Path, default=DEFAULT_SLIDE_INDEX)
    parser.add_argument("--official-output", type=Path, default=DEFAULT_OFFICIAL_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--canny-threshold", type=float, default=0.05)
    parser.add_argument("--grandqc-threshold", type=float, default=0.372)
    parser.add_argument("--official-grandqc-mpp", type=float, default=1.5)
    parser.add_argument("--slide-mpp", type=float, default=0.2425)
    parser.add_argument("--model-patch-size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--patch-level", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def load_slide_downsamples(index_path: Path, patch_level: int) -> dict[str, float]:
    from tiffslide import TiffSlide

    if not index_path.exists():
        raise FileNotFoundError(f"Slide index not found: {index_path}")
    index = pd.read_csv(index_path)
    if "id" not in index.columns or "slide_path" not in index.columns:
        raise ValueError(f"{index_path} must include id and slide_path columns")
    downsamples: dict[str, float] = {}
    for row in index.itertuples(index=False):
        slide_id = str(getattr(row, "id"))
        slide_path = Path(str(getattr(row, "slide_path")))
        with TiffSlide(str(slide_path)) as slide:
            downsamples[slide_id] = float(slide.level_downsamples[patch_level])
    return downsamples


def official_mask_path(official_output: Path, slide_id: str) -> Path:
    return official_output / "mask_qc" / f"{slide_id}.ome.tiff_mask.png"


def load_official_patch_classes(
    args: argparse.Namespace,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    downsamples = load_slide_downsamples(args.slide_index, args.patch_level)
    p_s = int(args.official_grandqc_mpp / args.slide_mpp * args.model_patch_size)
    rows: list[pd.DataFrame] = []
    for slide_id, slide_frame in manifest.groupby("slide_id", sort=True):
        path = official_mask_path(args.official_output, str(slide_id))
        if not path.exists():
            raise FileNotFoundError(f"Official GrandQC mask not found: {path}")
        if slide_id not in downsamples:
            raise ValueError(f"Slide {slide_id} missing from {args.slide_index}")
        mask = np.asarray(Image.open(path))
        downsample = downsamples[str(slide_id)]
        x_level0 = (
            slide_frame["column"].to_numpy(dtype=float) + 0.5
        ) * args.patch_size * downsample
        y_level0 = (
            slide_frame["row"].to_numpy(dtype=float) + 0.5
        ) * args.patch_size * downsample
        x_mask = np.clip(
            np.floor(x_level0 / p_s * args.model_patch_size).astype(int),
            0,
            mask.shape[1] - 1,
        )
        y_mask = np.clip(
            np.floor(y_level0 / p_s * args.model_patch_size).astype(int),
            0,
            mask.shape[0] - 1,
        )
        class_id = mask[y_mask, x_mask].astype(int)
        out = slide_frame[PATCH_ID_COLUMNS].copy()
        out["official_grandqc_artifact_class_id"] = class_id
        out["official_grandqc_artifact_class"] = [
            OFFICIAL_CLASSES.get(int(value), f"unknown_{int(value)}")
            for value in class_id
        ]
        out["official_grandqc_artifact_normal_tissue_keep"] = class_id == 1
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def add_official_rule_columns(frame: pd.DataFrame) -> None:
    frame["canny_or_resnet_and_official_artifact_normal_keep"] = (
        frame["canny_or_resnet_keep"]
        & frame["official_grandqc_artifact_normal_tissue_keep"]
    )
    frame["canny_or_resnet_and_grandqc_tissue_and_official_artifact_normal_keep"] = (
        frame["canny_or_resnet_and_grandqc_tissue_keep"]
        & frame["official_grandqc_artifact_normal_tissue_keep"]
    )


def evaluate_official_rules(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
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
        "canny_or_resnet_and_official_grandqc_artifact_normal": (
            "canny_or_resnet_and_official_artifact_normal_keep"
        ),
        "canny_or_resnet_and_grandqc_tissue_and_official_artifact_normal": (
            "canny_or_resnet_and_grandqc_tissue_and_official_artifact_normal_keep"
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


def main() -> None:
    args = parse_args()
    args.patch_encoder_root = Path("/unused")
    args.require_artifact_scores = False
    joined = join_inputs(args)
    add_rule_columns(joined, args)
    official = load_official_patch_classes(args, joined)
    joined = joined.merge(
        official,
        on=PATCH_ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    missing = joined["official_grandqc_artifact_class_id"].isna()
    if missing.any():
        raise ValueError(
            "Missing official GrandQC artifact class for "
            f"{int(missing.sum())} patch(es)"
        )
    add_official_rule_columns(joined)
    summary, reports = evaluate_official_rules(joined)
    write_outputs(args.output_dir, joined, summary, reports)
    class_counts = (
        joined.groupby(["official_grandqc_artifact_class_id", "official_grandqc_artifact_class"])
        .size()
        .reset_index(name="patch_count")
    )
    class_counts.to_csv(args.output_dir / "official_artifact_class_counts.csv", index=False)
    print((args.output_dir / "report.md").read_text())
    print("\nOfficial artifact class counts:")
    print(class_counts.to_string(index=False))


if __name__ == "__main__":
    main()
