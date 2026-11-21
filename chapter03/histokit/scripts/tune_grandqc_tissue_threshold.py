#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


DEFAULT_TRAIN_MANIFEST = Path(
    "/sharedscratch/dm236/data/icaird_cervical/resnet_patch_subset/"
    "stratified_120_consensus_100k/split_dataset/sample_manifest_train.parquet"
)
DEFAULT_SAMPLE_INDEX = Path(
    "/sharedscratch/dm236/data/icaird_cervical/stratified_120/"
    "index_sampled_stratified_120_ometiff.csv"
)
DEFAULT_SLIDES_ROOT = Path("/sharedscratch/dcchb-group/datasets/icaird_cervical/slides")
DEFAULT_OUTPUT_DIR = Path(
    "/sharedscratch/dm236/data/icaird_cervical/segmentation_thresholds/"
    "grandqc_train"
)
DEFAULT_WEIGHTS_DIR = Path(
    "/sharedscratch/dm236/data/icaird_cervical/grandqc_weights"
)
DEFAULT_MODEL_NAME = "Tissue_Detection_MPP10.pth"
DEFAULT_MODEL_URL = (
    "https://zenodo.org/records/14507273/files/"
    f"{DEFAULT_MODEL_NAME}?download=1"
)
CLASS_LABELS = ["background", "foreground"]
PATCH_ID_COLUMNS = ["slide_id", "row", "column"]
SCORE_COLUMN = "grandqc_tissue_score"


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
            "Cache GrandQC tissue probabilities for iCAIRD training slides and "
            "tune a foreground/background threshold against consensus labels."
        )
    )
    parser.add_argument(
        "--manifest",
        "--train-manifest",
        dest="manifest",
        type=Path,
        default=DEFAULT_TRAIN_MANIFEST,
        help="Consensus patch manifest to evaluate.",
    )
    parser.add_argument("--sample-index", type=Path, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--slides-root", type=Path, default=DEFAULT_SLIDES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR / DEFAULT_MODEL_NAME,
    )
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    parser.add_argument(
        "--download-model-if-missing",
        dest="download_model_if_missing",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-download-model",
        dest="download_model_if_missing",
        action="store_false",
    )
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--patch-level", type=int, default=1)
    parser.add_argument("--selection-metric", choices=["macro_f1"], default="macro_f1")
    parser.add_argument("--min-threshold", type=float, default=0.0)
    parser.add_argument("--max-threshold", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Optional comma-separated explicit thresholds, overriding min/max/step.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Evaluate one preselected threshold instead of sweeping/selecting.",
    )
    parser.add_argument(
        "--limit-slides",
        type=int,
        default=None,
        help="Optional smoke-test limit on number of training slides to score.",
    )
    parser.add_argument(
        "--only-score",
        action="store_true",
        help="Cache GrandQC scores but skip threshold tuning.",
    )
    parser.add_argument(
        "--only-tune",
        action="store_true",
        help="Skip score generation and tune from existing score parquets.",
    )
    parser.add_argument(
        "--overwrite-scores",
        action="store_true",
        help="Recompute existing per-slide score parquet files.",
    )
    parser.add_argument(
        "--require-complete",
        dest="require_complete",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--allow-incomplete",
        dest="require_complete",
        action="store_false",
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
        raise RuntimeError("This script requires scikit-learn.") from exc
    return balanced_accuracy_score, classification_report, confusion_matrix


def parse_thresholds(args: argparse.Namespace) -> list[float]:
    if args.fixed_threshold is not None:
        thresholds = [float(args.fixed_threshold)]
    elif args.thresholds:
        thresholds = sorted(
            {float(value.strip()) for value in args.thresholds.split(",")}
        )
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


def ensure_model(path: Path, *, url: str, download_if_missing: bool) -> Path:
    if path.exists():
        return path
    if not download_if_missing:
        raise FileNotFoundError(f"GrandQC model weights not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    print(f"Downloading GrandQC model weights to {path}")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(path)
    return path


def load_manifest(path: Path, *, split: str = "train") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    frame = pd.read_parquet(path)
    required = {*PATCH_ID_COLUMNS, "majority_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if "sample_split" in frame.columns:
        frame = frame.loc[frame["sample_split"] == split].copy()
    frame = frame.loc[frame["majority_label"].isin(CLASS_LABELS)].copy()
    frame["slide_id"] = frame["slide_id"].astype(str)
    frame["row"] = frame["row"].astype(int)
    frame["column"] = frame["column"].astype(int)
    return frame


def load_slide_index(path: Path, slides_root: Path, *, split: str = "train") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sample index not found: {path}")
    index = pd.read_csv(path)
    required = {"id", "slide", "sample_split"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"Sample index missing columns: {sorted(missing)}")
    index = index.loc[index["sample_split"] == split].copy()
    index["id"] = index["id"].astype(str)
    index["slide_path"] = index["slide"].map(lambda name: str(slides_root / str(name)))
    return index


def selected_slide_rows(
    manifest: pd.DataFrame,
    slide_index: pd.DataFrame,
    *,
    limit_slides: int | None,
) -> pd.DataFrame:
    slide_ids = sorted(manifest["slide_id"].unique())
    if limit_slides is not None:
        slide_ids = slide_ids[:limit_slides]
    selected = slide_index.loc[slide_index["id"].isin(slide_ids)].copy()
    missing = sorted(set(slide_ids) - set(selected["id"]))
    if missing:
        raise ValueError(f"Slides missing from sample index: {missing[:10]}")
    selected = selected.sort_values("id").reset_index(drop=True)
    return selected


def score_path(output_dir: Path, slide_id: str) -> Path:
    return output_dir / "scores" / f"{slide_id}.grandqc_scores.parquet"


def score_array_to_frame(slide_id: str, scores: np.ndarray) -> pd.DataFrame:
    rows, columns = np.indices(scores.shape)
    return pd.DataFrame(
        {
            "slide_id": slide_id,
            "row": rows.reshape(-1).astype(int),
            "column": columns.reshape(-1).astype(int),
            SCORE_COLUMN: scores.reshape(-1).astype(float),
        }
    )


def run_grandqc_on_slide(
    *,
    slide_path: Path,
    model_path: Path,
    patch_size: int,
    patch_level: int,
) -> np.ndarray:
    from PIL import Image
    from openslide import open_slide
    from tiffslide import TiffSlide

    from histokit.segmentation.grandqc import GrandQCDetector
    from histokit.utils.geometry import Size

    Image.MAX_IMAGE_PIXELS = None

    class HybridOmeTiffSlide:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.tsr = TiffSlide(str(path))
            self.osr = open_slide(str(path))

        def close(self) -> None:
            self.osr.close()
            self.tsr.close()

        @property
        def mpp(self) -> float | None:
            props = self.tsr.properties
            for prefix in ("tiffslide", "openslide"):
                try:
                    x = float(props[f"{prefix}.mpp-x"])
                    y = float(props[f"{prefix}.mpp-y"])
                    return (x + y) / 2.0
                except (KeyError, TypeError, ValueError):
                    continue
            return None

        @property
        def dimensions(self) -> list[Size]:
            return [Size(*dim) for dim in self.tsr.level_dimensions]

        def level_downsamples(self) -> list[float]:
            return list(self.tsr.level_downsamples)

        def size_in_patches(self, patch_size: int, patch_level: int) -> Size:
            dims = self.dimensions[patch_level]
            width_patches = (dims.width + patch_size - 1) // patch_size
            height_patches = (dims.height + patch_size - 1) // patch_size
            return Size(width_patches, height_patches)

        def get_thumbnail_for_size(
            self, width: int, height: int, white_background: bool = True
        ) -> np.ndarray:
            image = self.osr.get_thumbnail((width, height)).convert("RGB")
            image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
            array = np.asarray(image)
            if white_background:
                black_pixels = np.all(array == 0, axis=2)
                array = array.copy()
                array[black_pixels] = [255, 255, 255]
            return array

    detector = GrandQCDetector(
        patch_size=patch_size,
        patch_level=patch_level,
        model_path=model_path,
    )
    slide = HybridOmeTiffSlide(slide_path)
    try:
        return detector(slide)
    finally:
        slide.close()


def cache_grandqc_scores(
    *,
    slide_rows: pd.DataFrame,
    output_dir: Path,
    model_path: Path,
    patch_size: int,
    patch_level: int,
    overwrite: bool,
    scorer: Callable[..., np.ndarray] = run_grandqc_on_slide,
) -> list[Path]:
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for row in slide_rows.itertuples(index=False):
        slide_id = str(getattr(row, "id"))
        slide_path = Path(getattr(row, "slide_path"))
        if not slide_path.exists():
            raise FileNotFoundError(f"Slide not found for {slide_id}: {slide_path}")
        target = score_path(output_dir, slide_id)
        if target.exists() and not overwrite:
            print(f"Skipping existing scores for {slide_id}: {target}")
            written.append(target)
            continue
        print(f"Scoring {slide_id}: {slide_path}")
        scores = scorer(
            slide_path=slide_path,
            model_path=model_path,
            patch_size=patch_size,
            patch_level=patch_level,
        )
        frame = score_array_to_frame(slide_id, scores)
        frame.to_parquet(target, index=False)
        written.append(target)
        print(f"Wrote {len(frame)} GrandQC scores to {target}")
    return written


def load_score_frames(output_dir: Path, slide_ids: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for slide_id in sorted(set(slide_ids)):
        path = score_path(output_dir, slide_id)
        if not path.exists():
            missing.append(slide_id)
            continue
        frame = pd.read_parquet(path)
        required = {*PATCH_ID_COLUMNS, SCORE_COLUMN}
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")
        frame = frame[list(PATCH_ID_COLUMNS) + [SCORE_COLUMN]].copy()
        frame["slide_id"] = frame["slide_id"].astype(str)
        frame["row"] = frame["row"].astype(int)
        frame["column"] = frame["column"].astype(int)
        frames.append(frame)
    if missing:
        raise FileNotFoundError(f"Missing score files for slides: {missing[:10]}")
    if not frames:
        raise RuntimeError(f"No GrandQC score files found under {output_dir / 'scores'}")
    return pd.concat(frames, ignore_index=True)


def join_scores(
    manifest: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    require_complete: bool,
) -> pd.DataFrame:
    joined = manifest.merge(scores, on=PATCH_ID_COLUMNS, how="left", validate="one_to_one")
    missing = joined[SCORE_COLUMN].isna()
    if missing.any() and require_complete:
        examples = joined.loc[missing, PATCH_ID_COLUMNS].head(10).to_dict("records")
        raise ValueError(
            f"Missing GrandQC score for {int(missing.sum())} training patch(es). "
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
    scores = frame[SCORE_COLUMN].astype(float).to_numpy()
    rows: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    matrices: dict[float, np.ndarray] = {}
    for threshold in thresholds:
        y_pred = np.where(scores > threshold, "foreground", "background")
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


def select_best(metrics: pd.DataFrame) -> Selection:
    ordered = metrics.sort_values(
        ["macro_f1", "foreground_recall", "threshold"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    best = ordered.iloc[0].to_dict()
    return Selection(
        threshold=float(best["threshold"]),
        metric_value=float(best["macro_f1"]),
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
    manifest: Path,
    sample_index: Path,
    slides_root: Path,
    model_path: Path,
    thresholds: list[float],
    patch_size: int,
    patch_level: int,
    classification_report: Any,
    split: str,
    fixed_threshold: float | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joined = joined.copy()
    joined["grandqc_threshold"] = selection.threshold
    joined["grandqc_prediction"] = np.where(
        joined[SCORE_COLUMN].astype(float).to_numpy() > selection.threshold,
        "foreground",
        "background",
    )
    metrics.to_csv(output_dir / "threshold_metrics.csv", index=False)
    (output_dir / "classification_reports.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    joined.to_parquet(output_dir / f"joined_{split}_patch_scores.parquet", index=False)
    pd.DataFrame(
        matrices[selection.threshold], index=CLASS_LABELS, columns=CLASS_LABELS
    ).to_csv(output_dir / "best_confusion_matrix.csv")
    report_text = classification_report(
        joined["majority_label"],
        joined["grandqc_prediction"],
        labels=CLASS_LABELS,
        target_names=CLASS_LABELS,
        zero_division=0,
    )
    (output_dir / "best_classification_report.txt").write_text(report_text)
    selection_rule = (
        f"fixed threshold {fixed_threshold}"
        if fixed_threshold is not None
        else "maximize macro_f1; tie-break by higher foreground recall, then lower threshold"
    )
    summary = {
        "selection_metric": "macro_f1",
        "selection_rule": selection_rule,
        "split": split,
        "fixed_threshold": fixed_threshold,
        "best": asdict(selection),
        "manifest": str(manifest),
        "sample_index": str(sample_index),
        "slides_root": str(slides_root),
        "model_path": str(model_path),
        "patch_count": int(len(joined)),
        "slide_count": int(joined["slide_id"].nunique()),
        "threshold_count": int(len(thresholds)),
        "min_threshold": float(min(thresholds)),
        "max_threshold": float(max(thresholds)),
        "patch_size": patch_size,
        "patch_level": patch_level,
    }
    (output_dir / "best_threshold.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.only_score and args.only_tune:
        raise ValueError("--only-score and --only-tune cannot be used together")
    os.environ.setdefault("HISTOKIT_DEVICE", "cuda")

    (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    ) = require_metrics()
    thresholds = parse_thresholds(args)
    manifest = load_manifest(args.manifest, split=args.split)
    slide_index = load_slide_index(args.sample_index, args.slides_root, split=args.split)
    slide_rows = selected_slide_rows(
        manifest, slide_index, limit_slides=args.limit_slides
    )
    if args.limit_slides is not None:
        manifest = manifest.loc[manifest["slide_id"].isin(slide_rows["id"])].copy()

    model_path = ensure_model(
        args.model_path,
        url=args.model_url,
        download_if_missing=args.download_model_if_missing,
    )
    if not args.only_tune:
        cache_grandqc_scores(
            slide_rows=slide_rows,
            output_dir=args.output_dir,
            model_path=model_path,
            patch_size=args.patch_size,
            patch_level=args.patch_level,
            overwrite=args.overwrite_scores,
        )
    if args.only_score:
        print(f"Cached GrandQC scores for {len(slide_rows)} slide(s)")
        return

    score_frame = load_score_frames(args.output_dir, manifest["slide_id"].tolist())
    joined = join_scores(
        manifest,
        score_frame,
        require_complete=args.require_complete,
    )
    metrics, reports, matrices = evaluate_thresholds(
        joined,
        thresholds,
        balanced_accuracy_score=balanced_accuracy_score,
        classification_report=classification_report,
        confusion_matrix=confusion_matrix,
    )
    selection = select_best(metrics)
    write_outputs(
        output_dir=args.output_dir,
        joined=joined,
        metrics=metrics,
        reports=reports,
        matrices=matrices,
        selection=selection,
        manifest=args.manifest,
        sample_index=args.sample_index,
        slides_root=args.slides_root,
        model_path=model_path,
        thresholds=thresholds,
        patch_size=args.patch_size,
        patch_level=args.patch_level,
        classification_report=classification_report,
        split=args.split,
        fixed_threshold=args.fixed_threshold,
    )
    print(f"Evaluated {len(joined)} patches across {joined['slide_id'].nunique()} slides")
    print(f"Best threshold: {selection.threshold:.6g}")
    print(f"macro_f1: {selection.macro_f1:.6f}")
    print(f"foreground_recall: {selection.foreground_recall:.6f}")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
