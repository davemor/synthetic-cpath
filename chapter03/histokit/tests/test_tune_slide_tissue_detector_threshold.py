from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "tune_slide_tissue_detector_threshold.py"
)
SPEC = importlib.util.spec_from_file_location("tune_slide_tissue_detector_threshold", SCRIPT_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_score_array_to_frame_converts_binary_mask_to_float_scores() -> None:
    scores = np.array([[True, False], [False, True]])

    frame = module.score_array_to_frame("s1", "clam_segmentation_otsu", scores)

    assert frame.columns.tolist() == [
        "slide_id",
        "row",
        "column",
        "clam_segmentation_otsu_score",
    ]
    assert frame["row"].tolist() == [0, 0, 1, 1]
    assert frame["column"].tolist() == [0, 1, 0, 1]
    assert frame["clam_segmentation_otsu_score"].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_join_scores_preserves_patch_rows() -> None:
    manifest = pd.DataFrame(
        {
            "slide_id": ["s1", "s1"],
            "row": [0, 0],
            "column": [0, 1],
            "majority_label": ["foreground", "background"],
        }
    )
    scores = pd.DataFrame(
        {
            "slide_id": ["s1", "s1"],
            "row": [0, 0],
            "column": [0, 1],
            "otsu_hs_segmentation_score": [1.0, 0.0],
        }
    )

    joined = module.join_scores(
        manifest,
        scores,
        detector="otsu_hs_segmentation",
        require_complete=True,
    )

    assert len(joined) == 2
    assert joined["majority_label"].tolist() == ["foreground", "background"]
    assert joined["otsu_hs_segmentation_score"].tolist() == [1.0, 0.0]


def test_join_scores_fails_on_missing_score() -> None:
    manifest = pd.DataFrame(
        {
            "slide_id": ["s1"],
            "row": [0],
            "column": [1],
            "majority_label": ["background"],
        }
    )
    scores = pd.DataFrame(
        {
            "slide_id": ["s1"],
            "row": [0],
            "column": [0],
            "clam_segmentation_otsu_score": [1.0],
        }
    )

    with pytest.raises(ValueError, match="Missing clam_segmentation_otsu score"):
        module.join_scores(
            manifest,
            scores,
            detector="clam_segmentation_otsu",
            require_complete=True,
        )


def test_threshold_selection_uses_strict_greater_than_and_tie_break() -> None:
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )

    frame = pd.DataFrame(
        {
            "majority_label": ["background", "foreground"],
            "clam_segmentation_otsu_score": [0.0, 1.0],
        }
    )

    metrics, _, _ = module.evaluate_thresholds(
        frame,
        "clam_segmentation_otsu",
        [0.0, 0.5, 1.0],
        balanced_accuracy_score=balanced_accuracy_score,
        classification_report=classification_report,
        confusion_matrix=confusion_matrix,
    )
    selection = module.select_best(metrics)

    assert selection.threshold == 0.0
    assert metrics.loc[
        metrics["threshold"] == 1.0,
        "true_foreground_pred_background",
    ].item() == 1


def test_cache_detector_scores_uses_mocked_scorer(tmp_path: Path) -> None:
    slide_path = tmp_path / "s1.ome.tiff"
    slide_path.write_bytes(b"fake")
    slide_rows = pd.DataFrame({"id": ["s1"], "slide_path": [str(slide_path)]})

    def scorer(**_: object) -> np.ndarray:
        return np.array([[1.0, 0.0]])

    written = module.cache_detector_scores(
        detector="otsu_hs_segmentation",
        slide_rows=slide_rows,
        output_dir=tmp_path,
        patch_size=224,
        patch_level=1,
        features_level=1,
        overwrite=False,
        scorer=scorer,
    )

    assert len(written) == 1
    frame = pd.read_parquet(written[0])
    assert frame["otsu_hs_segmentation_score"].tolist() == [1.0, 0.0]


def test_write_outputs_writes_expected_files(tmp_path: Path) -> None:
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )

    joined = pd.DataFrame(
        {
            "slide_id": ["s1", "s1"],
            "row": [0, 0],
            "column": [0, 1],
            "majority_label": ["background", "foreground"],
            "otsu_hs_segmentation_score": [0.0, 1.0],
        }
    )
    metrics, reports, matrices = module.evaluate_thresholds(
        joined,
        "otsu_hs_segmentation",
        [0.0],
        balanced_accuracy_score=balanced_accuracy_score,
        classification_report=classification_report,
        confusion_matrix=confusion_matrix,
    )
    selection = module.select_best(metrics)

    module.write_outputs(
        output_dir=tmp_path,
        detector="otsu_hs_segmentation",
        joined=joined,
        metrics=metrics,
        reports=reports,
        matrices=matrices,
        selection=selection,
        manifest=tmp_path / "manifest.parquet",
        sample_index=tmp_path / "index.csv",
        slides_root=tmp_path / "slides",
        thresholds=[0.0],
        patch_size=224,
        patch_level=1,
        features_level=1,
        classification_report=classification_report,
        split="train",
        fixed_threshold=None,
    )

    assert (tmp_path / "threshold_metrics.csv").exists()
    assert (tmp_path / "classification_reports.json").exists()
    assert (tmp_path / "best_threshold.json").exists()
    assert (tmp_path / "best_classification_report.txt").exists()
    assert (tmp_path / "best_confusion_matrix.csv").exists()
    assert (tmp_path / "joined_train_patch_scores.parquet").exists()
