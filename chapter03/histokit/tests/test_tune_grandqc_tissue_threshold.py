from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "tune_grandqc_tissue_threshold.py"
SPEC = importlib.util.spec_from_file_location("tune_grandqc_tissue_threshold", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
tune = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tune
SPEC.loader.exec_module(tune)


def test_score_array_to_frame_uses_patch_grid_coordinates() -> None:
    frame = tune.score_array_to_frame("s1", np.array([[0.1, 0.2], [0.3, 0.4]]))

    assert frame.to_dict("records") == [
        {"slide_id": "s1", "row": 0, "column": 0, "grandqc_tissue_score": 0.1},
        {"slide_id": "s1", "row": 0, "column": 1, "grandqc_tissue_score": 0.2},
        {"slide_id": "s1", "row": 1, "column": 0, "grandqc_tissue_score": 0.3},
        {"slide_id": "s1", "row": 1, "column": 1, "grandqc_tissue_score": 0.4},
    ]


def test_cache_grandqc_scores_with_fake_scorer(tmp_path: Path) -> None:
    slide = tmp_path / "s1.ome.tiff"
    slide.write_bytes(b"fake")
    slide_rows = pd.DataFrame({"id": ["s1"], "slide_path": [str(slide)]})

    def fake_scorer(**_: object) -> np.ndarray:
        return np.array([[0.25, 0.75]])

    paths = tune.cache_grandqc_scores(
        slide_rows=slide_rows,
        output_dir=tmp_path / "out",
        model_path=tmp_path / "model.pth",
        patch_size=224,
        patch_level=1,
        overwrite=False,
        scorer=fake_scorer,
    )

    assert len(paths) == 1
    scores = pd.read_parquet(paths[0])
    assert scores["grandqc_tissue_score"].tolist() == [0.25, 0.75]


def test_join_scores_fails_when_required_score_missing() -> None:
    manifest = pd.DataFrame(
        {
            "slide_id": ["s1"],
            "row": [1],
            "column": [1],
            "majority_label": ["foreground"],
        }
    )
    scores = pd.DataFrame(
        {
            "slide_id": ["s1"],
            "row": [0],
            "column": [0],
            "grandqc_tissue_score": [0.9],
        }
    )

    with pytest.raises(ValueError, match="Missing GrandQC score"):
        tune.join_scores(manifest, scores, require_complete=True)


def test_threshold_predictions_are_strictly_greater_than() -> None:
    pytest.importorskip("sklearn")
    balanced_accuracy_score, classification_report, confusion_matrix = tune.require_metrics()
    frame = pd.DataFrame(
        {
            "majority_label": ["background", "foreground"],
            "grandqc_tissue_score": [0.5, 0.501],
        }
    )

    metrics, _, matrices = tune.evaluate_thresholds(
        frame,
        [0.5],
        balanced_accuracy_score=balanced_accuracy_score,
        classification_report=classification_report,
        confusion_matrix=confusion_matrix,
    )

    assert metrics.loc[0, "true_background_pred_background"] == 1
    assert metrics.loc[0, "true_foreground_pred_foreground"] == 1
    assert matrices[0.5].tolist() == [[1, 0], [0, 1]]


def test_select_best_tie_breaks_by_foreground_recall_then_lower_threshold() -> None:
    metrics = pd.DataFrame(
        {
            "threshold": [0.6, 0.4, 0.2],
            "macro_f1": [0.9, 0.9, 0.8],
            "foreground_recall": [0.7, 0.8, 1.0],
        }
    )

    selected = tune.select_best(metrics)

    assert selected.threshold == pytest.approx(0.4)
    assert selected.metric_value == pytest.approx(0.9)
