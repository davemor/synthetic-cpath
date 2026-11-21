from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "tune_canny_tissue_threshold.py"
SPEC = importlib.util.spec_from_file_location("tune_canny_tissue_threshold", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
tune = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tune
SPEC.loader.exec_module(tune)


def _write_patchset(root: Path, slide_id: str, frame: pd.DataFrame) -> None:
    patchset_dir = root / f"{slide_id}.patchset"
    patchset_dir.mkdir(parents=True)
    frame.to_parquet(patchset_dir / "frame.parquet", index=False)


def test_join_scores_preserves_requested_test_patches(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        {
            "slide_id": ["s1", "s1"],
            "row": [0, 1],
            "column": [1, 2],
            "majority_label": ["background", "foreground"],
        }
    )
    patchsets_root = tmp_path / "patchsets"
    _write_patchset(
        patchsets_root,
        "s1",
        pd.DataFrame(
            {
                "row": [0, 1, 2],
                "column": [1, 2, 3],
                "tissue_score": [0.01, 0.02, 0.99],
            }
        ),
    )

    scores = tune.load_patchset_scores(patchsets_root, ["s1"])
    joined = tune.join_scores(manifest, scores, require_complete=True)

    assert len(joined) == 2
    assert joined["tissue_score"].tolist() == [0.01, 0.02]


def test_join_scores_fails_on_missing_tissue_score(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        {
            "slide_id": ["s1"],
            "row": [4],
            "column": [5],
            "majority_label": ["foreground"],
        }
    )
    patchsets_root = tmp_path / "patchsets"
    _write_patchset(
        patchsets_root,
        "s1",
        pd.DataFrame({"row": [0], "column": [0], "tissue_score": [0.01]}),
    )

    scores = tune.load_patchset_scores(patchsets_root, ["s1"])
    with pytest.raises(ValueError, match="Missing cached tissue_score"):
        tune.join_scores(manifest, scores, require_complete=True)


def test_threshold_predictions_are_strictly_greater_than() -> None:
    pytest.importorskip("sklearn")
    balanced_accuracy_score, classification_report, confusion_matrix = tune.require_metrics()
    frame = pd.DataFrame(
        {
            "majority_label": ["background", "foreground"],
            "tissue_score": [0.02, 0.021],
        }
    )

    metrics, reports, matrices = tune.evaluate_thresholds(
        frame,
        [0.02],
        balanced_accuracy_score=balanced_accuracy_score,
        classification_report=classification_report,
        confusion_matrix=confusion_matrix,
    )

    assert metrics.loc[0, "true_background_pred_background"] == 1
    assert metrics.loc[0, "true_foreground_pred_foreground"] == 1
    assert "0.02" in reports
    assert matrices[0.02].tolist() == [[1, 0], [0, 1]]


def test_select_best_tie_breaks_by_foreground_recall_then_lower_threshold() -> None:
    metrics = pd.DataFrame(
        {
            "threshold": [0.03, 0.02, 0.01],
            "macro_f1": [0.9, 0.9, 0.8],
            "foreground_recall": [0.7, 0.8, 1.0],
        }
    )

    selected = tune.select_best(metrics, "macro_f1")

    assert selected.threshold == pytest.approx(0.02)
    assert selected.metric_value == pytest.approx(0.9)
