import numpy as np
import pandas as pd

from icaird_cervical_benchmark.metrics import compute_metrics, write_summary
from icaird_cervical_benchmark.metrics import (
    bootstrap_metric_intervals,
    prediction_arrays_from_frame,
)


def test_binary_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0


def test_multiclass_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0


def test_multiclass_metrics_renormalizes_probabilities():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    y_proba = np.array(
        [
            [0.8001, 0.1001, 0.1001],
            [0.1001, 0.8001, 0.1001],
            [0.1001, 0.1001, 0.8001],
            [0.7001, 0.2001, 0.1001],
            [0.2001, 0.7001, 0.1001],
            [0.1001, 0.2001, 0.7001],
        ]
    )

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0


def test_write_summary_reads_nested_metric_files(tmp_path):
    run_dir = tmp_path / "probes" / "manual" / "logistic" / "seed_0"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "task": "probe",
                "model_id": "manual:logistic",
                "embedding": "manual",
                "classifier": "logistic",
                "accuracy": 0.5,
                "auc": 0.75,
            }
        ]
    ).to_csv(run_dir / "metrics.csv", index=False)

    write_summary(tmp_path)

    assert (tmp_path / "all_metrics.csv").exists()
    assert (tmp_path / "summary.csv").exists()


def test_prediction_arrays_from_frame_extracts_probability_columns():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "true_label": [0, 1],
            "predicted_label": [0, 1],
            "proba_normal": [0.8, 0.2],
            "proba_high_grade": [0.2, 0.8],
        }
    )

    y_true, y_pred, y_proba, class_names = prediction_arrays_from_frame(frame)

    assert y_true.tolist() == [0, 1]
    assert y_pred.tolist() == [0, 1]
    assert y_proba.shape == (2, 2)
    assert class_names == ["normal", "high_grade"]


def test_bootstrap_metric_intervals_are_reproducible():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 1])
    y_proba = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.3, 0.7],
        ]
    )

    first = bootstrap_metric_intervals(y_true, y_pred, y_proba, n_resamples=100, seed=42)
    second = bootstrap_metric_intervals(y_true, y_pred, y_proba, n_resamples=100, seed=42)

    assert first == second
    assert first["accuracy"] == compute_metrics(y_true, y_pred, y_proba)["accuracy"]
    assert first["bootstrap_valid_resamples"] > 0
    assert first["accuracy_ci_low"] <= first["accuracy"] <= first["accuracy_ci_high"]
