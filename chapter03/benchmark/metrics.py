from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def prediction_arrays_from_frame(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract labels and class probabilities from a saved predictions.csv frame."""
    required = {"true_label", "predicted_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction frame is missing required columns: {missing}")
    probability_columns = [col for col in frame.columns if col.startswith("proba_")]
    if not probability_columns:
        raise ValueError("Prediction frame does not contain any proba_<class> columns.")
    class_names = [col.removeprefix("proba_") for col in probability_columns]
    y_true = frame["true_label"].to_numpy(dtype=int)
    y_pred = frame["predicted_label"].to_numpy(dtype=int)
    y_proba = frame[probability_columns].to_numpy(dtype=np.float64)
    return y_true, y_pred, y_proba, class_names


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    labels = np.unique(y_true)
    if y_proba.ndim != 2 or y_proba.shape[1] != len(labels):
        raise ValueError(
            f"Probability shape {y_proba.shape} does not match {len(labels)} classes in y_true."
        )
    if len(labels) == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        row_sums = y_proba.sum(axis=1, keepdims=True)
        y_proba = np.divide(y_proba, row_sums, out=np.zeros_like(y_proba), where=row_sums != 0)
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "auc": float(auc)}


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Bootstrap held-out predictions to estimate finite-test-set uncertainty.

    The intervals are conditional on the trained model. They do not include
    uncertainty due to retraining with a different random seed.
    """
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if len(y_true) != len(y_pred) or len(y_true) != len(y_proba):
        raise ValueError("y_true, y_pred, and y_proba must contain the same number of rows.")

    point = compute_metrics(y_true, y_pred, y_proba)
    full_labels = np.unique(y_true)
    rng = np.random.default_rng(seed)
    accuracy_values: list[float] = []
    auc_values: list[float] = []
    skipped = 0
    alpha = 1.0 - confidence_level
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    for _ in range(n_resamples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sample_true = y_true[indices]
        if len(np.unique(sample_true)) != len(full_labels):
            skipped += 1
            continue
        try:
            sample_metrics = compute_metrics(sample_true, y_pred[indices], y_proba[indices])
        except ValueError:
            skipped += 1
            continue
        accuracy_values.append(sample_metrics["accuracy"])
        auc_values.append(sample_metrics["auc"])

    if not accuracy_values:
        raise ValueError("All bootstrap resamples were invalid; check class support in the test set.")

    accuracy_array = np.asarray(accuracy_values, dtype=np.float64)
    auc_array = np.asarray(auc_values, dtype=np.float64)
    return {
        "accuracy": point["accuracy"],
        "auc": point["auc"],
        "accuracy_ci_low": float(np.percentile(accuracy_array, lower_q)),
        "accuracy_ci_high": float(np.percentile(accuracy_array, upper_q)),
        "auc_ci_low": float(np.percentile(auc_array, lower_q)),
        "auc_ci_high": float(np.percentile(auc_array, upper_q)),
        "bootstrap_resamples": n_resamples,
        "bootstrap_valid_resamples": len(accuracy_values),
        "bootstrap_skipped_resamples": skipped,
        "confidence_level": confidence_level,
    }


def bootstrap_metric_intervals_from_predictions(
    predictions_csv: Path,
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> tuple[dict[str, float | int], list[str]]:
    frame = pd.read_csv(predictions_csv)
    y_true, y_pred, y_proba, class_names = prediction_arrays_from_frame(frame)
    intervals = bootstrap_metric_intervals(
        y_true,
        y_pred,
        y_proba,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return intervals, class_names


def write_evaluation_artifacts(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    sample_ids: list[str],
    class_names: list[str],
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    (output_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    (output_dir / "classification_report.txt").write_text(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    try:
        _plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")
    except Exception as exc:
        (output_dir / "confusion_matrix_plot_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    rows = {
        "sample_id": sample_ids,
        "true_label": y_true,
        "predicted_label": y_pred,
    }
    for idx, class_name in enumerate(class_names):
        rows[f"proba_{class_name}"] = y_proba[:, idx]
    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    return metrics


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.8), max(5, len(class_names) * 0.7)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(output_dir: Path) -> None:
    metric_paths = sorted(output_dir.glob("**/metrics.csv"))
    rows = []
    for path in metric_paths:
        if path.parent == output_dir:
            continue
        df = pd.read_csv(path)
        if not df.empty:
            row = df.iloc[0].to_dict()
            row["artifact_dir"] = str(path.parent)
            rows.append(row)
    if not rows:
        return
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "all_metrics.csv", index=False)
    group_cols = [col for col in ["task", "model_id", "embedding", "classifier"] if col in metrics.columns]
    if not group_cols:
        group_cols = ["model_id"] if "model_id" in metrics.columns else ["artifact_dir"]
    summary = (
        metrics.groupby(group_cols, dropna=False)[["accuracy", "auc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    summary.to_csv(output_dir / "summary.csv", index=False)
