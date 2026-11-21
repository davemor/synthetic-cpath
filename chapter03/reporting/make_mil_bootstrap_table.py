#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


TASK_LABELS = {
    "malignant": "Malignancy",
    "category": "Category",
    "subcategory": "Subcategory",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MIL accuracy and ROC-AUC tables with non-parametric bootstrap "
            "confidence intervals from saved test-set predictions."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/icaird_mil_uni2h_bags_gpu_20260616_1335"),
        help="Local experiment run directory.",
    )
    parser.add_argument("--embedding", default="uni2_h", help="Embedding subdirectory.")
    parser.add_argument("--seed", type=int, default=0, help="Model seed directory.")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Bootstrap replicates.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level.")
    parser.add_argument("--random-seed", type=int, default=0, help="Bootstrap RNG seed.")
    return parser.parse_args()


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    accuracy = accuracy_score(y_true, y_pred)
    if y_proba.shape[1] == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        row_sums = y_proba.sum(axis=1, keepdims=True)
        y_proba = np.divide(y_proba, row_sums, out=np.zeros_like(y_proba), where=row_sums != 0)
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    return float(accuracy), float(auc)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], tuple[float, float], int]:
    accuracies = []
    aucs = []
    labels = np.unique(y_true)
    n_samples = len(y_true)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        sampled_true = y_true[indices]
        if len(np.unique(sampled_true)) != len(labels):
            continue
        try:
            accuracy, auc = metric_values(sampled_true, y_pred[indices], y_proba[indices])
        except ValueError:
            continue
        accuracies.append(accuracy)
        aucs.append(auc)
    alpha = 1.0 - confidence
    percentiles = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    accuracy_ci = tuple(np.percentile(accuracies, percentiles))
    auc_ci = tuple(np.percentile(aucs, percentiles))
    return accuracy_ci, auc_ci, len(accuracies)


def format_estimate(value: float, ci: tuple[float, float]) -> str:
    return f"{value:.3f} ({ci[0]:.3f}-{ci[1]:.3f})"


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def latex_table(frame: pd.DataFrame, confidence: float) -> str:
    ci_pct = int(round(confidence * 100))
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{MIL test-set performance with {ci_pct}\% non-parametric bootstrap confidence intervals.}}",
        r"\label{tab:mil_bootstrap_performance}",
        r"\begin{tabular}{lll}",
        r"\hline",
        rf"\textbf{{Task}} & \textbf{{Accuracy ({ci_pct}\% CI)}} & \textbf{{ROC-AUC ({ci_pct}\% CI)}} \\",
        r"\hline",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.task)} & {latex_escape(row.accuracy_ci)} & "
            f"{latex_escape(row.roc_auc_ci)} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)
    rows = []
    for task, task_label in TASK_LABELS.items():
        prediction_path = (
            args.run_dir
            / "mil"
            / args.embedding
            / task
            / f"seed_{args.seed}"
            / "test"
            / "predictions.csv"
        )
        predictions = pd.read_csv(prediction_path)
        proba_columns = [column for column in predictions.columns if column.startswith("proba_")]
        y_true = predictions["true_label"].to_numpy()
        y_pred = predictions["predicted_label"].to_numpy()
        y_proba = predictions[proba_columns].to_numpy(dtype=np.float64)
        accuracy, auc = metric_values(y_true, y_pred, y_proba)
        accuracy_ci, auc_ci, usable_bootstraps = bootstrap_ci(
            y_true,
            y_pred,
            y_proba,
            args.n_bootstrap,
            args.confidence,
            rng,
        )
        rows.append(
            {
                "task": task_label,
                "accuracy": accuracy,
                "accuracy_ci_low": accuracy_ci[0],
                "accuracy_ci_high": accuracy_ci[1],
                "roc_auc": auc,
                "roc_auc_ci_low": auc_ci[0],
                "roc_auc_ci_high": auc_ci[1],
                "accuracy_ci": format_estimate(accuracy, accuracy_ci),
                "roc_auc_ci": format_estimate(auc, auc_ci),
                "n_test": len(predictions),
                "n_bootstrap": args.n_bootstrap,
                "usable_bootstraps": usable_bootstraps,
            }
        )
    output_frame = pd.DataFrame(rows)
    output_csv = args.run_dir / "mil_test_bootstrap_performance.csv"
    output_tex = args.run_dir / "mil_test_bootstrap_performance.tex"
    output_frame.to_csv(output_csv, index=False)
    output_tex.write_text(latex_table(output_frame, args.confidence))
    print(output_frame[["task", "accuracy_ci", "roc_auc_ci", "n_test", "usable_bootstraps"]].to_string(index=False))
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_tex}")


if __name__ == "__main__":
    main()
