#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from icaird_cervical_benchmark.metrics import bootstrap_metric_intervals_from_predictions


TASK_ORDER = ["malignant", "category", "subcategory"]
TASK_LABELS = {
    "malignant": "Malignancy",
    "category": "Diagnostic category",
    "subcategory": "Diagnostic subcategory",
}
EMBEDDING_LABELS = {
    "uni2_h": "UNI2-h",
}
MODEL_LABEL = "Transformer MIL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a thesis-ready iCAIRD Cervical MIL results table with bootstrap "
            "confidence intervals from saved test predictions."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Existing bootstrap result CSV to reformat without recomputing intervals.",
    )
    parser.add_argument("--mil-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260702)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.input_csv is not None:
        frame = pd.read_csv(args.input_csv)
    else:
        if args.mil_run_dir is None:
            raise ValueError("--mil-run-dir is required unless --input-csv is given.")
        frame = pd.DataFrame(collect_rows(args))
    frame = sort_results(frame)
    csv_path = args.output_dir / "icaird_cervical_mil_results_bootstrap.csv"
    markdown_path = args.output_dir / "icaird_cervical_mil_results_bootstrap.md"
    latex_path = args.output_dir / "icaird_cervical_mil_results_bootstrap.tex"
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(to_markdown(frame, args.confidence_level), encoding="utf-8")
    latex_path.write_text(to_latex(frame, args.confidence_level), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {latex_path}")


def collect_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for predictions_csv in sorted(args.mil_run_dir.glob("mil/*/*/seed_*/test/predictions.csv")):
        task = predictions_csv.parent.parent.parent.name
        embedding = predictions_csv.parent.parent.parent.parent.name
        intervals, class_names = bootstrap_metric_intervals_from_predictions(
            predictions_csv,
            n_resamples=args.resamples,
            confidence_level=args.confidence_level,
            seed=args.seed,
        )
        rows.append(
            {
                "task": task,
                "display_task": TASK_LABELS.get(task, title_label(task)),
                "embedding": embedding,
                "display_embedding": EMBEDDING_LABELS.get(embedding, title_label(embedding)),
                "model": "transformer-mil",
                "display_model": MODEL_LABEL,
                "class_names": ";".join(class_names),
                "predictions_csv": str(predictions_csv),
                "sort_task": order_index(TASK_ORDER, task),
                **intervals,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No MIL test predictions found under {args.mil_run_dir}.")
    return rows


def sort_results(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["sort_task"]).reset_index(drop=True)


def to_markdown(frame: pd.DataFrame, confidence_level: float) -> str:
    confidence_pct = int(round(confidence_level * 100))
    lines = [
        "# iCAIRD Cervical MIL Results",
        "",
        f"Intervals are non-parametric {confidence_pct}% bootstrap confidence intervals from held-out test-set predictions. They quantify finite-test-set uncertainty conditional on the trained model and do not include retraining variability.",
        "",
        f"| Task | Accuracy, % | {confidence_pct}% CI | ROC-AUC | {confidence_pct}% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in frame.iterrows():
        lines.append(
                "| "
                f"{row['display_task']} | "
                f"{format_accuracy(row)} | "
                f"{format_accuracy_interval(row)} | "
                f"{format_auc(row)} | "
            f"{format_auc_interval(row)} |"
        )
    lines.append("")
    return "\n".join(lines)


def to_latex(frame: pd.DataFrame, confidence_level: float) -> str:
    confidence_pct = int(round(confidence_level * 100))
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Slide-level multiple-instance learning performance using UNI2-h embeddings on the iCAIRD Cervical held-out test set. Confidence intervals are non-parametric bootstrap confidence intervals from saved test-set predictions and quantify finite-test-set uncertainty conditional on the trained model.}",
        "\\label{tab:icaird_cervical_mil_results}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        f"Task & Accuracy, \\% & {confidence_pct}\\% CI & ROC-AUC & {confidence_pct}\\% CI \\\\",
        "\\midrule",
    ]
    best_accuracy = frame["accuracy"].max()
    best_auc = frame["auc"].max()
    for _, row in frame.iterrows():
        accuracy = format_accuracy(row)
        accuracy_interval = format_accuracy_interval(row)
        auc = format_auc(row)
        auc_interval = format_auc_interval(row)
        if float(row["accuracy"]) == float(best_accuracy):
            accuracy = f"\\textbf{{{accuracy}}}"
            accuracy_interval = f"\\textbf{{{accuracy_interval}}}"
        if float(row["auc"]) == float(best_auc):
            auc = f"\\textbf{{{auc}}}"
            auc_interval = f"\\textbf{{{auc_interval}}}"
        lines.append(
            f"{latex_escape(str(row['display_task']))} & "
            f"{accuracy} & {accuracy_interval} & {auc} & {auc_interval} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def format_accuracy(row: pd.Series) -> str:
    return f"{100.0 * float(row['accuracy']):.2f}"


def format_auc(row: pd.Series) -> str:
    return f"{float(row['auc']):.3f}"


def format_accuracy_interval(row: pd.Series) -> str:
    upper = 100.0 * (float(row["accuracy_ci_high"]) - float(row["accuracy"]))
    lower = 100.0 * (float(row["accuracy"]) - float(row["accuracy_ci_low"]))
    return f"+{upper:.2f} / -{lower:.2f}"


def format_auc_interval(row: pd.Series) -> str:
    upper = float(row["auc_ci_high"]) - float(row["auc"])
    lower = float(row["auc"]) - float(row["auc_ci_low"])
    return f"+{upper:.3f} / -{lower:.3f}"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def title_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def order_index(order: list[str], value: str) -> int:
    try:
        return order.index(value)
    except ValueError:
        return len(order)


if __name__ == "__main__":
    main()
