#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from icaird_cervical_benchmark.metrics import bootstrap_metric_intervals_from_predictions


EMBEDDING_ORDER = ["dinov2", "hibou", "manual_features", "uni2", "uni"]
CLASSIFIER_ORDER = ["logistic-regression", "mlp", "xgboost"]
IMAGE_ORDER = [
    "resnet18",
    "densenet121",
    "efficientnet_b4",
    "convnext_tiny",
    "swin_tiny_patch4_window7_224",
]

EMBEDDING_LABELS = {
    "dinov2": "DINOv2",
    "hibou": "Hibou",
    "manual_features": "Manual features",
    "uni2": "UNI2",
    "uni": "UNI",
}
CLASSIFIER_LABELS = {
    "logistic-regression": "Logistic regression",
    "mlp": "MLP",
    "xgboost": "XGBoost",
}
IMAGE_LABELS = {
    "resnet18": "ResNet-18",
    "densenet121": "DenseNet-121",
    "efficientnet_b4": "EfficientNet-B4",
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_tiny_patch4_window7_224": "Swin-Tiny",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build thesis-ready iCAIRD Cervical results tables with bootstrap confidence "
            "intervals from saved predictions.csv files."
        )
    )
    parser.add_argument("--probe-run-dir", type=Path, required=True)
    parser.add_argument("--image-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260702)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_probe_rows(args) + collect_image_rows(args)
    frame = pd.DataFrame(rows)
    frame = sort_results(frame)
    csv_path = args.output_dir / "icaird_cervical_thesis_results_bootstrap.csv"
    markdown_path = args.output_dir / "icaird_cervical_thesis_results_bootstrap.md"
    latex_path = args.output_dir / "icaird_cervical_thesis_results_bootstrap.tex"
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(to_markdown(frame), encoding="utf-8")
    latex_path.write_text(to_latex(frame, args.confidence_level), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {latex_path}")


def collect_probe_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for predictions_csv in sorted((args.probe_run_dir / "probes").glob("*/*/seed_*/predictions.csv")):
        classifier = predictions_csv.parent.parent.name
        embedding = predictions_csv.parent.parent.parent.name
        intervals, class_names = bootstrap_metric_intervals_from_predictions(
            predictions_csv,
            n_resamples=args.resamples,
            confidence_level=args.confidence_level,
            seed=args.seed,
        )
        rows.append(
            {
                "section": "Embedding-based models",
                "embedding_or_input": embedding,
                "classifier": classifier,
                "display_embedding_or_input": EMBEDDING_LABELS.get(embedding, title_label(embedding)),
                "display_classifier": CLASSIFIER_LABELS.get(classifier, title_label(classifier)),
                "sort_embedding": order_index(EMBEDDING_ORDER, embedding),
                "sort_classifier": order_index(CLASSIFIER_ORDER, classifier),
                "predictions_csv": str(predictions_csv),
                "class_names": ";".join(class_names),
                **intervals,
            }
        )
    return rows


def collect_image_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for predictions_csv in sorted((args.image_run_dir / "images").glob("*/seed_*/predictions.csv")):
        model = predictions_csv.parent.parent.name
        intervals, class_names = bootstrap_metric_intervals_from_predictions(
            predictions_csv,
            n_resamples=args.resamples,
            confidence_level=args.confidence_level,
            seed=args.seed,
        )
        rows.append(
            {
                "section": "Patch image classifiers",
                "embedding_or_input": "patch_images",
                "classifier": model,
                "display_embedding_or_input": "Patch images",
                "display_classifier": IMAGE_LABELS.get(model, title_label(model)),
                "sort_embedding": 0,
                "sort_classifier": order_index(IMAGE_ORDER, model),
                "predictions_csv": str(predictions_csv),
                "class_names": ";".join(class_names),
                **intervals,
            }
        )
    return rows


def sort_results(frame: pd.DataFrame) -> pd.DataFrame:
    section_order = {"Embedding-based models": 0, "Patch image classifiers": 1}
    frame = frame.copy()
    frame["sort_section"] = frame["section"].map(section_order).fillna(99)
    return frame.sort_values(["sort_section", "sort_embedding", "sort_classifier"]).drop(
        columns=["sort_section"]
    )


def to_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# iCAIRD Cervical Thesis Results",
        "",
        "Intervals are non-parametric 95% bootstrap confidence intervals from the held-out test-set predictions. They quantify finite-test-set uncertainty conditional on the trained model and do not include retraining variability.",
        "",
    ]
    for section, section_frame in frame.groupby("section", sort=False):
        lines.extend([f"## {section}", ""])
        display = section_frame[
            ["display_embedding_or_input", "display_classifier", "accuracy", "auc"]
        ].copy()
        display["accuracy"] = section_frame.apply(format_accuracy_ci, axis=1)
        display["auc"] = section_frame.apply(format_auc_ci, axis=1)
        display.columns = ["Embedding / Input", "Classifier", "Accuracy, % (95% CI)", "ROC-AUC (95% CI)"]
        lines.append(display.to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)


def to_latex(frame: pd.DataFrame, confidence_level: float) -> str:
    confidence_pct = int(round(confidence_level * 100))
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Patch-level classification performance on the iCAIRD Cervical 100K held-out test set. Confidence intervals are non-parametric bootstrap confidence intervals from saved test-set predictions and quantify finite-test-set uncertainty conditional on each trained model.}",
        "\\label{tab:icaird_cervical_results}",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        f"Embedding / Input & Classifier & Accuracy, \\% ({confidence_pct}\\% CI) & ROC-AUC ({confidence_pct}\\% CI) \\\\",
        "\\midrule",
        "\\multicolumn{4}{l}{\\textbf{Embedding-based models}} \\\\",
        "\\midrule",
    ]
    probe_frame = frame[frame["section"] == "Embedding-based models"]
    lines.extend(latex_multirow_block(probe_frame))
    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{4}{l}{\\textbf{Patch image classifiers}} \\\\",
            "\\midrule",
        ]
    )
    image_frame = frame[frame["section"] == "Patch image classifiers"]
    lines.extend(latex_multirow_block(image_frame))
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def latex_multirow_block(frame: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    best_accuracy = frame["accuracy"].max()
    best_auc = frame["auc"].max()
    groups = list(frame.groupby("display_embedding_or_input", sort=False))
    for group_index, (label, group) in enumerate(groups):
        group = group.reset_index(drop=True)
        row_count = len(group)
        for index, row in group.iterrows():
            embedding = f"\\multirow{{{row_count}}}{{*}}{{{latex_escape(label)}}}" if index == 0 else ""
            accuracy = format_accuracy_ci(row)
            auc = format_auc_ci(row)
            if float(row["accuracy"]) == float(best_accuracy):
                accuracy = f"\\textbf{{{accuracy}}}"
            if float(row["auc"]) == float(best_auc):
                auc = f"\\textbf{{{auc}}}"
            lines.append(
                f"{embedding} & {latex_escape(str(row['display_classifier']))} & {accuracy} & {auc} \\\\"
            )
        if group_index != len(groups) - 1:
            lines.append("\\addlinespace")
    return lines


def format_accuracy_ci(row: pd.Series) -> str:
    return (
        f"{100.0 * float(row['accuracy']):.2f} "
        f"({100.0 * float(row['accuracy_ci_low']):.2f}--"
        f"{100.0 * float(row['accuracy_ci_high']):.2f})"
    )


def format_auc_ci(row: pd.Series) -> str:
    return (
        f"{float(row['auc']):.3f} "
        f"({float(row['auc_ci_low']):.3f}--"
        f"{float(row['auc_ci_high']):.3f})"
    )


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
