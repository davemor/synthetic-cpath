#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from icaird_cervical_benchmark.metrics import bootstrap_metric_intervals_from_predictions


EMBEDDING_ORDER = ["uni", "uni2", "hibou", "dinov2", "manual_features"]
CLASSIFIER_ORDER = ["logistic-regression", "mlp", "xgboost"]
IMAGE_ORDER = [
    "resnet18",
    "densenet121",
    "efficientnet_b4",
    "convnext_tiny",
    "swin_tiny_patch4_window7_224",
]

EMBEDDING_LABELS = {
    "uni": "UNI",
    "uni2": "UNI2-h",
    "hibou": "Hibou-L",
    "dinov2": "DINOv2-L",
    "manual_features": "Image features",
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
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Existing bootstrap result CSV to reformat without recomputing intervals.",
    )
    parser.add_argument("--probe-run-dir", type=Path, default=None)
    parser.add_argument("--image-run-dir", type=Path, default=None)
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
        frame = refresh_display_columns(frame)
    else:
        if args.probe_run_dir is None or args.image_run_dir is None:
            raise ValueError("--probe-run-dir and --image-run-dir are required unless --input-csv is given.")
        rows = collect_probe_rows(args) + collect_image_rows(args)
        frame = pd.DataFrame(rows)
    frame = sort_results(frame)
    csv_path = args.output_dir / "icaird_cervical_thesis_results_bootstrap.csv"
    markdown_path = args.output_dir / "icaird_cervical_thesis_results_bootstrap.md"
    latex_path = args.output_dir / "icaird_cervical_thesis_results_bootstrap.tex"
    embedding_latex_path = args.output_dir / "icaird_cervical_embedding_results_bootstrap.tex"
    patch_image_latex_path = args.output_dir / "icaird_cervical_patch_image_results_bootstrap.tex"
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(to_markdown(frame), encoding="utf-8")
    latex_path.write_text(to_latex(frame, args.confidence_level), encoding="utf-8")
    embedding_latex_path.write_text(
        to_embedding_latex(frame, args.confidence_level),
        encoding="utf-8",
    )
    patch_image_latex_path.write_text(
        to_patch_image_latex(frame, args.confidence_level),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {latex_path}")
    print(f"Wrote {embedding_latex_path}")
    print(f"Wrote {patch_image_latex_path}")


def refresh_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["display_embedding_or_input"] = [
        "Patch images"
        if section == "Patch image classifiers"
        else EMBEDDING_LABELS.get(str(embedding), title_label(str(embedding)))
        for section, embedding in zip(frame["section"], frame["embedding_or_input"], strict=False)
    ]
    frame["display_classifier"] = [
        IMAGE_LABELS.get(str(classifier), title_label(str(classifier)))
        if section == "Patch image classifiers"
        else CLASSIFIER_LABELS.get(str(classifier), title_label(str(classifier)))
        for section, classifier in zip(frame["section"], frame["classifier"], strict=False)
    ]
    frame["sort_embedding"] = [
        0
        if section == "Patch image classifiers"
        else order_index(EMBEDDING_ORDER, str(embedding))
        for section, embedding in zip(frame["section"], frame["embedding_or_input"], strict=False)
    ]
    frame["sort_classifier"] = [
        order_index(IMAGE_ORDER, str(classifier))
        if section == "Patch image classifiers"
        else order_index(CLASSIFIER_ORDER, str(classifier))
        for section, classifier in zip(frame["section"], frame["classifier"], strict=False)
    ]
    return frame


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
        first_column = "Embedding" if section == "Embedding-based models" else "Input"
        lines.extend(
            [
                f"| {first_column} | Classifier | Accuracy, % | Accuracy 95% CI | ROC-AUC | ROC-AUC 95% CI |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in section_frame.iterrows():
            lines.append(
                "| "
                f"{row['display_embedding_or_input']} | "
                f"{row['display_classifier']} | "
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
        "\\caption{Patch-level classification performance on the iCAIRD Cervical 100K held-out test set. Confidence intervals are non-parametric bootstrap confidence intervals from saved test-set predictions and quantify finite-test-set uncertainty conditional on each trained model.}",
        "\\label{tab:icaird_cervical_results}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        f"Embedding / Input & Classifier & Accuracy, \\% & {confidence_pct}\\% CI & ROC-AUC & {confidence_pct}\\% CI \\\\",
        "\\midrule",
        "\\multicolumn{6}{l}{\\textbf{Embedding-based models}} \\\\",
        "\\midrule",
    ]
    probe_frame = frame[frame["section"] == "Embedding-based models"]
    lines.extend(latex_multirow_block(probe_frame))
    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{6}{l}{\\textbf{Patch image classifiers}} \\\\",
            "\\midrule",
        ]
    )
    image_frame = frame[frame["section"] == "Patch image classifiers"]
    lines.extend(latex_multirow_block(image_frame))
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def to_embedding_latex(frame: pd.DataFrame, confidence_level: float) -> str:
    confidence_pct = int(round(confidence_level * 100))
    probe_frame = frame[frame["section"] == "Embedding-based models"]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Embedding-based patch-level classification performance on the iCAIRD Cervical 100K held-out test set. Confidence intervals are non-parametric bootstrap confidence intervals from saved test-set predictions and quantify finite-test-set uncertainty conditional on each trained model.}",
        "\\label{tab:icaird_cervical_embedding_results}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        f"Embedding & Classifier & Accuracy, \\% & {confidence_pct}\\% CI & ROC-AUC & {confidence_pct}\\% CI \\\\",
        "\\midrule",
    ]
    lines.extend(latex_multirow_block(probe_frame))
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def to_patch_image_latex(frame: pd.DataFrame, confidence_level: float) -> str:
    confidence_pct = int(round(confidence_level * 100))
    image_frame = frame[frame["section"] == "Patch image classifiers"]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Patch image classifier performance on the iCAIRD Cervical 100K held-out test set. Confidence intervals are non-parametric bootstrap confidence intervals from saved test-set predictions and quantify finite-test-set uncertainty conditional on each trained model.}",
        "\\label{tab:icaird_cervical_patch_image_results}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        f"Classifier & Accuracy, \\% & {confidence_pct}\\% CI & ROC-AUC & {confidence_pct}\\% CI \\\\",
        "\\midrule",
    ]
    lines.extend(latex_flat_classifier_block(image_frame))
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
                f"{embedding} & {latex_escape(str(row['display_classifier']))} & "
                f"{accuracy} & {accuracy_interval} & {auc} & {auc_interval} \\\\"
            )
        if group_index != len(groups) - 1:
            lines.append("\\addlinespace")
    return lines


def latex_flat_classifier_block(frame: pd.DataFrame) -> list[str]:
    lines: list[str] = []
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
            f"{latex_escape(str(row['display_classifier']))} & "
            f"{accuracy} & {accuracy_interval} & {auc} & {auc_interval} \\\\"
        )
    return lines


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
