#!/usr/bin/env python3
"""Summarise dense logistic-regression threshold tables for thesis use."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


DATASETS = [
    ("kather100k", "Kather"),
    ("spider-colorectal", "CRC"),
    ("spider-breast", "Breast"),
    ("spider-skin", "Skin"),
    ("spider-thorax", "Thorax"),
    ("icaird100k", "iCAIRD"),
]
ENCODER_LABELS = {
    "uni": "UNI",
    "uni2": "UNI2",
    "hibou": "Hibou",
    "dinov2": "DINOv2",
}
PROJECTOR_LABELS = {
    "pca": "PCA",
    "umap": "UMAP",
    "grp": "GRP",
}
THRESHOLDS = [
    ("accuracy", "0.99", "99% Acc."),
    ("accuracy", "0.95", "95% Acc."),
    ("roc_auc", "0.99", "99% ROC-AUC"),
    ("roc_auc", "0.95", "95% ROC-AUC"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="logistic_regression_threshold_values.csv from the interpretation output.",
    )
    parser.add_argument(
        "--curve-comparison-csv",
        type=Path,
        default=None,
        help=(
            "curve_vs_generalised_lr_by_tuple.csv. Defaults to the file with that "
            "name in --output-dir when present."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def fmt(value: float) -> str:
    return f"{value:.3f}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else float("nan")
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def selection_key(row: dict[str, str], metric: str) -> tuple[float, float, int]:
    other_metric = "roc_auc" if metric == "accuracy" else "accuracy"
    return (
        float(row[f"{metric}_mean"]),
        float(row[f"{other_metric}_mean"]),
        -int(row["n_components"]),
    )


def best_rows_by_dataset(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for metric, threshold, threshold_label in THRESHOLDS:
        for dataset, dataset_label in DATASETS:
            candidates = [
                row
                for row in rows
                if row["selection_metric"] == metric
                and row["threshold"] == threshold
                and row["dataset"] == dataset
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: selection_key(row, metric))
            output.append(
                {
                    "dataset": dataset,
                    "dataset_label": dataset_label,
                    "threshold_metric": metric,
                    "threshold": threshold,
                    "threshold_label": threshold_label,
                    "encoder": best["encoder"],
                    "encoder_label": ENCODER_LABELS[best["encoder"]],
                    "projector": best["projector"],
                    "projector_label": PROJECTOR_LABELS[best["projector"]],
                    "n_components": int(best["n_components"]),
                    "accuracy_mean": float(best["accuracy_mean"]),
                    "accuracy_std": float(best["accuracy_std"]),
                    "roc_auc_mean": float(best["roc_auc_mean"]),
                    "roc_auc_std": float(best["roc_auc_std"]),
                    "classifier_params": best["classifier_params"],
                    "projector_params": best["projector_params"],
                    "params_stability": best["params_stability"],
                }
            )
    return output


def curve_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str, int], dict[str, str]]:
    lookup = {}
    for row in rows:
        lookup[
            (
                row["selection_metric"],
                row["threshold"],
                row["dataset"],
                row["encoder"],
                row["projector"],
                int(row["n_components"]),
            )
        ] = row
    return lookup


def projector_summary(
    rows: list[dict[str, str]],
    curve_by_tuple: dict[tuple[str, str, str, str, str, int], dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["selection_metric"], row["threshold"], row["projector"])].append(row)

    output = []
    for metric, threshold, threshold_label in THRESHOLDS:
        for projector in ["pca", "umap", "grp"]:
            values = grouped.get((metric, threshold, projector), [])
            acc = [float(row["accuracy_mean"]) for row in values]
            roc = [float(row["roc_auc_mean"]) for row in values]
            curve_acc = []
            curve_roc = []
            for row in values:
                curve_row = (curve_by_tuple or {}).get(
                    (
                        row["selection_metric"],
                        row["threshold"],
                        row["dataset"],
                        row["encoder"],
                        row["projector"],
                        int(row["n_components"]),
                    )
                )
                if curve_row is not None:
                    curve_acc.append(float(curve_row["curve_accuracy"]))
                    curve_roc.append(float(curve_row["curve_roc_auc"]))
            dims = [int(row["n_components"]) for row in values]
            output.append(
                {
                    "threshold_metric": metric,
                    "threshold": threshold,
                    "threshold_label": threshold_label,
                    "projector": projector,
                    "projector_label": PROJECTOR_LABELS[projector],
                    "n": len(values),
                    "median_dim": sorted(dims)[len(dims) // 2] if dims else "",
                    "curve_accuracy_mean": mean(curve_acc),
                    "curve_accuracy_std": sample_std(curve_acc),
                    "curve_roc_auc_mean": mean(curve_roc),
                    "curve_roc_auc_std": sample_std(curve_roc),
                    "accuracy_mean": mean(acc),
                    "accuracy_std": sample_std(acc),
                    "roc_auc_mean": mean(roc),
                    "roc_auc_std": sample_std(roc),
                }
            )
    return output


def best_cell(row: dict[str, object] | None) -> str:
    if row is None:
        return "---"
    return (
        r"\shortstack[l]{"
        + f"{row['encoder_label']}--{row['projector_label']}, {row['n_components']}D"
        + r"\\"
        + f"Acc. {fmt(float(row['accuracy_mean']))}, ROC-AUC {fmt(float(row['roc_auc_mean']))}"
        + "}"
    )


def write_best_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lookup = {
        (row["dataset"], row["threshold_metric"], row["threshold"]): row
        for row in rows
    }
    lines = [
        r"\begin{table}[!htpb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Dataset & 99\% Acc. & 95\% Acc. & 99\% ROC-AUC & 95\% ROC-AUC \\",
        r"\midrule",
    ]
    for dataset, dataset_label in DATASETS:
        cells = [latex_escape(dataset_label)]
        for metric, threshold, _threshold_label in THRESHOLDS:
            cells.append(best_cell(lookup.get((dataset, metric, threshold))))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\caption{Best held-out logistic-regression result at each "
                r"threshold-defined dimension. Within each dataset and threshold "
                r"definition, the selected row is the encoder--projector pair with "
                r"the highest value for the threshold metric; ties use the other "
                r"metric and then the lower dimensionality. Cells report the "
                r"selected encoder--projector, dimensionality, mean accuracy, and "
                r"mean ROC-AUC across evaluation seeds.}"
            ),
            r"\label{tab:lr-threshold-best-summary}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_projector_latex(path: Path, rows: list[dict[str, object]]) -> None:
    def sd_range(field: str) -> str:
        values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
        return f"{min(values):.3f}-{max(values):.3f}" if values else "---"

    grouped = {
        (row["threshold_metric"], row["threshold"]): []
        for row in rows
    }
    for row in rows:
        grouped[(row["threshold_metric"], row["threshold"])].append(row)

    lines = [
        r"\begin{table}[!htpb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Projector & Median $d$ & Curve Acc. & Held-out Acc. & Curve ROC-AUC & Held-out ROC-AUC \\",
        r"\midrule",
    ]

    for index, (metric, threshold, threshold_label) in enumerate(THRESHOLDS):
        if index > 0:
            lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{6}}{{l}}{{\textbf{{{latex_escape(threshold_label)} threshold}}}} \\"
        )
        for row in grouped.get((metric, threshold), []):
            lines.append(
                " & ".join(
                    [
                        latex_escape(row["projector_label"]),
                        str(row["median_dim"]),
                        fmt(float(row["curve_accuracy_mean"])),
                        fmt(float(row["accuracy_mean"])),
                        fmt(float(row["curve_roc_auc_mean"])),
                        fmt(float(row["roc_auc_mean"])),
                    ]
                )
                + r" \\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\caption{Aggregate curve-stage and held-out logistic-regression "
                r"performance at threshold-defined dimensions, grouped by projector. "
                r"Values are means across evaluated dataset--encoder threshold tuples. "
                rf"Omitted SDs ranged from {sd_range('curve_accuracy_std')} for curve "
                rf"accuracy, {sd_range('accuracy_std')} for held-out accuracy, "
                rf"{sd_range('curve_roc_auc_std')} for curve ROC-AUC, and "
                rf"{sd_range('roc_auc_std')} for held-out ROC-AUC."
                r"}"
            ),
            r"\label{tab:lr-threshold-projector-summary}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_paragraph(path: Path, best_rows: list[dict[str, object]], projector_rows: list[dict[str, object]]) -> None:
    projector_wins = Counter(row["projector_label"] for row in best_rows)
    encoder_wins = Counter(row["encoder_label"] for row in best_rows)
    acc = [float(row["accuracy_mean"]) for row in best_rows]
    roc = [float(row["roc_auc_mean"]) for row in best_rows]
    best_projector = max(projector_rows, key=lambda row: float(row["roc_auc_mean"]))
    paragraph = (
        "The dense logistic-regression threshold table was reduced by selecting the "
        "best encoder--projector combination within each dataset and threshold "
        "definition. Across these 24 selected rows, mean held-out accuracy was "
        f"{mean(acc):.3f} (SD {sample_std(acc):.3f}) and mean ROC-AUC was "
        f"{mean(roc):.3f} (SD {sample_std(roc):.3f}). The selected projectors were "
        + ", ".join(f"{name} ({count})" for name, count in projector_wins.most_common())
        + "; the selected encoders were "
        + ", ".join(f"{name} ({count})" for name, count in encoder_wins.most_common())
        + ". The highest aggregate projector-level ROC-AUC was observed for "
        f"{best_projector['projector_label']} under the {best_projector['threshold_label']} "
        f"threshold ({float(best_projector['roc_auc_mean']):.3f})."
    )
    path.write_text(paragraph + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.input_csv)
    curve_comparison_csv = args.curve_comparison_csv or (
        args.output_dir / "curve_vs_generalised_lr_by_tuple.csv"
    )
    curve_by_tuple = (
        curve_lookup(read_rows(curve_comparison_csv))
        if curve_comparison_csv.exists()
        else {}
    )
    best_rows = best_rows_by_dataset(rows)
    projector_rows = projector_summary(rows, curve_by_tuple)

    write_csv(
        args.output_dir / "logistic_regression_threshold_best_summary.csv",
        [
            {
                **row,
                "accuracy_mean": f"{float(row['accuracy_mean']):.4f}",
                "accuracy_std": f"{float(row['accuracy_std']):.4f}",
                "roc_auc_mean": f"{float(row['roc_auc_mean']):.4f}",
                "roc_auc_std": f"{float(row['roc_auc_std']):.4f}",
            }
            for row in best_rows
        ],
        [
            "dataset",
            "dataset_label",
            "threshold_metric",
            "threshold",
            "threshold_label",
            "encoder",
            "encoder_label",
            "projector",
            "projector_label",
            "n_components",
            "accuracy_mean",
            "accuracy_std",
            "roc_auc_mean",
            "roc_auc_std",
            "classifier_params",
            "projector_params",
            "params_stability",
        ],
    )
    write_csv(
        args.output_dir / "logistic_regression_threshold_projector_summary.csv",
        [
            {
                **row,
                "curve_accuracy_mean": f"{float(row['curve_accuracy_mean']):.4f}",
                "curve_accuracy_std": f"{float(row['curve_accuracy_std']):.4f}",
                "accuracy_mean": f"{float(row['accuracy_mean']):.4f}",
                "accuracy_std": f"{float(row['accuracy_std']):.4f}",
                "curve_roc_auc_mean": f"{float(row['curve_roc_auc_mean']):.4f}",
                "curve_roc_auc_std": f"{float(row['curve_roc_auc_std']):.4f}",
                "roc_auc_mean": f"{float(row['roc_auc_mean']):.4f}",
                "roc_auc_std": f"{float(row['roc_auc_std']):.4f}",
            }
            for row in projector_rows
        ],
        [
            "threshold_metric",
            "threshold",
            "threshold_label",
            "projector",
            "projector_label",
            "n",
            "median_dim",
            "curve_accuracy_mean",
            "curve_accuracy_std",
            "accuracy_mean",
            "accuracy_std",
            "curve_roc_auc_mean",
            "curve_roc_auc_std",
            "roc_auc_mean",
            "roc_auc_std",
        ],
    )
    write_best_latex(args.output_dir / "logistic_regression_threshold_best_summary.tex", best_rows)
    write_projector_latex(args.output_dir / "logistic_regression_threshold_projector_summary.tex", projector_rows)
    write_paragraph(args.output_dir / "logistic_regression_threshold_summary_paragraph.txt", best_rows, projector_rows)

    for name in [
        "logistic_regression_threshold_best_summary.tex",
        "logistic_regression_threshold_projector_summary.tex",
        "logistic_regression_threshold_summary_paragraph.txt",
    ]:
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
