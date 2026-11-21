#!/usr/bin/env python3
"""Create threshold-dimension CSV and LaTeX tables from hproj results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


DATASETS = [
    ("kather100k", "Kather"),
    ("spider-colorectal", "CRC"),
    ("spider-breast", "Breast"),
    ("spider-skin", "Skin"),
    ("spider-thorax", "Thorax"),
    ("icaird100k", "iCAIRD"),
]
ENCODERS = [
    ("uni", "UNI"),
    ("uni2", "UNI2"),
    ("hibou", "Hibou"),
    ("dinov2", "DINOv2"),
]
PROJECTORS = [
    ("pca", "P"),
    ("umap", "U"),
    ("grp", "G"),
]
METRICS = [
    ("accuracy", "Accuracy"),
    ("roc_auc", "ROC-AUC"),
]
CONDITIONS = ["0.95", "0.99"]
STACKED_CONDITIONS = ["0.99", "0.95"]
MISSING = "---"


def parse_args() -> argparse.Namespace:
    run_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=run_root / "curve" / "thresholds_results.csv",
        help="Input thresholds CSV produced by the curve stage.",
    )
    parser.add_argument(
        "--intrinsic-dims",
        type=Path,
        default=run_root / "intrinsic_dims.csv",
        help="Intrinsic dimensionality CSV.",
    )
    parser.add_argument(
        "--classifier",
        default="logistic-regression",
        help="Classifier to include from the threshold results.",
    )
    parser.add_argument(
        "--curve-dir",
        type=Path,
        default=run_root / "curve",
        help="Curve output directory containing per-projector task JSONs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=run_root / "tables" / "thresholds",
        help="Directory for generated CSV and LaTeX tables.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def condition_slug(condition: str) -> str:
    return condition.replace(".", "")


def condition_percent(condition: str) -> str:
    return f"{float(condition) * 100:.0f}\\%"


def format_dim(value: str) -> str:
    if value == "":
        return MISSING
    number = float(value)
    if not math.isfinite(number):
        return MISSING
    if math.isclose(number, round(number), abs_tol=1e-8):
        return str(int(round(number)))
    return f"{number:.1f}"


def format_intrinsic_dim(value: str) -> str:
    if value == "":
        return MISSING
    number = float(value)
    if not math.isfinite(number):
        return MISSING
    return str(int(round(number)))


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def build_intrinsic_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for row in rows:
        lookup[(row["dataset"], row["encoder"])] = format_intrinsic_dim(
            row["intrinsic_dim_mle"]
        )
    return lookup


def build_baseline_lookup(
    rows: list[dict[str, str]],
    classifier: str,
) -> dict[tuple[str, str, str, str], float]:
    lookup: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        if row["classifier"] != classifier:
            continue
        key = (row["metric"], row["dataset"], row["encoder"], row["projector"])
        lookup[key] = float(row["baseline"])
    return lookup


def parse_dataset_encoder(value: str) -> tuple[str, str]:
    return value.rsplit("_", 1)


def build_curve_lookup(
    curve_dir: Path,
    classifier: str,
) -> dict[tuple[str, str, str, str, int], float]:
    values: dict[tuple[str, str, str, str, int], list[float]] = defaultdict(list)
    for projector_key, _projector_label in PROJECTORS:
        task_dir = curve_dir / f"{projector_key}_{classifier}" / "tasks"
        for path in task_dir.glob("*.json"):
            item = json.loads(path.read_text())
            dataset, encoder = parse_dataset_encoder(item["dataset"])
            if item["classifier"] != classifier or item["projector"] != projector_key:
                continue
            n_components = int(item["n_components"])
            for metric, _metric_label in METRICS:
                values[
                    (metric, dataset, encoder, projector_key, n_components)
                ].append(float(item[metric]))

    return {
        key: sum(metric_values) / len(metric_values)
        for key, metric_values in values.items()
    }


def build_threshold_lookup(
    baselines: dict[tuple[str, str, str, str], float],
    curves: dict[tuple[str, str, str, str, int], float],
) -> dict[tuple[str, str, str, str, str], str]:
    lookup: dict[tuple[str, str, str, str, str], str] = {}
    for metric, _metric_label in METRICS:
        for condition in CONDITIONS:
            condition_value = float(condition)
            for dataset_key, _dataset_label in DATASETS:
                for encoder_key, _encoder_label in ENCODERS:
                    for projector_key, _projector_label in PROJECTORS:
                        baseline = baselines.get(
                            (metric, dataset_key, encoder_key, projector_key)
                        )
                        if baseline is None:
                            continue
                        threshold = baseline * condition_value
                        dims = sorted(
                            n_components
                            for curve_metric, curve_dataset, curve_encoder, curve_projector, n_components in curves
                            if curve_metric == metric
                            and curve_dataset == dataset_key
                            and curve_encoder == encoder_key
                            and curve_projector == projector_key
                        )
                        achieved = MISSING
                        for n_components in dims:
                            value = curves[
                                (
                                    metric,
                                    dataset_key,
                                    encoder_key,
                                    projector_key,
                                    n_components,
                                )
                            ]
                            if value >= threshold:
                                achieved = format_dim(str(n_components))
                                break
                        lookup[
                            (
                                metric,
                                condition,
                                dataset_key,
                                encoder_key,
                                projector_key,
                            )
                        ] = achieved
    return lookup


def build_rows(
    intrinsic: dict[tuple[str, str], str],
    thresholds: dict[tuple[str, str, str, str, str], str],
    metric: str,
    condition: str,
) -> tuple[list[str], list[list[str]]]:
    header = ["Dataset"]
    for _encoder_key, encoder_label in ENCODERS:
        header.append(f"{encoder_label}_d_id")
        for _projector_key, projector_label in PROJECTORS:
            header.append(f"{encoder_label}_{projector_label}")

    rows: list[list[str]] = []
    for dataset_key, dataset_label in DATASETS:
        row = [dataset_label]
        for encoder_key, _encoder_label in ENCODERS:
            row.append(intrinsic.get((dataset_key, encoder_key), MISSING))
            for projector_key, _projector_label in PROJECTORS:
                row.append(
                    thresholds.get(
                        (metric, condition, dataset_key, encoder_key, projector_key),
                        MISSING,
                    )
                )
        rows.append(row)
    return header, rows


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_latex(path: Path, rows: list[list[str]], metric: str, metric_label: str, condition: str) -> None:
    num_cols = 1 + len(ENCODERS) * (1 + len(PROJECTORS))
    column_spec = "l" + "c" * (num_cols - 1)
    midrules = " ".join(
        rf"\cmidrule(lr){{{start}-{start + len(PROJECTORS)}}}"
        for start in range(2, num_cols + 1, len(PROJECTORS) + 1)
    )

    lines = [
        r"\begin{table}[!htpb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
    ]
    group_header = [""]
    for _encoder_key, encoder_label in ENCODERS:
        group_header.append(
            rf"\multicolumn{{4}}{{c}}{{{latex_escape(encoder_label)}}}"
        )
    lines.append(" & ".join(group_header) + r" \\")
    lines.append(midrules)

    subheader = ["Dataset"]
    for _encoder_key, _encoder_label in ENCODERS:
        subheader.extend([r"$d_\text{ID}$", "P", "U", "G"])
    lines.append(" & ".join(subheader) + r" \\")
    lines.append(r"\midrule")

    for row in rows:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")

    label_metric = metric.replace("_", "-")
    label_condition = condition_slug(condition)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                rf"\caption{{Minimum dimensions for {condition_percent(condition)} "
                rf"{metric_label} recovery with intrinsic dimensionality included "
                rf"for comparison, where {condition_percent(condition)} is measured "
                r"relative to the full-dimensional baseline. "
                r"$d_\text{ID}$: MLE intrinsic dimensionality. "
                r"P: PCA. U: UMAP. G: GRP. "
                r"``---'': threshold never reached. "
                r"Full-dimensional embeddings are 1024-D (UNI, Hibou, DINOv2) "
                r"or 1536-D (UNI2).}"
            ),
            rf"\label{{tab:threshold-{label_metric}-{label_condition}}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def latex_encoder_header(num_cols: int) -> list[str]:
    midrules = " ".join(
        rf"\cmidrule(lr){{{start}-{start + len(PROJECTORS)}}}"
        for start in range(2, num_cols + 1, len(PROJECTORS) + 1)
    )
    group_header = [""]
    for _encoder_key, encoder_label in ENCODERS:
        group_header.append(
            rf"\multicolumn{{4}}{{c}}{{{latex_escape(encoder_label)}}}"
        )
    subheader = ["Dataset"]
    for _encoder_key, _encoder_label in ENCODERS:
        subheader.extend([r"$d_\text{ID}$", "P", "U", "G"])
    return [
        " & ".join(group_header) + r" \\",
        midrules,
        " & ".join(subheader) + r" \\",
    ]


def write_stacked_metric_latex(
    path: Path,
    metric: str,
    metric_label: str,
    rows_by_condition: dict[str, list[list[str]]],
) -> None:
    num_cols = 1 + len(ENCODERS) * (1 + len(PROJECTORS))
    column_spec = "l" + "c" * (num_cols - 1)
    lines = [
        r"\begin{table}[!htpb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        *latex_encoder_header(num_cols),
        r"\midrule",
    ]

    for idx, condition in enumerate(STACKED_CONDITIONS):
        if idx:
            lines.append(r"\addlinespace[3pt]")
        lines.append(
            rf"\multicolumn{{{num_cols}}}{{l}}{{\textbf{{{condition_percent(condition)} {metric_label} recovery}}}} \\"
        )
        for row in rows_by_condition[condition]:
            lines.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")

    label_metric = metric.replace("_", "-")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                rf"\caption{{Minimum dimensions for 95\% and 99\% {metric_label} "
                r"recovery with intrinsic dimensionality included for comparison, "
                r"where recovery is measured relative to the full-dimensional baseline. "
                r"$d_\text{ID}$: MLE intrinsic dimensionality. "
                r"P: PCA. U: UMAP. G: GRP. "
                r"``---'': threshold never reached. "
                r"Full-dimensional embeddings are 1024-D (UNI, Hibou, DINOv2) "
                r"or 1536-D (UNI2). The iCAIRD row reports the iCAIRD cervical "
                r"100K dataset.}"
            ),
            rf"\label{{tab:threshold-{label_metric}-stacked}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def validate_complete_inputs(
    intrinsic: dict[tuple[str, str], str],
    thresholds: dict[tuple[str, str, str, str, str], str],
) -> None:
    missing_intrinsic = [
        (dataset, encoder)
        for dataset, _dataset_label in DATASETS
        for encoder, _encoder_label in ENCODERS
        if intrinsic.get((dataset, encoder)) in (None, MISSING)
    ]
    if missing_intrinsic:
        raise SystemExit(f"Missing intrinsic dimensionality rows: {missing_intrinsic}")

    missing_threshold_keys = [
        (metric, condition, dataset, encoder, projector)
        for metric, _metric_label in METRICS
        for condition in CONDITIONS
        for dataset, _dataset_label in DATASETS
        for encoder, _encoder_label in ENCODERS
        for projector, _projector_label in PROJECTORS
        if (metric, condition, dataset, encoder, projector) not in thresholds
    ]
    if missing_threshold_keys:
        raise SystemExit(f"Missing threshold calculations: {missing_threshold_keys[:10]}")


def main() -> None:
    args = parse_args()
    threshold_rows = read_csv(args.thresholds)
    intrinsic_rows = read_csv(args.intrinsic_dims)
    intrinsic = build_intrinsic_lookup(intrinsic_rows)
    baselines = build_baseline_lookup(threshold_rows, args.classifier)
    curves = build_curve_lookup(args.curve_dir, args.classifier)
    thresholds = build_threshold_lookup(baselines, curves)
    validate_complete_inputs(intrinsic, thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric, metric_label in METRICS:
        rows_by_condition = {}
        for condition in CONDITIONS:
            header, rows = build_rows(intrinsic, thresholds, metric, condition)
            rows_by_condition[condition] = rows
            stem = f"threshold_{metric}_{condition_slug(condition)}"
            csv_path = args.output_dir / f"{stem}.csv"
            tex_path = args.output_dir / f"{stem}.tex"
            write_csv(csv_path, header, rows)
            write_latex(tex_path, rows, metric, metric_label, condition)
            print(f"Wrote {csv_path}")
            print(f"Wrote {tex_path}")
        stacked_path = args.output_dir / f"threshold_{metric}_stacked.tex"
        write_stacked_metric_latex(stacked_path, metric, metric_label, rows_by_condition)
        print(f"Wrote {stacked_path}")


if __name__ == "__main__":
    main()
