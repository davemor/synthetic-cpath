#!/usr/bin/env python3
"""Summarise generalisation performance at selected threshold dimensions.

For each threshold-defined representation tuple
(dataset, encoder, projector, dimension), this script averages repeated seeds
for each classifier/hyperparameter setting, selects the best held-out setting,
and compares it with the best full-dimensional baseline for the same dataset
and encoder.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


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
PROJECTORS = [("pca", "PCA"), ("umap", "UMAP"), ("grp", "GRP")]
METRICS = [("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC")]
CONDITIONS = ["0.99", "0.95"]
CONDITION_LABELS = {"0.99": "99%", "0.95": "95%"}
CLASSIFIER_LABELS = {
    "logistic-regression": "LR",
    "mlp": "MLP",
    "xgboost": "XGB",
}
BAR_COLOURS = {"0.99": "#1f77b4", "0.95": "#ff7f0e"}
MISSING = "---"


def parse_args() -> argparse.Namespace:
    run_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generalisation-results",
        type=Path,
        default=run_root / "generalisation_results.csv",
        help="Generalisation results CSV.",
    )
    parser.add_argument(
        "--curve-thresholds",
        type=Path,
        default=run_root / "curve" / "thresholds_results.csv",
        help="Raw curve threshold CSV, used for full-dimensional curve baselines.",
    )
    parser.add_argument(
        "--curve-dir",
        type=Path,
        default=run_root / "curve",
        help="Curve directory, used to compute missing metric/threshold combinations.",
    )
    parser.add_argument(
        "--classifier",
        default="logistic-regression",
        help="Curve classifier used for threshold dimension selection.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=run_root / "tables" / "generalisation_thresholds",
        help="Output directory for CSV and LaTeX tables.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=run_root / "report" / "plots",
        help="Output directory for SVG/PDF plots.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    value_mean = mean(values)
    return math.sqrt(sum((value - value_mean) ** 2 for value in values) / (len(values) - 1))


def fmt_num(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return MISSING
    return f"{value:.{digits}f}"


def fmt_dim(value: str | int | float) -> str:
    number = float(value)
    if math.isclose(number, round(number), abs_tol=1e-8):
        return str(int(round(number)))
    return f"{number:.1f}"


def fmt_recovery(value: float) -> str:
    if not math.isfinite(value):
        return MISSING
    return f"{value * 100:.1f}%"


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
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


def short_params(classifier: str, params_text: str) -> str:
    try:
        params = ast.literal_eval(params_text)
    except Exception:
        return params_text
    if classifier == "logistic-regression":
        return f"C={params.get('C')}"
    if classifier == "mlp":
        return f"alpha={params.get('alpha')}, layers={params.get('hidden_layer_sizes')}"
    if classifier == "xgboost":
        keys = ["n_estimators", "max_depth", "learning_rate", "subsample"]
        return ", ".join(f"{key}={params.get(key)}" for key in keys)
    return json.dumps(params, sort_keys=True)


def parse_dataset_encoder(value: str) -> tuple[str, str]:
    return value.rsplit("_", 1)


def build_curve_baselines(
    rows: list[dict[str, str]], classifier: str
) -> dict[tuple[str, str, str, str], float]:
    baselines: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        if row["classifier"] != classifier:
            continue
        baselines[(row["metric"], row["dataset"], row["encoder"], row["projector"])] = float(
            row["baseline"]
        )
    return baselines


def build_curve_means(
    curve_dir: Path, classifier: str
) -> dict[tuple[str, str, str, str, int], float]:
    values: dict[tuple[str, str, str, str, int], list[float]] = defaultdict(list)
    for projector_key, _projector_label in PROJECTORS:
        task_dir = curve_dir / f"{projector_key}_{classifier}" / "tasks"
        for path in task_dir.glob("*.json"):
            item = json.loads(path.read_text())
            if item["classifier"] != classifier or item["projector"] != projector_key:
                continue
            dataset, encoder = parse_dataset_encoder(item["dataset"])
            n_components = int(item["n_components"])
            for metric, _metric_label in METRICS:
                values[
                    (metric, dataset, encoder, projector_key, n_components)
                ].append(float(item[metric]))
    return {key: mean(vals) for key, vals in values.items()}


def build_threshold_dims(
    curve_baselines: dict[tuple[str, str, str, str], float],
    curve_means: dict[tuple[str, str, str, str, int], float],
) -> dict[tuple[str, str, str, str, str], int | None]:
    output: dict[tuple[str, str, str, str, str], int | None] = {}
    for metric, _metric_label in METRICS:
        for condition in CONDITIONS:
            condition_value = float(condition)
            for dataset, _dataset_label in DATASETS:
                for encoder, _encoder_label in ENCODERS:
                    for projector, _projector_label in PROJECTORS:
                        baseline = curve_baselines.get((metric, dataset, encoder, projector))
                        if baseline is None:
                            output[(metric, condition, dataset, encoder, projector)] = None
                            continue
                        threshold = baseline * condition_value
                        dims = sorted(
                            dim
                            for curve_metric, curve_dataset, curve_encoder, curve_projector, dim in curve_means
                            if curve_metric == metric
                            and curve_dataset == dataset
                            and curve_encoder == encoder
                            and curve_projector == projector
                        )
                        achieved = None
                        for dim in dims:
                            if curve_means[(metric, dataset, encoder, projector, dim)] >= threshold:
                                achieved = dim
                                break
                        output[(metric, condition, dataset, encoder, projector)] = achieved
    return output


def aggregate_generalisation_rows(
    rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    values: dict[tuple[str, str, int, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (
            row["dataset"],
            row["encoder"],
            int(float(row["n_components"])),
            row["projector"],
            row["classifier"],
            row["best_params"],
        )
        values[key]["accuracy"].append(float(row["accuracy"]))
        values[key]["roc_auc"].append(float(row["roc_auc"]))
        values[key]["cv_score"].append(float(row["cv_score"]))

    aggregated: list[dict[str, object]] = []
    for (dataset, encoder, dim, projector, classifier, params), metrics in values.items():
        aggregated.append(
            {
                "dataset": dataset,
                "encoder": encoder,
                "n_components": dim,
                "projector": projector,
                "classifier": classifier,
                "best_params": params,
                "params_short": short_params(classifier, params),
                "n_seeds": len(metrics["accuracy"]),
                "accuracy_mean": mean(metrics["accuracy"]),
                "accuracy_std": sample_std(metrics["accuracy"]),
                "roc_auc_mean": mean(metrics["roc_auc"]),
                "roc_auc_std": sample_std(metrics["roc_auc"]),
                "cv_score_mean": mean(metrics["cv_score"]),
            }
        )
    return aggregated


def select_best(
    rows: list[dict[str, object]],
    metric: str,
) -> dict[str, object] | None:
    if not rows:
        return None
    metric_col = f"{metric}_mean"
    other_col = "accuracy_mean" if metric == "roc_auc" else "roc_auc_mean"
    return max(
        rows,
        key=lambda row: (
            float(row[metric_col]),
            float(row[other_col]),
            -int(row["n_components"]),
        ),
    )


def build_full_baselines(
    aggregated: list[dict[str, object]]
) -> dict[tuple[str, str, str], dict[str, object]]:
    baselines: dict[tuple[str, str, str], dict[str, object]] = {}
    for metric, _metric_label in METRICS:
        for dataset, _dataset_label in DATASETS:
            for encoder, _encoder_label in ENCODERS:
                candidates = [
                    row
                    for row in aggregated
                    if row["dataset"] == dataset
                    and row["encoder"] == encoder
                    and row["projector"] == "none"
                ]
                best = select_best(candidates, metric)
                if best is not None:
                    baselines[(metric, dataset, encoder)] = best
    return baselines


def build_tuple_evidence(
    aggregated: list[dict[str, object]],
    baselines: dict[tuple[str, str, str], dict[str, object]],
    threshold_dims: dict[tuple[str, str, str, str, str], int | None],
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for metric, _metric_label in METRICS:
        metric_col = f"{metric}_mean"
        metric_std_col = f"{metric}_std"
        other_metric = "accuracy" if metric == "roc_auc" else "roc_auc"
        for condition in CONDITIONS:
            for dataset, dataset_label in DATASETS:
                for encoder, encoder_label in ENCODERS:
                    baseline = baselines.get((metric, dataset, encoder))
                    if baseline is None:
                        continue
                    baseline_value = float(baseline[metric_col])
                    for projector, projector_label in PROJECTORS:
                        dim = threshold_dims.get((metric, condition, dataset, encoder, projector))
                        if dim is None:
                            continue
                        candidates = [
                            row
                            for row in aggregated
                            if row["dataset"] == dataset
                            and row["encoder"] == encoder
                            and row["projector"] == projector
                            and row["n_components"] == dim
                        ]
                        best = select_best(candidates, metric)
                        if best is None:
                            continue
                        evidence.append(
                            {
                                "metric": metric,
                                "condition": condition,
                                "dataset": dataset,
                                "dataset_label": dataset_label,
                                "encoder": encoder,
                                "encoder_label": encoder_label,
                                "projector": projector,
                                "projector_label": projector_label,
                                "n_components": dim,
                                "classifier": best["classifier"],
                                "classifier_label": CLASSIFIER_LABELS.get(
                                    str(best["classifier"]), str(best["classifier"])
                                ),
                                "params_short": best["params_short"],
                                "best_params": best["best_params"],
                                "n_seeds": best["n_seeds"],
                                "metric_mean": float(best[metric_col]),
                                "metric_std": float(best[metric_std_col]),
                                "other_metric_mean": float(best[f"{other_metric}_mean"]),
                                "other_metric_std": float(best[f"{other_metric}_std"]),
                                "full_baseline": baseline_value,
                                "full_classifier": baseline["classifier"],
                                "full_classifier_label": CLASSIFIER_LABELS.get(
                                    str(baseline["classifier"]), str(baseline["classifier"])
                                ),
                                "full_params_short": baseline["params_short"],
                                "recovery": float(best[metric_col]) / baseline_value
                                if baseline_value
                                else float("nan"),
                            }
                        )
    return evidence


def build_coverage_rows(
    aggregated: list[dict[str, object]],
    threshold_dims: dict[tuple[str, str, str, str, str], int | None],
) -> list[dict[str, object]]:
    evaluated = {
        (
            str(row["dataset"]),
            str(row["encoder"]),
            str(row["projector"]),
            int(row["n_components"]),
        )
        for row in aggregated
    }
    rows: list[dict[str, object]] = []
    for metric, _metric_label in METRICS:
        for condition in CONDITIONS:
            for dataset, dataset_label in DATASETS:
                for encoder, encoder_label in ENCODERS:
                    for projector, projector_label in PROJECTORS:
                        dim = threshold_dims.get((metric, condition, dataset, encoder, projector))
                        present = (
                            dim is not None
                            and (dataset, encoder, projector, int(dim)) in evaluated
                        )
                        rows.append(
                            {
                                "metric": metric,
                                "condition": condition,
                                "dataset": dataset,
                                "dataset_label": dataset_label,
                                "encoder": encoder,
                                "encoder_label": encoder_label,
                                "projector": projector,
                                "projector_label": projector_label,
                                "threshold_dim": dim,
                                "evaluated": present,
                            }
                        )
    return rows


def choose_dataset_summaries(evidence: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for metric, _metric_label in METRICS:
        for condition in CONDITIONS:
            for dataset, _dataset_label in DATASETS:
                candidates = [
                    row
                    for row in evidence
                    if row["metric"] == metric
                    and row["condition"] == condition
                    and row["dataset"] == dataset
                ]
                if not candidates:
                    continue
                best = max(
                    candidates,
                    key=lambda row: (
                        float(row["metric_mean"]),
                        float(row["recovery"]),
                        -int(row["n_components"]),
                    ),
                )
                summaries.append(best)
    return summaries


def write_coverage_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "metric",
        "threshold",
        "dataset",
        "encoder",
        "projector",
        "threshold_dim",
        "evaluated_in_generalisation",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "metric": row["metric"],
                    "threshold": CONDITION_LABELS[str(row["condition"])],
                    "dataset": row["dataset_label"],
                    "encoder": row["encoder_label"],
                    "projector": row["projector_label"],
                    "threshold_dim": MISSING
                    if row["threshold_dim"] is None
                    else row["threshold_dim"],
                    "evaluated_in_generalisation": "yes" if row["evaluated"] else "no",
                }
            )


def write_coverage_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["metric", "threshold", "dataset", "evaluated", "total", "coverage"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric, _metric_label in METRICS:
            for condition in CONDITIONS:
                for dataset, dataset_label in DATASETS:
                    subset = [
                        row
                        for row in rows
                        if row["metric"] == metric
                        and row["condition"] == condition
                        and row["dataset"] == dataset
                    ]
                    evaluated_count = sum(1 for row in subset if row["evaluated"])
                    total_count = len(subset)
                    writer.writerow(
                        {
                            "metric": metric,
                            "threshold": CONDITION_LABELS[condition],
                            "dataset": dataset_label,
                            "evaluated": evaluated_count,
                            "total": total_count,
                            "coverage": f"{evaluated_count / total_count:.3f}"
                            if total_count
                            else MISSING,
                        }
                    )


def write_tuple_csv(path: Path, evidence: list[dict[str, object]]) -> None:
    fields = [
        "metric",
        "condition",
        "dataset",
        "encoder",
        "projector",
        "n_components",
        "classifier",
        "params_short",
        "metric_mean",
        "metric_std",
        "other_metric_mean",
        "other_metric_std",
        "full_baseline",
        "recovery",
        "full_classifier",
        "full_params_short",
        "n_seeds",
        "best_params",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in evidence:
            writer.writerow(
                {
                    "metric": row["metric"],
                    "condition": CONDITION_LABELS[str(row["condition"])],
                    "dataset": row["dataset_label"],
                    "encoder": row["encoder_label"],
                    "projector": row["projector_label"],
                    "n_components": row["n_components"],
                    "classifier": row["classifier_label"],
                    "params_short": row["params_short"],
                    "metric_mean": fmt_num(float(row["metric_mean"])),
                    "metric_std": fmt_num(float(row["metric_std"])),
                    "other_metric_mean": fmt_num(float(row["other_metric_mean"])),
                    "other_metric_std": fmt_num(float(row["other_metric_std"])),
                    "full_baseline": fmt_num(float(row["full_baseline"])),
                    "recovery": fmt_recovery(float(row["recovery"])),
                    "full_classifier": row["full_classifier_label"],
                    "full_params_short": row["full_params_short"],
                    "n_seeds": row["n_seeds"],
                    "best_params": row["best_params"],
                }
            )


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "metric",
        "threshold",
        "dataset",
        "encoder_projector",
        "n_components",
        "classifier",
        "params_short",
        "performance",
        "std",
        "full_baseline",
        "recovery",
        "other_metric",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "metric": row["metric"],
                    "threshold": CONDITION_LABELS[str(row["condition"])],
                    "dataset": row["dataset_label"],
                    "encoder_projector": f"{row['encoder_label']} + {row['projector_label']}",
                    "n_components": row["n_components"],
                    "classifier": row["classifier_label"],
                    "params_short": row["params_short"],
                    "performance": fmt_num(float(row["metric_mean"])),
                    "std": fmt_num(float(row["metric_std"])),
                    "full_baseline": fmt_num(float(row["full_baseline"])),
                    "recovery": fmt_recovery(float(row["recovery"])),
                    "other_metric": fmt_num(float(row["other_metric_mean"])),
                }
            )


def write_latex_summary(path: Path, rows: list[dict[str, object]], metric: str, label: str) -> None:
    lines = [
        r"\begin{table}[!htpb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lllrlrr}",
        r"\toprule",
        r"Threshold & Dataset & Encoder + projector & Dim. & Classifier & Test score & Recovery \\",
        r"\midrule",
    ]
    for condition in CONDITIONS:
        lines.append(
            rf"\multicolumn{{7}}{{l}}{{\textbf{{{CONDITION_LABELS[condition]} threshold dimensions}}}} \\"
        )
        for dataset, _dataset_label in DATASETS:
            matches = [
                row
                for row in rows
                if row["metric"] == metric
                and row["condition"] == condition
                and row["dataset"] == dataset
            ]
            if not matches:
                lines.append(
                    " & ".join(
                        [
                            CONDITION_LABELS[condition],
                            latex_escape(_dataset_label),
                            r"\multicolumn{5}{l}{not evaluated at exact threshold dimension}",
                        ]
                    )
                    + r" \\"
                )
                continue
            row = matches[0]
            lines.append(
                " & ".join(
                    [
                        CONDITION_LABELS[condition],
                        latex_escape(row["dataset_label"]),
                        latex_escape(f"{row['encoder_label']} + {row['projector_label']}"),
                        str(row["n_components"]),
                        latex_escape(row["classifier_label"]),
                        fmt_num(float(row["metric_mean"])),
                        latex_escape(fmt_recovery(float(row["recovery"]))),
                    ]
                )
                + r" \\"
            )
        if condition != CONDITIONS[-1]:
            lines.append(r"\addlinespace[3pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                rf"\caption{{Best held-out {label} results at the curve-selected "
                r"threshold dimensions. For each dataset and threshold, the table "
                r"reports the best encoder, projector, classifier, and hyperparameter "
                r"combination after averaging repeated seeds. Recovery is relative to "
                r"the best full-dimensional held-out result for the same dataset and "
                r"encoder under the same classifier search. Rows marked not evaluated "
                r"indicate that the exact curve-selected threshold dimension was not "
                r"present in the downloaded generalisation results.}"
            ),
            rf"\label{{tab:generalisation-{metric.replace('_', '-')}-thresholds}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            (
                "<style>"
                "text{font-family:Arial,Helvetica,sans-serif;fill:#222}"
                ".title{font-size:62px;font-weight:700}"
                ".axis{font-size:46px}"
                ".tick{font-size:36px;fill:#444}"
                ".legend{font-size:42px}"
                ".legend-title{font-size:42px;font-weight:600}"
                "</style>"
            ),
        ]

    def add(self, text: str) -> None:
        self.parts.append(text)

    def text(self, x: float, y: float, value: str, cls: str = "", anchor: str = "middle") -> None:
        attrs = f'x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}"'
        if cls:
            attrs += f' class="{cls}"'
        self.add(f"<text {attrs}>{escape(value)}</text>")

    def finish(self) -> str:
        return "\n".join(self.parts + ["</svg>", ""])


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    span = high - low
    raw_step = span / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(raw_step))
    step = min([1, 2, 5, 10], key=lambda m: abs(raw_step - m * power)) * power
    start = math.floor(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 0.5:
        if value >= low - step * 0.2:
            ticks.append(value)
        value += step
    return ticks


def render_recovery_plot(rows: list[dict[str, object]], metric: str, label: str) -> str:
    width, height = 2200, 1320
    svg = Svg(width, height)
    left, top = 170, 150
    plot_width, plot_height = 1840, 760
    plot_right, plot_bottom = left + plot_width, top + plot_height
    title = f"Generalisation Recovery: {label}"
    svg.text(width / 2, 86, title, cls="title")

    values_by_dataset_condition = {}
    values = []
    for row in rows:
        if row["metric"] != metric:
            continue
        key = (row["dataset"], row["condition"])
        value = float(row["recovery"])
        values_by_dataset_condition[key] = value
        values.append(value)

    y_min = min(0.9, min(values) - 0.015)
    y_max = max(1.02, max(values) + 0.015)

    def sy(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * plot_height

    def sx_dataset(index: int) -> float:
        return left + (index + 0.5) / len(DATASETS) * plot_width

    for tick in nice_ticks(y_min, y_max):
        y = sy(tick)
        svg.add(
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{plot_right:.2f}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        svg.text(left - 16, y + 12, f"{tick * 100:.0f}%", cls="tick", anchor="end")

    for condition, dash in [("0.99", "4 8"), ("0.95", "10 8")]:
        y = sy(float(condition))
        svg.add(
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{plot_right:.2f}" y2="{y:.2f}" '
            f'stroke="#555" stroke-width="2" stroke-dasharray="{dash}"/>'
        )

    group_width = plot_width / len(DATASETS)
    bar_width = group_width * 0.24
    offsets = {"0.99": -bar_width * 0.65, "0.95": bar_width * 0.65}
    for dataset_index, (dataset, dataset_label) in enumerate(DATASETS):
        x_center = sx_dataset(dataset_index)
        for condition in CONDITIONS:
            value = values_by_dataset_condition.get((dataset, condition))
            if value is None:
                continue
            x = x_center + offsets[condition]
            y = sy(value)
            bar_height = plot_bottom - y
            svg.add(
                f'<rect x="{x - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
                f'fill="{BAR_COLOURS[condition]}" opacity="0.85"/>'
            )
            svg.text(x, y - 12, f"{value * 100:.1f}", cls="tick")
        svg.text(x_center, plot_bottom + 56, dataset_label, cls="tick")

    svg.add(
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{plot_width:.2f}" height="{plot_height:.2f}" fill="none" stroke="#222" stroke-width="1"/>'
    )
    svg.text(width / 2, plot_bottom + 120, "Dataset", cls="axis")
    svg.add(
        f'<text x="62" y="{top + plot_height / 2:.2f}" class="axis" text-anchor="middle" transform="rotate(-90 62 {top + plot_height / 2:.2f})">Held-out recovery</text>'
    )

    legend_x, legend_y = width / 2, 1180
    box_width, box_height = 760, 132
    box_x, box_y = legend_x - box_width / 2, legend_y - box_height / 2
    svg.add(
        f'<rect x="{box_x:.2f}" y="{box_y:.2f}" width="{box_width:.2f}" height="{box_height:.2f}" fill="white" stroke="#222" stroke-width="1.2"/>'
    )
    svg.text(legend_x, box_y + 42, "Threshold", cls="legend-title")
    item_width = 250
    start_x = legend_x - item_width
    for index, condition in enumerate(CONDITIONS):
        item_x = start_x + index * item_width
        svg.add(
            f'<rect x="{item_x:.2f}" y="{box_y + 72:.2f}" width="58" height="30" fill="{BAR_COLOURS[condition]}" opacity="0.85"/>'
        )
        svg.text(item_x + 78, box_y + 98, CONDITION_LABELS[condition], cls="legend", anchor="start")

    return svg.finish()


def convert_svg_to_pdf(svg_path: Path, pdf_path: Path) -> None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise SystemExit("rsvg-convert is required to create PDF plots.")
    subprocess.run([converter, "-f", "pdf", "-o", str(pdf_path), str(svg_path)], check=True)


def write_plots(plots_dir: Path, summaries: list[dict[str, object]]) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for metric, label in METRICS:
        svg = render_recovery_plot(summaries, metric, label)
        slug = metric.replace("_", "-")
        svg_path = plots_dir / f"generalisation_{slug}_threshold_recovery.svg"
        pdf_path = plots_dir / f"generalisation_{slug}_threshold_recovery.pdf"
        svg_path.write_text(svg)
        convert_svg_to_pdf(svg_path, pdf_path)
        print(f"Wrote {svg_path}")
        print(f"Wrote {pdf_path}")


def main() -> None:
    args = parse_args()
    generalisation_rows = read_csv(args.generalisation_results)
    curve_threshold_rows = read_csv(args.curve_thresholds)
    curve_baselines = build_curve_baselines(curve_threshold_rows, args.classifier)
    curve_means = build_curve_means(args.curve_dir, args.classifier)
    threshold_dims = build_threshold_dims(curve_baselines, curve_means)

    aggregated = aggregate_generalisation_rows(generalisation_rows)
    full_baselines = build_full_baselines(aggregated)
    coverage_rows = build_coverage_rows(aggregated, threshold_dims)
    evidence = build_tuple_evidence(aggregated, full_baselines, threshold_dims)
    summaries = choose_dataset_summaries(evidence)

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    tuple_csv = args.tables_dir / "generalisation_best_hyperparams_by_tuple.csv"
    summary_csv = args.tables_dir / "generalisation_best_by_dataset_threshold.csv"
    coverage_csv = args.tables_dir / "generalisation_threshold_coverage.csv"
    coverage_summary_csv = args.tables_dir / "generalisation_threshold_coverage_summary.csv"
    write_tuple_csv(tuple_csv, evidence)
    write_summary_csv(summary_csv, summaries)
    write_coverage_csv(coverage_csv, coverage_rows)
    write_coverage_summary_csv(coverage_summary_csv, coverage_rows)
    print(f"Wrote {tuple_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {coverage_csv}")
    print(f"Wrote {coverage_summary_csv}")

    for metric, label in METRICS:
        tex_path = args.tables_dir / f"generalisation_{metric}_threshold_summary.tex"
        write_latex_summary(tex_path, summaries, metric, label)
        print(f"Wrote {tex_path}")

    write_plots(args.plots_dir, summaries)


if __name__ == "__main__":
    main()
