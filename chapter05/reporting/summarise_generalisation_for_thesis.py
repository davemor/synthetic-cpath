#!/usr/bin/env python3
"""Summarise threshold-validation generalisation results for thesis tables."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import product
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
    ("pca", "PCA"),
    ("umap", "UMAP"),
    ("grp", "GRP"),
]
METRICS = [
    ("accuracy", "Accuracy"),
    ("roc_auc", "ROC-AUC"),
]
CONDITIONS = ["0.99", "0.95"]
CLASSIFIER_LABELS = {
    "logistic-regression": "LR",
    "mlp": "MLP",
    "xgboost": "XGB",
}
MISSING = "---"
COMBINED_MANIFEST_NAME = (
    "task_manifest_principled_reduced_icaird_full_plus_threshold_completion_20260622.jsonl"
)


def parse_args() -> argparse.Namespace:
    run_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=run_root,
        help="hproj run root containing generalisation_results.csv and curve outputs.",
    )
    parser.add_argument(
        "--classifier",
        default="logistic-regression",
        help="Curve classifier used to define threshold dimensions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-root>/tables/generalisation_thesis_summary.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_params(text: str) -> dict[str, object]:
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return {"raw": text}
    return parsed if isinstance(parsed, dict) else {"raw": parsed}


def params_key(params: dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def params_display(params: dict[str, object], classifier: str) -> str:
    if classifier == "logistic-regression":
        return f"C={params.get('C')}"
    if classifier == "mlp":
        return f"layers={params.get('hidden_layer_sizes')}, alpha={params.get('alpha')}"
    if classifier == "xgboost":
        keys = ["n_estimators", "max_depth", "learning_rate", "subsample"]
        return ", ".join(f"{key}={params.get(key)}" for key in keys)
    return params_key(params)


def projector_params_display(params: dict[str, object], projector: str) -> str:
    if not params:
        return ""
    if projector == "umap":
        keys = ["n_neighbors", "min_dist", "metric", "subsample"]
        return ", ".join(f"{key}={params.get(key)}" for key in keys if key in params)
    return params_key(params)


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


def fmt_pm(mean_value: float, std_value: float) -> str:
    return f"{fmt_num(mean_value)} $\\pm$ {fmt_num(std_value)}"


def fmt_recovery(value: float) -> str:
    if not math.isfinite(value):
        return MISSING
    return f"{value * 100:.1f}\\%"


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


def parse_dataset_encoder(value: str) -> tuple[str, str]:
    return value.rsplit("_", 1)


def build_curve_baselines(
    threshold_rows: list[dict[str, str]], classifier: str
) -> dict[tuple[str, str, str, str], float]:
    values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in threshold_rows:
        if row["classifier"] != classifier:
            continue
        values[(row["metric"], row["dataset"], row["encoder"], row["projector"])].append(
            float(row["baseline"])
        )
    return {key: mean(items) for key, items in values.items()}


def build_curve_means(
    curve_dir: Path, classifier: str
) -> dict[tuple[str, str, str, str, int], float]:
    values: dict[tuple[str, str, str, str, int], list[float]] = defaultdict(list)
    for projector, _projector_label in PROJECTORS:
        task_dir = curve_dir / f"{projector}_{classifier}" / "tasks"
        for path in task_dir.glob("*.json"):
            item = json.loads(path.read_text())
            if item["classifier"] != classifier or item["projector"] != projector:
                continue
            dataset, encoder = parse_dataset_encoder(item["dataset"])
            dim = int(item["n_components"])
            for metric, _metric_label in METRICS:
                values[(metric, dataset, encoder, projector, dim)].append(float(item[metric]))
    return {key: mean(items) for key, items in values.items()}


def build_threshold_dims(
    baselines: dict[tuple[str, str, str, str], float],
    curve_means: dict[tuple[str, str, str, str, int], float],
) -> dict[tuple[str, str, str, str, str], int | None]:
    output: dict[tuple[str, str, str, str, str], int | None] = {}
    for metric, _metric_label in METRICS:
        for condition in CONDITIONS:
            for dataset, _dataset_label in DATASETS:
                for encoder, _encoder_label in ENCODERS:
                    for projector, _projector_label in PROJECTORS:
                        baseline = baselines.get((metric, dataset, encoder, projector))
                        if baseline is None:
                            output[(metric, condition, dataset, encoder, projector)] = None
                            continue
                        target = baseline * float(condition)
                        dims = sorted(
                            dim
                            for curve_metric, curve_dataset, curve_encoder, curve_projector, dim in curve_means
                            if (curve_metric, curve_dataset, curve_encoder, curve_projector)
                            == (metric, dataset, encoder, projector)
                        )
                        achieved = None
                        for dim in dims:
                            if curve_means[(metric, dataset, encoder, projector, dim)] >= target:
                                achieved = dim
                                break
                        output[(metric, condition, dataset, encoder, projector)] = achieved
    return output


def read_manifest_task_rows(run_root: Path) -> list[dict[str, object]] | None:
    manifest_path = run_root / "generalisation" / COMBINED_MANIFEST_NAME
    if not manifest_path.exists():
        return None

    rows = []
    with manifest_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            local_task_path = (
                run_root / "generalisation" / "tasks" / f"{record['task_id']}.json"
            )
            task_path = local_task_path
            if not task_path.exists():
                task_path = Path(record["task_path"])
            if not task_path.exists():
                raise SystemExit(f"Missing task JSON for manifest record: {task_path}")
            result = json.loads(task_path.read_text())
            row = {
                "dataset": record["dataset"],
                "encoder": record["encoder"],
                "n_components": int(record["n_components"]),
                "projector": result["projector"],
                "projector_name": record["projector_name"],
                "projector_params": record["projector_params"],
                "classifier": record["classifier_name"],
                "best_params": result["best_params"],
                "cv_score": float(result["cv_score"]),
                "base_seed": int(record["base_seed"]),
                "accuracy": float(result["accuracy"]),
                "roc_auc": float(result["roc_auc"]),
            }
            rows.append(row)
    return rows


def read_generalisation_csv_rows(run_root: Path) -> list[dict[str, object]]:
    rows = []
    for row in read_csv(run_root / "generalisation_results.csv"):
        rows.append(
            {
                "dataset": row["dataset"],
                "encoder": row["encoder"],
                "n_components": int(float(row["n_components"])),
                "projector": row["projector"],
                "projector_name": row["projector"],
                "projector_params": {},
                "classifier": row["classifier"],
                "best_params": row["best_params"],
                "cv_score": float(row["cv_score"]),
                "base_seed": int(float(row["base_seed"])),
                "accuracy": float(row["accuracy"]),
                "roc_auc": float(row["roc_auc"]),
            }
        )
    return rows


def load_generalisation_rows(run_root: Path) -> list[dict[str, object]]:
    manifest_rows = read_manifest_task_rows(run_root)
    if manifest_rows is not None:
        return manifest_rows
    return read_generalisation_csv_rows(run_root)


def aggregate_generalisation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values: dict[tuple[str, str, int, str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    params_by_key: dict[tuple[str, str, int, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    projector_params_by_key: dict[tuple[str, str, int, str, str, str, str], dict[str, object]] = {}

    for row in rows:
        projector_params = row.get("projector_params", {})
        projector_params_key = params_key(projector_params)
        key = (
            str(row["dataset"]),
            str(row["encoder"]),
            int(row["n_components"]),
            str(row["projector"]),
            str(row["projector_name"]),
            projector_params_key,
            str(row["classifier"]),
        )
        values[key]["accuracy"].append(float(row["accuracy"]))  # type: ignore[arg-type]
        values[key]["roc_auc"].append(float(row["roc_auc"]))  # type: ignore[arg-type]
        values[key]["cv_score"].append(float(row["cv_score"]))  # type: ignore[arg-type]
        params_by_key[key].append(parse_params(str(row["best_params"])))
        projector_params_by_key[key] = projector_params

    aggregated = []
    for (
        dataset,
        encoder,
        dim,
        projector,
        projector_name,
        projector_params_key,
        classifier,
    ), metric_values in values.items():
        param_counts = Counter(
            params_key(params)
            for params in params_by_key[
                (
                    dataset,
                    encoder,
                    dim,
                    projector,
                    projector_name,
                    projector_params_key,
                    classifier,
                )
            ]
        )
        modal_key, modal_count = param_counts.most_common(1)[0]
        modal_params = json.loads(modal_key)
        n_seeds = len(metric_values["accuracy"])
        projector_params = projector_params_by_key[
            (
                dataset,
                encoder,
                dim,
                projector,
                projector_name,
                projector_params_key,
                classifier,
            )
        ]
        aggregated.append(
            {
                "dataset": dataset,
                "encoder": encoder,
                "n_components": dim,
                "projector": projector,
                "projector_name": projector_name,
                "projector_params": projector_params,
                "projector_params_key": projector_params_key,
                "projector_params_short": projector_params_display(
                    projector_params, projector_name
                ),
                "classifier": classifier,
                "classifier_label": CLASSIFIER_LABELS.get(classifier, classifier),
                "accuracy_mean": mean(metric_values["accuracy"]),
                "accuracy_std": sample_std(metric_values["accuracy"]),
                "roc_auc_mean": mean(metric_values["roc_auc"]),
                "roc_auc_std": sample_std(metric_values["roc_auc"]),
                "cv_score_mean": mean(metric_values["cv_score"]),
                "n_seeds": n_seeds,
                "modal_params": modal_params,
                "modal_params_short": params_display(modal_params, classifier),
                "modal_params_count": modal_count,
                "modal_params_frequency": modal_count / n_seeds if n_seeds else float("nan"),
                "distinct_params": len(param_counts),
            }
        )
    return aggregated


def best_full_baselines(
    aggregated: list[dict[str, object]]
) -> dict[tuple[str, str, str], dict[str, object]]:
    baselines = {}
    for metric, _metric_label in METRICS:
        metric_col = f"{metric}_mean"
        other_col = "roc_auc_mean" if metric == "accuracy" else "accuracy_mean"
        for dataset, _dataset_label in DATASETS:
            for encoder, _encoder_label in ENCODERS:
                candidates = [
                    row
                    for row in aggregated
                    if row["dataset"] == dataset
                    and row["encoder"] == encoder
                    and row["projector"] == "none"
                ]
                if not candidates:
                    continue
                baselines[(metric, dataset, encoder)] = max(
                    candidates,
                    key=lambda row: (
                        float(row[metric_col]),
                        float(row[other_col]),
                    ),
                )
    return baselines


def build_appendix_rows(
    aggregated: list[dict[str, object]],
    threshold_dims: dict[tuple[str, str, str, str, str], int | None],
    full_baselines: dict[tuple[str, str, str], dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in aggregated:
        grouped[
            (
                str(row["dataset"]),
                str(row["encoder"]),
                int(row["n_components"]),
                str(row["projector"]),
                str(row["classifier"]),
            )
        ].append(row)
    lookup: dict[tuple[str, str, int, str, str], list[dict[str, object]]] = dict(grouped)
    _legacy_lookup: dict[tuple[str, str, int, str, str], dict[str, object]] = {
        (
            str(row["dataset"]),
            str(row["encoder"]),
            int(row["n_components"]),
            str(row["projector"]),
            str(row["classifier"]),
        ): row
        for row in aggregated
    }
    rows = []
    for metric, metric_label in METRICS:
        metric_col = f"{metric}_mean"
        for condition in CONDITIONS:
            for dataset, dataset_label in DATASETS:
                for encoder, encoder_label in ENCODERS:
                    full = full_baselines.get((metric, dataset, encoder))
                    full_value = float(full[metric_col]) if full else float("nan")
                    for projector, projector_label in PROJECTORS:
                        dim = threshold_dims.get((metric, condition, dataset, encoder, projector))
                        if dim is None:
                            continue
                        for classifier in sorted(CLASSIFIER_LABELS):
                            items = lookup.get((dataset, encoder, int(dim), projector, classifier), [])
                            if not items:
                                continue
                            for item in items:
                                rows.append(
                                    {
                                        "selection_metric": metric,
                                        "selection_metric_label": metric_label,
                                        "threshold": condition,
                                        "dataset": dataset,
                                        "dataset_label": dataset_label,
                                        "encoder": encoder,
                                        "encoder_label": encoder_label,
                                        "projector": projector,
                                        "projector_label": projector_label,
                                        "projector_params_short": item[
                                            "projector_params_short"
                                        ],
                                        "projector_params_json": params_key(
                                            item["projector_params"]
                                        ),
                                        "n_components": int(dim),
                                        "classifier": classifier,
                                        "classifier_label": item["classifier_label"],
                                        "modal_params_short": item["modal_params_short"],
                                        "modal_params_json": params_key(item["modal_params"]),
                                        "modal_params_count": item["modal_params_count"],
                                        "modal_params_frequency": item[
                                            "modal_params_frequency"
                                        ],
                                        "distinct_params": item["distinct_params"],
                                        "n_seeds": item["n_seeds"],
                                        "accuracy_mean": item["accuracy_mean"],
                                        "accuracy_std": item["accuracy_std"],
                                        "roc_auc_mean": item["roc_auc_mean"],
                                        "roc_auc_std": item["roc_auc_std"],
                                        "cv_score_mean": item["cv_score_mean"],
                                        "full_baseline": full_value,
                                        "recovery": float(item[metric_col]) / full_value
                                        if full_value
                                        else float("nan"),
                                        "full_classifier": full["classifier_label"]
                                        if full
                                        else MISSING,
                                        "full_params_short": full["modal_params_short"]
                                        if full
                                        else MISSING,
                                    }
                                )
    return rows


def choose_best_rows(appendix_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for metric, _metric_label in METRICS:
        metric_col = f"{metric}_mean"
        other_col = "roc_auc_mean" if metric == "accuracy" else "accuracy_mean"
        for condition in CONDITIONS:
            for dataset, _dataset_label in DATASETS:
                candidates = [
                    row
                    for row in appendix_rows
                    if row["selection_metric"] == metric
                    and row["threshold"] == condition
                    and row["dataset"] == dataset
                ]
                if not candidates:
                    continue
                output.append(
                    max(
                        candidates,
                        key=lambda row: (
                            float(row[metric_col]),
                            float(row["recovery"]),
                            float(row[other_col]),
                            -int(row["n_components"]),
                        ),
                    )
                )
    return output


def write_appendix_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "selection_metric",
        "threshold",
        "dataset",
        "encoder",
        "projector",
        "projector_params",
        "n_components",
        "classifier",
        "modal_params_short",
        "modal_params_count",
        "modal_params_frequency",
        "distinct_params",
        "n_seeds",
        "accuracy_mean",
        "accuracy_std",
        "roc_auc_mean",
        "roc_auc_std",
        "cv_score_mean",
        "full_baseline",
        "recovery",
        "full_classifier",
        "full_params_short",
        "modal_params_json",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "selection_metric": row["selection_metric"],
                    "threshold": row["threshold"],
                    "dataset": row["dataset_label"],
                    "encoder": row["encoder_label"],
                    "projector": row["projector_label"],
                    "projector_params": row["projector_params_short"],
                    "n_components": row["n_components"],
                    "classifier": row["classifier_label"],
                    "modal_params_short": row["modal_params_short"],
                    "modal_params_count": row["modal_params_count"],
                    "modal_params_frequency": fmt_num(float(row["modal_params_frequency"])),
                    "distinct_params": row["distinct_params"],
                    "n_seeds": row["n_seeds"],
                    "accuracy_mean": fmt_num(float(row["accuracy_mean"])),
                    "accuracy_std": fmt_num(float(row["accuracy_std"])),
                    "roc_auc_mean": fmt_num(float(row["roc_auc_mean"])),
                    "roc_auc_std": fmt_num(float(row["roc_auc_std"])),
                    "cv_score_mean": fmt_num(float(row["cv_score_mean"])),
                    "full_baseline": fmt_num(float(row["full_baseline"])),
                    "recovery": fmt_num(float(row["recovery"])),
                    "full_classifier": row["full_classifier"],
                    "full_params_short": row["full_params_short"],
                    "modal_params_json": row["modal_params_json"],
                }
            )


def write_best_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "metric",
        "threshold",
        "dataset",
        "encoder",
        "projector",
        "projector_params",
        "n_components",
        "classifier",
        "modal_params_short",
        "params_stability",
        "accuracy",
        "roc_auc",
        "full_baseline",
        "recovery",
        "full_classifier",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "metric": row["selection_metric_label"],
                    "threshold": f"{float(row['threshold']) * 100:.0f}%",
                    "dataset": row["dataset_label"],
                    "encoder": row["encoder_label"],
                    "projector": row["projector_label"],
                    "projector_params": row["projector_params_short"],
                    "n_components": row["n_components"],
                    "classifier": row["classifier_label"],
                    "modal_params_short": row["modal_params_short"],
                    "params_stability": f"{row['modal_params_count']}/{row['n_seeds']}",
                    "accuracy": f"{fmt_num(float(row['accuracy_mean']))} ({fmt_num(float(row['accuracy_std']))})",
                    "roc_auc": f"{fmt_num(float(row['roc_auc_mean']))} ({fmt_num(float(row['roc_auc_std']))})",
                    "full_baseline": fmt_num(float(row["full_baseline"])),
                    "recovery": f"{float(row['recovery']) * 100:.1f}%",
                    "full_classifier": row["full_classifier"],
                }
            )


def write_best_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[!htpb]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lllrlp{3.2cm}p{3.8cm}rrr}",
        r"\toprule",
        r"Metric & Dataset & Enc.+proj. & Dim. & Clf. & Projector params & Classifier hyperparameters & Acc. & ROC-AUC & Recovery \\",
        r"\midrule",
    ]
    for metric, metric_label in METRICS:
        lines.append(rf"\multicolumn{{10}}{{l}}{{\textbf{{{metric_label}}}}} \\")
        for condition in CONDITIONS:
            lines.append(
                rf"\multicolumn{{10}}{{l}}{{{float(condition) * 100:.0f}\% threshold dimension}} \\"
            )
            for row in rows:
                if row["selection_metric"] != metric or row["threshold"] != condition:
                    continue
                lines.append(
                    " & ".join(
                        [
                            metric_label,
                            latex_escape(row["dataset_label"]),
                            latex_escape(
                                f"{row['encoder_label']} + {row['projector_label']}"
                            ),
                            str(row["n_components"]),
                            latex_escape(row["classifier_label"]),
                            latex_escape(row["projector_params_short"] or MISSING),
                            latex_escape(
                                f"{row['modal_params_short']} ({row['modal_params_count']}/{row['n_seeds']})"
                            ),
                            fmt_pm(float(row["accuracy_mean"]), float(row["accuracy_std"])),
                            fmt_pm(float(row["roc_auc_mean"]), float(row["roc_auc_std"])),
                            fmt_recovery(float(row["recovery"])),
                        ]
                    )
                    + r" \\"
                )
            lines.append(r"\addlinespace[2pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\caption{Best held-out generalisation result for each dataset, "
                r"metric, and threshold-defined dimension. Projector parameters are "
                r"shown where applicable. Classifier hyperparameters are the modal "
                r"cross-validation-selected setting across the five evaluation seeds, "
                r"with the stability count shown in parentheses. Recovery is measured "
                r"relative to the best full-dimensional baseline for the same dataset "
                r"and encoder.}"
            ),
            r"\label{tab:generalisation-threshold-validation-best}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def build_hyperparameter_summary(appendix_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[str, str]] = Counter()
    for row in appendix_rows:
        params = json.loads(str(row["modal_params_json"]))
        classifier = str(row["classifier"])
        for key, value in params.items():
            value_text = json.dumps(value, sort_keys=True, default=str)
            counts[(classifier, key)][value_text] += 1
            totals[(classifier, key)] += 1

    summary = []
    for (classifier, param), counter in sorted(counts.items()):
        total = totals[(classifier, param)]
        top_value, top_count = counter.most_common(1)[0]
        summary.append(
            {
                "classifier": classifier,
                "classifier_label": CLASSIFIER_LABELS.get(classifier, classifier),
                "parameter": param,
                "most_common_value": top_value,
                "count": top_count,
                "total": total,
                "frequency": top_count / total if total else float("nan"),
                "distinct_values": len(counter),
            }
        )
    return summary


def write_hyperparameter_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "classifier",
        "parameter",
        "most_common_value",
        "count",
        "total",
        "frequency",
        "distinct_values",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "classifier": row["classifier_label"],
                    "parameter": row["parameter"],
                    "most_common_value": row["most_common_value"],
                    "count": row["count"],
                    "total": row["total"],
                    "frequency": fmt_num(float(row["frequency"])),
                    "distinct_values": row["distinct_values"],
                }
            )


def write_hyperparameter_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[!htpb]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Classifier & Parameter & Most common value & Frequency & Distinct values \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(row["classifier_label"]),
                    latex_escape(row["parameter"]),
                    latex_escape(row["most_common_value"]),
                    f"{row['count']}/{row['total']}",
                    str(row["distinct_values"]),
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
                r"\caption{Most common cross-validation-selected hyperparameter "
                r"values among the exact threshold-dimension generalisation results.}"
            ),
            r"\label{tab:generalisation-hyperparameter-selection}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_paragraph(path: Path, best_rows: list[dict[str, object]]) -> None:
    recovery_values = [float(row["recovery"]) for row in best_rows if math.isfinite(float(row["recovery"]))]
    acc_values = [float(row["accuracy_mean"]) for row in best_rows]
    roc_values = [float(row["roc_auc_mean"]) for row in best_rows]
    classifier_counts = Counter(str(row["classifier_label"]) for row in best_rows)
    projector_counts = Counter(str(row["projector_label"]) for row in best_rows)
    paragraph = (
        "Across the best held-out threshold-validation configurations, mean recovery "
        f"relative to the matched full-dimensional baseline was {mean(recovery_values) * 100:.1f}% "
        f"(SD {sample_std(recovery_values) * 100:.1f}%). The selected configurations "
        f"had mean accuracy {mean(acc_values):.4f} (SD {sample_std(acc_values):.4f}) "
        f"and mean ROC-AUC {mean(roc_values):.4f} (SD {sample_std(roc_values):.4f}). "
        "The most frequently selected classifiers were "
        + ", ".join(f"{name} ({count})" for name, count in classifier_counts.most_common())
        + ", and the most frequently selected projectors were "
        + ", ".join(f"{name} ({count})" for name, count in projector_counts.most_common())
        + "."
    )
    path.write_text(paragraph + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_root / "tables" / "generalisation_thesis_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_rows = read_csv(args.run_root / "curve" / "thresholds_results.csv")
    generalisation_rows = load_generalisation_rows(args.run_root)

    curve_baselines = build_curve_baselines(threshold_rows, args.classifier)
    curve_means = build_curve_means(args.run_root / "curve", args.classifier)
    threshold_dims = build_threshold_dims(curve_baselines, curve_means)
    aggregated = aggregate_generalisation_rows(generalisation_rows)
    full_baselines = best_full_baselines(aggregated)
    appendix_rows = build_appendix_rows(aggregated, threshold_dims, full_baselines)
    best_rows = choose_best_rows(appendix_rows)
    hyperparameter_rows = build_hyperparameter_summary(appendix_rows)

    write_appendix_csv(output_dir / "generalisation_threshold_validation_full_appendix.csv", appendix_rows)
    write_best_csv(output_dir / "generalisation_threshold_validation_best.csv", best_rows)
    write_best_tex(output_dir / "generalisation_threshold_validation_best.tex", best_rows)
    write_hyperparameter_csv(output_dir / "generalisation_hyperparameter_selection_summary.csv", hyperparameter_rows)
    write_hyperparameter_tex(output_dir / "generalisation_hyperparameter_selection_summary.tex", hyperparameter_rows)
    write_paragraph(output_dir / "generalisation_threshold_validation_paragraph.txt", best_rows)

    print(output_dir / "generalisation_threshold_validation_best.csv")
    print(output_dir / "generalisation_threshold_validation_best.tex")
    print(output_dir / "generalisation_threshold_validation_full_appendix.csv")
    print(output_dir / "generalisation_hyperparameter_selection_summary.csv")
    print(output_dir / "generalisation_hyperparameter_selection_summary.tex")
    print(output_dir / "generalisation_threshold_validation_paragraph.txt")


if __name__ == "__main__":
    main()
