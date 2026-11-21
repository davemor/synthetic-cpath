#!/usr/bin/env python3
"""Create interpretation tables for hproj threshold generalisation results."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarise_generalisation_for_thesis as thesis


CLASSIFIERS = ["logistic-regression", "mlp", "xgboost"]
CLASSIFIER_LABELS = thesis.CLASSIFIER_LABELS
MISSING = thesis.MISSING


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run-root>/tables/generalisation_interpretation.",
    )
    parser.add_argument(
        "--curve-classifier",
        default="logistic-regression",
        help="Curve classifier that defined the threshold dimensions.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt_num(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return MISSING
    return f"{value:.{digits}f}"


def fmt_num4(value: float) -> str:
    return thesis.fmt_num(value, 4)


def fmt_pm_text(mean_value: float, std_value: float) -> str:
    if not math.isfinite(mean_value):
        return MISSING
    return f"{mean_value:.3f} ({std_value:.3f})"


def fmt_cell(item: dict[str, object] | None) -> str:
    if item is None:
        return MISSING
    return (
        r"\shortstack{"
        + f"A {fmt_num(float(item['accuracy_mean']))} ({fmt_num(float(item['accuracy_std']))})"
        + r"\\"
        + f"R {fmt_num(float(item['roc_auc_mean']))} ({fmt_num(float(item['roc_auc_std']))})"
        + "}"
    )


def latex_escape(value: object) -> str:
    return thesis.latex_escape(value)


def condition_percent(condition: str) -> str:
    return f"{float(condition) * 100:.0f}\\%"


def build_intrinsic_lookup(run_root: Path) -> dict[tuple[str, str], str]:
    rows = read_csv(run_root / "intrinsic_dims.csv")
    lookup = {}
    for row in rows:
        value = float(row["intrinsic_dim_mle"])
        lookup[(row["dataset"], row["encoder"])] = str(int(round(value)))
    return lookup


def selection_key(item: dict[str, object], metric: str) -> tuple[float, float, int]:
    other_metric = "roc_auc" if metric == "accuracy" else "accuracy"
    return (
        float(item[f"{metric}_mean"]),
        float(item[f"{other_metric}_mean"]),
        -int(item["n_components"]),
    )


def build_aggregated_lookup(
    aggregated: list[dict[str, object]],
) -> dict[tuple[str, str, int, str, str], list[dict[str, object]]]:
    lookup: dict[tuple[str, str, int, str, str], list[dict[str, object]]] = defaultdict(list)
    for item in aggregated:
        lookup[
            (
                str(item["dataset"]),
                str(item["encoder"]),
                int(item["n_components"]),
                str(item["projector"]),
                str(item["classifier"]),
            )
        ].append(item)
    return lookup


def best_for_classifier(
    lookup: dict[tuple[str, str, int, str, str], list[dict[str, object]]],
    dataset: str,
    encoder: str,
    projector: str,
    dim: int,
    classifier: str,
    metric: str,
) -> dict[str, object] | None:
    candidates = lookup.get((dataset, encoder, dim, projector, classifier), [])
    if not candidates:
        return None
    return max(candidates, key=lambda item: selection_key(item, metric))


def threshold_panel_keys(
    threshold_dims: dict[tuple[str, str, str, str, str], int | None],
) -> list[tuple[str, str, str, str, str, int]]:
    keys = []
    for metric, _metric_label in thesis.METRICS:
        for condition in thesis.CONDITIONS:
            for dataset, _dataset_label in thesis.DATASETS:
                for encoder, _encoder_label in thesis.ENCODERS:
                    for projector, _projector_label in thesis.PROJECTORS:
                        dim = threshold_dims.get((metric, condition, dataset, encoder, projector))
                        if dim is not None:
                            keys.append((metric, condition, dataset, encoder, projector, int(dim)))
    return keys


def write_classifier_threshold_csvs(
    output_dir: Path,
    threshold_dims: dict[tuple[str, str, str, str, str], int | None],
    lookup: dict[tuple[str, str, int, str, str], list[dict[str, object]]],
) -> dict[str, dict[tuple[str, str, str, str, str], dict[str, object] | None]]:
    fields = [
        "classifier",
        "selection_metric",
        "threshold",
        "dataset",
        "encoder",
        "projector",
        "n_components",
        "projector_params",
        "classifier_params",
        "params_stability",
        "accuracy_mean",
        "accuracy_std",
        "roc_auc_mean",
        "roc_auc_std",
    ]
    best_by_classifier = {}
    for classifier in CLASSIFIERS:
        rows = []
        best_map = {}
        for metric, condition, dataset, encoder, projector, dim in threshold_panel_keys(threshold_dims):
            best = best_for_classifier(lookup, dataset, encoder, projector, dim, classifier, metric)
            best_map[(metric, condition, dataset, encoder, projector)] = best
            if best is None:
                continue
            rows.append(
                {
                    "classifier": CLASSIFIER_LABELS[classifier],
                    "selection_metric": metric,
                    "threshold": condition,
                    "dataset": dataset,
                    "encoder": encoder,
                    "projector": projector,
                    "n_components": dim,
                    "projector_params": best.get("projector_params_short", ""),
                    "classifier_params": best["modal_params_short"],
                    "params_stability": f"{best['modal_params_count']}/{best['n_seeds']}",
                    "accuracy_mean": fmt_num4(float(best["accuracy_mean"])),
                    "accuracy_std": fmt_num4(float(best["accuracy_std"])),
                    "roc_auc_mean": fmt_num4(float(best["roc_auc_mean"])),
                    "roc_auc_std": fmt_num4(float(best["roc_auc_std"])),
                }
            )
        best_by_classifier[classifier] = best_map
        slug = classifier.replace("-", "_")
        write_csv(output_dir / f"{slug}_threshold_values.csv", rows, fields)
    return best_by_classifier


def latex_encoder_header(num_cols: int) -> list[str]:
    midrules = " ".join(
        rf"\cmidrule(lr){{{start}-{start + len(thesis.PROJECTORS)}}}"
        for start in range(2, num_cols + 1, len(thesis.PROJECTORS) + 1)
    )
    group_header = [""]
    for _encoder_key, encoder_label in thesis.ENCODERS:
        group_header.append(rf"\multicolumn{{4}}{{c}}{{{latex_escape(encoder_label)}}}")
    subheader = ["Dataset"]
    for _encoder_key, _encoder_label in thesis.ENCODERS:
        subheader.extend([r"$d_\text{ID}$", "P", "U", "G"])
    return [
        " & ".join(group_header) + r" \\",
        midrules,
        " & ".join(subheader) + r" \\",
    ]


def write_classifier_threshold_tex(
    path: Path,
    classifier: str,
    best_map: dict[tuple[str, str, str, str, str], dict[str, object] | None],
    threshold_dims: dict[tuple[str, str, str, str, str], int | None],
    intrinsic: dict[tuple[str, str], str],
) -> None:
    num_cols = 1 + len(thesis.ENCODERS) * (1 + len(thesis.PROJECTORS))
    lines = [
        r"\begin{table}[!htpb]",
        r"\centering",
        r"\tiny",
        r"\setlength{\tabcolsep}{1.7pt}",
        r"\renewcommand{\arraystretch}{0.9}",
        rf"\begin{{tabular}}{{l{'c' * (num_cols - 1)}}}",
        r"\toprule",
        *latex_encoder_header(num_cols),
        r"\midrule",
    ]
    for metric, metric_label in thesis.METRICS:
        for condition in thesis.CONDITIONS:
            lines.append(
                rf"\multicolumn{{{num_cols}}}{{l}}{{\textbf{{{condition_percent(condition)} {metric_label} threshold}}}} \\"
            )
            for dataset, dataset_label in thesis.DATASETS:
                row = [dataset_label]
                for encoder, _encoder_label in thesis.ENCODERS:
                    row.append(intrinsic.get((dataset, encoder), MISSING))
                    for projector, _projector_label in thesis.PROJECTORS:
                        dim = threshold_dims.get((metric, condition, dataset, encoder, projector))
                        item = None
                        if dim is not None:
                            item = best_map.get((metric, condition, dataset, encoder, projector))
                        row.append(fmt_cell(item))
                lines.append(" & ".join(latex_escape(cell) if cell != MISSING and not str(cell).startswith(r"\shortstack") else str(cell) for cell in row) + r" \\")
            lines.append(r"\addlinespace[2pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                rf"\caption{{Held-out {CLASSIFIER_LABELS[classifier]} performance at "
                r"threshold-defined dimensions. Each cell reports Accuracy and "
                r"ROC-AUC as mean (SD) across evaluation seeds. P: PCA. U: UMAP. "
                r"G: GRP. ``---'': threshold not reached or not evaluated.}"
            ),
            rf"\label{{tab:{classifier.replace('-', '-')}-threshold-values}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_appendix_tex(
    path: Path,
    best_by_classifier: dict[str, dict[tuple[str, str, str, str, str], dict[str, object] | None]],
    threshold_dims: dict[tuple[str, str, str, str, str], int | None],
    intrinsic: dict[tuple[str, str], str],
) -> None:
    parts = []
    for classifier in CLASSIFIERS:
        tmp = path.with_name(f"_{classifier}_tmp.tex")
        write_classifier_threshold_tex(tmp, classifier, best_by_classifier[classifier], threshold_dims, intrinsic)
        parts.append(tmp.read_text())
        tmp.unlink()
    path.write_text("\n".join(parts))


def build_classifier_variation(
    threshold_dims: dict[tuple[str, str, str, str, str], int | None],
    best_by_classifier: dict[str, dict[tuple[str, str, str, str, str], dict[str, object] | None]],
    full_baselines: dict[tuple[str, str, str], dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_classifier: dict[str, dict[str, list[float]]] = {
        classifier: defaultdict(list) for classifier in CLASSIFIERS
    }
    wins = Counter()
    best_rows = []
    for metric, condition, dataset, encoder, projector, dim in threshold_panel_keys(threshold_dims):
        candidates = []
        for classifier in CLASSIFIERS:
            item = best_by_classifier[classifier].get((metric, condition, dataset, encoder, projector))
            if item is None:
                continue
            full = full_baselines.get((metric, dataset, encoder))
            full_value = float(full[f"{metric}_mean"]) if full else float("nan")
            recovery = float(item[f"{metric}_mean"]) / full_value if full_value else float("nan")
            per_classifier[classifier]["accuracy"].append(float(item["accuracy_mean"]))
            per_classifier[classifier]["roc_auc"].append(float(item["roc_auc_mean"]))
            per_classifier[classifier]["recovery"].append(recovery)
            candidates.append((classifier, item, recovery))
        if candidates:
            classifier, item, recovery = max(
                candidates,
                key=lambda entry: (
                    float(entry[1][f"{metric}_mean"]),
                    float(entry[2]),
                    float(entry[1]["roc_auc_mean" if metric == "accuracy" else "accuracy_mean"]),
                ),
            )
            wins[classifier] += 1
            best_rows.append(
                {
                    "selection_metric": metric,
                    "threshold": condition,
                    "dataset": dataset,
                    "encoder": encoder,
                    "projector": projector,
                    "n_components": dim,
                    "best_classifier": CLASSIFIER_LABELS[classifier],
                    "projector_params": item.get("projector_params_short", ""),
                    "classifier_params": item["modal_params_short"],
                    "params_stability": f"{item['modal_params_count']}/{item['n_seeds']}",
                    "accuracy_mean": fmt_num4(float(item["accuracy_mean"])),
                    "accuracy_std": fmt_num4(float(item["accuracy_std"])),
                    "roc_auc_mean": fmt_num4(float(item["roc_auc_mean"])),
                    "roc_auc_std": fmt_num4(float(item["roc_auc_std"])),
                    "recovery": fmt_num4(recovery),
                }
            )
    summary = []
    for classifier in CLASSIFIERS:
        values = per_classifier[classifier]
        summary.append(
            {
                "classifier": CLASSIFIER_LABELS[classifier],
                "n": len(values["accuracy"]),
                "accuracy_mean": fmt_num4(thesis.mean(values["accuracy"])),
                "accuracy_sd": fmt_num4(thesis.sample_std(values["accuracy"])),
                "roc_auc_mean": fmt_num4(thesis.mean(values["roc_auc"])),
                "roc_auc_sd": fmt_num4(thesis.sample_std(values["roc_auc"])),
                "recovery_mean": fmt_num4(thesis.mean(values["recovery"])),
                "recovery_sd": fmt_num4(thesis.sample_std(values["recovery"])),
                "wins": wins[classifier],
            }
        )
    return summary, best_rows


def write_classifier_variation_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[!htpb]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Classifier & Accuracy & ROC-AUC & Recovery & Wins \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(row["classifier"]),
                    f"{row['accuracy_mean']} ({row['accuracy_sd']})",
                    f"{row['roc_auc_mean']} ({row['roc_auc_sd']})",
                    f"{float(row['recovery_mean']) * 100:.1f}\\% ({float(row['recovery_sd']) * 100:.1f}\\%)",
                    str(row["wins"]),
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
                r"\caption{Classifier variation across exact threshold-dimension "
                r"evaluations. Values are mean (SD) across threshold tuples after "
                r"selecting the best hyperparameters within each classifier. Wins "
                r"count the number of threshold tuples where the classifier achieved "
                r"the highest score for the threshold-selection metric.}"
            ),
            r"\label{tab:classifier-variation-summary}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def selected_c(params_text: str) -> str:
    try:
        params = ast.literal_eval(params_text)
    except Exception:
        return MISSING
    return str(params.get("C", MISSING))


def build_lr_sensitivity(
    raw_rows: list[dict[str, object]],
    threshold_dims: dict[tuple[str, str, str, str, str], int | None],
) -> list[dict[str, object]]:
    threshold_keys = set(threshold_panel_keys(threshold_dims))
    grouped: dict[tuple[str, str, str], dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in raw_rows:
        if row["classifier"] != "logistic-regression":
            continue
        for metric, condition, dataset, encoder, projector, dim in threshold_keys:
            if (
                row["dataset"] == dataset
                and row["encoder"] == encoder
                and row["projector"] == projector
                and int(row["n_components"]) == dim
            ):
                c_value = selected_c(str(row["best_params"]))
                grouped[(metric, condition, c_value)]["items"].append(
                    {
                        "accuracy": float(row["accuracy"]),
                        "roc_auc": float(row["roc_auc"]),
                    }
                )
    output = []
    for metric, condition in [(m, c) for m, _ in thesis.METRICS for c in thesis.CONDITIONS]:
        by_c = {
            key[2]: value["items"]
            for key, value in grouped.items()
            if key[0] == metric and key[1] == condition
        }
        for c_value in sorted(by_c, key=lambda value: float(value) if value != MISSING else 999):
            items = by_c[c_value]
            acc = [item["accuracy"] for item in items]
            roc = [item["roc_auc"] for item in items]
            output.append(
                {
                    "selection_metric": metric,
                    "threshold": condition,
                    "C": c_value,
                    "n": len(items),
                    "accuracy_mean": fmt_num4(thesis.mean(acc)),
                    "accuracy_sd": fmt_num4(thesis.sample_std(acc)),
                    "roc_auc_mean": fmt_num4(thesis.mean(roc)),
                    "roc_auc_sd": fmt_num4(thesis.sample_std(roc)),
                }
            )
    return output


def write_lr_sensitivity_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[!htpb]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Selection metric & Threshold & $C$ & $n$ & Accuracy & ROC-AUC \\",
        r"\midrule",
    ]
    for row in rows:
        metric_label = dict(thesis.METRICS)[str(row["selection_metric"])]
        lines.append(
            " & ".join(
                [
                    metric_label,
                    f"{float(row['threshold']) * 100:.0f}\\%",
                    str(row["C"]),
                    str(row["n"]),
                    f"{row['accuracy_mean']} ({row['accuracy_sd']})",
                    f"{row['roc_auc_mean']} ({row['roc_auc_sd']})",
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
                r"\caption{Observed logistic-regression performance grouped by the "
                r"cross-validation-selected regularisation parameter. This is an "
                r"observational sensitivity summary of selected $C$ values, not a "
                r"paired refit of every $C$ value for every seed.}"
            ),
            r"\label{tab:lr-hyperparameter-sensitivity}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def build_curve_vs_generalised_lr(
    curve_means: dict[tuple[str, str, str, str, int], float],
    threshold_dims: dict[tuple[str, str, str, str, str], int | None],
    best_lr: dict[tuple[str, str, str, str, str], dict[str, object] | None],
) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    rows = []
    for metric, condition, dataset, encoder, projector, dim in threshold_panel_keys(threshold_dims):
        item = best_lr.get((metric, condition, dataset, encoder, projector))
        if item is None:
            continue
        curve_acc = curve_means.get(("accuracy", dataset, encoder, projector, dim), float("nan"))
        curve_roc = curve_means.get(("roc_auc", dataset, encoder, projector, dim), float("nan"))
        gen_acc = float(item["accuracy_mean"])
        gen_roc = float(item["roc_auc_mean"])
        rows.append(
            {
                "selection_metric": metric,
                "threshold": condition,
                "dataset": dataset,
                "encoder": encoder,
                "projector": projector,
                "n_components": dim,
                "curve_accuracy": curve_acc,
                "generalised_accuracy": gen_acc,
                "delta_accuracy": gen_acc - curve_acc,
                "curve_roc_auc": curve_roc,
                "generalised_roc_auc": gen_roc,
                "delta_roc_auc": gen_roc - curve_roc,
            }
        )
    summary = []
    for metric, condition in [(m, c) for m, _ in thesis.METRICS for c in thesis.CONDITIONS]:
        subset = [row for row in rows if row["selection_metric"] == metric and row["threshold"] == condition]
        acc_delta = [float(row["delta_accuracy"]) for row in subset]
        roc_delta = [float(row["delta_roc_auc"]) for row in subset]
        summary.append(
            {
                "selection_metric": metric,
                "threshold": condition,
                "n": len(subset),
                "delta_accuracy_mean": fmt_num4(thesis.mean(acc_delta)),
                "delta_accuracy_sd": fmt_num4(thesis.sample_std(acc_delta)),
                "delta_roc_auc_mean": fmt_num4(thesis.mean(roc_delta)),
                "delta_roc_auc_sd": fmt_num4(thesis.sample_std(roc_delta)),
            }
        )
    all_acc = [float(row["delta_accuracy"]) for row in rows]
    all_roc = [float(row["delta_roc_auc"]) for row in rows]
    paragraph = (
        "Comparing the curve-stage logistic-regression scores with the held-out "
        "generalisation logistic-regression scores at the same threshold dimensions, "
        f"the mean accuracy change was {thesis.mean(all_acc):.4f} "
        f"(SD {thesis.sample_std(all_acc):.4f}) and the mean ROC-AUC change was "
        f"{thesis.mean(all_roc):.4f} (SD {thesis.sample_std(all_roc):.4f}). "
        "Positive values indicate that the held-out generalisation evaluation was "
        "higher than the curve estimate; negative values indicate lower held-out "
        "performance."
    )
    return rows, summary, paragraph


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_root / "tables" / "generalisation_interpretation"
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_rows = read_csv(args.run_root / "curve" / "thresholds_results.csv")
    curve_baselines = thesis.build_curve_baselines(threshold_rows, args.curve_classifier)
    curve_means = thesis.build_curve_means(args.run_root / "curve", args.curve_classifier)
    threshold_dims = thesis.build_threshold_dims(curve_baselines, curve_means)
    raw_rows = thesis.load_generalisation_rows(args.run_root)
    aggregated = thesis.aggregate_generalisation_rows(raw_rows)
    full_baselines = thesis.best_full_baselines(aggregated)
    lookup = build_aggregated_lookup(aggregated)
    intrinsic = build_intrinsic_lookup(args.run_root)

    best_by_classifier = write_classifier_threshold_csvs(output_dir, threshold_dims, lookup)

    for classifier in CLASSIFIERS:
        slug = classifier.replace("-", "_")
        write_classifier_threshold_tex(
            output_dir / f"{slug}_threshold_values.tex",
            classifier,
            best_by_classifier[classifier],
            threshold_dims,
            intrinsic,
        )
    write_appendix_tex(
        output_dir / "classifier_threshold_value_tables_appendix.tex",
        best_by_classifier,
        threshold_dims,
        intrinsic,
    )

    variation_rows, best_classifier_rows = build_classifier_variation(
        threshold_dims, best_by_classifier, full_baselines
    )
    write_csv(
        output_dir / "classifier_variation_summary.csv",
        variation_rows,
        [
            "classifier",
            "n",
            "accuracy_mean",
            "accuracy_sd",
            "roc_auc_mean",
            "roc_auc_sd",
            "recovery_mean",
            "recovery_sd",
            "wins",
        ],
    )
    write_classifier_variation_tex(output_dir / "classifier_variation_summary.tex", variation_rows)
    write_csv(
        output_dir / "best_classifier_by_tuple.csv",
        best_classifier_rows,
        [
            "selection_metric",
            "threshold",
            "dataset",
            "encoder",
            "projector",
            "n_components",
            "best_classifier",
            "projector_params",
            "classifier_params",
            "params_stability",
            "accuracy_mean",
            "accuracy_std",
            "roc_auc_mean",
            "roc_auc_std",
            "recovery",
        ],
    )

    lr_rows = build_lr_sensitivity(raw_rows, threshold_dims)
    write_csv(
        output_dir / "lr_hyperparameter_sensitivity.csv",
        lr_rows,
        [
            "selection_metric",
            "threshold",
            "C",
            "n",
            "accuracy_mean",
            "accuracy_sd",
            "roc_auc_mean",
            "roc_auc_sd",
        ],
    )
    write_lr_sensitivity_tex(output_dir / "lr_hyperparameter_sensitivity.tex", lr_rows)

    curve_rows, curve_summary, curve_paragraph = build_curve_vs_generalised_lr(
        curve_means, threshold_dims, best_by_classifier["logistic-regression"]
    )
    write_csv(
        output_dir / "curve_vs_generalised_lr_by_tuple.csv",
        [
            {
                **row,
                "curve_accuracy": fmt_num4(float(row["curve_accuracy"])),
                "generalised_accuracy": fmt_num4(float(row["generalised_accuracy"])),
                "delta_accuracy": fmt_num4(float(row["delta_accuracy"])),
                "curve_roc_auc": fmt_num4(float(row["curve_roc_auc"])),
                "generalised_roc_auc": fmt_num4(float(row["generalised_roc_auc"])),
                "delta_roc_auc": fmt_num4(float(row["delta_roc_auc"])),
            }
            for row in curve_rows
        ],
        [
            "selection_metric",
            "threshold",
            "dataset",
            "encoder",
            "projector",
            "n_components",
            "curve_accuracy",
            "generalised_accuracy",
            "delta_accuracy",
            "curve_roc_auc",
            "generalised_roc_auc",
            "delta_roc_auc",
        ],
    )
    write_csv(
        output_dir / "curve_vs_generalised_lr_summary.csv",
        curve_summary,
        [
            "selection_metric",
            "threshold",
            "n",
            "delta_accuracy_mean",
            "delta_accuracy_sd",
            "delta_roc_auc_mean",
            "delta_roc_auc_sd",
        ],
    )
    (output_dir / "curve_vs_generalised_lr_paragraph.txt").write_text(
        curve_paragraph + "\n"
    )

    outputs = [
        "logistic_regression_threshold_values.tex",
        "mlp_threshold_values.tex",
        "xgboost_threshold_values.tex",
        "classifier_threshold_value_tables_appendix.tex",
        "classifier_variation_summary.tex",
        "lr_hyperparameter_sensitivity.tex",
        "curve_vs_generalised_lr_paragraph.txt",
        "best_classifier_by_tuple.csv",
    ]
    for name in outputs:
        print(output_dir / name)


if __name__ == "__main__":
    main()
