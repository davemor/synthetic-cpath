#!/usr/bin/env python3
"""Create compact thesis summaries for hproj generalisation results."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd


DATASET_ORDER = ["Kather", "CRC", "Breast", "Skin", "Thorax", "iCAIRD"]
METRIC_ORDER = ["accuracy", "roc_auc"]
METRIC_LABELS = {"accuracy": "Accuracy", "roc_auc": "ROC-AUC"}
CLASSIFIER_ORDER = ["LR", "MLP", "XGB"]
THRESHOLD_ORDER = ["0.99", "0.95"]
PROJECTOR_ORDER = ["UMAP", "PCA", "GRP"]
MISSING = "---"


def parse_args() -> argparse.Namespace:
    run_root = Path(
        "/Users/dm236/Documents/Research/AllResults/Chapter5-dimensionality-reduction/"
        "results_with_icaird/hproj-2026-05-25_16-41-27-bb2a5203-icaird100k"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=run_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def metric_col(metric: str) -> str:
    return f"{metric}_mean"


def other_metric_col(metric: str) -> str:
    return "roc_auc_mean" if metric == "accuracy" else "accuracy_mean"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def fmt_num(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return MISSING
    return f"{value:.{digits}f}"


def fmt_pm(mean_value: float, std_value: float) -> str:
    return rf"{fmt_num(mean_value)} $\pm$ {fmt_num(std_value)}"


def fmt_pm_bold(mean_value: float, std_value: float, bold: bool = False) -> str:
    text = fmt_pm(mean_value, std_value)
    return rf"\textbf{{{text}}}" if bold else text


def fmt_percent(value: float) -> str:
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


def load_appendix(run_root: Path) -> pd.DataFrame:
    path = (
        run_root
        / "tables"
        / "generalisation_thesis_summary"
        / "generalisation_threshold_validation_full_appendix.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for col in [
        "threshold",
        "n_components",
        "accuracy_mean",
        "accuracy_std",
        "roc_auc_mean",
        "roc_auc_std",
        "full_baseline",
        "recovery",
    ]:
        df[col] = pd.to_numeric(df[col])
    return df


def sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset_order"] = out["dataset"].map({name: i for i, name in enumerate(DATASET_ORDER)})
    out["threshold_order"] = out["threshold"].astype(str).map(
        {name: i for i, name in enumerate(THRESHOLD_ORDER)}
    )
    if "classifier" in out:
        out["classifier_order"] = out["classifier"].map(
            {name: i for i, name in enumerate(CLASSIFIER_ORDER)}
        )
    if "projector" in out:
        out["projector_order"] = out["projector"].map(
            {name: i for i, name in enumerate(PROJECTOR_ORDER)}
        )
    return out


def choose_best(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = sort_frame(df)
    for metric in METRIC_ORDER:
        mcol = metric_col(metric)
        ocol = other_metric_col(metric)
        for threshold in [0.99, 0.95]:
            for dataset in DATASET_ORDER:
                candidates = ordered[
                    (ordered["selection_metric"] == metric)
                    & (ordered["threshold"].round(2) == threshold)
                    & (ordered["dataset"] == dataset)
                ].copy()
                if candidates.empty:
                    continue
                candidates = candidates.sort_values(
                    by=[mcol, "recovery", ocol, "n_components"],
                    ascending=[False, False, False, True],
                )
                rows.append(candidates.iloc[0].to_dict())
    return pd.DataFrame(rows)


def write_compact_best(best: pd.DataFrame, output_dir: Path) -> None:
    csv_rows = []
    for _, row in best.iterrows():
        metric = row["selection_metric"]
        csv_rows.append(
            {
                "metric": METRIC_LABELS[metric],
                "threshold": f"{int(round(float(row['threshold']) * 100))}%",
                "dataset": row["dataset"],
                "encoder_projector": f"{row['encoder']} + {row['projector']}",
                "dimension": int(row["n_components"]),
                "classifier": row["classifier"],
                "performance_mean": f"{float(row[metric_col(metric)]):.6f}",
                "performance_std": f"{float(row[f'{metric}_std']):.6f}",
                "full_baseline": f"{float(row['full_baseline']):.6f}",
                "recovery": f"{float(row['recovery']):.6f}",
                "accuracy_mean": f"{float(row['accuracy_mean']):.6f}",
                "roc_auc_mean": f"{float(row['roc_auc_mean']):.6f}",
            }
        )
    csv_path = output_dir / "generalisation_best_compact.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        r"\begin{table}[!htpb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lllrlccc}",
        r"\toprule",
        r"Metric & Threshold & Dataset & Dim. & Enc.+proj. & Clf. & Test perf. & Recovery \\",
        r"\midrule",
    ]
    for metric in METRIC_ORDER:
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textbf{{{METRIC_LABELS[metric]}}}}} \\")
        for threshold in [0.99, 0.95]:
            subset = best[
                (best["selection_metric"] == metric)
                & (best["threshold"].round(2) == threshold)
            ]
            for _, row in sort_frame(subset).sort_values("dataset_order").iterrows():
                perf = fmt_pm(float(row[metric_col(metric)]), float(row[f"{metric}_std"]))
                lines.append(
                    " & ".join(
                        [
                            METRIC_LABELS[metric],
                            f"{int(round(threshold * 100))}\\%",
                            latex_escape(row["dataset"]),
                            str(int(row["n_components"])),
                            latex_escape(f"{row['encoder']} + {row['projector']}"),
                            latex_escape(row["classifier"]),
                            perf,
                            fmt_percent(float(row["recovery"])),
                        ]
                    )
                    + r" \\"
                )
            lines.append(r"\addlinespace[1pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\caption{Best held-out generalisation configuration for each "
                r"dataset, metric, and curve-selected threshold. Test performance "
                r"is reported as mean $\pm$ SD over evaluation seeds after selecting "
                r"the best hyperparameters within each classifier family. Recovery "
                r"is relative to the matched full-dimensional baseline.}"
            ),
            r"\label{tab:generalisation-best-compact}",
            r"\end{table}",
            "",
        ]
    )
    (output_dir / "generalisation_best_compact.tex").write_text("\n".join(lines))


def classifier_robustness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped_cols = ["selection_metric", "threshold", "dataset", "classifier"]
    for key, group in df.groupby(grouped_cols, sort=False):
        metric, threshold, dataset, classifier = key
        mcol = metric_col(metric)
        best = group.sort_values(
            by=[mcol, "recovery", other_metric_col(metric), "n_components"],
            ascending=[False, False, False, True],
        ).iloc[0]
        rows.append(
            {
                "selection_metric": metric,
                "threshold": float(threshold),
                "dataset": dataset,
                "classifier": classifier,
                "performance": float(best[mcol]),
                "performance_std": float(best[f"{metric}_std"]),
                "recovery": float(best["recovery"]),
                "n_components": int(best["n_components"]),
                "encoder_projector": f"{best['encoder']} + {best['projector']}",
            }
        )
    per_classifier = pd.DataFrame(rows)

    out_rows = []
    for (metric, threshold, dataset), group in per_classifier.groupby(
        ["selection_metric", "threshold", "dataset"], sort=False
    ):
        values = {row["classifier"]: row for _, row in group.iterrows()}
        performances = [float(row["performance"]) for _, row in group.iterrows()]
        best_row = max(values.values(), key=lambda row: float(row["performance"]))
        out_rows.append(
            {
                "selection_metric": metric,
                "threshold": float(threshold),
                "dataset": dataset,
                "lr": values.get("LR", {}).get("performance", float("nan")),
                "lr_std": values.get("LR", {}).get("performance_std", float("nan")),
                "mlp": values.get("MLP", {}).get("performance", float("nan")),
                "mlp_std": values.get("MLP", {}).get("performance_std", float("nan")),
                "xgb": values.get("XGB", {}).get("performance", float("nan")),
                "xgb_std": values.get("XGB", {}).get("performance_std", float("nan")),
                "range": max(performances) - min(performances),
                "best_classifier": best_row["classifier"],
            }
        )
    return pd.DataFrame(out_rows)


def write_robustness(robust: pd.DataFrame, df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    csv_path = output_dir / "generalisation_classifier_robustness_by_dataset.csv"
    robust_out = robust.copy()
    robust_out["metric"] = robust_out["selection_metric"].map(METRIC_LABELS)
    robust_out["threshold_label"] = robust_out["threshold"].map(lambda v: f"{int(round(v * 100))}%")
    robust_out[
        [
            "metric",
            "threshold_label",
            "dataset",
            "lr",
            "lr_std",
            "mlp",
            "mlp_std",
            "xgb",
            "xgb_std",
            "range",
            "best_classifier",
        ]
    ].to_csv(csv_path, index=False)

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lllcccr}",
        r"\toprule",
        r"Metric & Threshold & Dataset & LR & MLP & XGB & Range \\",
        r"\midrule",
    ]
    ordered = sort_frame(robust)
    for metric in METRIC_ORDER:
        for threshold in [0.99, 0.95]:
            subset = ordered[
                (ordered["selection_metric"] == metric)
                & (ordered["threshold"].round(2) == threshold)
            ].sort_values("dataset_order")
            for row_index, (_, row) in enumerate(subset.iterrows()):
                metric_cell = (
                    rf"\multirow{{{len(subset)}}}{{*}}{{{METRIC_LABELS[metric]}}}"
                    if row_index == 0
                    else ""
                )
                threshold_cell = (
                    rf"\multirow{{{len(subset)}}}{{*}}{{{int(round(threshold * 100))}\%}}"
                    if row_index == 0
                    else ""
                )
                lines.append(
                    " & ".join(
                        [
                            metric_cell,
                            threshold_cell,
                            latex_escape(row["dataset"]),
                            fmt_pm_bold(
                                float(row["lr"]),
                                float(row["lr_std"]),
                                row["best_classifier"] == "LR",
                            ),
                            fmt_pm_bold(
                                float(row["mlp"]),
                                float(row["mlp_std"]),
                                row["best_classifier"] == "MLP",
                            ),
                            fmt_pm_bold(
                                float(row["xgb"]),
                                float(row["xgb_std"]),
                                row["best_classifier"] == "XGB",
                            ),
                            fmt_num(float(row["range"])),
                        ]
                    )
                    + r" \\"
                )
            if not (metric == METRIC_ORDER[-1] and math.isclose(threshold, 0.95)):
                lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\caption{Classifier robustness at the curve-selected threshold "
                r"dimensions. Within each classifier family, the best encoder, "
                r"projector, dimension, and hyperparameters are selected for the "
                r"reported metric. Classifier entries are mean $\pm$ SD over "
                r"evaluation seeds, with the best classifier-family mean in each "
                r"row shown in bold. Range is the difference between the best and "
                r"worst classifier-family mean performance for that dataset and "
                r"threshold.}"
            ),
            r"\label{tab:generalisation-classifier-robustness}",
            r"\end{table}",
            "",
        ]
    )
    (output_dir / "generalisation_classifier_robustness_by_dataset.tex").write_text(
        "\n".join(lines)
    )

    summary_rows = []
    for (metric, threshold, classifier), group in df.groupby(
        ["selection_metric", "threshold", "classifier"], sort=False
    ):
        summary_rows.append(
            {
                "metric": metric,
                "threshold": float(threshold),
                "classifier": classifier,
                "performance_mean": float(group[metric_col(metric)].mean()),
                "performance_std": float(group[metric_col(metric)].std(ddof=1)),
                "recovery_mean": float(group["recovery"].mean()),
                "recovery_std": float(group["recovery"].std(ddof=1)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "generalisation_classifier_recovery_summary.csv", index=False)
    return summary


def write_lr_sensitivity(run_root: Path, output_dir: Path) -> None:
    src = (
        run_root
        / "tables"
        / "generalisation_interpretation"
        / "curve_vs_generalised_lr_summary.csv"
    )
    para_src = (
        run_root
        / "tables"
        / "generalisation_interpretation"
        / "curve_vs_generalised_lr_paragraph.txt"
    )
    if not src.exists():
        return
    df = pd.read_csv(src)
    df.to_csv(output_dir / "generalisation_lr_curve_vs_sweep_summary.csv", index=False)
    lines = [
        r"\begin{table}[!htpb]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Selection metric & Threshold & $\Delta$ Accuracy & $\Delta$ ROC-AUC \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            " & ".join(
                [
                    METRIC_LABELS[str(row["selection_metric"])],
                    f"{int(round(float(row['threshold']) * 100))}\\%",
                    fmt_pm(float(row["delta_accuracy_mean"]), float(row["delta_accuracy_sd"])),
                    fmt_pm(float(row["delta_roc_auc_mean"]), float(row["delta_roc_auc_sd"])),
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
                r"\caption{Difference between the held-out generalisation "
                r"logistic-regression sweep and the curve-stage logistic-regression "
                r"estimate at matched threshold dimensions. Positive values indicate "
                r"higher held-out generalisation performance.}"
            ),
            r"\label{tab:generalisation-lr-curve-vs-sweep}",
            r"\end{table}",
            "",
        ]
    )
    (output_dir / "generalisation_lr_curve_vs_sweep_summary.tex").write_text("\n".join(lines))
    if para_src.exists():
        (output_dir / "generalisation_lr_curve_vs_sweep_paragraph.txt").write_text(
            para_src.read_text()
        )


def svg_text(x: float, y: float, text: str, cls: str = "", anchor: str = "middle") -> str:
    klass = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}"{klass}>{escape(text)}</text>'


def make_recovery_svg(summary: pd.DataFrame, output_path: Path) -> None:
    width, height = 1900, 1120
    margin_l, margin_r = 210, 70
    panel_top, panel_h = 95, 660
    panel_gap = 90
    panel_w = (width - margin_l - margin_r - panel_gap) / 2
    plot_bottom = panel_top + panel_h
    y_min = max(0.88, float(summary["recovery_mean"].min() - summary["recovery_std"].max() - 0.02))
    y_max = min(1.03, float(summary["recovery_mean"].max() + summary["recovery_std"].max() + 0.02))
    if y_max < 1.01:
        y_max = 1.01
    if y_min > 0.90:
        y_min = 0.90

    def y(value: float) -> float:
        return panel_top + (y_max - value) / (y_max - y_min) * panel_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            "<style>"
            "text{font-family:Arial,Helvetica,sans-serif;fill:#222}"
            ".axis{font-size:40px}.tick{font-size:30px;fill:#444}"
            ".panel{font-size:48px;font-weight:600}.legend{font-size:38px}"
            "</style>"
        ),
    ]
    colours = {"0.99": "#1f77b4", "0.95": "#ff7f0e"}
    ticks = [0.90, 0.95, 0.99, 1.00]
    for panel_i, metric in enumerate(METRIC_ORDER):
        left = margin_l + panel_i * (panel_w + panel_gap)
        parts.append(svg_text(left + panel_w / 2, 55, METRIC_LABELS[metric], "panel"))
        for tick in ticks:
            if y_min <= tick <= y_max:
                yy = y(tick)
                colour = "#555" if tick in {0.95, 0.99, 1.0} else "#dddddd"
                dash = ' stroke-dasharray="10 8"' if tick in {0.95, 0.99, 1.0} else ""
                parts.append(
                    f'<line x1="{left:.2f}" y1="{yy:.2f}" x2="{left + panel_w:.2f}" y2="{yy:.2f}" '
                    f'stroke="{colour}" stroke-width="2"{dash}/>'
                )
                if panel_i == 0:
                    parts.append(svg_text(left - 18, yy + 12, f"{tick * 100:.0f}%", "tick", "end"))
        parts.append(
            f'<rect x="{left:.2f}" y="{panel_top:.2f}" width="{panel_w:.2f}" height="{panel_h:.2f}" '
            'fill="none" stroke="#222" stroke-width="1.4"/>'
        )
        group_w = panel_w / len(CLASSIFIER_ORDER)
        bar_w = group_w * 0.26
        for ci, classifier in enumerate(CLASSIFIER_ORDER):
            cx = left + group_w * (ci + 0.5)
            for ti, threshold in enumerate(THRESHOLD_ORDER):
                row = summary[
                    (summary["metric"] == metric)
                    & (summary["threshold"].round(2) == float(threshold))
                    & (summary["classifier"] == classifier)
                ]
                if row.empty:
                    continue
                rec = float(row.iloc[0]["recovery_mean"])
                err = float(row.iloc[0]["recovery_std"])
                bx = cx + (ti - 0.5) * (bar_w * 1.35)
                top = y(rec)
                bh = plot_bottom - top
                parts.append(
                    f'<rect x="{bx - bar_w / 2:.2f}" y="{top:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" '
                    f'fill="{colours[threshold]}" opacity="0.9"/>'
                )
                ey1, ey2 = y(rec + err), y(rec - err)
                parts.append(
                    f'<line x1="{bx:.2f}" y1="{ey1:.2f}" x2="{bx:.2f}" y2="{ey2:.2f}" '
                    'stroke="#222" stroke-width="2.6"/>'
                )
                parts.append(
                    f'<line x1="{bx - bar_w * 0.24:.2f}" y1="{ey1:.2f}" x2="{bx + bar_w * 0.24:.2f}" y2="{ey1:.2f}" '
                    'stroke="#222" stroke-width="2.6"/>'
                )
                parts.append(
                    f'<line x1="{bx - bar_w * 0.24:.2f}" y1="{ey2:.2f}" x2="{bx + bar_w * 0.24:.2f}" y2="{ey2:.2f}" '
                    'stroke="#222" stroke-width="2.6"/>'
                )
            parts.append(svg_text(cx, plot_bottom + 55, classifier, "tick"))
        parts.append(svg_text(left + panel_w / 2, plot_bottom + 118, "Classifier", "axis"))
    parts.append(
        f'<text x="52" y="{panel_top + panel_h / 2:.2f}" class="axis" text-anchor="middle" '
        f'transform="rotate(-90 52 {panel_top + panel_h / 2:.2f})">Held-out recovery</text>'
    )
    leg_x, leg_y, leg_w, leg_h = width / 2 - 300, height - 215, 600, 112
    parts.append(
        f'<rect x="{leg_x:.2f}" y="{leg_y:.2f}" width="{leg_w:.2f}" height="{leg_h:.2f}" '
        'fill="white" stroke="#222" stroke-width="1.2"/>'
    )
    for i, threshold in enumerate(THRESHOLD_ORDER):
        x = leg_x + 115 + i * 240
        parts.append(
            f'<rect x="{x:.2f}" y="{leg_y + 42:.2f}" width="58" height="30" '
            f'fill="{colours[threshold]}" opacity="0.9"/>'
        )
        parts.append(svg_text(x + 78, leg_y + 68, f"{int(float(threshold) * 100)}%", "legend", "start"))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts))


def write_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    svg_path = output_dir / "generalisation_classifier_recovery.pdf.svg"
    pdf_path = output_dir / "generalisation_classifier_recovery.pdf"
    make_recovery_svg(summary, svg_path)
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf_path), str(svg_path)], check=True)
    final_svg = output_dir / "generalisation_classifier_recovery.svg"
    svg_path.replace(final_svg)


def write_summary_paragraph(best: pd.DataFrame, robust: pd.DataFrame, output_dir: Path) -> None:
    recovery = [float(v) for v in best["recovery"]]
    ranges = [float(v) for v in robust["range"]]
    classifier_counts = best["classifier"].value_counts().to_dict()
    projector_counts = best["projector"].value_counts().to_dict()
    paragraph = (
        "Across the compact best held-out threshold configurations, mean recovery "
        f"was {mean(recovery) * 100:.1f}% (SD {std(recovery) * 100:.1f}%). "
        f"The mean spread between classifier families within matched dataset, metric, "
        f"and threshold summaries was {mean(ranges):.4f} (SD {std(ranges):.4f}). "
        "The selected classifiers were "
        + ", ".join(f"{name} ({count})" for name, count in classifier_counts.items())
        + "; selected projectors were "
        + ", ".join(f"{name} ({count})" for name, count in projector_counts.items())
        + "."
    )
    (output_dir / "generalisation_compact_summary_paragraph.txt").write_text(paragraph + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_root / "tables" / "generalisation_summary_package"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_appendix(args.run_root)
    best = choose_best(df)
    write_compact_best(best, output_dir)
    robust = classifier_robustness(df)
    summary = write_robustness(robust, df, output_dir)
    write_lr_sensitivity(args.run_root, output_dir)
    write_plot(summary, output_dir)
    write_summary_paragraph(best, robust, output_dir)
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()
