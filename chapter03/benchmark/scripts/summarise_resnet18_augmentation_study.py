#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from icaird_cervical_benchmark.metrics import bootstrap_metric_intervals_from_predictions


AUGMENTATION_ORDER = ["none", "geometric", "colour", "hed", "combined"]
AUGMENTATION_LABELS = {
    "none": "None",
    "geometric": "Geometric / morphological",
    "colour": "Colour",
    "hed": "HED",
    "combined": "Combined",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise ResNet-18 augmentation ablation runs for thesis reporting."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="DATASET=RUN_DIR",
        help="Completed run directory labelled by dataset, e.g. icaird=/path/to/run.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_spec in args.run:
        dataset, run_dir = parse_run_spec(run_spec)
        rows.extend(collect_metric_rows(dataset, run_dir, args))
    if not rows:
        raise FileNotFoundError("No ResNet-18 augmentation metrics were found.")

    per_seed = pd.DataFrame(rows)
    summary = summarise(per_seed)
    per_seed_path = args.output_dir / "resnet18_augmentation_per_seed.csv"
    summary_path = args.output_dir / "resnet18_augmentation_summary.csv"
    markdown_path = args.output_dir / "resnet18_augmentation_summary.md"
    latex_path = args.output_dir / "resnet18_augmentation_summary.tex"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    markdown_path.write_text(to_markdown(summary, args.confidence_level), encoding="utf-8")
    latex_path.write_text(to_latex(summary, args.confidence_level), encoding="utf-8")
    print(f"Wrote {per_seed_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {latex_path}")


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Run spec must be DATASET=RUN_DIR, got {spec!r}.")
    dataset, run_dir = spec.split("=", 1)
    dataset = dataset.strip()
    if not dataset:
        raise ValueError(f"Run spec has an empty dataset label: {spec!r}.")
    return dataset, Path(run_dir)


def collect_metric_rows(dataset: str, run_dir: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    metric_paths = sorted((run_dir / "images" / "resnet18").glob("**/metrics.csv"))
    rows = []
    for metric_path in metric_paths:
        seed_dir = metric_path.parent
        if not seed_dir.name.startswith("seed_"):
            continue
        augmentation_suite = infer_augmentation_suite(seed_dir)
        metric_frame = pd.read_csv(metric_path)
        if metric_frame.empty:
            continue
        metric_row = metric_frame.iloc[0].to_dict()
        predictions_csv = seed_dir / "predictions.csv"
        bootstrap = {}
        if predictions_csv.exists():
            intervals, _class_names = bootstrap_metric_intervals_from_predictions(
                predictions_csv,
                n_resamples=args.resamples,
                confidence_level=args.confidence_level,
                seed=args.seed,
            )
            bootstrap = {
                "accuracy_ci_low": intervals["accuracy_ci_low"],
                "accuracy_ci_high": intervals["accuracy_ci_high"],
                "auc_ci_low": intervals["auc_ci_low"],
                "auc_ci_high": intervals["auc_ci_high"],
                "bootstrap_valid_resamples": intervals["bootstrap_valid_resamples"],
                "bootstrap_skipped_resamples": intervals["bootstrap_skipped_resamples"],
            }
        rows.append(
            {
                "dataset": dataset,
                "augmentation_suite": augmentation_suite,
                "augmentation_label": AUGMENTATION_LABELS.get(augmentation_suite, augmentation_suite),
                "seed": int(str(seed_dir.name).removeprefix("seed_")),
                "accuracy": float(metric_row["accuracy"]),
                "auc": float(metric_row["auc"]),
                "valid_auc": metric_row.get("valid_auc"),
                "runtime_seconds": metric_row.get("runtime_seconds"),
                "artifact_dir": str(seed_dir),
                **bootstrap,
            }
        )
    return rows


def infer_augmentation_suite(seed_dir: Path) -> str:
    parent = seed_dir.parent
    if parent.name == "resnet18":
        return "none"
    return parent.name


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, suite), group in per_seed.groupby(["dataset", "augmentation_suite"], sort=False):
        accuracy = group["accuracy"].astype(float)
        auc = group["auc"].astype(float)
        row = {
            "dataset": dataset,
            "augmentation_suite": suite,
            "augmentation_label": AUGMENTATION_LABELS.get(suite, suite),
            "n_seeds": len(group),
            "accuracy_mean": accuracy.mean(),
            "accuracy_std": accuracy.std(ddof=1) if len(group) > 1 else float("nan"),
            "auc_mean": auc.mean(),
            "auc_std": auc.std(ddof=1) if len(group) > 1 else float("nan"),
            "runtime_seconds_mean": group["runtime_seconds"].astype(float).mean(),
        }
        if {"accuracy_ci_low", "accuracy_ci_high", "auc_ci_low", "auc_ci_high"} <= set(group.columns):
            row.update(
                {
                    "accuracy_ci_upper_mean": (group["accuracy_ci_high"] - group["accuracy"]).mean(),
                    "accuracy_ci_lower_mean": (group["accuracy"] - group["accuracy_ci_low"]).mean(),
                    "auc_ci_upper_mean": (group["auc_ci_high"] - group["auc"]).mean(),
                    "auc_ci_lower_mean": (group["auc"] - group["auc_ci_low"]).mean(),
                }
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["sort_dataset"] = pd.factorize(summary["dataset"])[0]
    summary["sort_augmentation"] = summary["augmentation_suite"].map(
        {value: index for index, value in enumerate(AUGMENTATION_ORDER)}
    ).fillna(len(AUGMENTATION_ORDER))
    return summary.sort_values(["sort_dataset", "sort_augmentation"]).drop(
        columns=["sort_dataset", "sort_augmentation"]
    )


def to_markdown(summary: pd.DataFrame, confidence_level: float) -> str:
    confidence_pct = int(round(confidence_level * 100))
    lines = [
        "# ResNet-18 Augmentation Study",
        "",
        f"Values are mean and standard deviation over seeds. CI columns are average non-parametric {confidence_pct}% bootstrap half-widths from held-out test predictions.",
        "",
        f"| Dataset | Augmentation | Seeds | Accuracy, % | SD | {confidence_pct}% CI | ROC-AUC | SD | {confidence_pct}% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            f"{row['dataset']} | "
            f"{row['augmentation_label']} | "
            f"{int(row['n_seeds'])} | "
            f"{format_pct(row['accuracy_mean'])} | "
            f"{format_pct(row['accuracy_std'])} | "
            f"{format_pct_interval(row, 'accuracy')} | "
            f"{format_auc(row['auc_mean'])} | "
            f"{format_auc(row['auc_std'])} | "
            f"{format_auc_interval(row)} |"
        )
    lines.append("")
    return "\n".join(lines)


def to_latex(summary: pd.DataFrame, confidence_level: float) -> str:
    confidence_pct = int(round(confidence_level * 100))
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{ResNet-18 augmentation ablation across patch classification datasets. Accuracy and ROC-AUC are reported as mean and standard deviation over seeds. Confidence intervals are average non-parametric bootstrap half-widths from held-out test predictions.}",
        "\\label{tab:resnet18_augmentation_ablation}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        f"Dataset & Augmentation & Accuracy, \\% & SD & {confidence_pct}\\% CI & ROC-AUC & SD & {confidence_pct}\\% CI \\\\",
        "\\midrule",
    ]
    grouped = list(summary.groupby("dataset", sort=False))
    for dataset_index, (dataset, group) in enumerate(grouped):
        group = group.reset_index(drop=True)
        for row_index, row in group.iterrows():
            dataset_cell = f"\\multirow{{{len(group)}}}{{*}}{{{latex_escape(str(dataset))}}}" if row_index == 0 else ""
            lines.append(
                f"{dataset_cell} & "
                f"{latex_escape(str(row['augmentation_label']))} & "
                f"{format_pct(row['accuracy_mean'])} & "
                f"{format_pct(row['accuracy_std'])} & "
                f"{format_pct_interval(row, 'accuracy')} & "
                f"{format_auc(row['auc_mean'])} & "
                f"{format_auc(row['auc_std'])} & "
                f"{format_auc_interval(row)} \\\\"
            )
        if dataset_index != len(grouped) - 1:
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def format_pct(value: object) -> str:
    value = float(value)
    if pd.isna(value):
        return "--"
    return f"{100.0 * value:.2f}"


def format_pct_interval(row: pd.Series, metric: str) -> str:
    upper = float(row.get(f"{metric}_ci_upper_mean", float("nan")))
    lower = float(row.get(f"{metric}_ci_lower_mean", float("nan")))
    if pd.isna(upper) or pd.isna(lower):
        return "--"
    return f"+{100.0 * upper:.2f} / -{100.0 * lower:.2f}"


def format_auc(value: object) -> str:
    value = float(value)
    if pd.isna(value):
        return "--"
    return f"{value:.3f}"


def format_auc_interval(row: pd.Series) -> str:
    upper = float(row.get("auc_ci_upper_mean", float("nan")))
    lower = float(row.get("auc_ci_lower_mean", float("nan")))
    if pd.isna(upper) or pd.isna(lower):
        return "--"
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


if __name__ == "__main__":
    main()
