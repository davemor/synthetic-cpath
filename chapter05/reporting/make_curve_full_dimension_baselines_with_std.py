#!/usr/bin/env python3
"""Generate full-dimensional curve-stage logistic-regression baseline tables.

The curve stage writes the same full-dimensional baseline inside each
projector-specific output. This script collapses duplicate rows within each
projector/seed, averages the projector-specific baselines per seed, and then
reports the mean and standard deviation over curve seeds.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


PROJECTORS = ("pca", "umap", "grp")
ENCODER_ORDER = ("uni", "uni2", "hibou", "dinov2")
DATASET_ORDER = (
    "kather100k",
    "spider-colorectal",
    "spider-breast",
    "spider-skin",
    "spider-thorax",
    "icaird100k",
)
DATASET_LABELS = {
    "kather100k": "Kather",
    "spider-colorectal": "CRC",
    "spider-breast": "Breast",
    "spider-skin": "Skin",
    "spider-thorax": "Thorax",
    "icaird100k": "iCAIRD",
}
ENCODER_LABELS = {
    "uni": "UNI",
    "uni2": "UNI2",
    "hibou": "Hibou",
    "dinov2": "DINOv2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="hproj run directory containing curve/*_logistic-regression outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated CSV/LaTeX outputs.",
    )
    return parser.parse_args()


def split_dataset_encoder(dataset_encoder: str) -> tuple[str, str]:
    for encoder in sorted(ENCODER_ORDER, key=len, reverse=True):
        suffix = f"_{encoder}"
        if dataset_encoder.endswith(suffix):
            return dataset_encoder[: -len(suffix)], encoder
    raise ValueError(f"Could not split dataset/encoder field: {dataset_encoder}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def fmt_mean(value: float, bold: bool = False) -> str:
    text = f"{value:.4f}"
    return rf"\textbf{{{text}}}" if bold else text


def fmt_sd(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e")


def fmt_cell(mean_value: float, sd_value: float, bold: bool = False) -> str:
    return rf"{fmt_mean(mean_value, bold)}\,{{\scriptsize $\pm$\,{fmt_sd(sd_value)}}}"


def load_seed_values(run_root: Path) -> dict[tuple[str, str], dict[str, object]]:
    repeated = defaultdict(lambda: {"accuracy": [], "roc_auc": [], "dimension": None})
    for projector in PROJECTORS:
        csv_path = run_root / "curve" / f"{projector}_logistic-regression" / "unprojected_scores.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (projector, row["dataset"], row["base_seed"])
                repeated[key]["accuracy"].append(float(row["accuracy"]))
                repeated[key]["roc_auc"].append(float(row["roc_auc"]))
                repeated[key]["dimension"] = int(float(row["n_components"]))

    by_seed = defaultdict(lambda: {"accuracy": [], "roc_auc": [], "dimension": None})
    for (_projector, dataset_encoder, seed), values in repeated.items():
        key = (dataset_encoder, seed)
        by_seed[key]["accuracy"].append(mean(values["accuracy"]))
        by_seed[key]["roc_auc"].append(mean(values["roc_auc"]))
        by_seed[key]["dimension"] = values["dimension"]

    by_dataset_encoder = defaultdict(lambda: {"accuracy": [], "roc_auc": [], "dimension": None})
    for (dataset_encoder, _seed), values in by_seed.items():
        by_dataset_encoder[dataset_encoder]["accuracy"].append(mean(values["accuracy"]))
        by_dataset_encoder[dataset_encoder]["roc_auc"].append(mean(values["roc_auc"]))
        by_dataset_encoder[dataset_encoder]["dimension"] = values["dimension"]

    stats = {}
    for dataset_encoder, values in by_dataset_encoder.items():
        stats[split_dataset_encoder(dataset_encoder)] = {
            "dimension": values["dimension"],
            "n_seeds": len(values["accuracy"]),
            "accuracy_mean": mean(values["accuracy"]),
            "accuracy_std": sd(values["accuracy"]),
            "roc_auc_mean": mean(values["roc_auc"]),
            "roc_auc_std": sd(values["roc_auc"]),
        }
    return stats


def ordered_rows(stats: dict[tuple[str, str], dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for dataset in DATASET_ORDER:
        for encoder in ENCODER_ORDER:
            values = stats[(dataset, encoder)]
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "encoder": encoder,
                    "encoder_label": ENCODER_LABELS[encoder],
                    **values,
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "dataset",
        "encoder",
        "dimension",
        "n_seeds",
        "accuracy_mean",
        "accuracy_std",
        "roc_auc_mean",
        "roc_auc_std",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset_label"],
                    "encoder": row["encoder_label"],
                    "dimension": row["dimension"],
                    "n_seeds": row["n_seeds"],
                    "accuracy_mean": f"{row['accuracy_mean']:.8f}",
                    "accuracy_std": f"{row['accuracy_std']:.8g}",
                    "roc_auc_mean": f"{row['roc_auc_mean']:.8f}",
                    "roc_auc_std": f"{row['roc_auc_std']:.8g}",
                }
            )


def write_latex(rows: list[dict[str, object]], output_path: Path) -> None:
    best_by_dataset = {}
    for dataset in DATASET_ORDER:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        best_by_dataset[(dataset, "accuracy")] = max(row["accuracy_mean"] for row in dataset_rows)
        best_by_dataset[(dataset, "roc_auc")] = max(row["roc_auc_mean"] for row in dataset_rows)

    lines = [
        r"\begin{table}[!htpb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrcc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Encoder} & \textbf{D} & \textbf{Accuracy} & \textbf{ROC-AUC} \\",
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        if dataset_index:
            lines.append(r"\midrule")
        for row in [row for row in rows if row["dataset"] == dataset]:
            acc_bold = row["accuracy_mean"] == best_by_dataset[(dataset, "accuracy")]
            auc_bold = row["roc_auc_mean"] == best_by_dataset[(dataset, "roc_auc")]
            lines.append(
                " & ".join(
                    [
                        row["dataset_label"],
                        row["encoder_label"],
                        str(row["dimension"]),
                        fmt_cell(row["accuracy_mean"], row["accuracy_std"], acc_bold),
                        fmt_cell(row["roc_auc_mean"], row["roc_auc_std"], auc_bold),
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
                r"\caption{Full-dimensional curve-stage logistic-regression "
                r"baselines per dataset and encoder, reported as mean "
                r"$\pm$ SD over curve seeds. Baselines are collapsed within "
                r"each projector-specific curve output and then averaged "
                r"because the full-dimensional reference is "
                r"projector-independent.}"
            ),
            r"\label{tab:curve-full-dim-baselines}",
            r"\end{table}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_root / "tables" / "curve_full_dimension_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = ordered_rows(load_seed_values(args.run_root))
    write_csv(rows, output_dir / "curve_full_dimension_baselines.csv")
    write_latex(rows, output_dir / "curve_full_dimension_baselines.tex")
    print(output_dir / "curve_full_dimension_baselines.csv")
    print(output_dir / "curve_full_dimension_baselines.tex")


if __name__ == "__main__":
    main()
