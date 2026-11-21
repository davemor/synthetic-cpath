#!/usr/bin/env python3
"""Select recommended projector/encoder/dimension settings from hproj curves.

The recommendation is deliberately parsimonious: for each dataset, find the
smallest dimensionality that reaches at least 99% of both the best observed
ROC-AUC and the best observed accuracy on that dataset. This avoids choosing
very high-dimensional points that only improve a metric by tiny numerical
amounts, while still requiring near-peak predictive performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


DATASET_LABELS = {
    "kather100k": "Kather",
    "spider-colorectal": "CRC",
    "spider-breast": "Breast",
    "spider-skin": "Skin",
    "spider-thorax": "Thorax",
    "icaird100k": "iCAIRD",
}
DATASET_ORDER = list(DATASET_LABELS)
ENCODER_LABELS = {
    "uni": "UNI",
    "uni2": "UNI2",
    "hibou": "Hibou",
    "dinov2": "DINOv2",
}
PROJECTOR_LABELS = {"pca": "PCA", "umap": "UMAP", "grp": "GRP"}
PROJECTOR_ORDER = ["pca", "umap", "grp"]
METRICS = ["roc_auc", "accuracy"]
RECOVERY_FRACTION = 0.99


def parse_args() -> argparse.Namespace:
    run_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curve-dir",
        type=Path,
        default=run_root / "curve",
        help="Curve output directory containing per-projector task JSONs.",
    )
    parser.add_argument(
        "--classifier",
        default="logistic-regression",
        help="Classifier curve outputs to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=run_root / "tables" / "best_projector_encoder",
        help="Directory for generated evidence tables.",
    )
    return parser.parse_args()


def parse_dataset_encoder(value: str) -> tuple[str, str]:
    return value.rsplit("_", 1)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    value_mean = mean(values)
    return math.sqrt(sum((value - value_mean) ** 2 for value in values) / (len(values) - 1))


def fmt_metric(value: float) -> str:
    return f"{value:.4f}"


def fmt_percent(value: float) -> str:
    return f"{value * 100:.2f}"


def read_curve_records(curve_dir: Path, classifier: str) -> list[dict[str, object]]:
    values: dict[tuple[str, str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for projector in PROJECTOR_ORDER:
        task_dir = curve_dir / f"{projector}_{classifier}" / "tasks"
        for path in task_dir.glob("*.json"):
            item = json.loads(path.read_text())
            if item["classifier"] != classifier or item["projector"] != projector:
                continue
            dataset, encoder = parse_dataset_encoder(item["dataset"])
            key = (dataset, encoder, projector, int(item["n_components"]))
            for metric in METRICS:
                values[key][metric].append(float(item[metric]))

    records: list[dict[str, object]] = []
    for (dataset, encoder, projector, dim), metric_values in values.items():
        records.append(
            {
                "dataset": dataset,
                "encoder": encoder,
                "projector": projector,
                "dim": dim,
                "n": len(metric_values["roc_auc"]),
                "roc_auc_mean": mean(metric_values["roc_auc"]),
                "roc_auc_std": sample_std(metric_values["roc_auc"]),
                "accuracy_mean": mean(metric_values["accuracy"]),
                "accuracy_std": sample_std(metric_values["accuracy"]),
            }
        )
    return records


def display_pair(record: dict[str, object]) -> str:
    return (
        f"{ENCODER_LABELS[str(record['encoder'])]} + "
        f"{PROJECTOR_LABELS[str(record['projector'])]}"
    )


def select_recommendations(records: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        dataset_records = [record for record in records if record["dataset"] == dataset]
        if not dataset_records:
            raise SystemExit(f"No curve records found for expected dataset: {dataset}")

        best_roc_record = max(
            dataset_records,
            key=lambda record: (
                float(record["roc_auc_mean"]),
                float(record["accuracy_mean"]),
                -int(record["dim"]),
            ),
        )
        best_accuracy_record = max(
            dataset_records,
            key=lambda record: (
                float(record["accuracy_mean"]),
                float(record["roc_auc_mean"]),
                -int(record["dim"]),
            ),
        )
        best_roc = float(best_roc_record["roc_auc_mean"])
        best_accuracy = float(best_accuracy_record["accuracy_mean"])

        eligible = [
            record
            for record in dataset_records
            if float(record["roc_auc_mean"]) >= RECOVERY_FRACTION * best_roc
            and float(record["accuracy_mean"]) >= RECOVERY_FRACTION * best_accuracy
        ]
        if not eligible:
            raise SystemExit(f"No eligible recommendation found for dataset: {dataset}")

        recommendation = min(
            eligible,
            key=lambda record: (
                int(record["dim"]),
                -float(record["roc_auc_mean"]),
                -float(record["accuracy_mean"]),
            ),
        )
        output.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "recommended_encoder": recommendation["encoder"],
                "recommended_projector": recommendation["projector"],
                "recommended_pair": display_pair(recommendation),
                "recommended_dim": recommendation["dim"],
                "recommended_roc_auc_mean": recommendation["roc_auc_mean"],
                "recommended_roc_auc_std": recommendation["roc_auc_std"],
                "recommended_accuracy_mean": recommendation["accuracy_mean"],
                "recommended_accuracy_std": recommendation["accuracy_std"],
                "best_roc_pair": display_pair(best_roc_record),
                "best_roc_dim": best_roc_record["dim"],
                "best_roc_auc_mean": best_roc,
                "best_roc_accuracy_mean": best_roc_record["accuracy_mean"],
                "best_accuracy_pair": display_pair(best_accuracy_record),
                "best_accuracy_dim": best_accuracy_record["dim"],
                "best_accuracy_mean": best_accuracy,
                "best_accuracy_roc_auc_mean": best_accuracy_record["roc_auc_mean"],
                "roc_auc_recovery": float(recommendation["roc_auc_mean"]) / best_roc,
                "accuracy_recovery": float(recommendation["accuracy_mean"]) / best_accuracy,
                "eligible_count": len(eligible),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "dataset",
        "recommended_pair",
        "recommended_dim",
        "recommended_roc_auc_mean",
        "recommended_roc_auc_std",
        "recommended_accuracy_mean",
        "recommended_accuracy_std",
        "roc_auc_recovery",
        "accuracy_recovery",
        "best_roc_pair",
        "best_roc_dim",
        "best_roc_auc_mean",
        "best_roc_accuracy_mean",
        "best_accuracy_pair",
        "best_accuracy_dim",
        "best_accuracy_mean",
        "best_accuracy_roc_auc_mean",
        "eligible_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset_label"],
                    "recommended_pair": row["recommended_pair"],
                    "recommended_dim": row["recommended_dim"],
                    "recommended_roc_auc_mean": fmt_metric(float(row["recommended_roc_auc_mean"])),
                    "recommended_roc_auc_std": fmt_metric(float(row["recommended_roc_auc_std"])),
                    "recommended_accuracy_mean": fmt_metric(float(row["recommended_accuracy_mean"])),
                    "recommended_accuracy_std": fmt_metric(float(row["recommended_accuracy_std"])),
                    "roc_auc_recovery": fmt_percent(float(row["roc_auc_recovery"])),
                    "accuracy_recovery": fmt_percent(float(row["accuracy_recovery"])),
                    "best_roc_pair": row["best_roc_pair"],
                    "best_roc_dim": row["best_roc_dim"],
                    "best_roc_auc_mean": fmt_metric(float(row["best_roc_auc_mean"])),
                    "best_roc_accuracy_mean": fmt_metric(float(row["best_roc_accuracy_mean"])),
                    "best_accuracy_pair": row["best_accuracy_pair"],
                    "best_accuracy_dim": row["best_accuracy_dim"],
                    "best_accuracy_mean": fmt_metric(float(row["best_accuracy_mean"])),
                    "best_accuracy_roc_auc_mean": fmt_metric(
                        float(row["best_accuracy_roc_auc_mean"])
                    ),
                    "eligible_count": row["eligible_count"],
                }
            )


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


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[!htpb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (
            r"Dataset & Recommended & Dim. & ROC-AUC & Acc. & "
            r"ROC rec. & Acc. rec. & Peak ROC-AUC \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(row["dataset_label"]),
                    latex_escape(row["recommended_pair"]),
                    latex_escape(row["recommended_dim"]),
                    fmt_metric(float(row["recommended_roc_auc_mean"])),
                    fmt_metric(float(row["recommended_accuracy_mean"])),
                    fmt_percent(float(row["roc_auc_recovery"])) + r"\%",
                    fmt_percent(float(row["accuracy_recovery"])) + r"\%",
                    (
                        f"{latex_escape(row['best_roc_pair'])}, "
                        f"{latex_escape(row['best_roc_dim'])}D "
                        f"({fmt_metric(float(row['best_roc_auc_mean']))})"
                    ),
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
                r"\caption{Recommended projector and encoder pair for each dataset. "
                r"The selected dimensionality is the smallest dimension that recovers "
                r"at least 99\% of both the best observed ROC-AUC and the best observed "
                r"accuracy for that dataset. Peak ROC-AUC shows the absolute best "
                r"ROC-AUC configuration for comparison.}"
            ),
            r"\label{tab:best-projector-encoder}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    records = read_curve_records(args.curve_dir, args.classifier)
    rows = select_recommendations(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "best_projector_encoder_recommendations.csv"
    tex_path = args.output_dir / "best_projector_encoder_recommendations.tex"
    write_csv(csv_path, rows)
    write_latex(tex_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
