from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "final-results" / "quality-results-with-nufd.csv"
FULL_OUTPUT = ROOT / "final-results" / "quality-results-fid-ip-ir-authenticity.tex"
SUMMARY_CSV = ROOT / "final-results" / "quality-metric-range-best-worst.csv"
SUMMARY_TEX = ROOT / "final-results" / "quality-metric-range-best-worst.tex"
RQ2_OUTPUT = ROOT / "final-results" / "rq2-normalized-quality.tex"
DATASETS = {
    "icaird_cervical100k": "iCAIRD",
    "pathmnist224": "Kather 100K",
    "pcam": "PCam",
}
ARCHITECTURES = {
    "stylegan2ada": "StyleGAN2-ADA",
    "sd_lora": "SD-LoRA",
    "dcgan": "DCGAN",
}
SHORT_ARCHITECTURES = {
    "stylegan2ada": "SG2",
    "sd_lora": "SD",
    "dcgan": "DCG",
}
CONDITIONING = {
    "unconditioned": "Unconditioned",
    "pseudo_label": "Global cluster",
    "true_label": "Class label",
    "label_global_cluster": "Class + global",
    "label_local_cluster": "Class + local",
}
METRICS = {
    "FID": ("min", 2),
    "nUFD": ("min", 2),
    "IP": ("max", 4),
    "IR": ("max", 4),
    "Authenticity": ("max", 4),
}


def render_value(value: float, decimals: int, bold: bool = False) -> str:
    text = f"{value:.{decimals}f}"
    return rf"\textbf{{{text}}}" if bold else text


def full_table(frame: pd.DataFrame) -> str:
    lines = [
        "% Requires \\usepackage{booktabs,multirow}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "Dataset & Architecture & Conditioning & FID $\\downarrow$ & nUFD $\\downarrow$ & IP $\\uparrow$ & IR $\\uparrow$ & Auth $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for dataset_index, (dataset, dataset_name) in enumerate(DATASETS.items()):
        dataset_rows = frame[frame["Dataset"] == dataset]
        for architecture_index, (architecture, architecture_name) in enumerate(
            ARCHITECTURES.items()
        ):
            group = dataset_rows[dataset_rows["Architecture"] == architecture]
            unique_best: dict[str, float | None] = {}
            for metric, (direction, _) in METRICS.items():
                target = (
                    group[metric].min() if direction == "min" else group[metric].max()
                )
                unique_best[metric] = (
                    target if np.isclose(group[metric], target).sum() == 1 else None
                )
            for condition_index, (condition, condition_name) in enumerate(
                CONDITIONING.items()
            ):
                row = group[group["Conditioning"] == condition].iloc[0]
                dataset_cell = (
                    rf"\multirow{{15}}{{*}}{{{dataset_name}}}"
                    if architecture_index == 0 and condition_index == 0
                    else ""
                )
                architecture_cell = (
                    rf"\multirow{{5}}{{*}}{{{architecture_name}}}"
                    if condition_index == 0
                    else ""
                )
                values = [
                    render_value(
                        row[metric],
                        decimals,
                        unique_best[metric] is not None
                        and np.isclose(row[metric], unique_best[metric]),
                    )
                    for metric, (_, decimals) in METRICS.items()
                ]
                lines.append(
                    " & ".join(
                        [dataset_cell, architecture_cell, condition_name, *values]
                    )
                    + r" \\"
                )
            if architecture_index < 2:
                lines.append(r"\cmidrule(lr){2-8}")
        if dataset_index < 2:
            lines.append(r"\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Quality results for each dataset, architecture, and conditioning regime. nUFD denotes normalized Fr\\'echet distance in UNI feature space. IP and IR denote improved precision and recall in UNI feature space using $k=3$. Auth denotes Authenticity calculated in UNI space. Bold denotes the unique best result within each dataset--architecture. Results are from one independently trained generator seed.}",
            "\\label{tab:original-quality-authenticity}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def extremum_label(group: pd.DataFrame, metric: str, target: float) -> str:
    matches = group[np.isclose(group[metric], target)]
    if len(matches) > 1:
        return f"Multiple ({len(matches)})"
    row = matches.iloc[0]
    return f"{SHORT_ARCHITECTURES[row['Architecture']]}, {CONDITIONING[row['Conditioning']]}"


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, dataset_name in DATASETS.items():
        group = frame[frame["Dataset"] == dataset]
        for metric, (direction, decimals) in METRICS.items():
            minimum, maximum = group[metric].min(), group[metric].max()
            best = minimum if direction == "min" else maximum
            worst = maximum if direction == "min" else minimum
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Metric": metric,
                    "Range": f"{minimum:.{decimals}f}--{maximum:.{decimals}f}",
                    "Best configuration": extremum_label(group, metric, best),
                    "Best value": best,
                    "Worst configuration": extremum_label(group, metric, worst),
                    "Worst value": worst,
                }
            )
    return pd.DataFrame(rows)


def summary_table(frame: pd.DataFrame) -> str:
    lines = [
        "% Requires \\usepackage{booktabs,multirow}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lllrlr}",
        "\\toprule",
        "Dataset & Metric & Range & Best configuration & Best & Worst configuration / value \\\\",
        "\\midrule",
    ]
    for dataset_index, (dataset, group) in enumerate(frame.groupby("Dataset", sort=False)):
        for row_index, row in enumerate(group.itertuples(index=False)):
            dataset_cell = rf"\multirow{{5}}{{*}}{{{dataset}}}" if row_index == 0 else ""
            decimals = METRICS[row.Metric][1]
            worst = f"{row._5} / {row._6:.{decimals}f}"
            lines.append(
                f"{dataset_cell} & {row.Metric} & {row.Range} & {row._3} & "
                f"{row._4:.{decimals}f} & {worst} \\\\"
            )
        if dataset_index < 2:
            lines.append(r"\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Observed metric ranges and best and worst generator configurations within each dataset. SG2 denotes StyleGAN2-ADA, SD denotes SD-LoRA, and DCG denotes DCGAN. Multiple indicates an exact tie. FID extrema are reported only within datasets because raw FID is not comparable across datasets.}",
            "\\label{tab:quality-range-best-worst}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def rq2_table(frame: pd.DataFrame) -> str:
    conditions = ["true_label", "label_global_cluster", "label_local_cluster"]
    metrics = {
        "nUFD": ("min", 2),
        "IP": ("max", 4),
        "IR": ("max", 4),
        "Authenticity": ("max", 4),
    }
    lines = [
        "% Requires \\usepackage{booktabs,multirow}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Dataset & Architecture & Conditioning & nUFD $\\downarrow$ & IP $\\uparrow$ & IR $\\uparrow$ & Auth $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for dataset_index, (dataset, dataset_name) in enumerate(DATASETS.items()):
        dataset_rows = frame[frame["Dataset"] == dataset]
        for architecture_index, (architecture, architecture_name) in enumerate(
            ARCHITECTURES.items()
        ):
            group = dataset_rows[
                (dataset_rows["Architecture"] == architecture)
                & dataset_rows["Conditioning"].isin(conditions)
            ]
            unique_best: dict[str, float | None] = {}
            for metric, (direction, _) in metrics.items():
                target = (
                    group[metric].min() if direction == "min" else group[metric].max()
                )
                unique_best[metric] = (
                    target if np.isclose(group[metric], target).sum() == 1 else None
                )
            for condition_index, condition in enumerate(conditions):
                row = group[group["Conditioning"] == condition].iloc[0]
                dataset_cell = (
                    rf"\multirow{{9}}{{*}}{{{dataset_name}}}"
                    if architecture_index == 0 and condition_index == 0
                    else ""
                )
                architecture_cell = (
                    rf"\multirow{{3}}{{*}}{{{architecture_name}}}"
                    if condition_index == 0
                    else ""
                )
                values = [
                    render_value(
                        row[metric],
                        decimals,
                        unique_best[metric] is not None
                        and np.isclose(row[metric], unique_best[metric]),
                    )
                    for metric, (_, decimals) in metrics.items()
                ]
                lines.append(
                    " & ".join(
                        [
                            dataset_cell,
                            architecture_cell,
                            CONDITIONING[condition],
                            *values,
                        ]
                    )
                    + r" \\"
                )
            if architecture_index < 2:
                lines.append(r"\cmidrule(lr){2-7}")
        if dataset_index < 2:
            lines.append(r"\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Quality results for class-label conditioning and conditioning that combines class labels with global or class-local cluster pseudo-labels. nUFD denotes normalized Fr\\'echet distance in UNI feature space. IP and IR denote improved precision and recall in UNI feature space using $k=3$. Auth denotes Authenticity calculated in UNI space. Bold denotes the unique best conditioning result within each dataset--architecture group; tied values are not bolded. Results are from one independently trained generator seed.}",
            "\\label{tab:rq2-normalized-quality}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    frame = pd.read_csv(INPUT)
    FULL_OUTPUT.write_text(full_table(frame), encoding="utf-8")
    summary_frame = summary(frame)
    summary_frame.to_csv(SUMMARY_CSV, index=False)
    SUMMARY_TEX.write_text(summary_table(summary_frame), encoding="utf-8")
    RQ2_OUTPUT.write_text(rq2_table(frame), encoding="utf-8")


if __name__ == "__main__":
    main()
