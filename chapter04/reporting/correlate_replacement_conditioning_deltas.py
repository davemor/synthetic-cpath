from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "final-results" / "replacement-vs-real-paired-deltas.csv"
CSV_OUTPUT = ROOT / "final-results" / "replacement-conditioning-correlations.csv"
TEX_OUTPUT = ROOT / "final-results" / "replacement-conditioning-correlations.tex"

CONDITIONS = ["Class", "Class + global", "Class + local"]
KEYS = ["Dataset", "Architecture", "Seed", "Metric"]


def holm_adjust(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * ranked
    ).clip(max=1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def calculate_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot(index=KEYS, columns="Conditioning", values="Delta")
    missing = [condition for condition in CONDITIONS if condition not in wide]
    if missing:
        raise ValueError(f"Missing conditioning regimes: {missing}")

    rows: list[dict[str, object]] = []
    for metric in ("Accuracy", "ROC-AUC"):
        metric_frame = wide.xs(metric, level="Metric")[CONDITIONS].dropna()
        for first, second in combinations(CONDITIONS, 2):
            rho, p_value = spearmanr(metric_frame[first], metric_frame[second])
            rows.append(
                {
                    "Metric": metric,
                    "Conditioning 1": first,
                    "Conditioning 2": second,
                    "Spearman rho": float(rho),
                    "p": float(p_value),
                    "n": len(metric_frame),
                }
            )

    result = pd.DataFrame(rows)
    result["Holm-adjusted p"] = holm_adjust(result["p"])
    return result


def write_latex(frame: pd.DataFrame) -> None:
    def format_p(value: float) -> str:
        return "$<0.001$" if value < 0.001 else f"{value:.3f}"

    lines = [
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "Metric & Conditioning 1 & Conditioning 2 & Spearman's $\\rho$ & $p$ & $p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule",
    ]
    for index, row in frame.iterrows():
        if index == 3:
            lines.append("\\midrule")
        lines.append(
            f"{row['Metric']} & {row['Conditioning 1']} & {row['Conditioning 2']} & "
            f"{row['Spearman rho']:.3f} & {format_p(row['p'])} & "
            f"{format_p(row['Holm-adjusted p'])} \\\\"
        )
    is_augmentation = "augmentation" in INPUT.name
    training_label = "augmentation" if is_augmentation else "synthetic-only"
    table_label = (
        "augmentation-conditioning-correlations"
        if is_augmentation
        else "replacement-conditioning-correlations"
    )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{Pairwise Spearman correlations between {training_label} utility "
            "differences from the matched real-only baseline. Each correlation uses "
            "$n=27$ matched dataset--architecture--seed observations. Holm-adjusted "
            "$p$-values account for the six reported tests.}",
            f"\\label{{tab:{table_label}}}",
            "\\end{table}",
            "",
        ]
    )
    TEX_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate conditioning-regime utility differences."
    )
    parser.add_argument(
        "--training-regime",
        choices=("replacement", "augmentation"),
        default="replacement",
    )
    return parser.parse_args()


def main() -> None:
    global CSV_OUTPUT, INPUT, TEX_OUTPUT

    args = parse_args()
    if args.training_regime == "augmentation":
        INPUT = ROOT / "final-results" / "augmentation-vs-real-paired-deltas.csv"
        CSV_OUTPUT = ROOT / "final-results" / "augmentation-conditioning-correlations.csv"
        TEX_OUTPUT = ROOT / "final-results" / "augmentation-conditioning-correlations.tex"

    frame = pd.read_csv(INPUT)
    duplicate = frame.duplicated(KEYS + ["Conditioning"])
    if duplicate.any():
        raise ValueError("Duplicate matched conditioning observations found.")
    result = calculate_correlations(frame)
    result.to_csv(CSV_OUTPUT, index=False)
    write_latex(result)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
