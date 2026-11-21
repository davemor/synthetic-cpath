#!/usr/bin/env python3
"""Render hproj intrinsic dimensionality estimates as a LaTeX table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DATASET_LABELS = {
    "kather100k": "Kather100K",
    "spider-colorectal": "SPIDER colorectal",
    "spider-breast": "SPIDER breast",
    "spider-skin": "SPIDER skin",
    "spider-thorax": "SPIDER thorax",
    "icaird100k": "iCAIRD100K",
}

ENCODER_LABELS = {
    "uni": "UNI",
    "uni2": "UNI2",
    "hibou": "Hibou",
    "dinov2": "DINOv2",
}

DATASET_ORDER = [
    "kather100k",
    "spider-colorectal",
    "spider-breast",
    "spider-skin",
    "spider-thorax",
    "icaird100k",
]

ENCODER_ORDER = ["uni", "uni2", "hibou", "dinov2"]


def parse_args() -> argparse.Namespace:
    run_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=run_root / "intrinsic_dims.csv",
        help="Input intrinsic_dims.csv path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=run_root / "tables" / "intrinsic_dims_table.tex",
        help="Output LaTeX table path.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Decimal places for intrinsic dimensionality estimates.",
    )
    return parser.parse_args()


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def format_number(value: str, digits: int) -> str:
    if value is None or value == "":
        return "--"
    number = float(value)
    if math.isnan(number):
        return "--"
    return f"{number:.{digits}f}"


def sort_key(row: dict[str, str]) -> tuple[int, int]:
    dataset = row["dataset"]
    encoder = row["encoder"]
    dataset_index = DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else 999
    encoder_index = ENCODER_ORDER.index(encoder) if encoder in ENCODER_ORDER else 999
    return dataset_index, encoder_index


def render_table(rows: list[dict[str, str]], digits: int) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{l l r r r r}",
        r"\hline",
        (
            r"\textbf{Dataset} & \textbf{Embedder} & \textbf{D} & "
            r"\textbf{n} & \textbf{MLE} & \textbf{TwoNN} \\"
        ),
        r"\hline",
    ]

    previous_dataset = None
    for row in sorted(rows, key=sort_key):
        dataset = row["dataset"]
        if previous_dataset is not None and dataset != previous_dataset:
            lines.append(r"\hline")

        dataset_label = DATASET_LABELS.get(dataset, dataset)
        encoder_label = ENCODER_LABELS.get(row["encoder"], row["encoder"])
        fields = [
            latex_escape(dataset_label),
            latex_escape(encoder_label),
            str(int(float(row["num_dimensions"]))),
            str(int(float(row["n_samples"]))),
            format_number(row["intrinsic_dim_mle"], digits),
            format_number(row["intrinsic_dim_twonn"], digits),
        ]
        lines.append(" & ".join(fields) + r" \\")
        previous_dataset = dataset

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Intrinsic dimensionality estimates for each dataset and embedder.}",
            r"\label{tab:intrinsic_dimensionality_hproj}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    table = render_table(rows, args.digits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table)
    print(f"Wrote {args.output}")
    print(table)


if __name__ == "__main__":
    main()
