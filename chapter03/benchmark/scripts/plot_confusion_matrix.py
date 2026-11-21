#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import ConfusionMatrixDisplay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot readable confusion matrix PNGs from icaird-cervical-benchmark "
            "confusion_matrix.csv files."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to one confusion_matrix.csv or to a directory containing confusion_matrix.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path for a single CSV. Defaults to replacing confusion_matrix.png "
            "beside each CSV."
        ),
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional filename suffix before .png when plotting a directory, e.g. '_readable'.",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "true", "pred", "all"],
        default="none",
        help=(
            "Normalization mode: none = counts, true = row/recall normalization, "
            "pred = column/precision normalization, all = fraction of all samples."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure title. Defaults to a title inferred from the CSV path.",
    )
    parser.add_argument("--cmap", default="Blues", help="Matplotlib colour map.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI.")
    return parser.parse_args()


def load_confusion_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(path, index_col=0)
    labels = [readable_label(str(label)) for label in frame.index.tolist()]
    matrix = frame.to_numpy(dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square confusion matrix, got shape {matrix.shape}.")
    if len(labels) != matrix.shape[0]:
        raise ValueError("Number of row labels does not match confusion matrix shape.")
    return labels, matrix


def readable_label(label: str) -> str:
    overrides = {
        "normal_inflammation": "Normal /\ninflammation",
        "low_grade": "Low grade",
        "high_grade": "High grade",
        "hpv": "HPV",
        "cin1": "CIN1",
        "cin2": "CIN2",
        "cin3": "CIN3",
        "cgin": "CGIN",
        "squamous_carcinoma": "Squamous\ncarcinoma",
    }
    if label in overrides:
        return overrides[label]
    return label.replace("_", " ").capitalize()


def normalize_matrix(matrix: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return matrix
    if mode == "true":
        denominator = matrix.sum(axis=1, keepdims=True)
    elif mode == "pred":
        denominator = matrix.sum(axis=0, keepdims=True)
    elif mode == "all":
        denominator = np.array([[matrix.sum()]])
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")
    return np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator != 0)


def annotation_text(count: float, value: float, normalize: str) -> str:
    if normalize == "none":
        return f"{int(count)}"
    return f"{value:.2f}\n({int(count)})"


def iter_confusion_matrix_csvs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("**/confusion_matrix.csv"))


def infer_title(path: Path, normalize: str) -> str:
    parts = path.parts
    split = path.parent.name
    task = path.parent.parent.parent.name if len(parts) >= 4 else "MIL"
    task_title = readable_label(task).replace("\n", " ")
    if normalize == "none":
        suffix = "counts"
    elif normalize == "true":
        suffix = "normalised by true label"
    elif normalize == "pred":
        suffix = "normalised by predicted label"
    else:
        suffix = "normalised by all samples"
    return f"{task_title} {split} confusion matrix ({suffix})"


def plot_confusion_matrix(
    labels: list[str],
    counts: np.ndarray,
    values: np.ndarray,
    output: Path,
    title: str,
    normalize: str,
    cmap: str,
    dpi: int,
) -> None:
    fig_size = max(5.5, 1.15 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(confusion_matrix=values, display_labels=labels)
    disp.plot(
        ax=ax,
        cmap=cmap,
        colorbar=False,
        values_format=".0f" if normalize == "none" else ".2f",
        im_kw={"vmin": 0.0},
    )
    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes("right", size="4%", pad=0.12)
    colorbar = fig.colorbar(disp.im_, cax=colorbar_ax)
    colorbar.set_label("Count" if normalize == "none" else "Proportion", color="black")
    colorbar.ax.tick_params(colors="black")

    if normalize != "none" and disp.text_ is not None:
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                disp.text_[row, col].set_text(annotation_text(counts[row, col], values[row, col], normalize))

    if disp.text_ is not None:
        threshold = float(values.max()) / 2.0 if values.size else 0.0
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                text = disp.text_[row, col]
                text.set_color("white" if values[row, col] > threshold else "black")
                text.set_fontsize(9)

    ax.set_title(title, color="black")
    ax.set_xlabel("Predicted label", color="black")
    ax.set_ylabel("True label", color="black")
    ax.tick_params(axis="both", colors="black")
    ax.set_xticklabels(labels, rotation=35, ha="right", color="black")
    ax.set_yticklabels(labels, color="black")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_paths = iter_confusion_matrix_csvs(args.input)
    if not csv_paths:
        raise FileNotFoundError(f"No confusion_matrix.csv files found under {args.input}.")
    if args.output is not None and len(csv_paths) > 1:
        raise ValueError("--output can only be used when plotting a single CSV.")
    for input_csv in csv_paths:
        output = args.output or input_csv.with_name(f"confusion_matrix{args.suffix}.png")
        labels, counts = load_confusion_matrix(input_csv)
        values = normalize_matrix(counts, args.normalize)
        plot_confusion_matrix(
            labels=labels,
            counts=counts,
            values=values,
            output=output,
            title=args.title or infer_title(input_csv, args.normalize),
            normalize=args.normalize,
            cmap=args.cmap,
            dpi=args.dpi,
        )
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
