from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "final-results" / "quality_results_fid_ip_ir_authenticity.csv"
OUTPUT_STEM = ROOT / "final-results" / "rq1-global-vs-unconditioned-quality"
FONT_SIZE = 24

DATASETS = [
    ("icaird_cervical100k", "iCAIRD"),
    ("pathmnist224", "Kather"),
    ("pcam", "PCam"),
]
ARCHITECTURES = [
    ("stylegan2ada", "SGAN"),
    ("sd_lora", "SD"),
    ("dcgan", "DCG"),
]
METRICS = [
    ("FID", "FID", 2),
    ("IP", "IP", 4),
    ("IR", "IR", 4),
    ("Authenticity", "Authenticity", 4),
]


def build_delta_frame() -> pd.DataFrame:
    frame = pd.read_csv(INPUT)
    key = ["Dataset", "Architecture"]
    unconditioned = (
        frame[frame["Conditioning"] == "unconditioned"]
        .set_index(key)
        .sort_index()
    )
    global_cluster = (
        frame[frame["Conditioning"] == "pseudo_label"]
        .set_index(key)
        .sort_index()
    )

    if not unconditioned.index.equals(global_cluster.index):
        raise ValueError("Unconditioned and global-cluster cells are not aligned.")

    rows: list[dict[str, str | float]] = []
    for dataset, dataset_label in DATASETS:
        for architecture, architecture_label in ARCHITECTURES:
            index = (dataset, architecture)
            if index not in unconditioned.index:
                raise ValueError(f"Missing quality cell: {index}")
            baseline = unconditioned.loc[index]
            alternative = global_cluster.loc[index]
            rows.append(
                {
                    "Dataset": dataset_label,
                    "Architecture": architecture_label,
                    # Positive values always denote improvement.
                    "FID": baseline["FID"] - alternative["FID"],
                    "IP": alternative["IP"] - baseline["IP"],
                    "IR": alternative["IR"] - baseline["IR"],
                    "Authenticity": (
                        alternative["Authenticity"] - baseline["Authenticity"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def metric_matrix(frame: pd.DataFrame, metric: str) -> np.ndarray:
    matrix = frame.pivot(index="Dataset", columns="Architecture", values=metric)
    return matrix.loc[
        [label for _, label in DATASETS],
        [label for _, label in ARCHITECTURES],
    ].to_numpy(float)


def plot(frame: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "rq1_delta", ["#D55E00", "#F7F7F7", "#0072B2"]
    )
    fig, axes = plt.subplots(1, 4, figsize=(23.0, 8.0), constrained_layout=True)

    for panel, (ax, (metric, title, decimals)) in enumerate(
        zip(axes, METRICS, strict=True)
    ):
        values = metric_matrix(frame, metric)
        limit = float(np.max(np.abs(values)))
        if limit == 0:
            limit = 1.0
        ax.imshow(
            values,
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            aspect="equal",
        )

        ax.set_title(
            rf"$\Delta$ {title}",
            fontsize=FONT_SIZE,
            pad=12,
            fontweight="bold",
        )
        ax.set_xticks(
            range(len(ARCHITECTURES)),
            [label for _, label in ARCHITECTURES],
            fontsize=FONT_SIZE,
            fontweight="bold",
        )
        ax.set_yticks(range(len(DATASETS)))
        if panel == 0:
            ax.set_yticklabels(
                [label for _, label in DATASETS],
                fontsize=FONT_SIZE,
                fontweight="bold",
            )
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0, pad=7)

        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                colour = "white" if abs(value) / limit > 0.62 else "#202020"
                if abs(value) < 0.5 * 10**-decimals:
                    annotation = "0.0"
                else:
                    annotation = f"{value:.{decimals}f}"
                ax.text(
                    column,
                    row,
                    annotation,
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=FONT_SIZE,
                    fontweight="bold",
                )

        for spine in ax.spines.values():
            spine.set_visible(False)

    legend = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=cmap),
        ax=axes,
        orientation="horizontal",
        fraction=0.08,
        pad=0.13,
        shrink=0.5,
        aspect=55,
        ticks=[-1, 0, 1],
    )
    legend.ax.set_xticklabels(["Deterioration", "No change", "Improvement"])
    legend.ax.tick_params(labelsize=FONT_SIZE, length=0, pad=8)
    for label in legend.ax.get_xticklabels():
        label.set_fontweight("bold")
    legend.outline.set_linewidth(0.6)

    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    deltas = build_delta_frame()
    deltas.to_csv(OUTPUT_STEM.with_suffix(".csv"), index=False)
    plot(deltas)
