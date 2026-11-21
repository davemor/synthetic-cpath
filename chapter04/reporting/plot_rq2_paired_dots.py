from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "final-results" / "quality-results-with-nufd.csv"
OUTPUT = ROOT / "final-results" / "rq2-class-cluster-paired-dots"
DATASETS = {
    "icaird_cervical100k": "iCAIRD",
    "pathmnist224": "Kather",
    "pcam": "PCam",
}
ARCHITECTURES = {
    "stylegan2ada": "SG2",
    "sd_lora": "SD",
    "dcgan": "DCG",
}
COMPARISONS = {
    "label_global_cluster": ("Global", "#0173B2", "o"),
    "label_local_cluster": ("Local", "#DE8F05", "s"),
}
METRICS = [
    ("nUFD", "nUFD", "lower is better"),
    ("IP", "IP", "higher is better"),
    ("IR", "IR", "higher is better"),
    ("Authenticity", "Auth", "higher is better"),
]


def differences(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["Dataset", "Architecture"]
    baseline = (
        frame[frame["Conditioning"] == "true_label"]
        .set_index(keys)
        .sort_index()
    )
    rows = []
    for conditioning in COMPARISONS:
        comparison = (
            frame[frame["Conditioning"] == conditioning]
            .set_index(keys)
            .sort_index()
        )
        for index in baseline.index:
            row = {
                "Dataset": index[0],
                "Architecture": index[1],
                "Comparison": conditioning,
            }
            for metric, _, _ in METRICS:
                row[metric] = comparison.loc[index, metric] - baseline.loc[index, metric]
            rows.append(row)
    return pd.DataFrame(rows)


def plot(frame: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 17,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=False)
    figure.subplots_adjust(
        left=0.09,
        right=0.99,
        top=0.94,
        bottom=0.18,
        wspace=0.20,
        hspace=0.46,
    )
    x_positions = np.arange(9)
    ordered_keys = [
        (dataset, architecture)
        for dataset in DATASETS
        for architecture in ARCHITECTURES
    ]

    for axis, (metric, title, direction) in zip(
        axes.ravel(), METRICS, strict=True
    ):
        values_by_comparison = {}
        for comparison, (label, colour, marker) in COMPARISONS.items():
            selected = frame[frame["Comparison"] == comparison].set_index(
                ["Dataset", "Architecture"]
            )
            values = np.array([selected.loc[key, metric] for key in ordered_keys])
            values_by_comparison[comparison] = values
            axis.scatter(
                x_positions,
                values,
                s=115,
                color=colour,
                edgecolor="#202020",
                linewidth=0.9,
                marker=marker,
                label=label,
                zorder=4,
            )

        global_values = values_by_comparison["label_global_cluster"]
        local_values = values_by_comparison["label_local_cluster"]
        for x, global_value, local_value in zip(
            x_positions, global_values, local_values, strict=True
        ):
            axis.plot(
                [x, x],
                [global_value, local_value],
                color="#606060",
                linewidth=1.5,
                zorder=2,
            )

        axis.axhline(0, color="#202020", linewidth=1.5, zorder=1)
        axis.axvline(2.5, color="#808080", linewidth=1.1)
        axis.axvline(5.5, color="#808080", linewidth=1.1)
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.8, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(-0.6, 8.6)
        axis.set_xticks(x_positions, list(ARCHITECTURES.values()) * 3)
        axis.tick_params(axis="x", length=0, pad=7, labelsize=15)
        axis.tick_params(axis="y", labelsize=16)
        axis.set_title(f"$\\Delta$ {title} ({direction})", fontsize=21, pad=10)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontweight("bold")
        for centre, dataset_label in zip((1, 4, 7), DATASETS.values(), strict=True):
            axis.text(
                centre,
                -0.18,
                dataset_label,
                ha="center",
                va="top",
                transform=axis.get_xaxis_transform(),
                fontsize=18,
                fontweight="bold",
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        fontsize=18,
        handletextpad=0.5,
        columnspacing=2.0,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    plot(differences(pd.read_csv(INPUT)))


if __name__ == "__main__":
    main()
