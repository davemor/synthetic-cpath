from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "final-results" / "quality-results-with-nufd.csv"
OUTPUT = ROOT / "final-results" / "quality-metrics-grouped-bars"
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
CONDITIONING = {
    "unconditioned": "None",
    "pseudo_label": "Global",
    "true_label": "Class",
    "label_global_cluster": "Class + G",
    "label_local_cluster": "Class + L",
}
COLORS = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC"]
METRICS = [
    ("nUFD", "nUFD $\\downarrow$"),
    ("IP", "IP $\\uparrow$"),
    ("IR", "IR $\\uparrow$"),
    ("Authenticity", "Auth $\\uparrow$"),
]


def plot(frame: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 20,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=False)
    figure.subplots_adjust(
        left=0.08,
        right=0.99,
        top=0.95,
        bottom=0.20,
        wspace=0.18,
        hspace=0.46,
    )
    group_centres = np.arange(9, dtype=float)
    width = 0.15
    offsets = (np.arange(5) - 2) * width

    ordered = frame.set_index(["Dataset", "Architecture", "Conditioning"])
    for axis, (metric, title) in zip(axes.ravel(), METRICS, strict=True):
        for condition_index, (condition, condition_label) in enumerate(
            CONDITIONING.items()
        ):
            values = []
            for dataset in DATASETS:
                for architecture in ARCHITECTURES:
                    values.append(ordered.loc[(dataset, architecture, condition), metric])
            axis.bar(
                group_centres + offsets[condition_index],
                values,
                width=width,
                color=COLORS[condition_index],
                edgecolor="none",
                linewidth=0,
                label=condition_label,
                zorder=3,
            )

        axis.set_title(title, fontsize=24, pad=10)
        axis.set_xticks(
            group_centres,
            list(ARCHITECTURES.values()) * len(DATASETS),
            rotation=0,
        )
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.8, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.axvline(2.5, color="#606060", linewidth=1.2)
        axis.axvline(5.5, color="#606060", linewidth=1.2)
        axis.set_xlim(-0.6, 8.6)
        axis.tick_params(axis="x", length=0, pad=7, labelsize=16)
        axis.tick_params(axis="y", labelsize=18)
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
                fontsize=20,
                fontweight="bold",
            )

        if metric in {"IP", "IR", "Authenticity"}:
            axis.set_ylim(0, 1)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=19,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    plot(pd.read_csv(INPUT))


if __name__ == "__main__":
    main()
