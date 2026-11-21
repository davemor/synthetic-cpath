from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "final-results" / "utility_results_accuracy_roc_auc.csv"
OUTPUT = ROOT / "final-results" / "utility-metrics-grouped-bars"

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
    "true_label": "Class",
    "label_global_cluster": "Class + G",
    "label_local_cluster": "Class + L",
}
COLORS = ["#029E73", "#D55E00", "#CC78BC"]
PANELS = [
    (
        "Accuracy_mean_Synthetic-only",
        "Accuracy_sd_Synthetic-only",
        r"Synthetic-only accuracy $\uparrow$",
    ),
    (
        "ROC_AUC_mean_Synthetic-only",
        "ROC_AUC_sd_Synthetic-only",
        r"Synthetic-only ROC-AUC $\uparrow$",
    ),
    (
        "Accuracy_mean_Real_+_synthetic",
        "Accuracy_sd_Real_+_synthetic",
        r"Real + synthetic accuracy $\uparrow$",
    ),
    (
        "ROC_AUC_mean_Real_+_synthetic",
        "ROC_AUC_sd_Real_+_synthetic",
        r"Real + synthetic ROC-AUC $\uparrow$",
    ),
]


def validate(frame: pd.DataFrame) -> None:
    required = {
        "Dataset",
        "Architecture",
        "Conditioning",
        *(column for mean, sd, _ in PANELS for column in (mean, sd)),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required utility columns: {sorted(missing)}")

    expected = pd.MultiIndex.from_product(
        [DATASETS, ARCHITECTURES, CONDITIONING],
        names=["Dataset", "Architecture", "Conditioning"],
    )
    indexed = frame.set_index(list(expected.names))
    missing_cells = expected.difference(indexed.index)
    if len(missing_cells):
        raise ValueError(f"Missing utility cells: {missing_cells.tolist()}")
    if indexed.index.duplicated().any():
        duplicates = indexed.index[indexed.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate utility cells: {duplicates}")


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
        bottom=0.21,
        wspace=0.18,
        hspace=0.48,
    )
    group_centres = np.arange(9, dtype=float)
    width = 0.22
    offsets = (np.arange(3) - 1) * width
    ordered = frame.set_index(["Dataset", "Architecture", "Conditioning"])

    for axis, (mean_column, sd_column, title) in zip(
        axes.ravel(), PANELS, strict=True
    ):
        for condition_index, (condition, condition_label) in enumerate(
            CONDITIONING.items()
        ):
            means = []
            standard_deviations = []
            for dataset in DATASETS:
                for architecture in ARCHITECTURES:
                    row = ordered.loc[(dataset, architecture, condition)]
                    means.append(float(row[mean_column]))
                    standard_deviations.append(float(row[sd_column]))
            axis.bar(
                group_centres + offsets[condition_index],
                means,
                yerr=standard_deviations,
                width=width,
                color=COLORS[condition_index],
                edgecolor="none",
                linewidth=0,
                error_kw={
                    "ecolor": "#202020",
                    "elinewidth": 1.2,
                    "capsize": 2.5,
                    "capthick": 1.2,
                },
                label=condition_label,
                zorder=3,
            )

        axis.set_title(title, fontsize=23, pad=10)
        axis.set_xticks(
            group_centres,
            list(ARCHITECTURES.values()) * len(DATASETS),
        )
        axis.set_ylim(0, 1)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.8, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.axvline(2.5, color="#606060", linewidth=1.2)
        axis.axvline(5.5, color="#606060", linewidth=1.2)
        axis.set_xlim(-0.6, 8.6)
        axis.tick_params(axis="x", length=0, pad=7, labelsize=16)
        axis.tick_params(axis="y", labelsize=17)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontweight("bold")
        for centre, dataset_label in zip(
            (1, 4, 7), DATASETS.values(), strict=True
        ):
            axis.text(
                centre,
                -0.18,
                dataset_label,
                ha="center",
                va="top",
                transform=axis.get_xaxis_transform(),
                fontsize=19,
                fontweight="bold",
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        fontsize=19,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    frame = pd.read_csv(INPUT)
    validate(frame)
    plot(frame)


if __name__ == "__main__":
    main()
