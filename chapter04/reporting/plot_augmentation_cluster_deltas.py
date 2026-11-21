from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUTS = [
    Path("utility-report-sources/icaird.csv"),
    Path("utility-report-sources/kather.csv"),
    Path("utility-report-sources/pcam.csv"),
]
OUTPUT = Path("final-results")

DATASETS = [
    ("icaird_cervical100k", "iCAIRD Cervical 100K"),
    ("pathmnist224", "Kather 100K"),
    ("pcam", "PCam"),
]
ARCHITECTURES = [
    ("stylegan2ada", "StyleGAN2-ADA"),
    ("sd_lora", "SD-LoRA"),
    ("dcgan", "DCGAN"),
]
CONDITIONS = [
    ("true_label", "Class label", "#5B5F97"),
    ("label_global_cluster", "Class + global", "#2878B5"),
    ("label_local_cluster", "Class + local", "#D97706"),
]
METRICS = [("accuracy", "Accuracy delta"), ("macro_roc_auc_ovr", "ROC-AUC delta")]


def load_seed_results() -> pd.DataFrame:
    results = pd.concat([pd.read_csv(path) for path in INPUTS], ignore_index=True)
    results = results[
        (results["family"] == "utility")
        & results["metric"].isin([metric for metric, _ in METRICS])
    ].copy()
    results["value"] = pd.to_numeric(results["value"], errors="raise")
    key = ["dataset", "generator_model", "conditioning", "seed", "metric"]
    conflicts = results.groupby(key)["value"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting duplicate utility results in source reports")
    return results.drop_duplicates(key)


def calculate_deltas(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, dataset_label in DATASETS:
        for architecture, architecture_label in ARCHITECTURES:
            subset = results[
                (results["dataset"] == dataset)
                & (results["generator_model"] == architecture)
            ]
            for condition, condition_label, _ in CONDITIONS:
                condition_name = f"real_plus_synthetic_{condition}"
                for metric, metric_label in METRICS:
                    augmented = subset[
                        (subset["conditioning"] == condition_name)
                        & (subset["metric"] == metric)
                    ].set_index("seed")["value"]
                    baseline = results[
                        (results["dataset"] == dataset)
                        & (results["generator_model"] == "real")
                        & (results["conditioning"] == "real_only")
                        & (results["metric"] == metric)
                    ].set_index("seed")["value"]
                    paired = augmented.subtract(baseline, fill_value=np.nan).dropna()
                    if list(paired.index.sort_values()) != [0, 1, 2]:
                        raise ValueError(
                            f"Missing paired seeds for {dataset}, {architecture}, "
                            f"{condition}, {metric}"
                        )
                    rows.append(
                        {
                            "dataset": dataset_label,
                            "architecture": architecture_label,
                            "conditioning": condition_label,
                            "metric": metric_label,
                            "delta": paired.mean(),
                            "sd": paired.std(ddof=1),
                            "n_seeds": len(paired),
                        }
                    )
    return pd.DataFrame(rows)


def plot(deltas: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 18,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.0), sharey="row")
    x = np.arange(len(ARCHITECTURES))
    width = 0.25

    for row, (_, metric_label) in enumerate(METRICS):
        metric_data = deltas[deltas["metric"] == metric_label]
        extent = float((metric_data["delta"].abs() + metric_data["sd"]).max()) * 1.13
        for column, (_, dataset_label) in enumerate(DATASETS):
            ax = axes[row, column]
            panel = metric_data[metric_data["dataset"] == dataset_label]
            for offset, (_, condition_label, colour) in enumerate(CONDITIONS):
                values = (
                    panel[panel["conditioning"] == condition_label]
                    .set_index("architecture")
                    .reindex([label for _, label in ARCHITECTURES])
                )
                ax.bar(
                    x + (offset - 1) * width,
                    values["delta"],
                    width,
                    yerr=values["sd"],
                    capsize=3,
                    color=colour,
                    edgecolor="white",
                    linewidth=0.6,
                    label=condition_label,
                    error_kw={"elinewidth": 1.0, "capthick": 1.0},
                )
            ax.axhline(0, color="#202020", linewidth=0.9)
            ax.set_ylim(-extent, extent)
            ax.set_xticks(x, [label for _, label in ARCHITECTURES])
            if row == 1:
                ax.set_xlabel("Architecture")
            ax.grid(axis="y", color="#d8d8d8", linewidth=0.6, alpha=0.8)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(dataset_label, fontweight="bold")
            if column == 0:
                short_metric = "accuracy" if row == 0 else "ROC-AUC"
                ax.set_ylabel(rf"$\Delta$ {short_metric} vs real-only")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.2, h_pad=1.5)
    fig.savefig(OUTPUT / "augmentation-cluster-delta-bars.png", dpi=300)
    fig.savefig(OUTPUT / "augmentation-cluster-delta-bars.pdf")
    plt.close(fig)


def plot_each_dataset(deltas: pd.DataFrame) -> None:
    x = np.arange(len(ARCHITECTURES))
    width = 0.25
    for _, dataset_label in DATASETS:
        fig, axes = plt.subplots(2, 1, figsize=(8.8, 10.8))
        dataset_data = deltas[deltas["dataset"] == dataset_label]
        for row, (_, metric_label) in enumerate(METRICS):
            ax = axes[row]
            panel = dataset_data[dataset_data["metric"] == metric_label]
            extent = float((panel["delta"].abs() + panel["sd"]).max()) * 1.13
            for offset, (_, condition_label, colour) in enumerate(CONDITIONS):
                values = (
                    panel[panel["conditioning"] == condition_label]
                    .set_index("architecture")
                    .reindex([label for _, label in ARCHITECTURES])
                )
                ax.bar(
                    x + (offset - 1) * width,
                    values["delta"],
                    width,
                    yerr=values["sd"],
                    capsize=4,
                    color=colour,
                    edgecolor="white",
                    linewidth=0.7,
                    label=condition_label,
                    error_kw={"elinewidth": 1.2, "capthick": 1.2},
                )
            ax.axhline(0, color="#202020", linewidth=1.0)
            ax.set_ylim(-extent, extent)
            ax.set_xticks(x, [label for _, label in ARCHITECTURES])
            ax.set_xlabel("Architecture")
            short_metric = "accuracy" if row == 0 else "ROC-AUC"
            ax.set_ylabel(rf"$\Delta$ {short_metric} vs real-only")
            ax.set_title(short_metric.capitalize(), fontweight="bold")
            ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=3,
            frameon=False,
        )
        fig.suptitle(dataset_label, fontsize=22, fontweight="bold", y=0.875)
        fig.subplots_adjust(
            left=0.15,
            right=0.98,
            bottom=0.09,
            top=0.78,
            hspace=0.50,
        )
        slug = {
            "iCAIRD Cervical 100K": "icaird-cervical100k",
            "Kather 100K": "kather100k",
            "PCam": "pcam",
        }[dataset_label]
        fig.savefig(OUTPUT / f"augmentation-cluster-delta-bars-{slug}.png", dpi=300)
        fig.savefig(OUTPUT / f"augmentation-cluster-delta-bars-{slug}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    frame = calculate_deltas(load_seed_results())
    frame.to_csv(OUTPUT / "augmentation-cluster-delta-bars.csv", index=False)
    plot(frame)
    plot_each_dataset(frame)
