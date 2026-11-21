from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "utility-report-sources"
OUTPUT = ROOT / "final-results" / "replacement-vs-real-utility-grid"
DELTA_OUTPUT = ROOT / "final-results" / "replacement-vs-real-paired-deltas.csv"

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
CONDITIONS = {
    "synthetic_only_true_label": "Class",
    "synthetic_only_label_global_cluster": "Class + global",
    "synthetic_only_label_local_cluster": "Class + local",
}
METRICS = {
    "accuracy": "Accuracy",
    "macro_roc_auc_ovr": "ROC-AUC",
}


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(SOURCE_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No utility sources found in {SOURCE_DIR}")
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    common = (
        frame["family"].eq("utility")
        & frame["dataset"].isin(DATASETS)
        & frame["metric"].isin(METRICS)
    )
    synthetic = frame.loc[
        common
        & frame["generator_model"].isin(ARCHITECTURES)
        & frame["conditioning"].isin(CONDITIONS)
    ].copy()
    real = frame.loc[common & frame["conditioning"].eq("real_only")].copy()

    synthetic_keys = [
        "dataset",
        "generator_model",
        "conditioning",
        "seed",
        "metric",
    ]
    real_keys = ["dataset", "conditioning", "seed", "metric"]
    for selected, keys, label in (
        (synthetic, synthetic_keys, "synthetic"),
        (real, real_keys, "real-only"),
    ):
        conflicts = selected.groupby(keys)["value"].nunique().gt(1)
        if conflicts.any():
            raise ValueError(f"Conflicting duplicate {label} utility records.")

    synthetic = synthetic.drop_duplicates(synthetic_keys).reset_index(drop=True)
    real = real.drop_duplicates(real_keys).reset_index(drop=True)
    synthetic["value"] = pd.to_numeric(synthetic["value"], errors="raise")
    real["value"] = pd.to_numeric(real["value"], errors="raise")
    if len(synthetic) != 162:
        raise ValueError(f"Expected 162 synthetic values, found {len(synthetic)}.")
    if len(real) != 18:
        raise ValueError(f"Expected 18 real-only values, found {len(real)}.")
    return synthetic, real


def paired_deltas(synthetic: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    baseline = real.set_index(["dataset", "seed", "metric"])["value"]
    rows = []
    for record in synthetic.itertuples(index=False):
        baseline_value = baseline.loc[(record.dataset, record.seed, record.metric)]
        rows.append(
            {
                "Dataset": record.dataset,
                "Architecture": record.generator_model,
                "Conditioning": CONDITIONS[record.conditioning],
                "Seed": int(record.seed),
                "Metric": METRICS[record.metric],
                "Synthetic-only": float(record.value),
                "Real-only": float(baseline_value),
                "Delta": float(record.value - baseline_value),
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != 162:
        raise ValueError(f"Expected 162 paired differences, found {len(output)}.")
    return output


def value_text(value: float) -> str:
    return "0.0" if abs(value) < 0.00005 else f"{value:.4f}"


def plot(frame: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 17,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )
    is_augmentation = "augmentation" in OUTPUT.name
    colours = (
        ["#D55E00", "#F7F7F7", "#0072B2"]
        if is_augmentation
        else ["#D55E00", "#F7F7F7"]
    )
    cmap = LinearSegmentedColormap.from_list("utility_delta", colours)
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(10, 14),
        constrained_layout=False,
        gridspec_kw={"wspace": -0.03, "hspace": 0.22},
    )
    figure.subplots_adjust(left=0.17, right=0.98, top=0.94, bottom=0.13)

    cell_means = (
        frame.groupby(["Metric", "Conditioning", "Dataset", "Architecture"])[
            "Delta"
        ]
        .mean()
        .to_numpy(float)
    )
    limit = max(float(np.abs(cell_means).max()), np.finfo(float).eps)
    if is_augmentation:
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    else:
        if float(cell_means.max()) > np.finfo(float).eps:
            raise ValueError("One-sided replacement scale requires non-positive deltas.")
        norm = Normalize(vmin=-limit, vmax=0.0)

    for row, condition in enumerate(CONDITIONS.values()):
        for column, metric in enumerate(METRICS.values()):
            axis = axes[row, column]
            means = np.empty((3, 3), dtype=float)
            standard_deviations = np.empty((3, 3), dtype=float)
            for dataset_index, dataset in enumerate(DATASETS):
                for architecture_index, architecture in enumerate(ARCHITECTURES):
                    values = frame.loc[
                        frame["Metric"].eq(metric)
                        & frame["Conditioning"].eq(condition)
                        & frame["Dataset"].eq(dataset)
                        & frame["Architecture"].eq(architecture),
                        "Delta",
                    ].to_numpy(float)
                    if len(values) != 3:
                        raise ValueError(
                            f"Expected three paired seeds for {metric}, {condition}, "
                            f"{dataset}, {architecture}; found {len(values)}."
                        )
                    means[dataset_index, architecture_index] = values.mean()
                    standard_deviations[dataset_index, architecture_index] = values.std(
                        ddof=1
                    )

            axis.imshow(
                means,
                cmap=cmap,
                norm=norm,
                aspect="equal",
                extent=(-0.5, 2.5, 2.5, -0.5),
            )
            axis.set_xlim(-0.5, 3.5)
            axis.set_ylim(3.5, -0.5)
            if row == 0:
                axis.set_title(rf"$\Delta$ {metric} $\uparrow$", fontsize=20, pad=10)
            axis.set_xticks(
                range(4),
                [*ARCHITECTURES.values(), r"$\mu \pm \sigma$"],
                fontsize=13,
            )
            axis.set_yticks(range(4))
            axis.set_yticklabels(
                [*DATASETS.values(), r"$\mu \pm \sigma$"]
                if column == 0
                else ["", "", "", ""],
                fontsize=15,
            )
            axis.tick_params(length=0, pad=7)
            axis.set_xticks(np.arange(-0.5, 4, 1), minor=True)
            axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
            axis.grid(which="minor", color="#B8B8B8", linewidth=1.0)
            axis.tick_params(which="minor", bottom=False, left=False)

            for dataset_index in range(3):
                for architecture_index in range(3):
                    mean = means[dataset_index, architecture_index]
                    colour = "white" if abs(mean) / limit > 0.64 else "#202020"
                    axis.text(
                        architecture_index,
                        dataset_index,
                        f"{value_text(mean)}\n"
                        rf"$\pm$ {value_text(standard_deviations[dataset_index, architecture_index])}",
                        ha="center",
                        va="center",
                        color=colour,
                        fontsize=10,
                        fontweight="bold",
                    )

            summaries: list[tuple[int, int, np.ndarray]] = []
            for dataset_index, dataset in enumerate(DATASETS):
                values = frame.loc[
                    frame["Metric"].eq(metric)
                    & frame["Conditioning"].eq(condition)
                    & frame["Dataset"].eq(dataset),
                    "Delta",
                ].to_numpy(float)
                summaries.append((3, dataset_index, values))
            for architecture_index, architecture in enumerate(ARCHITECTURES):
                values = frame.loc[
                    frame["Metric"].eq(metric)
                    & frame["Conditioning"].eq(condition)
                    & frame["Architecture"].eq(architecture),
                    "Delta",
                ].to_numpy(float)
                summaries.append((architecture_index, 3, values))
            overall = frame.loc[
                frame["Metric"].eq(metric)
                & frame["Conditioning"].eq(condition),
                "Delta",
            ].to_numpy(float)
            summaries.append((3, 3, overall))

            for summary_column, summary_row, values in summaries:
                mean = values.mean()
                standard_deviation = values.std(ddof=1)
                axis.add_patch(
                    Rectangle(
                        (summary_column - 0.5, summary_row - 0.5),
                        1,
                        1,
                        facecolor=cmap(norm(mean)),
                        edgecolor="#B8B8B8",
                        linewidth=1.0,
                        zorder=2,
                    )
                )
                colour = "white" if abs(mean) / limit > 0.64 else "#202020"
                axis.text(
                    summary_column,
                    summary_row,
                    f"{value_text(mean)}\n"
                    rf"$\pm$ {value_text(standard_deviation)}",
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=9,
                    fontweight="bold",
                    zorder=3,
                )

            axis.add_patch(
                Rectangle(
                    (-0.5, -0.5),
                    3,
                    3,
                    fill=False,
                    edgecolor="#202020",
                    linewidth=4.5,
                    clip_on=False,
                    zorder=5,
                )
            )
            for spine in axis.spines.values():
                spine.set_visible(False)

        axes[row, 0].set_ylabel(
            condition,
            fontsize=17,
            fontweight="bold",
            labelpad=48,
        )

    legend_axis = figure.add_axes((0.34, 0.045, 0.44, 0.014))
    legend_ticks = [-limit, 0, limit] if is_augmentation else [-limit, 0]
    legend_labels = (
        ["Worse than real", "No difference", "Better than real"]
        if is_augmentation
        else ["Worse than real", "No difference"]
    )
    legend = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=legend_axis,
        orientation="horizontal",
        ticks=legend_ticks,
    )
    legend.ax.set_xticklabels(legend_labels)
    legend.ax.tick_params(labelsize=15, length=0, pad=7)
    for label in legend.ax.get_xticklabels():
        label.set_fontweight("bold")

    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot utility differences from the matched real-only baseline."
    )
    parser.add_argument(
        "--training-regime",
        choices=("replacement", "augmentation"),
        default="replacement",
    )
    return parser.parse_args()


def main() -> None:
    global CONDITIONS, DELTA_OUTPUT, OUTPUT

    args = parse_args()
    if args.training_regime == "augmentation":
        CONDITIONS = {
            "real_plus_synthetic_true_label": "Class",
            "real_plus_synthetic_label_global_cluster": "Class + global",
            "real_plus_synthetic_label_local_cluster": "Class + local",
        }
        OUTPUT = ROOT / "final-results" / "augmentation-vs-real-utility-grid"
        DELTA_OUTPUT = (
            ROOT / "final-results" / "augmentation-vs-real-paired-deltas.csv"
        )
    synthetic, real = load_results()
    deltas = paired_deltas(synthetic, real)
    deltas.to_csv(DELTA_OUTPUT, index=False)
    plot(deltas)


if __name__ == "__main__":
    main()
