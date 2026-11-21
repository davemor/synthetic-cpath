from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "final-results" / "quality-results-with-nufd.csv"
OUTPUTS = {
    "fidelity-authenticity": ROOT
    / "final-results"
    / "rq2-paired-subcell-nufd-authenticity",
    "precision-recall": ROOT / "final-results" / "rq2-paired-subcell-ip-ir",
}
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
    "label_global_cluster": "G",
    "label_local_cluster": "L",
}
METRIC_GROUPS = {
    "fidelity-authenticity": [
        ("nUFD", r"$\Delta$ nUFD $\downarrow$", 2, -1.0),
        ("Authenticity", r"$\Delta$ Authenticity $\uparrow$", 4, 1.0),
    ],
    "precision-recall": [
        ("IP", r"$\Delta$ IP $\uparrow$", 4, 1.0),
        ("IR", r"$\Delta$ IR $\uparrow$", 4, 1.0),
    ],
}


def paired_differences(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["Dataset", "Architecture"]
    baseline = frame[frame["Conditioning"] == "true_label"].set_index(keys)
    rows = []
    for comparison in COMPARISONS:
        compared = frame[frame["Conditioning"] == comparison].set_index(keys)
        for dataset in DATASETS:
            for architecture in ARCHITECTURES:
                key = (dataset, architecture)
                row = {
                    "Dataset": dataset,
                    "Architecture": architecture,
                    "Comparison": comparison,
                }
                for metric_group in METRIC_GROUPS.values():
                    for metric, _, _, _ in metric_group:
                        row[metric] = compared.loc[key, metric] - baseline.loc[key, metric]
                rows.append(row)
    return pd.DataFrame(rows)


def value_text(value: float, precision: int) -> str:
    threshold = 0.5 * 10**-precision
    return "0.0" if abs(value) < threshold else f"{value:.{precision}f}"


def metric_matrix(frame: pd.DataFrame, metric: str) -> np.ndarray:
    values = np.empty((3, 6), dtype=float)
    indexed = frame.set_index(["Dataset", "Architecture", "Comparison"])
    for row, dataset in enumerate(DATASETS):
        for architecture_index, architecture in enumerate(ARCHITECTURES):
            for comparison_index, comparison in enumerate(COMPARISONS):
                column = architecture_index * 2 + comparison_index
                values[row, column] = indexed.loc[
                    (dataset, architecture, comparison), metric
                ]
    return values


def summary_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    global_values = values[:, [0, 2, 4]]
    local_values = values[:, [1, 3, 5]]
    means = np.column_stack(
        [global_values.mean(axis=1), local_values.mean(axis=1)]
    )
    standard_deviations = np.column_stack(
        [global_values.std(axis=1, ddof=1), local_values.std(axis=1, ddof=1)]
    )
    return means, standard_deviations


def plot_portrait_group(frame: pd.DataFrame, group: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "font.weight": "bold",
            "axes.titleweight": "bold",
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "paired_delta", ["#D55E00", "#F7F7F7", "#0072B2"]
    )
    metrics = METRIC_GROUPS[group]
    figure, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=False)
    figure.subplots_adjust(
        left=0.14,
        right=0.98,
        top=0.96,
        bottom=0.18,
        hspace=0.48,
    )

    for panel, (axis, (metric, title, precision, improvement_sign)) in enumerate(
        zip(axes.ravel(), metrics, strict=True)
    ):
        scale = 1.0
        values = metric_matrix(frame, metric) * scale
        means, standard_deviations = summary_columns(values)
        column_means = values.mean(axis=0)
        column_standard_deviations = values.std(axis=0, ddof=1)
        global_values = values[:, [0, 2, 4]].ravel()
        local_values = values[:, [1, 3, 5]].ravel()
        overall_means = np.array([global_values.mean(), local_values.mean()])
        overall_standard_deviations = np.array(
            [global_values.std(ddof=1), local_values.std(ddof=1)]
        )
        improvement = improvement_sign * values
        limit = max(float(np.max(np.abs(improvement))), np.finfo(float).eps)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        display_columns = [0, 2, 4, 1, 3, 5]
        display_values = values[:, display_columns]
        display_improvement = improvement[:, display_columns]
        gutter = 0.10
        column_positions = np.array(
            [0.0, 1.0, 2.0, 3.0, 4.0 + gutter, 5.0 + gutter, 6.0 + gutter, 7.0 + gutter]
        )
        axis.imshow(
            display_improvement[:, :3],
            cmap=cmap,
            norm=norm,
            aspect="equal",
            extent=(-0.5, 2.5, 2.5, -0.5),
        )
        axis.imshow(
            display_improvement[:, 3:],
            cmap=cmap,
            norm=norm,
            aspect="equal",
            extent=(3.5 + gutter, 6.5 + gutter, 2.5, -0.5),
        )
        axis.set_xlim(-0.5, 7.5 + gutter)
        axis.set_ylim(3.5, -0.5)
        axis.set_title(title, fontsize=22, pad=9)
        axis.set_xticks(
            column_positions,
            [
                "SG2",
                "SD",
                "DCG",
                "$\\mu_G$\n$\\pm\\sigma_G$",
                "SG2",
                "SD",
                "DCG",
                "$\\mu_L$\n$\\pm\\sigma_L$",
            ],
            fontsize=15,
        )
        axis.set_yticks(range(4))
        axis.set_yticklabels(
            [*DATASETS.values(), r"$\mu \pm \sigma$"],
            fontsize=17,
        )
        axis.tick_params(length=0, pad=6)

        for row in range(3):
            for column in range(6):
                display_column = (
                    column if column < 3 else column + 1 + gutter
                )
                colour = (
                    "white"
                    if abs(display_improvement[row, column]) / limit > 0.65
                    else "#202020"
                )
                axis.text(
                    display_column,
                    row,
                    value_text(display_values[row, column], precision),
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=13,
                    fontweight="bold",
                    zorder=3,
                )
            for summary_column in range(2):
                mean = means[row, summary_column]
                encoded_mean = improvement_sign * mean
                column = 3 + summary_column * (4 + gutter)
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor=cmap(norm(encoded_mean)),
                        edgecolor="#A0A0A0",
                        linewidth=1.0,
                        zorder=2,
                    )
                )
                colour = (
                    "white" if abs(encoded_mean) / limit > 0.65 else "#202020"
                )
                axis.text(
                    column,
                    row,
                    f"{value_text(mean, precision)}\n"
                    rf"$\pm$ {value_text(standard_deviations[row, summary_column], precision)}",
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=12,
                    fontweight="bold",
                    zorder=3,
                )

        for display_index, source_column in enumerate(display_columns):
            column = (
                display_index
                if display_index < 3
                else display_index + 1 + gutter
            )
            encoded_mean = improvement_sign * column_means[source_column]
            axis.add_patch(
                Rectangle(
                    (column - 0.5, 2.5),
                    1,
                    1,
                    facecolor=cmap(norm(encoded_mean)),
                    edgecolor="#A0A0A0",
                    linewidth=1.0,
                    zorder=2,
                )
            )
            colour = "white" if abs(encoded_mean) / limit > 0.65 else "#202020"
            axis.text(
                column,
                3,
                f"{value_text(column_means[source_column], precision)}\n"
                rf"$\pm$ {value_text(column_standard_deviations[source_column], precision)}",
                ha="center",
                va="center",
                color=colour,
                fontsize=11,
                fontweight="bold",
                zorder=3,
            )

        for summary_column in range(2):
            encoded_mean = improvement_sign * overall_means[summary_column]
            column = 3 + summary_column * (4 + gutter)
            axis.add_patch(
                Rectangle(
                    (column - 0.5, 2.5),
                    1,
                    1,
                    facecolor=cmap(norm(encoded_mean)),
                    edgecolor="#A0A0A0",
                    linewidth=1.0,
                    zorder=2,
                )
            )
            colour = "white" if abs(encoded_mean) / limit > 0.65 else "#202020"
            axis.text(
                column,
                3,
                f"{value_text(overall_means[summary_column], precision)}\n"
                rf"$\pm$ {value_text(overall_standard_deviations[summary_column], precision)}",
                ha="center",
                va="center",
                color=colour,
                fontsize=11,
                fontweight="bold",
                zorder=3,
            )

        for x in (-0.5, 0.5, 1.5, 2.5, 3.5):
            axis.plot([x, x], [-0.5, 3.5], color="#B8B8B8", linewidth=0.9)
        for x in np.array((3.5, 4.5, 5.5, 6.5, 7.5)) + gutter:
            axis.plot([x, x], [-0.5, 3.5], color="#B8B8B8", linewidth=0.9)
        for y in (-0.5, 0.5, 1.5, 2.5, 3.5):
            axis.plot([-0.5, 3.5], [y, y], color="#B8B8B8", linewidth=0.9)
            axis.plot(
                [3.5 + gutter, 7.5 + gutter],
                [y, y],
                color="#B8B8B8",
                linewidth=0.9,
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
        axis.add_patch(
            Rectangle(
                (3.5 + gutter, -0.5),
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
        for centre, conditioning in (
            (1.0, "Global"),
            (5.0 + gutter, "Local"),
        ):
            axis.text(
                centre,
                -0.22,
                conditioning,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=17,
                fontweight="bold",
            )

    legend_axis = figure.add_axes((0.30, 0.030, 0.50, 0.012))
    legend = figure.colorbar(
        ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=cmap),
        cax=legend_axis,
        orientation="horizontal",
        ticks=[-1, 0, 1],
    )
    legend.ax.set_xticklabels(["Deterioration", "No change", "Improvement"])
    legend.ax.tick_params(labelsize=15, length=0, pad=6)
    for label in legend.ax.get_xticklabels():
        label.set_fontweight("bold")

    output = OUTPUTS[group]
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_group(frame: pd.DataFrame, group: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 19,
            "font.weight": "bold",
            "axes.titleweight": "bold",
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "paired_delta", ["#D55E00", "#F7F7F7", "#0072B2"]
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=False)
    figure.subplots_adjust(left=0.10, right=0.99, top=0.91, bottom=0.31, wspace=0.16)

    for panel, (axis, (metric, title, precision, improvement_sign)) in enumerate(
        zip(axes, METRIC_GROUPS[group], strict=True)
    ):
        values = metric_matrix(frame, metric)
        improvement = improvement_sign * values
        limit = max(float(np.max(np.abs(improvement))), np.finfo(float).eps)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        axis.imshow(improvement, cmap=cmap, norm=norm, aspect="equal")
        axis.set_title(title, fontsize=25, pad=12)
        axis.set_xticks(range(6), ["G", "L"] * 3, fontsize=18)
        axis.set_yticks(range(3))
        axis.set_yticklabels(
            list(DATASETS.values()) if panel == 0 else ["", "", ""],
            fontsize=20,
        )
        axis.tick_params(length=0, pad=8)

        for row in range(3):
            for column in range(6):
                colour = (
                    "white"
                    if abs(improvement[row, column]) / limit > 0.62
                    else "#202020"
                )
                axis.text(
                    column,
                    row,
                    value_text(values[row, column], precision),
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=16,
                    fontweight="bold",
                )

        axis.set_xticks(np.arange(-0.5, 6, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        axis.grid(which="minor", color="#B8B8B8", linewidth=1.0)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.axvline(1.5, color="#202020", linewidth=3.0)
        axis.axvline(3.5, color="#202020", linewidth=3.0)
        axis.add_patch(
            Rectangle(
                (-0.5, -0.5),
                6,
                3,
                fill=False,
                edgecolor="#202020",
                linewidth=5.0,
                clip_on=False,
                zorder=5,
            )
        )
        for spine in axis.spines.values():
            spine.set_visible(False)
        for centre, architecture in zip((0.5, 2.5, 4.5), ARCHITECTURES.values(), strict=True):
            axis.text(
                centre,
                -0.23,
                architecture,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=20,
                fontweight="bold",
            )

    legend_axis = figure.add_axes((0.34, 0.065, 0.40, 0.022))
    legend = figure.colorbar(
        ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=cmap),
        cax=legend_axis,
        orientation="horizontal",
        ticks=[-1, 0, 1],
    )
    legend.ax.set_xticklabels(["Deterioration", "No change", "Improvement"])
    legend.ax.tick_params(labelsize=18, length=0, pad=8)
    for label in legend.ax.get_xticklabels():
        label.set_fontweight("bold")

    output = OUTPUTS[group]
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    frame = paired_differences(pd.read_csv(INPUT))
    for group in METRIC_GROUPS:
        plot_portrait_group(frame, group)


if __name__ == "__main__":
    main()
