from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle


DATASETS = {
    "icaird_cervical100k": "iCAIRD",
    "pathmnist224": "Kather",
    "pcam": "PCam",
}
ARCHITECTURES = {
    "stylegan2ada": "SG2",
    "sd_lora": "SD-LoRA",
    "dcgan": "DCGAN",
}
METRICS = [
    ("nUFD", r"$\Delta$ nUFD $\downarrow$", 2, -1.0),
    ("IP", r"$\Delta$ IP $\uparrow$", 4, 1.0),
    ("IR", r"$\Delta$ IR $\uparrow$", 4, 1.0),
    ("Authenticity", r"$\Delta$ Authenticity $\uparrow$", 4, 1.0),
]
METRIC_GROUPS = {
    "all": METRICS,
    "fidelity-authenticity": [METRICS[0], METRICS[3]],
    "precision-recall": [METRICS[1], METRICS[2]],
}
FONT_SIZE = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("1x4", "2x2", "1x2"), default="1x4")
    parser.add_argument(
        "--metric-group", choices=tuple(METRIC_GROUPS), default="all"
    )
    return parser.parse_args()


def value_text(value: float, precision: int) -> str:
    threshold = 0.5 * 10**-precision
    return "0.0" if abs(value) < threshold else f"{value:.{precision}f}"


def summary_text(values: np.ndarray, precision: int) -> str:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    return f"{value_text(mean, precision)}\n$\\pm$ {value_text(sd, precision)}"


def plot(
    frame: pd.DataFrame,
    output: Path,
    layout: str = "1x4",
    metric_group: str = "all",
) -> None:
    dataset_labels = list(DATASETS.values())
    architecture_labels = list(ARCHITECTURES.values())
    display = frame.replace({"Dataset": DATASETS, "Architecture": ARCHITECTURES})
    plt.rcParams.update(
        {
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "rq1_delta", ["#D55E00", "#F7F7F7", "#0072B2"]
    )
    metrics = METRIC_GROUPS[metric_group]
    if layout == "1x4":
        figure, axes_grid = plt.subplots(
            1, 4, figsize=(31, 9), constrained_layout=True
        )
        panel_font_size = FONT_SIZE
        summary_font_size = FONT_SIZE - 2
    elif layout == "1x2":
        figure, axes_grid = plt.subplots(
            1,
            2,
            figsize=(14, 7),
            constrained_layout=False,
            gridspec_kw={"wspace": -0.04},
        )
        figure.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.18)
        panel_font_size = 17
        summary_font_size = 14
    else:
        figure, axes_grid = plt.subplots(
            2,
            2,
            figsize=(14, 14),
            constrained_layout=False,
            gridspec_kw={"wspace": -0.04, "hspace": 0.30},
        )
        figure.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.14)
        panel_font_size = 17
        summary_font_size = 14
    axes = np.asarray(axes_grid).ravel()

    for panel, (axis, (metric, title, precision, improvement_sign)) in enumerate(
        zip(axes, metrics, strict=True)
    ):
        values = display.pivot(
            index="Dataset", columns="Architecture", values=metric
        ).loc[dataset_labels, architecture_labels].to_numpy(float)
        improvement = improvement_sign * values
        limit = max(float(np.max(np.abs(improvement))), np.finfo(float).eps)
        axis.imshow(
            improvement,
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            aspect="equal",
            extent=(-0.5, 2.5, 2.5, -0.5),
        )
        axis.set_xlim(-0.5, 3.5)
        axis.set_ylim(3.5, -0.5)
        axis.set_title(title, fontsize=panel_font_size + 2, pad=10)
        axis.set_xticks(
            range(4),
            [*architecture_labels, "$\\mu \\pm \\sigma$"],
            fontsize=panel_font_size,
        )
        axis.set_yticks(range(4))
        show_y_labels = panel == 0 if layout in {"1x4", "1x2"} else panel % 2 == 0
        axis.set_yticklabels(
            [*dataset_labels, "$\\mu \\pm \\sigma$"]
            if show_y_labels
            else ["", "", "", ""],
            fontsize=panel_font_size,
        )
        axis.tick_params(length=0, pad=7)

        for row in range(3):
            for column in range(3):
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
                    fontsize=panel_font_size,
                    fontweight="bold",
                )

        panel_norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

        def draw_summary(column: int, row: int, summary_values: np.ndarray) -> None:
            summary_mean = float(np.mean(summary_values))
            encoded_mean = improvement_sign * summary_mean
            axis.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=cmap(panel_norm(encoded_mean)),
                    edgecolor="#9A9A9A",
                    linewidth=1.2,
                    zorder=2,
                )
            )
            text_colour = "white" if abs(encoded_mean) / limit > 0.62 else "#202020"
            axis.text(
                column,
                row,
                summary_text(summary_values, precision),
                ha="center",
                va="center",
                color=text_colour,
                fontsize=summary_font_size,
                fontweight="bold",
                zorder=3,
            )

        for row in range(3):
            draw_summary(3, row, values[row, :])
        for column in range(3):
            draw_summary(column, 3, values[:, column])
        draw_summary(3, 3, values.ravel())

        axis.add_patch(
            Rectangle(
                (-0.5, -0.5),
                3,
                3,
                fill=False,
                edgecolor="#202020",
                linewidth=5.5,
                clip_on=False,
                zorder=5,
            )
        )

        axis.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        axis.grid(which="minor", color="#B8B8B8", linewidth=1.0)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)

    legend_kwargs = {
        "mappable": ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=cmap),
        "orientation": "horizontal",
        "ticks": [-1, 0, 1],
    }
    if layout == "1x4":
        legend = figure.colorbar(
            ax=axes,
            fraction=0.07,
            pad=0.13,
            shrink=0.42,
            aspect=55,
            **legend_kwargs,
        )
    elif layout == "2x2":
        legend_axis = figure.add_axes((0.36, 0.055, 0.40, 0.016))
        legend = figure.colorbar(cax=legend_axis, **legend_kwargs)
    else:
        legend_axis = figure.add_axes((0.36, 0.055, 0.40, 0.022))
        legend = figure.colorbar(cax=legend_axis, **legend_kwargs)
    legend.ax.set_xticklabels(["Deterioration", "No change", "Improvement"])
    legend.ax.tick_params(labelsize=panel_font_size, length=0, pad=8)
    for label in legend.ax.get_xticklabels():
        label.set_fontweight("bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    plot(
        pd.read_csv(args.input),
        args.output,
        layout=args.layout,
        metric_group=args.metric_group,
    )


if __name__ == "__main__":
    main()
