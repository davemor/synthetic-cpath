from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "final-results" / "quality-results-with-nufd.csv"
OUTPUT = ROOT / "final-results" / "rq3-quality-by-architecture"
SUMMARY_OUTPUT = ROOT / "final-results" / "rq3-quality-by-architecture-summary.csv"

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
CONDITIONING = [
    "unconditioned",
    "pseudo_label",
    "true_label",
    "label_global_cluster",
    "label_local_cluster",
]
METRICS = [
    ("nUFD", r"nUFD $\downarrow$", 2, -1.0),
    ("IP", r"IP $\uparrow$", 4, 1.0),
    ("IR", r"IR $\uparrow$", 4, 1.0),
    ("Authenticity", r"Authenticity $\uparrow$", 4, 1.0),
]


def validate_input(frame: pd.DataFrame) -> None:
    required = {
        "Dataset",
        "Architecture",
        "Conditioning",
        *(metric for metric, _, _, _ in METRICS),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    expected = pd.MultiIndex.from_product(
        [DATASETS, ARCHITECTURES, CONDITIONING],
        names=["Dataset", "Architecture", "Conditioning"],
    )
    indexed = frame.set_index(list(expected.names))
    missing_cells = expected.difference(indexed.index)
    if len(missing_cells):
        raise ValueError(f"Missing matched quality cells: {missing_cells.tolist()}")
    if indexed.index.duplicated().any():
        duplicates = indexed.index[indexed.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate quality cells: {duplicates}")


def architecture_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index(["Dataset", "Architecture", "Conditioning"])
    rows: list[dict[str, str | float | int]] = []
    for dataset, dataset_label in DATASETS.items():
        for architecture, architecture_label in ARCHITECTURES.items():
            for metric, _, _, _ in METRICS:
                values = indexed.loc[
                    [
                        (dataset, architecture, condition)
                        for condition in CONDITIONING
                    ],
                    metric,
                ].to_numpy(float)
                rows.append(
                    {
                        "Dataset": dataset,
                        "Dataset label": dataset_label,
                        "Architecture": architecture,
                        "Architecture label": architecture_label,
                        "Metric": metric,
                        "Mean": values.mean(),
                        "SD across conditioning regimes": values.std(ddof=1),
                        "Number of conditioning regimes": len(values),
                    }
                )
    return pd.DataFrame(rows)


def value_text(value: float, precision: int) -> str:
    threshold = 0.5 * 10**-precision
    return "0.0" if abs(value) < threshold else f"{value:.{precision}f}"


def plot(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    metrics: list[tuple[str, str, int, float]],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 17,
            "font.weight": "bold",
            "axes.titleweight": "bold",
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "architecture_delta", ["#D55E00", "#F7F7F7", "#0072B2"]
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 14),
        constrained_layout=False,
        gridspec_kw={"wspace": -0.04, "hspace": 0.30},
    )
    figure.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.96,
        bottom=0.14,
    )

    for panel, (axis, (metric, title, precision, improvement_sign)) in enumerate(
        zip(axes.ravel(), metrics, strict=True)
    ):
        metric_summary = summary[summary["Metric"] == metric]
        means = (
            metric_summary.pivot(
                index="Dataset label", columns="Architecture label", values="Mean"
            )
            .loc[list(DATASETS.values()), list(ARCHITECTURES.values())]
            .to_numpy(float)
        )
        standard_deviations = (
            metric_summary.pivot(
                index="Dataset label",
                columns="Architecture label",
                values="SD across conditioning regimes",
            )
            .loc[list(DATASETS.values()), list(ARCHITECTURES.values())]
            .to_numpy(float)
        )
        quality = improvement_sign * means
        quality_min = float(quality.min())
        quality_max = float(quality.max())
        if np.isclose(quality_min, quality_max):
            quality_min -= 0.5
            quality_max += 0.5
        norm = Normalize(vmin=quality_min, vmax=quality_max)
        axis.imshow(
            quality,
            cmap=cmap,
            norm=norm,
            aspect="equal",
            extent=(-0.5, 2.5, 2.5, -0.5),
        )
        axis.set_xlim(-0.5, 3.5)
        axis.set_ylim(3.5, -0.5)
        axis.set_title(title, fontsize=22, pad=10)
        axis.set_xticks(
            range(4),
            [*ARCHITECTURES.values(), r"$\mu \pm \sigma$"],
            fontsize=14,
        )
        axis.set_yticks(range(4))
        axis.set_yticklabels(
            [*DATASETS.values(), r"$\mu \pm \sigma$"]
            if panel % 2 == 0
            else ["", "", "", ""],
            fontsize=17,
        )
        axis.tick_params(length=0, pad=7)
        axis.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        axis.grid(which="minor", color="#B8B8B8", linewidth=1.0)
        axis.tick_params(which="minor", bottom=False, left=False)

        for row in range(3):
            for column in range(3):
                colour = (
                    "white"
                    if norm(quality[row, column]) < 0.18
                    or norm(quality[row, column]) > 0.82
                    else "#202020"
                )
                axis.text(
                    column,
                    row,
                    f"{value_text(means[row, column], precision)}\n"
                    rf"$\pm$ {value_text(standard_deviations[row, column], precision)}",
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=14,
                    fontweight="bold",
                )

        row_values = [
            frame.loc[
                (frame["Dataset"] == dataset)
                & frame["Architecture"].isin(ARCHITECTURES)
                & frame["Conditioning"].isin(CONDITIONING),
                metric,
            ].to_numpy(float)
            for dataset in DATASETS
        ]
        column_values = [
            frame.loc[
                frame["Dataset"].isin(DATASETS)
                & (frame["Architecture"] == architecture)
                & frame["Conditioning"].isin(CONDITIONING),
                metric,
            ].to_numpy(float)
            for architecture in ARCHITECTURES
        ]
        overall_values = frame.loc[
            frame["Dataset"].isin(DATASETS)
            & frame["Architecture"].isin(ARCHITECTURES)
            & frame["Conditioning"].isin(CONDITIONING),
            metric,
        ].to_numpy(float)
        summaries = [
            *(
                (3, row, row_values[row].mean(), row_values[row].std(ddof=1))
                for row in range(3)
            ),
            *(
                (
                    column,
                    3,
                    column_values[column].mean(),
                    column_values[column].std(ddof=1),
                )
                for column in range(3)
            ),
            (3, 3, overall_values.mean(), overall_values.std(ddof=1)),
        ]
        for column, row, mean, standard_deviation in summaries:
            encoded_mean = improvement_sign * mean
            axis.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=cmap(norm(encoded_mean)),
                    edgecolor="#B8B8B8",
                    linewidth=1.0,
                    zorder=2,
                )
            )
            colour = (
                "white"
                if norm(encoded_mean) < 0.18 or norm(encoded_mean) > 0.82
                else "#202020"
            )
            axis.text(
                column,
                row,
                f"{value_text(mean, precision)}\n"
                rf"$\pm$ {value_text(standard_deviation, precision)}",
                ha="center",
                va="center",
                color=colour,
                fontsize=11,
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

    legend_axis = figure.add_axes((0.36, 0.055, 0.40, 0.016))
    legend = figure.colorbar(
        ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=cmap),
        cax=legend_axis,
        orientation="horizontal",
        ticks=[-1, 0, 1],
    )
    legend.ax.set_xticklabels(["Worse", "Mid-range", "Better"])
    legend.ax.tick_params(labelsize=16, length=0, pad=7)
    for label in legend.ax.get_xticklabels():
        label.set_fontweight("bold")

    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    frame = pd.read_csv(INPUT)
    validate_input(frame)
    summary = architecture_summaries(frame)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    plot(frame, summary, METRICS, OUTPUT)


if __name__ == "__main__":
    main()
