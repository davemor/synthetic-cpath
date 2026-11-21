from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "final-results"
EARLIER_QUALITY = RESULTS / "quality-results-with-nufd.csv"
LOW_D_QUALITY = RESULTS / "low-d-global-quality-results.csv"
LOW_D_UTILITY_SEEDS = RESULTS / "low-d-global-utility-seed-results.csv"
LOW_D_UTILITY_LONG_SOURCE = Path(
    "/private/tmp/balanced_global_results/"
    "main_pathmnist224_balanced_global_cluster/utility_metrics.csv"
)
EARLIER_UTILITY_SOURCES = ROOT / "utility-report-sources"
QUALITY_OUTPUT = RESULTS / "class-global-vs-low-d-global-quality-delta-table"
UTILITY_OUTPUT = RESULTS / "class-global-vs-low-d-global-utility-delta-table"
QUALITY_DATA_OUTPUT = RESULTS / "class-global-vs-low-d-global-quality-deltas.csv"
UTILITY_DATA_OUTPUT = RESULTS / "class-global-vs-low-d-global-utility-deltas.csv"
SUPERSEDED_OUTPUTS = [
    RESULTS / "class-global-vs-low-d-global-metric-delta-grid.png",
    RESULTS / "class-global-vs-low-d-global-metric-delta-grid.pdf",
    RESULTS / "class-global-vs-low-d-global-metric-deltas.csv",
]

DATASETS = {
    "icaird_cervical100k": "iCAIRD",
    "pathmnist224": "Kather 100K",
}
ARCHITECTURES = {
    "stylegan2ada": "SG2",
    "sd_lora": "SD-LoRA",
    "dcgan": "DCGAN",
}
LOW_D_DATASETS = {value: key for key, value in DATASETS.items()}
LOW_D_ARCHITECTURES = {
    "StyleGAN2-ADA": "stylegan2ada",
    "SD-LoRA": "sd_lora",
    "DCGAN": "dcgan",
}

QUALITY_METRICS = [
    ("nUFD", r"$\Delta$ nUFD $\downarrow$", 2, -1.0),
    ("IP", r"$\Delta$ IP $\uparrow$", 4, 1.0),
    ("IR", r"$\Delta$ IR $\uparrow$", 4, 1.0),
    ("Authenticity", r"$\Delta$ Authenticity $\uparrow$", 4, 1.0),
]
UTILITY_METRICS = [
    ("Replacement accuracy", r"$\Delta$ Replacement accuracy $\uparrow$", 4, 1.0),
    ("Replacement ROC-AUC", r"$\Delta$ Replacement ROC-AUC $\uparrow$", 4, 1.0),
    ("Augmented accuracy", r"$\Delta$ Augmented accuracy $\uparrow$", 4, 1.0),
    ("Augmented ROC-AUC", r"$\Delta$ Augmented ROC-AUC $\uparrow$", 4, 1.0),
]


def normalise_low_d_quality(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["Dataset"] = output["Dataset"].map(LOW_D_DATASETS)
    output["Architecture"] = output["Architecture"].map(LOW_D_ARCHITECTURES)
    if output[["Dataset", "Architecture"]].isna().any().any():
        raise ValueError("Unknown dataset or architecture in low-d quality results.")
    return output


def quality_deltas() -> pd.DataFrame:
    earlier = pd.read_csv(EARLIER_QUALITY)
    earlier = earlier[
        earlier["Dataset"].isin(DATASETS)
        & earlier["Architecture"].isin(ARCHITECTURES)
        & earlier["Conditioning"].eq("label_global_cluster")
    ].set_index(["Dataset", "Architecture"])
    low_d = normalise_low_d_quality(pd.read_csv(LOW_D_QUALITY)).set_index(
        ["Dataset", "Architecture"]
    )
    expected = pd.MultiIndex.from_product(
        [DATASETS, ARCHITECTURES], names=["Dataset", "Architecture"]
    )
    if not expected.isin(earlier.index).all() or not expected.isin(low_d.index).all():
        raise ValueError("Quality comparison is incomplete.")

    rows = []
    for dataset, architecture in expected:
        for metric, _, _, improvement_sign in QUALITY_METRICS:
            earlier_value = float(earlier.loc[(dataset, architecture), metric])
            low_d_value = float(low_d.loc[(dataset, architecture), metric])
            delta = low_d_value - earlier_value
            rows.append(
                {
                    "Family": "Quality",
                    "Dataset": dataset,
                    "Architecture": architecture,
                    "Metric": metric,
                    "Seed": 0,
                    "Earlier_class_global": earlier_value,
                    "Low_d_class_global": low_d_value,
                    "Delta_low_d_minus_earlier": delta,
                    "Improvement_oriented_delta": improvement_sign * delta,
                }
            )
    return pd.DataFrame(rows)


def load_earlier_utility() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(EARLIER_UTILITY_SOURCES.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(EARLIER_UTILITY_SOURCES)
    frame = pd.concat(frames, ignore_index=True)
    selected = frame[
        frame["family"].eq("utility")
        & frame["dataset"].isin(DATASETS)
        & frame["generator_model"].isin(ARCHITECTURES)
        & frame["conditioning"].isin(
            [
                "synthetic_only_label_global_cluster",
                "real_plus_synthetic_label_global_cluster",
            ]
        )
        & frame["metric"].isin(["accuracy", "macro_roc_auc_ovr"])
    ].copy()
    selected["value"] = pd.to_numeric(selected["value"], errors="raise")
    keys = ["dataset", "generator_model", "conditioning", "seed", "metric"]
    inconsistent = selected.groupby(keys)["value"].nunique().gt(1)
    if inconsistent.any():
        raise ValueError("Earlier utility exports contain inconsistent duplicates.")
    return selected.drop_duplicates(keys)


def build_low_d_utility_seeds() -> pd.DataFrame:
    if LOW_D_UTILITY_SEEDS.exists():
        return pd.read_csv(LOW_D_UTILITY_SEEDS)
    if not LOW_D_UTILITY_LONG_SOURCE.exists():
        raise FileNotFoundError(LOW_D_UTILITY_LONG_SOURCE)
    frame = pd.read_csv(LOW_D_UTILITY_LONG_SOURCE)
    selected = frame[
        frame["dataset"].isin(DATASETS)
        & frame["generator_model"].isin(ARCHITECTURES)
        & frame["conditioning"].isin(
            [
                "synthetic_only_label_global_cluster",
                "real_plus_synthetic_label_global_cluster",
            ]
        )
        & frame["metric"].isin(["accuracy", "macro_roc_auc_ovr"])
    ].copy()
    selected["Conditioning"] = "Class + low-d global"
    selected = selected.rename(
        columns={
            "dataset": "Dataset",
            "generator_model": "Architecture",
            "conditioning": "Training_regime",
            "seed": "Classifier_seed",
            "metric": "Metric",
            "value": "Value",
        }
    )[
        [
            "Dataset",
            "Architecture",
            "Conditioning",
            "Training_regime",
            "Classifier_seed",
            "Metric",
            "Value",
        ]
    ]
    selected.to_csv(LOW_D_UTILITY_SEEDS, index=False)
    return selected


def utility_deltas() -> pd.DataFrame:
    earlier = load_earlier_utility().rename(
        columns={
            "dataset": "Dataset",
            "generator_model": "Architecture",
            "conditioning": "Training_regime",
            "seed": "Classifier_seed",
            "metric": "Metric",
            "value": "Earlier_class_global",
        }
    )
    low_d = build_low_d_utility_seeds().copy()
    low_d = low_d.rename(columns={"Value": "Low_d_class_global"})
    keys = [
        "Dataset",
        "Architecture",
        "Training_regime",
        "Classifier_seed",
        "Metric",
    ]
    merged = earlier.merge(low_d, on=keys, how="inner", validate="one_to_one")
    expected = len(DATASETS) * len(ARCHITECTURES) * 2 * 3 * 2
    if len(merged) != expected:
        raise ValueError(f"Expected {expected} paired utility values, found {len(merged)}.")
    merged["Delta_low_d_minus_earlier"] = (
        merged["Low_d_class_global"] - merged["Earlier_class_global"]
    )
    metric_names = {
        ("synthetic_only_label_global_cluster", "accuracy"): "Replacement accuracy",
        ("synthetic_only_label_global_cluster", "macro_roc_auc_ovr"): "Replacement ROC-AUC",
        ("real_plus_synthetic_label_global_cluster", "accuracy"): "Augmented accuracy",
        ("real_plus_synthetic_label_global_cluster", "macro_roc_auc_ovr"): "Augmented ROC-AUC",
    }
    merged["Metric_display"] = [
        metric_names[(regime, metric)]
        for regime, metric in zip(merged["Training_regime"], merged["Metric"], strict=True)
    ]
    return pd.DataFrame(
        {
            "Family": "Utility",
            "Dataset": merged["Dataset"],
            "Architecture": merged["Architecture"],
            "Metric": merged["Metric_display"],
            "Seed": merged["Classifier_seed"].astype(int),
            "Earlier_class_global": merged["Earlier_class_global"].astype(float),
            "Low_d_class_global": merged["Low_d_class_global"].astype(float),
            "Delta_low_d_minus_earlier": merged["Delta_low_d_minus_earlier"].astype(float),
            "Improvement_oriented_delta": merged["Delta_low_d_minus_earlier"].astype(float),
        }
    )


def value_text(value: float, precision: int) -> str:
    threshold = 0.5 * 10**-precision
    return "0.0" if abs(value) < threshold else f"{value:.{precision}f}"


def cell_text(values: np.ndarray, precision: int) -> str:
    if len(values) == 1:
        return value_text(float(values[0]), precision)
    return (
        f"{value_text(float(values.mean()), precision)}\n"
        rf"$\pm$ {value_text(float(values.std(ddof=1)), precision)}"
    )


def draw_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    title: str,
    precision: int,
    improvement_sign: float,
    show_y_labels: bool,
) -> None:
    dataset_keys = list(DATASETS)
    architecture_keys = list(ARCHITECTURES)
    cell_values: dict[tuple[int, int], np.ndarray] = {}
    means = np.empty((len(dataset_keys), len(architecture_keys)), dtype=float)
    for row, dataset in enumerate(dataset_keys):
        for column, architecture in enumerate(architecture_keys):
            values = frame.loc[
                frame["Dataset"].eq(dataset)
                & frame["Architecture"].eq(architecture)
                & frame["Metric"].eq(metric),
                "Delta_low_d_minus_earlier",
            ].to_numpy(float)
            if len(values) not in {1, 3}:
                raise ValueError(f"Unexpected observations for {metric}/{dataset}/{architecture}: {len(values)}")
            cell_values[(row, column)] = values
            means[row, column] = values.mean()

    improvement = improvement_sign * means
    limit = max(float(np.max(np.abs(improvement))), np.finfo(float).eps)
    cmap = LinearSegmentedColormap.from_list(
        "low_d_delta", ["#D55E00", "#F7F7F7", "#0072B2"]
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    axis.imshow(
        improvement,
        cmap=cmap,
        norm=norm,
        aspect="equal",
        extent=(-0.5, 2.5, 1.5, -0.5),
    )
    axis.set_xlim(-0.5, 3.5)
    axis.set_ylim(2.5, -0.5)
    axis.set_title(title, fontsize=19, pad=10, fontweight="bold")
    axis.set_xticks(range(4), [*ARCHITECTURES.values(), r"$\mu \pm \sigma$"], fontsize=17)
    axis.set_yticks(range(3))
    axis.set_yticklabels(
        [*DATASETS.values(), r"$\mu \pm \sigma$"] if show_y_labels else ["", "", ""],
        fontsize=17,
    )
    axis.tick_params(length=0, pad=7)

    for row in range(2):
        for column in range(3):
            encoded = improvement[row, column]
            colour = "white" if abs(encoded) / limit > 0.64 else "#202020"
            axis.text(
                column,
                row,
                cell_text(cell_values[(row, column)], precision),
                ha="center",
                va="center",
                color=colour,
                fontsize=14 if len(cell_values[(row, column)]) > 1 else 17,
                fontweight="bold",
                zorder=3,
            )

    summaries: list[tuple[int, int, np.ndarray]] = []
    for row, dataset in enumerate(dataset_keys):
        summaries.append(
            (
                3,
                row,
                frame.loc[
                    frame["Dataset"].eq(dataset) & frame["Metric"].eq(metric),
                    "Delta_low_d_minus_earlier",
                ].to_numpy(float),
            )
        )
    for column, architecture in enumerate(architecture_keys):
        summaries.append(
            (
                column,
                2,
                frame.loc[
                    frame["Architecture"].eq(architecture) & frame["Metric"].eq(metric),
                    "Delta_low_d_minus_earlier",
                ].to_numpy(float),
            )
        )
    summaries.append(
        (
            3,
            2,
            frame.loc[frame["Metric"].eq(metric), "Delta_low_d_minus_earlier"].to_numpy(float),
        )
    )
    for column, row, values in summaries:
        mean = float(values.mean())
        encoded = improvement_sign * mean
        axis.add_patch(
            Rectangle(
                (column - 0.5, row - 0.5),
                1,
                1,
                facecolor=cmap(norm(encoded)),
                edgecolor="#9A9A9A",
                linewidth=1.0,
                zorder=2,
            )
        )
        colour = "white" if abs(encoded) / limit > 0.64 else "#202020"
        axis.text(
            column,
            row,
            cell_text(values, precision),
            ha="center",
            va="center",
            color=colour,
            fontsize=14,
            fontweight="bold",
            zorder=3,
        )

    axis.add_patch(
        Rectangle(
            (-0.5, -0.5),
            3,
            2,
            fill=False,
            edgecolor="#202020",
            linewidth=5.5,
            clip_on=False,
            zorder=5,
        )
    )
    axis.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    axis.grid(which="minor", color="#B8B8B8", linewidth=0.9)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_table(
    deltas: pd.DataFrame,
    specifications: list[tuple[str, str, int, float]],
    title: str,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )
    figure = plt.figure(figsize=(14, 14), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        left=0.12,
        right=0.98,
        top=0.86,
        bottom=0.15,
        wspace=-0.04,
        hspace=0.46,
    )
    axes = [
        figure.add_subplot(grid[index // 2, index % 2])
        for index in range(len(specifications))
    ]
    for index, (axis, specification) in enumerate(zip(axes, specifications, strict=True)):
        draw_panel(axis, deltas, *specification, show_y_labels=index % 2 == 0)

    figure.suptitle(
        title,
        fontsize=21,
        fontweight="bold",
        y=0.955,
    )
    cmap = LinearSegmentedColormap.from_list(
        "low_d_delta", ["#D55E00", "#F7F7F7", "#0072B2"]
    )
    legend_axis = figure.add_axes((0.30, 0.045, 0.40, 0.014))
    legend = figure.colorbar(
        ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=cmap),
        cax=legend_axis,
        orientation="horizontal",
        ticks=[-1, 0, 1],
    )
    legend.ax.set_xticklabels(["Deterioration", "No change", "Improvement"])
    legend.ax.tick_params(labelsize=13, length=0, pad=7)
    for label in legend.ax.get_xticklabels():
        label.set_fontweight("bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    quality = quality_deltas()
    utility = utility_deltas()
    quality.to_csv(QUALITY_DATA_OUTPUT, index=False)
    utility.to_csv(UTILITY_DATA_OUTPUT, index=False)
    plot_table(
        quality,
        QUALITY_METRICS,
        "Quality deltas: class + low-d global minus class + global",
        QUALITY_OUTPUT,
    )
    plot_table(
        utility,
        UTILITY_METRICS,
        "Utility deltas: class + low-d global minus class + global",
        UTILITY_OUTPUT,
    )
    for path in SUPERSEDED_OUTPUTS:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
