from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "utility-report-sources"
OUTPUT = ROOT / "final-results" / "replacement-utility-cluster-deltas-grid"
DELTA_OUTPUT = ROOT / "final-results" / "replacement-utility-paired-deltas.csv"

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
    "synthetic_only_label_global_cluster": "Global",
    "synthetic_only_label_local_cluster": "Local",
}
BASELINE = "synthetic_only_true_label"
METRICS = {
    "accuracy": ("Accuracy", r"$\Delta$ synthetic-only accuracy $\uparrow$"),
    "macro_roc_auc_ovr": (
        "ROC-AUC",
        r"$\Delta$ synthetic-only ROC-AUC $\uparrow$",
    ),
}


def load_seed_results() -> pd.DataFrame:
    paths = sorted(SOURCE_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No utility source CSV files found in {SOURCE_DIR}")
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    conditions = [BASELINE, *CONDITIONS]
    selected = frame.loc[
        frame["family"].eq("utility")
        & frame["dataset"].isin(DATASETS)
        & frame["generator_model"].isin(ARCHITECTURES)
        & frame["conditioning"].isin(conditions)
        & frame["metric"].isin(METRICS)
    ].copy()
    selected["value"] = pd.to_numeric(selected["value"], errors="raise")
    keys = ["dataset", "generator_model", "conditioning", "seed", "metric"]
    inconsistent = selected.groupby(keys)["value"].nunique().gt(1)
    if inconsistent.any():
        raise ValueError("Duplicate utility exports contain inconsistent values.")
    selected = selected.drop_duplicates(keys).reset_index(drop=True)

    expected = len(DATASETS) * len(ARCHITECTURES) * len(conditions) * 3 * len(METRICS)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} seed-level values, found {len(selected)}.")
    counts = selected.groupby(
        ["dataset", "generator_model", "conditioning", "metric"]
    )["seed"].nunique()
    if not counts.eq(3).all():
        raise ValueError("Every utility cell must contain exactly three classifier seeds.")
    return selected


def paired_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "generator_model", "seed", "metric"]
    baseline = frame[frame["conditioning"] == BASELINE].set_index(keys)["value"]
    rows = []
    for condition, comparison in CONDITIONS.items():
        compared = frame[frame["conditioning"] == condition].set_index(keys)["value"]
        if not baseline.sort_index().index.equals(compared.sort_index().index):
            raise ValueError(f"Seed alignment failed for {condition}.")
        delta = compared.sort_index() - baseline.sort_index()
        for (dataset, architecture, seed, metric), value in delta.items():
            rows.append(
                {
                    "Dataset": dataset,
                    "Architecture": architecture,
                    "Seed": int(seed),
                    "Metric": METRICS[metric][0],
                    "Comparison": comparison,
                    "Delta": float(value),
                }
            )
    output = pd.DataFrame(rows)
    expected = len(DATASETS) * len(ARCHITECTURES) * 3 * len(METRICS) * len(CONDITIONS)
    if len(output) != expected:
        raise ValueError(f"Expected {expected} paired deltas, found {len(output)}.")
    return output


def value_text(value: float) -> str:
    return "0.0" if abs(value) < 0.00005 else f"{value:.4f}"


def cell_text(values: np.ndarray) -> str:
    return (
        f"{value_text(values.mean())}\n"
        rf"$\pm$ {value_text(values.std(ddof=1))}"
    )


def values_for(
    frame: pd.DataFrame,
    metric: str,
    dataset: str | None = None,
    architecture: str | None = None,
    comparison: str | None = None,
) -> np.ndarray:
    selected = frame[frame["Metric"] == metric]
    if dataset is not None:
        selected = selected[selected["Dataset"] == dataset]
    if architecture is not None:
        selected = selected[selected["Architecture"] == architecture]
    if comparison is not None:
        selected = selected[selected["Comparison"] == comparison]
    return selected["Delta"].to_numpy(float)


def plot(frame: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "font.weight": "bold",
            "axes.titleweight": "bold",
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "utility_delta", ["#D55E00", "#F7F7F7", "#0072B2"]
    )
    figure, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=False)
    figure.subplots_adjust(
        left=0.14,
        right=0.98,
        top=0.96,
        bottom=0.18,
        hspace=0.48,
    )
    gutter = 0.10
    column_positions = np.array(
        [0.0, 1.0, 2.0, 3.0, 4.0 + gutter, 5.0 + gutter, 6.0 + gutter, 7.0 + gutter]
    )

    for panel, (axis, (_, (metric, title))) in enumerate(
        zip(axes, METRICS.items(), strict=True)
    ):
        cell_means = np.empty((3, 6), dtype=float)
        cell_values: dict[tuple[int, int], np.ndarray] = {}
        for row, dataset in enumerate(DATASETS):
            for comparison_index, comparison in enumerate(CONDITIONS.values()):
                for architecture_index, architecture in enumerate(ARCHITECTURES):
                    source_column = comparison_index * 3 + architecture_index
                    values = values_for(
                        frame,
                        metric,
                        dataset=dataset,
                        architecture=architecture,
                        comparison=comparison,
                    )
                    cell_means[row, source_column] = values.mean()
                    cell_values[(row, source_column)] = values

        limit = max(float(np.abs(cell_means).max()), np.finfo(float).eps)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        axis.imshow(
            cell_means[:, :3],
            cmap=cmap,
            norm=norm,
            aspect="equal",
            extent=(-0.5, 2.5, 2.5, -0.5),
        )
        axis.imshow(
            cell_means[:, 3:],
            cmap=cmap,
            norm=norm,
            aspect="equal",
            extent=(3.5 + gutter, 6.5 + gutter, 2.5, -0.5),
        )
        axis.set_xlim(-0.5, 7.5 + gutter)
        axis.set_ylim(3.5, -0.5)
        axis.set_title(title, fontsize=21, pad=9)
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
            fontsize=14,
        )
        axis.set_yticks(range(4))
        axis.set_yticklabels(
            [*DATASETS.values(), r"$\mu \pm \sigma$"],
            fontsize=17,
        )
        axis.tick_params(length=0, pad=6)

        for row in range(3):
            for source_column in range(6):
                display_column = (
                    source_column
                    if source_column < 3
                    else source_column + 1 + gutter
                )
                mean = cell_means[row, source_column]
                colour = "white" if abs(mean) / limit > 0.64 else "#202020"
                axis.text(
                    display_column,
                    row,
                    cell_text(cell_values[(row, source_column)]),
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=11,
                    fontweight="bold",
                    zorder=3,
                )

        summaries: list[tuple[float, int, np.ndarray]] = []
        for row, dataset in enumerate(DATASETS):
            for comparison_index, comparison in enumerate(CONDITIONS.values()):
                column = 3 + comparison_index * (4 + gutter)
                summaries.append(
                    (
                        column,
                        row,
                        values_for(
                            frame,
                            metric,
                            dataset=dataset,
                            comparison=comparison,
                        ),
                    )
                )
        for comparison_index, comparison in enumerate(CONDITIONS.values()):
            for architecture_index, architecture in enumerate(ARCHITECTURES):
                column = (
                    architecture_index
                    if comparison_index == 0
                    else architecture_index + 4 + gutter
                )
                summaries.append(
                    (
                        column,
                        3,
                        values_for(
                            frame,
                            metric,
                            architecture=architecture,
                            comparison=comparison,
                        ),
                    )
                )
            summaries.append(
                (
                    3 + comparison_index * (4 + gutter),
                    3,
                    values_for(frame, metric, comparison=comparison),
                )
            )

        for column, row, values in summaries:
            mean = values.mean()
            axis.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=cmap(norm(mean)),
                    edgecolor="#B8B8B8",
                    linewidth=0.9,
                    zorder=2,
                )
            )
            colour = "white" if abs(mean) / limit > 0.64 else "#202020"
            axis.text(
                column,
                row,
                cell_text(values),
                ha="center",
                va="center",
                color=colour,
                fontsize=9,
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
        for centre, label in ((1.0, "Global"), (5.0 + gutter, "Local")):
            axis.text(
                centre,
                -0.22,
                label,
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

    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    seed_results = load_seed_results()
    deltas = paired_deltas(seed_results)
    deltas.to_csv(DELTA_OUTPUT, index=False)
    plot(deltas)


if __name__ == "__main__":
    main()
