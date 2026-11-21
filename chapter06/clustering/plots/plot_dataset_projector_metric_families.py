from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_exp.dimension_plots import (
    select_best_dimension_configs,
    select_best_raw_configs,
    summarise_dimension_trends,
    summarise_raw_baselines,
)
from cluster_exp.table1 import METHOD_LABELS, METHOD_ORDER

from plot_rq1_dimension_metric_families import COMBINED_METRICS, METHOD_COLOURS, _raw_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/Users/dm236/Documents/Research/AllResults/Chapter6-clustering/res"),
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="uni2")
    parser.add_argument("--min-dimension", type=int, default=1)
    parser.add_argument("--max-dimension", type=int, default=50)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    stability = pd.read_csv(args.run_dir / "clustering_stability.csv")
    raw_baselines = summarise_raw_baselines(
        select_best_raw_configs(stability, dataset=args.dataset, encoder=args.encoder)
    )
    trends_by_projector = {}
    for projector in ["pca", "umap"]:
        selected = select_best_dimension_configs(
            stability,
            projector=projector,
            dataset=args.dataset,
            encoder=args.encoder,
            min_dimension=args.min_dimension,
            max_dimension=args.max_dimension,
        )
        trends_by_projector[projector] = summarise_dimension_trends(selected)

    output = (
        args.run_dir
        / f"clustering-results-rq1-{args.dataset}-{args.encoder}-pca-umap-combined-dimension-metrics.png"
    )
    _plot_projector_comparison(
        trends_by_projector=trends_by_projector,
        raw_baselines=raw_baselines,
        dataset=args.dataset,
        encoder=args.encoder,
        output=output,
        dpi=args.dpi,
    )
    pd.concat(
        [trends.assign(projector=projector) for projector, trends in trends_by_projector.items()],
        ignore_index=True,
    ).to_csv(
        args.run_dir
        / f"clustering-results-rq1-{args.dataset}-{args.encoder}-pca-umap-dimension-trends-plotted-values.csv",
        index=False,
    )
    print(output)


def _plot_projector_comparison(
    trends_by_projector: dict[str, pd.DataFrame],
    raw_baselines: pd.DataFrame,
    dataset: str,
    encoder: str,
    output: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            "axes.linewidth": 0.85,
        }
    )

    fig, axes = plt.subplots(4, 3, figsize=(13.2, 15.6), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.935, bottom=0.14, hspace=0.54, wspace=0.20)

    clusterers = [
        clusterer
        for clusterer in METHOD_ORDER
        if any(clusterer in set(trends["clusterer"]) for trends in trends_by_projector.values())
    ]
    for projector_index, projector in enumerate(["pca", "umap"]):
        trends = trends_by_projector[projector]
        row_offset = projector_index * 2
        for metric_index, (metric_key, metric_label, direction) in enumerate(COMBINED_METRICS):
            row = row_offset + (metric_index // 3)
            col = metric_index % 3
            axis = axes[row, col]
            baseline_specs = []
            for clusterer in clusterers:
                group = trends[trends["clusterer"] == clusterer].sort_values("dimension")
                if group.empty:
                    continue
                x = group["dimension"].to_numpy(dtype=float)
                y = group[f"{metric_key}_mean"].to_numpy(dtype=float)
                y_sd = group[f"{metric_key}_sd"].to_numpy(dtype=float)
                colour = METHOD_COLOURS.get(clusterer)
                axis.plot(x, y, color=colour, linewidth=2.35, zorder=3)
                valid = np.isfinite(y) & np.isfinite(y_sd)
                if np.any(valid):
                    lower = y[valid] - y_sd[valid]
                    upper = y[valid] + y_sd[valid]
                    if metric_key == "sd_index":
                        positive = y[np.isfinite(y) & (y > 0)]
                        floor = float(np.nanmin(positive)) * 0.5 if positive.size else 1e-6
                        lower = np.maximum(lower, floor)
                    axis.fill_between(
                        x[valid],
                        lower,
                        upper,
                        color=colour,
                        alpha=0.12,
                        linewidth=0,
                        zorder=1,
                    )
                baseline = _raw_baseline(raw_baselines, clusterer, metric_key)
                if baseline is not None:
                    baseline_specs.append((baseline, colour))

            for baseline, colour in baseline_specs:
                axis.axhline(
                    baseline,
                    color=colour,
                    linestyle="--",
                    linewidth=2.2,
                    alpha=0.98,
                    zorder=5,
                )

            arrow = "↑" if direction == "higher" else "↓"
            axis.set_title(f"{metric_label} {arrow}", fontweight="bold", pad=8)
            axis.set_xlabel("Dimensions")
            axis.set_ylabel(metric_label)
            axis.set_xlim(float(trends["dimension"].min()), float(trends["dimension"].max()))
            axis.set_xticks([1, 10, 20, 30, 40, 50])
            axis.grid(True, axis="y", alpha=0.26, linewidth=0.85)
            axis.grid(True, axis="x", alpha=0.12, linewidth=0.65)
            for spine in axis.spines.values():
                spine.set_color("#666666")
            if metric_key == "sd_index":
                positive = trends[f"{metric_key}_mean"].dropna()
                positive = positive[positive > 0]
                if not positive.empty:
                    axis.set_yscale("log")
            else:
                axis.set_ylim(bottom=0.0)

        axes[row_offset, 0].text(
            -0.24,
            0.5,
            projector.upper(),
            transform=axes[row_offset, 0].transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=16,
            fontweight="bold",
        )
        axes[row_offset + 1, 0].text(
            -0.24,
            0.5,
            projector.upper(),
            transform=axes[row_offset + 1, 0].transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=16,
            fontweight="bold",
        )

    handles = []
    labels = []
    for clusterer in clusterers:
        colour = METHOD_COLOURS.get(clusterer)
        label = METHOD_LABELS.get(clusterer, clusterer)
        handles.append(Line2D([0], [0], color=colour, linewidth=2.35, linestyle="-"))
        labels.append(label)
    for clusterer in clusterers:
        colour = METHOD_COLOURS.get(clusterer)
        label = METHOD_LABELS.get(clusterer, clusterer)
        handles.append(Line2D([0], [0], color=colour, linewidth=2.2, linestyle="--"))
        labels.append(f"{label} full")
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        fancybox=False,
        title="Clustering method",
        bbox_to_anchor=(0.5, 0.025),
        borderpad=0.8,
        columnspacing=1.15,
        handlelength=2.3,
    )
    legend.get_frame().set_edgecolor("#666666")
    legend.get_frame().set_linewidth(0.85)

    fig.suptitle(
        f"{dataset} / {encoder}: PCA and UMAP validation metrics vs dimensions",
        fontsize=17,
        fontweight="bold",
        y=0.982,
    )
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
