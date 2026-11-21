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


METHOD_COLOURS = {
    "kmeans": "#0072B2",
    "spherical_kmeans": "#E69F00",
    "hdbscan": "#009E73",
    "leiden": "#CC79A7",
}

EXTERNAL_METRICS = [
    ("ami", "AMI", "higher"),
    ("ari", "ARI", "higher"),
    ("v_measure", "V-measure", "higher"),
]

INTERNAL_METRICS = [
    ("silhouette", "Silhouette", "higher"),
    ("stability", "Stability", "higher"),
    ("sd_index", "SD-index", "lower"),
]

COMBINED_METRICS = EXTERNAL_METRICS + INTERNAL_METRICS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/Users/dm236/Documents/Research/AllResults/Chapter6-clustering/res"),
    )
    parser.add_argument("--min-dimension", type=int, default=1)
    parser.add_argument("--max-dimension", type=int, default=50)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    stability = pd.read_csv(args.run_dir / "clustering_stability.csv")
    raw_baselines = summarise_raw_baselines(
        select_best_raw_configs(
            stability,
            dataset=args.dataset,
            encoder=args.encoder,
        )
    )

    all_trends = []
    for projector in ["pca", "umap"]:
        selected = select_best_dimension_configs(
            stability,
            projector=projector,
            dataset=args.dataset,
            encoder=args.encoder,
            min_dimension=args.min_dimension,
            max_dimension=args.max_dimension,
        )
        trends = summarise_dimension_trends(selected)
        all_trends.append(trends)
        output_parts = ["clustering-results-rq1", projector]
        if args.dataset:
            output_parts.append(args.dataset)
        if args.encoder:
            output_parts.append(args.encoder)
        output_parts.extend(["combined", "dimension", "metrics"])
        _plot_metric_family(
            trends=trends,
            raw_baselines=raw_baselines,
            projector=projector,
            dataset=args.dataset,
            encoder=args.encoder,
            metrics=COMBINED_METRICS,
            output=args.run_dir / ("-".join(output_parts) + ".png"),
            dpi=args.dpi,
        )

    csv_parts = ["clustering-results-rq1"]
    if args.dataset:
        csv_parts.append(args.dataset)
    if args.encoder:
        csv_parts.append(args.encoder)
    csv_parts.extend(["dimension", "trends", "plotted", "values"])
    pd.concat(all_trends, ignore_index=True).to_csv(
        args.run_dir / ("-".join(csv_parts) + ".csv"),
        index=False,
    )


def _plot_metric_family(
    trends: pd.DataFrame,
    raw_baselines: pd.DataFrame,
    projector: str,
    dataset: str | None,
    encoder: str | None,
    metrics: list[tuple[str, str, str]],
    output: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 13,
            "axes.linewidth": 0.9,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.2), sharex=True)
    axes_flat = list(np.ravel(axes))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.875, bottom=0.245, hspace=0.34, wspace=0.18)
    legend_y = 0.035
    legend_ncol = 4

    clusterers = [clusterer for clusterer in METHOD_ORDER if clusterer in set(trends["clusterer"])]
    for axis, (metric_key, metric_label, direction) in zip(axes_flat, metrics):
        baseline_specs = []
        for clusterer in clusterers:
            if metric_key == "noise" and clusterer != "hdbscan":
                continue
            group = trends[trends["clusterer"] == clusterer].sort_values("dimension")
            if group.empty:
                continue
            x = group["dimension"].to_numpy(dtype=float)
            y = group[f"{metric_key}_mean"].to_numpy(dtype=float)
            y_sd = group[f"{metric_key}_sd"].to_numpy(dtype=float)
            colour = METHOD_COLOURS.get(clusterer)
            axis.plot(
                x,
                y,
                label=METHOD_LABELS.get(clusterer, clusterer),
                color=colour,
                linewidth=2.7,
                zorder=3,
            )
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
                baseline_specs.append((clusterer, baseline, colour))

        for clusterer, baseline, colour in baseline_specs:
            axis.axhline(
                baseline,
                color=colour,
                linestyle="--",
                linewidth=2.6,
                alpha=0.98,
                zorder=5,
                label="_nolegend_",
            )

        arrow = "↑" if direction == "higher" else "↓"
        axis.set_title(f"{metric_label} {arrow}", fontweight="bold", pad=10)
        axis.set_xlabel("Dimensions")
        axis.set_ylabel(metric_label)
        axis.set_box_aspect(0.82)
        axis.set_xlim(float(trends["dimension"].min()), float(trends["dimension"].max()))
        axis.set_xticks([1, 10, 20, 30, 40, 50])
        axis.grid(True, axis="y", alpha=0.26, linewidth=0.9)
        axis.grid(True, axis="x", alpha=0.12, linewidth=0.7)
        for spine in axis.spines.values():
            spine.set_color("#666666")
        if metric_key == "sd_index":
            positive = trends[f"{metric_key}_mean"].dropna()
            positive = positive[positive > 0]
            if not positive.empty:
                axis.set_yscale("log")
        if metric_key in {"ami", "ari", "v_measure", "silhouette", "stability", "noise"}:
            axis.set_ylim(bottom=0.0)

    handles = []
    labels = []
    for clusterer in clusterers:
        colour = METHOD_COLOURS.get(clusterer)
        method_label = METHOD_LABELS.get(clusterer, clusterer)
        handles.append(Line2D([0], [0], color=colour, linewidth=2.7, linestyle="-"))
        labels.append(method_label)
    for clusterer in clusterers:
        colour = METHOD_COLOURS.get(clusterer)
        method_label = METHOD_LABELS.get(clusterer, clusterer)
        handles.append(Line2D([0], [0], color=colour, linewidth=2.6, linestyle="--"))
        labels.append(f"{method_label} full")
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=legend_ncol,
        frameon=True,
        fancybox=False,
        title="Clustering method",
        bbox_to_anchor=(0.5, legend_y),
        borderpad=0.85,
        columnspacing=1.25,
        handlelength=2.4,
    )
    legend.get_frame().set_edgecolor("#666666")
    legend.get_frame().set_linewidth(0.9)

    projector_label = projector.upper()
    title_parts = [projector_label]
    if dataset or encoder:
        title_parts.append(" / ".join(part for part in [dataset, encoder] if part))
    fig.suptitle(
        f"{': '.join(title_parts)} validation metrics vs dimensions",
        fontsize=18,
        fontweight="bold",
        y=0.955,
    )
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _raw_baseline(
    raw_baselines: pd.DataFrame,
    clusterer: str,
    metric_key: str,
) -> float | None:
    if raw_baselines.empty:
        return None
    group = raw_baselines[raw_baselines["clusterer"] == clusterer]
    raw_col = f"{metric_key}_mean"
    if group.empty or raw_col not in group.columns:
        return None
    value = group.iloc[0][raw_col]
    if pd.isna(value):
        return None
    value = float(value)
    if metric_key == "sd_index" and value <= 0:
        return None
    return value


if __name__ == "__main__":
    main()
