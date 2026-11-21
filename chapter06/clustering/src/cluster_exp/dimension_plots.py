from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .table1 import METHOD_LABELS, METHOD_ORDER, PROJECTOR_LABELS, RANK_COLUMNS


DIMENSION_METRICS = [
    ("ami", "AMI", "mean_label_ami", "higher"),
    ("ari", "ARI", "mean_label_ari", "higher"),
    ("v_measure", "V-measure", "mean_label_v_measure", "higher"),
    ("silhouette", "Silhouette", "mean_silhouette", "higher"),
    ("sd_index", "SD-index", "mean_sd_index", "lower"),
    ("stability", "Stability", "mean_pairwise_ari", "higher"),
    ("clusters", "Clusters", "mean_n_clusters", "none"),
    ("noise", "Noise", "mean_noise_fraction", "lower"),
]

STABILITY_TO_RESULT_METRIC = {
    "ami": "label_ami",
    "ari": "label_ari",
    "v_measure": "label_v_measure",
    "silhouette": "silhouette",
    "sd_index": "sd_index",
    "clusters": "n_clusters",
    "noise": "noise_fraction",
}

SELECTION_JOIN_COLS = [
    "dataset",
    "encoder",
    "projector",
    "dimension",
    "target_name",
    "source_metric",
    "condition",
    "clusterer",
    "cluster_params_json",
]

METHOD_COLOURS = {
    "kmeans": "#1f77b4",
    "spherical_kmeans": "#ff7f0e",
    "hdbscan": "#2ca02c",
    "leiden": "#9467bd",
}


@dataclass(frozen=True)
class DimensionPlotOutputs:
    figure_path: Path
    csv_path: Path


def plot_dimension_metrics(
    run_dir: Path,
    projector: str = "pca",
    clusterer: str | None = None,
    dataset: str | None = None,
    encoder: str | None = None,
    error_source: str = "selected",
    include_raw_baseline: bool = True,
    output: Path | None = None,
    output_csv: Path | None = None,
    stability_csv: Path | None = None,
    results_csv: Path | None = None,
    min_dimension: int = 1,
    max_dimension: int = 50,
    show_error_band: bool = True,
    dpi: int = 300,
) -> DimensionPlotOutputs:
    """Plot metric trends over projection dimension for one projector.

    The input is the stability summary. Within each
    dataset/encoder/clusterer/projector/dimension group, the best clustering
    configuration is selected using the same ranking rule as the summary table.
    The plotted values are then averaged across dataset-encoder pairs.
    """

    run_dir = Path(run_dir)
    stability_csv = stability_csv or run_dir / "clustering_stability.csv"
    name_parts = ["clustering_results", projector]
    if dataset is not None:
        name_parts.append(dataset)
    if encoder is not None:
        name_parts.append(encoder)
    if clusterer is not None:
        name_parts.append(clusterer)
    if error_source == "seeds":
        name_parts.append("seed_variation")
    name_parts.append("dimension_metric_trends")
    output_stem = "_".join(name_parts)
    output = output or run_dir / f"{output_stem}.png"
    output_csv = output_csv or run_dir / f"{output_stem}.csv"

    stability = pd.read_csv(stability_csv)
    selected = select_best_dimension_configs(
        stability,
        projector=projector,
        clusterer=clusterer,
        dataset=dataset,
        encoder=encoder,
        min_dimension=min_dimension,
        max_dimension=max_dimension,
    )
    if error_source == "seeds":
        results_csv = results_csv or run_dir / "clustering_results.csv"
        results = pd.read_csv(results_csv)
        trends = summarise_seed_dimension_trends(selected, results)
        raw_baselines = (
            summarise_seed_raw_baselines(
                select_best_raw_configs(
                    stability,
                    clusterer=clusterer,
                    dataset=dataset,
                    encoder=encoder,
                ),
                results,
            )
            if include_raw_baseline
            else None
        )
    elif error_source == "selected":
        trends = summarise_dimension_trends(selected)
        raw_baselines = (
            summarise_raw_baselines(
                select_best_raw_configs(
                    stability,
                    clusterer=clusterer,
                    dataset=dataset,
                    encoder=encoder,
                )
            )
            if include_raw_baseline
            else None
        )
    else:
        raise ValueError("error_source must be 'selected' or 'seeds'.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    trends.to_csv(output_csv, index=False)
    _plot_trends(
        trends,
        projector=projector,
        dataset=dataset,
        encoder=encoder,
        raw_baselines=raw_baselines,
        output=output,
        show_error_band=show_error_band,
        dpi=dpi,
    )
    return DimensionPlotOutputs(figure_path=output, csv_path=output_csv)


def select_best_dimension_configs(
    stability: pd.DataFrame,
    projector: str = "pca",
    clusterer: str | None = None,
    dataset: str | None = None,
    encoder: str | None = None,
    min_dimension: int = 1,
    max_dimension: int = 50,
) -> pd.DataFrame:
    required = ["dataset", "encoder", "clusterer", "projector", "dimension"]
    missing = [column for column in required if column not in stability.columns]
    if missing:
        raise ValueError(f"Missing required stability columns: {', '.join(missing)}")

    frame = stability.copy()
    frame = frame[frame["clusterer"].isin(METHOD_ORDER)]
    frame = frame[frame["projector"] == projector]
    if clusterer is not None:
        frame = frame[frame["clusterer"] == clusterer]
    if dataset is not None:
        frame = frame[frame["dataset"] == dataset]
    if encoder is not None:
        frame = frame[frame["encoder"] == encoder]
    frame["dimension"] = pd.to_numeric(frame["dimension"], errors="coerce")
    frame = frame[frame["dimension"].notna()]
    frame["dimension"] = frame["dimension"].astype(int)
    frame = frame[(frame["dimension"] >= min_dimension) & (frame["dimension"] <= max_dimension)]
    if frame.empty:
        raise ValueError(
            f"No rows found for projector={projector!r}, clusterer={clusterer!r}, "
            f"dataset={dataset!r}, encoder={encoder!r}, and dimensions {min_dimension}..{max_dimension}."
        )

    for column, ascending, fill_value in RANK_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[f"__sort_{column}"] = frame[column].fillna(fill_value)

    group_cols = ["dataset", "encoder", "clusterer", "projector", "dimension"]
    sort_cols = group_cols + [f"__sort_{column}" for column, _, _ in RANK_COLUMNS]
    ascending = [True] * len(group_cols) + [item[1] for item in RANK_COLUMNS]
    selected = frame.sort_values(sort_cols, ascending=ascending).groupby(
        group_cols, dropna=False
    ).head(1)
    return selected.drop(columns=[col for col in selected.columns if col.startswith("__sort_")])


def select_best_raw_configs(
    stability: pd.DataFrame,
    clusterer: str | None = None,
    dataset: str | None = None,
    encoder: str | None = None,
) -> pd.DataFrame:
    required = ["dataset", "encoder", "clusterer", "projector"]
    missing = [column for column in required if column not in stability.columns]
    if missing:
        raise ValueError(f"Missing required stability columns: {', '.join(missing)}")

    frame = stability.copy()
    frame = frame[frame["clusterer"].isin(METHOD_ORDER)]
    frame = frame[frame["projector"] == "raw"]
    if clusterer is not None:
        frame = frame[frame["clusterer"] == clusterer]
    if dataset is not None:
        frame = frame[frame["dataset"] == dataset]
    if encoder is not None:
        frame = frame[frame["encoder"] == encoder]
    if frame.empty:
        return frame

    for column, ascending, fill_value in RANK_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[f"__sort_{column}"] = frame[column].fillna(fill_value)

    group_cols = ["dataset", "encoder", "clusterer", "projector"]
    sort_cols = group_cols + [f"__sort_{column}" for column, _, _ in RANK_COLUMNS]
    ascending = [True] * len(group_cols) + [item[1] for item in RANK_COLUMNS]
    selected = frame.sort_values(sort_cols, ascending=ascending).groupby(
        group_cols, dropna=False
    ).head(1)
    return selected.drop(columns=[col for col in selected.columns if col.startswith("__sort_")])


def summarise_dimension_trends(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["projector", "dimension", "clusterer"]
    for (projector, dimension, clusterer), group in selected.groupby(group_cols, dropna=False):
        row: dict[str, object] = {
            "projector": projector,
            "dimension": int(dimension),
            "clusterer": clusterer,
            "method": METHOD_LABELS.get(str(clusterer), str(clusterer)),
            "n": int(len(group)),
        }
        for metric_key, _, source_col, _ in DIMENSION_METRICS:
            values = (
                pd.to_numeric(group[source_col], errors="coerce").dropna()
                if source_col in group.columns
                else pd.Series(dtype=float)
            )
            row[f"{metric_key}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric_key}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)

    trends = pd.DataFrame(rows)
    trends["__method_order"] = trends["clusterer"].map({value: idx for idx, value in enumerate(METHOD_ORDER)})
    trends = trends.sort_values(["dimension", "__method_order"]).drop(columns=["__method_order"])
    return trends.reset_index(drop=True)


def summarise_raw_baselines(selected_raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if selected_raw.empty:
        return pd.DataFrame(rows)

    for clusterer, group in selected_raw.groupby("clusterer", dropna=False):
        row: dict[str, object] = {
            "projector": "raw",
            "clusterer": clusterer,
            "method": METHOD_LABELS.get(str(clusterer), str(clusterer)),
            "n": int(len(group)),
        }
        for metric_key, _, source_col, _ in DIMENSION_METRICS:
            values = (
                pd.to_numeric(group[source_col], errors="coerce").dropna()
                if source_col in group.columns
                else pd.Series(dtype=float)
            )
            row[f"{metric_key}_mean"] = float(values.mean()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_seed_dimension_trends(
    selected: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    matched = _match_selected_to_results(selected, results)
    if matched.empty:
        raise ValueError("No seed-level result rows matched the selected dimension configurations.")

    rows: list[dict[str, object]] = []
    group_cols = ["projector", "dimension", "clusterer"]
    for (projector, dimension, clusterer), group in matched.groupby(group_cols, dropna=False):
        row: dict[str, object] = {
            "projector": projector,
            "dimension": int(dimension),
            "clusterer": clusterer,
            "method": METHOD_LABELS.get(str(clusterer), str(clusterer)),
            "n": int(len(group)),
        }
        for metric_key, _, stability_col, _ in DIMENSION_METRICS:
            if metric_key == "stability":
                values = pd.to_numeric(group["mean_pairwise_ari"], errors="coerce").dropna()
            else:
                result_col = STABILITY_TO_RESULT_METRIC.get(metric_key, stability_col)
                values = (
                    pd.to_numeric(group[result_col], errors="coerce").dropna()
                    if result_col in group.columns
                    else pd.Series(dtype=float)
                )
            row[f"{metric_key}_mean"] = float(values.mean()) if len(values) else np.nan
            if metric_key == "stability":
                row[f"{metric_key}_sd"] = np.nan
            else:
                row[f"{metric_key}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)

    trends = pd.DataFrame(rows)
    trends["__method_order"] = trends["clusterer"].map({value: idx for idx, value in enumerate(METHOD_ORDER)})
    trends = trends.sort_values(["dimension", "__method_order"]).drop(columns=["__method_order"])
    return trends.reset_index(drop=True)


def summarise_seed_raw_baselines(
    selected_raw: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    if selected_raw.empty:
        return pd.DataFrame()
    matched = _match_selected_to_results(selected_raw, results)
    if matched.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for clusterer, group in matched.groupby("clusterer", dropna=False):
        row: dict[str, object] = {
            "projector": "raw",
            "clusterer": clusterer,
            "method": METHOD_LABELS.get(str(clusterer), str(clusterer)),
            "n": int(len(group)),
        }
        for metric_key, _, _, _ in DIMENSION_METRICS:
            if metric_key == "stability":
                values = pd.to_numeric(group["mean_pairwise_ari"], errors="coerce").dropna()
            else:
                result_col = STABILITY_TO_RESULT_METRIC.get(metric_key)
                values = (
                    pd.to_numeric(group[result_col], errors="coerce").dropna()
                    if result_col in group.columns
                    else pd.Series(dtype=float)
                )
            row[f"{metric_key}_mean"] = float(values.mean()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _match_selected_to_results(selected: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    selected = selected.copy()
    results = results.copy()
    if "dimension" in selected.columns and "dimension" in results.columns:
        selected["dimension"] = pd.to_numeric(selected["dimension"], errors="coerce").astype("Int64")
        results["dimension"] = pd.to_numeric(results["dimension"], errors="coerce").astype("Int64")
    if "cluster_params_json" not in results.columns and "cluster_params" in results.columns:
        results["cluster_params_json"] = results["cluster_params"]

    join_cols = [column for column in SELECTION_JOIN_COLS if column in selected.columns and column in results.columns]
    if "dimension" in join_cols and selected["dimension"].isna().all():
        join_cols.remove("dimension")
    if not join_cols:
        raise ValueError("No common columns found for joining selected configs to seed-level results.")

    extra_cols = ["mean_pairwise_ari"] if "mean_pairwise_ari" in selected.columns else []
    selected_keys = selected[join_cols + extra_cols].drop_duplicates()
    return results.merge(selected_keys, on=join_cols, how="inner")


def _plot_trends(
    trends: pd.DataFrame,
    projector: str,
    dataset: str | None,
    encoder: str | None,
    raw_baselines: pd.DataFrame | None,
    output: Path,
    show_error_band: bool,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 15,
            "legend.title_fontsize": 16,
            "axes.linewidth": 0.9,
        }
    )

    fig, axes = plt.subplots(4, 2, figsize=(16.0, 21.0), sharex=True)
    axes_flat = axes.ravel()
    projector_label = PROJECTOR_LABELS.get(projector, projector.upper())
    clusterers = [clusterer for clusterer in METHOD_ORDER if clusterer in set(trends["clusterer"])]

    for axis, (metric_key, metric_label, _, direction) in zip(axes_flat, DIMENSION_METRICS):
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
            )
            if show_error_band:
                valid = np.isfinite(y) & np.isfinite(y_sd)
                if np.any(valid):
                    lower = y[valid] - y_sd[valid]
                    upper = y[valid] + y_sd[valid]
                    if metric_key in {"sd_index", "clusters"}:
                        positive = y[np.isfinite(y) & (y > 0)]
                        floor = float(np.nanmin(positive)) * 0.5 if positive.size else 1e-6
                        lower = np.maximum(lower, floor)
                    axis.fill_between(
                        x[valid],
                        lower,
                        upper,
                        color=colour,
                        alpha=0.16,
                        linewidth=0,
                    )
            if raw_baselines is not None and not raw_baselines.empty:
                raw_group = raw_baselines[raw_baselines["clusterer"] == clusterer]
                raw_col = f"{metric_key}_mean"
                if not raw_group.empty and raw_col in raw_group.columns:
                    raw_value = raw_group.iloc[0][raw_col]
                    if not pd.isna(raw_value):
                        raw_value = float(raw_value)
                        if metric_key not in {"sd_index", "clusters"} or raw_value > 0:
                            axis.axhline(
                                raw_value,
                                color=colour,
                                linestyle=(0, (1.4, 2.4)),
                                linewidth=2.1,
                                alpha=0.9,
                                label="_nolegend_",
                            )

        arrow = "↑" if direction == "higher" else "↓" if direction == "lower" else ""
        axis.set_title(f"{metric_label} {arrow}".strip(), fontweight="bold", pad=12)
        axis.set_xlabel("Dimensions")
        axis.set_ylabel(metric_label)
        axis.grid(True, axis="y", alpha=0.26, linewidth=0.9)
        axis.grid(True, axis="x", alpha=0.12, linewidth=0.7)
        axis.set_xlim(
            float(trends["dimension"].min()),
            float(trends["dimension"].max()),
        )
        axis.set_xticks([1, 10, 20, 30, 40, 50])
        for spine in axis.spines.values():
            spine.set_color("#666666")
        if metric_key in {"sd_index", "clusters"}:
            positive = trends[f"{metric_key}_mean"].dropna()
            positive = positive[positive > 0]
            if not positive.empty:
                axis.set_yscale("log")
        if metric_key in {"ami", "ari", "v_measure", "silhouette", "stability", "noise"}:
            axis.set_ylim(bottom=0.0)
        if metric_key == "noise" and "hdbscan" not in clusterers:
            axis.text(
                0.5,
                0.5,
                "Only applicable to HDBSCAN",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=15,
                color="#555555",
            )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if raw_baselines is not None and not raw_baselines.empty:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle=(0, (1.4, 2.4)),
                linewidth=2.1,
                label="Raw baseline",
            )
        )
        labels.append("Raw baseline")
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        fancybox=False,
        title="Clustering method",
        bbox_to_anchor=(0.5, 0.012),
        borderpad=0.9,
        columnspacing=1.4,
        handlelength=2.4,
    )
    legend.get_frame().set_edgecolor("#666666")
    legend.get_frame().set_linewidth(0.9)
    if len(clusterers) == 1:
        clusterer_label = METHOD_LABELS.get(clusterers[0], clusterers[0])
        title = f"{projector_label} {clusterer_label} Metrics vs Dimensions"
    else:
        title = f"{projector_label} Clustering Metrics vs Dimensions"
    if dataset is not None or encoder is not None:
        context = " / ".join(part for part in [dataset, encoder] if part is not None)
        title = f"{title}: {context}"
    fig.suptitle(
        title,
        y=0.992,
        fontsize=24,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.065, 1.0, 0.965), h_pad=2.2, w_pad=2.1)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
