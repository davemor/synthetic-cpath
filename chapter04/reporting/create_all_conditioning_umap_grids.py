#!/usr/bin/env python3
"""Render thesis-ready real-vs-synthetic UMAP grids for all conditioning modes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.spatial import ConvexHull


DATASETS = (
    ("pathmnist224", "Kather 100K"),
    ("icaird_cervical100k", "iCAIRD Cervical 100K"),
    ("pcam", "PCam"),
)
MODELS = (
    ("stylegan2ada", "StyleGAN2-ADA"),
    ("sd_lora", "SD-LoRA"),
    ("dcgan", "DCGAN"),
)
CONDITIONINGS = (
    ("unconditioned", "Unconditioned"),
    ("true_label", "Class label"),
    ("pseudo_label", "Global cluster"),
    ("label_global_cluster", "Class + global"),
    ("label_local_cluster", "Class + local"),
)
REAL_COLOUR = "#0072B2"
SYNTHETIC_COLOUR = "#D55E00"


def hull(frame: pd.DataFrame) -> np.ndarray:
    points = frame[["umap_1", "umap_2"]].to_numpy(dtype=np.float64)
    vertices = ConvexHull(points).vertices
    polygon = points[vertices]
    return np.vstack([polygon, polygon[0]])


def draw_panel(axis: plt.Axes, frame: pd.DataFrame) -> None:
    styles = {
        "Real": (REAL_COLOUR, "o", 5.0, 0.28),
        "Synthetic": (SYNTHETIC_COLOUR, "^", 6.5, 0.36),
    }
    for source, (colour, marker, size, alpha) in styles.items():
        subset = frame[frame["source"] == source]
        polygon = hull(subset)
        axis.fill(
            polygon[:, 0], polygon[:, 1], color=colour, alpha=0.08, zorder=1
        )
        axis.plot(
            polygon[:, 0], polygon[:, 1], color=colour, linewidth=2.0, zorder=3
        )
        axis.scatter(
            subset["umap_1"],
            subset["umap_2"],
            c=colour,
            marker=marker,
            s=size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    axis.tick_params(
        axis="both",
        labelsize=15,
        length=4.5,
        width=1.0,
        color="#888888",
        labelcolor="#333333",
    )
    axis.set_aspect("equal", adjustable="datalim")
    for spine in axis.spines.values():
        spine.set_color("#B8B8B8")
        spine.set_linewidth(1.0)


def metric_text(axis: plt.Axes, metrics: dict[str, float]) -> None:
    axis.text(
        0.5,
        -0.19,
        (
            f"FID {metrics['fid']:.1f}  |  "
            f"IP {metrics['improved_precision']:.4f}  |  "
            f"IR {metrics['improved_recall']:.4f}"
        ),
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=18,
        fontweight="semibold",
        clip_on=False,
    )


def arm_metrics(
    metrics: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    conditioning: str,
) -> dict[str, float]:
    rows = metrics[
        (metrics["dataset"] == dataset)
        & (metrics["model"] == model)
        & (metrics["conditioning"] == conditioning)
        & (metrics["seed"] == 0)
    ]
    values = rows.set_index("metric")["value"].to_dict()
    expected = {"fid", "precision", "recall"}
    if set(values) != expected:
        raise ValueError(
            f"Expected {expected} for {dataset}/{model}/{conditioning}, "
            f"found {set(values)}"
        )
    return {
        "fid": float(values["fid"]),
        "improved_precision": float(values["precision"]),
        "improved_recall": float(values["recall"]),
    }


def render_dataset(
    input_root: Path,
    output_root: Path,
    metrics: pd.DataFrame,
    dataset_id: str,
    dataset_name: str,
) -> None:
    figure, axes = plt.subplots(5, 3, figsize=(18, 27), facecolor="white")
    for row, (conditioning, _) in enumerate(CONDITIONINGS):
        for column, (model_id, _) in enumerate(MODELS):
            arm = input_root / conditioning / f"{dataset_id}_{model_id}"
            draw_panel(axes[row, column], pd.read_csv(arm / "coordinates.csv"))
            metric_text(
                axes[row, column],
                arm_metrics(
                    metrics,
                    dataset=dataset_id,
                    model=model_id,
                    conditioning=conditioning,
                ),
            )

    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.075,
        top=0.915,
        wspace=0.18,
        hspace=0.37,
    )
    figure.suptitle(
        dataset_name,
        x=0.53,
        y=0.984,
        fontsize=31,
        fontweight="semibold",
    )
    for column, (_, model_name) in enumerate(MODELS):
        position = axes[0, column].get_position()
        figure.text(
            (position.x0 + position.x1) / 2,
            0.943,
            model_name,
            ha="center",
            va="center",
            fontsize=27,
            fontweight="semibold",
        )
    for row, (_, conditioning_name) in enumerate(CONDITIONINGS):
        position = axes[row, 0].get_position()
        figure.text(
            0.028,
            (position.y0 + position.y1) / 2,
            conditioning_name,
            ha="center",
            va="center",
            rotation=90,
            fontsize=24,
            fontweight="semibold",
        )
    handles = [
        Line2D(
            [0], [0], marker="o", color=REAL_COLOUR, linewidth=3, markersize=13,
            markerfacecolor=REAL_COLOUR, label="Real"
        ),
        Line2D(
            [0], [0], marker="^", color=SYNTHETIC_COLOUR, linewidth=3,
            markersize=14, markerfacecolor=SYNTHETIC_COLOUR, label="Synthetic"
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.012),
        ncol=2,
        frameon=False,
        fontsize=20,
        handlelength=2.5,
        columnspacing=3.0,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_slug = dataset_id.replace("_", "-")
    stem = output_root / f"synthetic-umap-{dataset_slug}-5x3-grid"
    figure.savefig(stem.with_suffix(".pdf"), dpi=300, facecolor="white")
    figure.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    args = parser.parse_args()
    metrics = pd.read_csv(args.metrics_csv)
    for dataset_id, dataset_name in DATASETS:
        render_dataset(
            args.input_root,
            args.output_root,
            metrics,
            dataset_id,
            dataset_name,
        )


if __name__ == "__main__":
    main()
