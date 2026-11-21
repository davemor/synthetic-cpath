#!/usr/bin/env python3
"""Create a clean 3x3 real-vs-synthetic UMAP comparison PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.spatial import ConvexHull


DATASETS = ("Kather100K", "iCAIRD", "PCam")
MODELS = ("StyleGAN2-ADA", "SD-LoRA", "DCGAN")
DATASET_DISPLAY = {
    "Kather100K": "Kather100K",
    "iCAIRD": "iCAIRD Cervical 100k",
    "PCam": "PCam",
}
# Seed-0 image FID and UNI-space improved precision/recall.
QUALITY_METRICS = {
    "class_label": {
        ("StyleGAN2-ADA", "Kather100K"): (28.975714016974734, 0.0578, 0.0013),
        ("StyleGAN2-ADA", "iCAIRD"): (33.4543452250702, 0.2549, 0.0004),
        ("StyleGAN2-ADA", "PCam"): (42.41820565777937, 0.1271, 0.0003),
        ("SD-LoRA", "Kather100K"): (119.76575563468094, 0.0060, 0.0035),
        ("SD-LoRA", "iCAIRD"): (106.27485776851623, 0.0037, 0.0000),
        ("SD-LoRA", "PCam"): (159.10929687226778, 0.0005, 0.0357),
        ("DCGAN", "Kather100K"): (182.15247219678216, 0.0000, 0.0000),
        ("DCGAN", "iCAIRD"): (182.56794606183155, 0.2721, 0.0000),
        ("DCGAN", "PCam"): (155.2061676634186, 0.0956, 0.0000),
    },
    "class_local": {
        ("StyleGAN2-ADA", "Kather100K"): (30.83288027722773, 0.0994, 0.0016),
        ("StyleGAN2-ADA", "iCAIRD"): (20.285424844734962, 0.2539, 0.0226),
        ("StyleGAN2-ADA", "PCam"): (16.70366663292316, 0.2043, 0.0009),
        ("SD-LoRA", "Kather100K"): (83.53563329581186, 0.0100, 0.0049),
        ("SD-LoRA", "iCAIRD"): (93.09585638578608, 0.0012, 0.0000),
        ("SD-LoRA", "PCam"): (119.76806105819475, 0.0005, 0.0000),
        ("DCGAN", "Kather100K"): (188.98407523555915, 0.0000, 0.0000),
        ("DCGAN", "iCAIRD"): (132.34946832615043, 0.1681, 0.0000),
        ("DCGAN", "PCam"): (191.13803607752203, 0.0051, 0.0000),
    },
}
REAL_COLOUR = "#1F77B4"
SYNTHETIC_COLOUR = "#D62728"


def coordinate_paths(
    root: Path, conditioning: str
) -> dict[tuple[str, str], Path]:
    """Return coordinate tables keyed by (model, dataset)."""
    if conditioning == "class_label":
        return {
        ("StyleGAN2-ADA", "iCAIRD"): root
        / "icaird_stylegan_vs_real_uni_umap_coordinates.csv",
        ("SD-LoRA", "iCAIRD"): root
        / "icaird_sd_lora/icaird_sd_lora_vs_real_uni_umap_coordinates.csv",
        ("DCGAN", "iCAIRD"): root
        / "icaird_dcgan/icaird_dcgan_vs_real_uni_umap_coordinates.csv",
        ("StyleGAN2-ADA", "PCam"): root
        / "pcam_stylegan2ada/pcam_stylegan2ada_vs_real_uni_umap_coordinates.csv",
        ("SD-LoRA", "PCam"): root
        / "pcam_sd_lora/pcam_sd_lora_vs_real_uni_umap_coordinates.csv",
        ("DCGAN", "PCam"): root
        / "pcam_dcgan/pcam_dcgan_vs_real_uni_umap_coordinates.csv",
        ("StyleGAN2-ADA", "Kather100K"): root
        / "kather100k_stylegan2ada/"
        "kather100k_stylegan2ada_vs_real_uni_umap_coordinates.csv",
        ("SD-LoRA", "Kather100K"): root
        / "kather100k_sd_lora/kather100k_sd_lora_vs_real_uni_umap_coordinates.csv",
        ("DCGAN", "Kather100K"): root
        / "kather100k_dcgan/kather100k_dcgan_vs_real_uni_umap_coordinates.csv",
        }
    if conditioning == "class_local":
        dataset_ids = {
            "Kather100K": "pathmnist224",
            "iCAIRD": "icaird_cervical100k",
            "PCam": "pcam",
        }
        model_ids = {
            "StyleGAN2-ADA": "stylegan2ada",
            "SD-LoRA": "sd_lora",
            "DCGAN": "dcgan",
        }
        return {
            (model, dataset): root
            / f"{dataset_ids[dataset]}_{model_ids[model]}"
            / (
                f"{dataset_ids[dataset]}_{model_ids[model]}_"
                "class_local_uni_umap_coordinates.csv"
            )
            for model in MODELS
            for dataset in DATASETS
        }
    raise ValueError(f"Unsupported conditioning preset: {conditioning}")


def convex_hull_polygon(frame: pd.DataFrame) -> np.ndarray:
    coordinates = frame[["umap_1", "umap_2"]].to_numpy(
        dtype=np.float64, copy=False
    )
    if len(coordinates) < 3 or not np.isfinite(coordinates).all():
        raise ValueError("Each source requires at least three finite coordinates")
    hull = ConvexHull(coordinates)
    polygon = coordinates[hull.vertices]
    return np.vstack([polygon, polygon[0]])


def draw_panel(axis: plt.Axes, frame: pd.DataFrame, model: str) -> None:
    """Draw one panel without a title, caption, repeated legend, or ticks."""
    expected_sources = {"Real", model}
    if set(frame["source"].unique()) != expected_sources:
        raise ValueError(
            f"Expected sources {sorted(expected_sources)}, found "
            f"{sorted(frame['source'].unique())}"
        )

    styles = {
        "Real": (REAL_COLOUR, "o", 5.0, 0.30),
        model: (SYNTHETIC_COLOUR, "^", 6.5, 0.38),
    }
    for source in ("Real", model):
        subset = frame[frame["source"] == source]
        colour, marker, size, alpha = styles[source]
        polygon = convex_hull_polygon(subset)
        axis.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor=colour,
            edgecolor=colour,
            alpha=0.09,
            linewidth=1.8,
            zorder=1,
        )
        axis.plot(
            polygon[:, 0],
            polygon[:, 1],
            color=colour,
            linewidth=1.7,
            alpha=0.9,
            zorder=4,
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
            zorder=2 if source == "Real" else 3,
        )

    axis.set_xlabel("UMAP dimension 1", fontsize=10.5, color="#666666", labelpad=7)
    axis.set_ylabel("UMAP dimension 2", fontsize=10.5, color="#666666", labelpad=7)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    axis.tick_params(
        axis="both",
        which="major",
        color="#B8B8B8",
        labelcolor="#666666",
        labelsize=8.5,
        length=3.5,
        width=0.8,
        direction="out",
    )
    axis.set_aspect("equal", adjustable="datalim")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#D0D0D0")
        spine.set_linewidth(0.9)


def add_metric_caption(
    axis: plt.Axes,
    *,
    fid: float,
    precision: float,
    recall: float,
) -> None:
    axis.text(
        0.5,
        -0.175,
        f"FID {fid:.1f}  |  Precision {precision:.4f}  |  Recall {recall:.4f}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=15,
        fontweight="semibold",
        color="#000000",
        clip_on=False,
    )


def add_legend(figure: plt.Figure) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=REAL_COLOUR,
            linestyle="-",
            linewidth=2.5,
            markersize=11,
            markerfacecolor=REAL_COLOUR,
            markeredgewidth=0,
            label="Real",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color=SYNTHETIC_COLOUR,
            linestyle="-",
            linewidth=2.5,
            markersize=12,
            markerfacecolor=SYNTHETIC_COLOUR,
            markeredgewidth=0,
            label="Synthetic",
        ),
    ]
    legend = figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        frameon=False,
        fontsize=15,
        handlelength=2.5,
        columnspacing=2.8,
        handletextpad=0.7,
    )
    legend._legend_box.align = "left"


def create_grid(input_root: Path, output_pdf: Path, conditioning: str) -> None:
    paths = coordinate_paths(input_root, conditioning)
    metrics = QUALITY_METRICS[conditioning]
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing coordinate tables: {missing}")

    figure, axes = plt.subplots(3, 3, figsize=(18, 18), facecolor="white")
    for row, model in enumerate(MODELS):
        for column, dataset in enumerate(DATASETS):
            frame = pd.read_csv(paths[(model, dataset)])
            draw_panel(axes[row, column], frame, model)
            add_metric_caption(
                axes[row, column],
                fid=metrics[(model, dataset)][0],
                precision=metrics[(model, dataset)][1],
                recall=metrics[(model, dataset)][2],
            )

    # One horizontal legend serves the complete grid.
    add_legend(figure)

    figure.subplots_adjust(
        left=0.090,
        right=0.985,
        bottom=0.105,
        top=0.935,
        wspace=0.180,
        hspace=0.320,
    )

    for column, dataset in enumerate(DATASETS):
        position = axes[0, column].get_position()
        figure.text(
            (position.x0 + position.x1) / 2,
            0.966,
            DATASET_DISPLAY[dataset],
            ha="center",
            va="center",
            fontsize=22,
            fontweight="semibold",
        )

    for row, model in enumerate(MODELS):
        position = axes[row, 0].get_position()
        figure.text(
            0.030,
            (position.y0 + position.y1) / 2,
            model,
            ha="center",
            va="center",
            rotation=90,
            fontsize=21,
            fontweight="semibold",
        )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_pdf,
        format="pdf",
        dpi=300,
        facecolor="white",
        metadata={
            "Title": "Real and synthetic UNI embedding manifold grid",
            "Subject": (
                "UMAP comparison by dataset and generative model; "
                f"conditioning={conditioning}"
            ),
            "Creator": "Synthetic Histopathology UMAP grid script",
        },
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("figures/synthetic_patches_umap"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("output/pdf/synthetic_umap_3x3_grid.pdf"),
    )
    parser.add_argument(
        "--conditioning",
        choices=sorted(QUALITY_METRICS),
        default="class_label",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    create_grid(arguments.input_root, arguments.output_pdf, arguments.conditioning)
