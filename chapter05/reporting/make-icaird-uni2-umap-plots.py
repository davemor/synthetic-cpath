#!/usr/bin/env python3
"""Make iCAIRD UNI2 + UMAP 1D/2D class-label visualisations.

The inputs are curve-stage projection cache files from base seed 0, fold 0.
They contain a stratified 50k curve sample split into train and validation
folds. The plots combine both fold partitions for a single projection view.
"""

from __future__ import annotations

import html
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np


RUN_DIR = Path(
    "/Users/dm236/Documents/Research/AllResults/Chapter5-dimensionality-reduction/"
    "results_with_icaird/hproj-2026-05-25_16-41-27-bb2a5203-icaird100k"
)
PLOT_DIR = RUN_DIR / "report" / "plots"
CACHE_DIR = PLOT_DIR / "icaird-uni2-umap-cache"

INPUT_1D = CACHE_DIR / "icaird-uni2-umap-1d-seed0-fold0.npz"
INPUT_2D = CACHE_DIR / "icaird-uni2-umap-2d-seed0-fold0.npz"

OUT_1D_SVG = PLOT_DIR / "icaird-uni2-umap-1d-histogram.svg"
OUT_1D_PDF = PLOT_DIR / "icaird-uni2-umap-1d-histogram.pdf"
OUT_1D_PNG = PLOT_DIR / "icaird-uni2-umap-1d-histogram.png"
OUT_2D_SVG = PLOT_DIR / "icaird-uni2-umap-2d-scatter.svg"
OUT_2D_PDF = PLOT_DIR / "icaird-uni2-umap-2d-scatter.pdf"
OUT_2D_PNG = PLOT_DIR / "icaird-uni2-umap-2d-scatter.png"

LABELS = {
    0: "Normal",
    1: "Low grade",
    2: "High grade",
    3: "Malignant",
}

COLORS = {
    0: "#2F6FB0",
    1: "#E08D2D",
    2: "#3C8D5A",
    3: "#B94748",
}


def load_fold_projection(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        features = np.concatenate([data["train_features"], data["valid_features"]], axis=0)
        labels = np.concatenate([data["train_labels"], data["valid_labels"]], axis=0)
    return features.astype(float), labels.astype(int)


def nice_limits(values: np.ndarray, pad_fraction: float = 0.04) -> tuple[float, float]:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if math.isclose(lo, hi):
        return lo - 1.0, hi + 1.0
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def fmt_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Helvetica, Arial, sans-serif; fill: #202124; }",
        ".axis { stroke: #202124; stroke-width: 1.4; }",
        ".grid { stroke: #d9d9d9; stroke-width: 0.8; }",
        ".tick { font-size: 17px; }",
        ".label { font-size: 22px; font-weight: 600; }",
        ".legend { font-size: 18px; }",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def draw_axes(
    parts: list[str],
    *,
    left: int,
    top: int,
    plot_w: int,
    plot_h: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
) -> tuple:
    x0, x1 = xlim
    y0, y1 = ylim

    def sx(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - y0) / (y1 - y0) * plot_h

    for y in ticks(y0, y1):
        py = sy(y)
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{py:.2f}" y2="{py:.2f}"/>')
        parts.append(f'<text class="tick" x="{left - 12}" y="{py + 6:.2f}" text-anchor="end">{fmt_tick(y)}</text>')
    for x in ticks(x0, x1):
        px = sx(x)
        parts.append(f'<line class="grid" y1="{top}" y2="{top + plot_h}" x1="{px:.2f}" x2="{px:.2f}"/>')
        parts.append(f'<text class="tick" x="{px:.2f}" y="{top + plot_h + 28}" text-anchor="middle">{fmt_tick(x)}</text>')

    parts.append(f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}"/>')
    parts.append(f'<text class="label" x="{left + plot_w / 2:.2f}" y="{top + plot_h + 70}" text-anchor="middle">{html.escape(xlabel)}</text>')
    parts.append(
        f'<text class="label" transform="translate({left - 70},{top + plot_h / 2:.2f}) rotate(-90)" '
        f'text-anchor="middle">{html.escape(ylabel)}</text>'
    )
    return sx, sy


def add_legend(parts: list[str], *, x: int, y: int) -> None:
    item_w = 145
    for i, label in enumerate(LABELS):
        lx = x + i * item_w
        parts.append(f'<circle cx="{lx}" cy="{y}" r="7" fill="{COLORS[label]}"/>')
        parts.append(f'<text class="legend" x="{lx + 14}" y="{y + 6}">{html.escape(LABELS[label])}</text>')


def make_scatter(features: np.ndarray, labels: np.ndarray) -> None:
    width, height = 980, 965
    left, top, plot_w, plot_h = 110, 55, 760, 760
    x = features[:, 0]
    y = features[:, 1]
    xlim = nice_limits(x)
    ylim = nice_limits(y)

    parts = svg_header(width, height)
    sx, sy = draw_axes(
        parts,
        left=left,
        top=top,
        plot_w=plot_w,
        plot_h=plot_h,
        xlim=xlim,
        ylim=ylim,
        xlabel="UMAP 1",
        ylabel="UMAP 2",
    )

    for label in LABELS:
        mask = labels == label
        color = COLORS[label]
        coords = zip(x[mask], y[mask])
        group = [f'<g fill="{color}" opacity="0.55">']
        for px, py in coords:
            group.append(f'<circle cx="{sx(float(px)):.2f}" cy="{sy(float(py)):.2f}" r="1.45"/>')
        group.append("</g>")
        parts.extend(group)

    add_legend(parts, x=left + 70, y=height - 42)
    parts.append("</svg>")
    OUT_2D_SVG.write_text("\n".join(parts))


def make_histogram(features: np.ndarray, labels: np.ndarray) -> None:
    width, height = 980, 650
    left, top, plot_w, plot_h = 110, 55, 760, 410
    values = features[:, 0]
    xlim = nice_limits(values)
    bins = np.linspace(xlim[0], xlim[1], 55)
    densities = {}
    y_max = 0.0
    for label in LABELS:
        hist, edges = np.histogram(values[labels == label], bins=bins, density=True)
        densities[label] = (hist, edges)
        y_max = max(y_max, float(np.max(hist)))
    ylim = (0.0, y_max * 1.12)

    parts = svg_header(width, height)
    sx, sy = draw_axes(
        parts,
        left=left,
        top=top,
        plot_w=plot_w,
        plot_h=plot_h,
        xlim=xlim,
        ylim=ylim,
        xlabel="UMAP 1",
        ylabel="Density",
    )

    for label in LABELS:
        hist, edges = densities[label]
        color = COLORS[label]
        points = [f"{sx(float(edges[0])):.2f},{sy(0):.2f}"]
        for i, h in enumerate(hist):
            x_left = sx(float(edges[i]))
            x_right = sx(float(edges[i + 1]))
            y_top = sy(float(h))
            points.append(f"{x_left:.2f},{y_top:.2f}")
            points.append(f"{x_right:.2f},{y_top:.2f}")
        points.append(f"{sx(float(edges[-1])):.2f},{sy(0):.2f}")
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="{color}" fill-opacity="0.18" '
            f'stroke="{color}" stroke-width="3.0" stroke-linejoin="round"/>'
        )

    add_legend(parts, x=left + 70, y=height - 42)
    parts.append("</svg>")
    OUT_1D_SVG.write_text("\n".join(parts))


def convert(svg_path: Path, pdf_path: Path, png_path: Path) -> None:
    rsvg = shutil.which("rsvg-convert")
    if rsvg is None:
        raise RuntimeError("rsvg-convert is required to write PDF/PNG outputs.")
    subprocess.run([rsvg, "-f", "pdf", "-o", str(pdf_path), str(svg_path)], check=True)
    subprocess.run([rsvg, "-f", "png", "-o", str(png_path), str(svg_path)], check=True)


def main() -> None:
    features_1d, labels_1d = load_fold_projection(INPUT_1D)
    features_2d, labels_2d = load_fold_projection(INPUT_2D)
    make_histogram(features_1d, labels_1d)
    make_scatter(features_2d, labels_2d)
    convert(OUT_1D_SVG, OUT_1D_PDF, OUT_1D_PNG)
    convert(OUT_2D_SVG, OUT_2D_PDF, OUT_2D_PNG)
    print(f"Wrote {OUT_1D_PDF}")
    print(f"Wrote {OUT_2D_PDF}")
    print(f"Plotted {features_1d.shape[0]} 1D points and {features_2d.shape[0]} 2D points.")


if __name__ == "__main__":
    main()
