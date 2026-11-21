#!/usr/bin/env python3
"""Draw a three-panel illustration of distribution precision and recall.

The picture is conceptual: red denotes a generated distribution P_g and blue
denotes a real distribution P_r.  The random seed makes the output exactly
reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


RED = "#d62728"
BLUE = "#1f5aa6"


def gaussian_pdf(x: np.ndarray, mean: tuple[float, float], cov: np.ndarray) -> np.ndarray:
    """Evaluate a two-dimensional Gaussian density on points in ``x``."""
    delta = x - np.asarray(mean)
    inv = np.linalg.inv(cov)
    exponent = np.einsum("...i,ij,...j->...", delta, inv, delta)
    return np.exp(-0.5 * exponent) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))


def mixture_pdf(
    x: np.ndarray,
    means: list[tuple[float, float]],
    covariances: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    return sum(w * gaussian_pdf(x, m, c) for m, c, w in zip(means, covariances, weights))


def blob(
    ax: plt.Axes,
    vertices: list[tuple[float, float]],
    color: str,
    *,
    alpha: float = 0.24,
    dashed: bool = False,
    fill: bool = True,
    linewidth: float = 1.1,
) -> PathPatch:
    """Add a smooth closed blob and return it for point-in-path tests."""
    # Chaikin corner cutting turns the control polygon into a smooth closed
    # curve without adding a SciPy dependency.
    verts = np.asarray(vertices[:-1] if vertices[0] == vertices[-1] else vertices, dtype=float)
    for _ in range(4):
        following = np.roll(verts, -1, axis=0)
        verts = np.stack((0.75 * verts + 0.25 * following,
                          0.25 * verts + 0.75 * following), axis=1).reshape(-1, 2)
    verts = np.vstack((verts, verts[0]))
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    patch = PathPatch(
        MplPath(verts, codes),
        facecolor=color if fill else "none",
        edgecolor=color,
        alpha=alpha if fill else 0.65,
        lw=linewidth,
        linestyle=(0, (4, 3)) if dashed else "solid",
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def sample_mixture(
    rng: np.random.Generator,
    n: int,
    means: list[tuple[float, float]],
    covariances: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    components = rng.choice(len(means), size=n, p=np.asarray(weights) / np.sum(weights))
    result = np.empty((n, 2))
    for index in range(len(means)):
        selected = components == index
        result[selected] = rng.multivariate_normal(means[index], covariances[index], selected.sum())
    return result


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-2.7, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, y=-0.09, fontsize=14)


def draw_distributions(ax: plt.Axes) -> None:
    x, y = np.meshgrid(np.linspace(-3.5, 3.5, 300), np.linspace(-2.7, 2.8, 260))
    points = np.stack((x, y), axis=-1)

    red_means = [(-1.55, 0.55), (0.05, -0.15), (1.05, -0.75)]
    red_covs = [np.array([[0.55, 0.08], [0.08, 0.50]]),
                 np.array([[0.65, -0.20], [-0.20, 0.35]]),
                 np.array([[0.55, -0.18], [-0.18, 0.45]])]
    blue_means = [(-1.65, -0.80), (0.85, 0.82)]
    blue_covs = [np.array([[0.65, 0.03], [0.03, 0.38]]),
                  np.array([[0.85, 0.12], [0.12, 0.74]])]

    for density, color in [
        (mixture_pdf(points, blue_means, blue_covs, [0.30, 0.70]), BLUE),
        (mixture_pdf(points, red_means, red_covs, [0.34, 0.42, 0.24]), RED),
    ]:
        levels = np.linspace(density.max() * 0.10, density.max() * 0.90, 9)
        ax.contourf(x, y, density, levels=np.r_[levels, density.max() * 1.01], colors=[color], alpha=0.055)
        ax.contour(x, y, density, levels=levels, colors=color, linewidths=0.8, alpha=0.62)

    ax.text(-3.15, 1.95, r"$P_g$", fontsize=19)
    ax.text(2.35, 2.35, r"$P_r$", fontsize=19)
    style_axis(ax, "(a) Example distributions")


def draw_precision(ax: plt.Axes, rng: np.random.Generator) -> None:
    blue_support = blob(
        ax,
        [(-3.0, -1.25), (-3.25, 0.05), (-1.55, 0.80), (-0.05, 2.35),
         (1.55, 2.55), (2.40, 1.65), (2.15, 0.45), (1.90, -0.35),
         (0.45, -0.55), (-1.15, -2.10), (-3.0, -1.25)],
        BLUE,
    )
    generated_support = blob(
        ax,
        [(-2.70, 1.60), (-1.60, 2.25), (-0.65, 1.35), (0.20, 1.10),
         (1.45, 0.15), (2.15, -1.25), (0.75, -1.65), (-0.20, -1.70),
         (-1.20, -0.55), (-2.35, 0.05), (-2.70, 1.60)],
        RED,
        dashed=True,
        fill=False,
        linewidth=2.0,
    )

    candidates = sample_mixture(
        rng,
        3000,
        [(-1.25, 0.55), (0.20, -0.35), (1.15, -0.95)],
        [np.array([[0.48, -0.10], [-0.10, 0.34]]),
         np.array([[0.58, -0.24], [-0.24, 0.31]]),
         np.array([[0.32, -0.08], [-0.08, 0.28]])],
        [0.34, 0.43, 0.23],
    )
    points = candidates[generated_support.get_path().contains_points(candidates)][:430]
    if len(points) < 430:
        raise RuntimeError("Generated support contains too few sampled points")
    inside = blue_support.get_path().contains_points(points)
    ax.scatter(points[~inside, 0], points[~inside, 1], s=10, facecolors="white", edgecolors=RED, linewidths=0.7, zorder=3)
    ax.scatter(points[inside, 0], points[inside, 1], s=11, c="red", edgecolors="#9b1010", linewidths=0.35, zorder=4)
    style_axis(ax, "(b) Precision")


def draw_recall(ax: plt.Axes, rng: np.random.Generator) -> None:
    red_support = blob(
        ax,
        [(-2.70, 1.55), (-1.60, 2.30), (-0.20, 1.25), (0.75, 0.95),
         (1.95, 0.15), (2.15, -1.15), (0.85, -1.75), (-0.05, -1.15),
         (-1.10, -0.55), (-2.35, 0.20), (-2.70, 1.55)],
        RED,
    )
    real_support = blob(
        ax,
        [(-3.05, -0.55), (-2.30, 0.30), (-1.35, 0.85), (-0.25, 2.15),
         (1.20, 2.70), (2.65, 2.15), (2.80, 0.70), (3.00, -0.10),
         (1.35, -0.45), (0.35, -0.85), (-1.35, -1.80), (-3.05, -0.55)],
        BLUE,
        dashed=True,
        fill=False,
        linewidth=2.0,
    )

    candidates = sample_mixture(
        rng,
        3000,
        [(-1.35, -0.65), (0.15, 0.35), (1.35, 1.05)],
        [np.array([[0.42, 0.08], [0.08, 0.32]]),
         np.array([[0.62, 0.12], [0.12, 0.48]]),
         np.array([[0.55, -0.08], [-0.08, 0.62]])],
        [0.18, 0.35, 0.47],
    )
    points = candidates[real_support.get_path().contains_points(candidates)][:430]
    if len(points) < 430:
        raise RuntimeError("Real support contains too few sampled points")
    inside = red_support.get_path().contains_points(points)
    ax.scatter(points[~inside, 0], points[~inside, 1], s=10, facecolors="white", edgecolors=BLUE, linewidths=0.7, zorder=3)
    ax.scatter(points[inside, 0], points[inside, 1], s=11, c="#0878d1", edgecolors="#073e7f", linewidths=0.35, zorder=4)
    style_axis(ax, "(c) Recall")


def make_figure(seed: int = 7) -> plt.Figure:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.25), constrained_layout=True)
    draw_distributions(axes[0])
    draw_precision(axes[1], rng)
    draw_recall(axes[2], rng)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("precision_recall_distributions.png"))
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    figure = make_figure(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.output}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
