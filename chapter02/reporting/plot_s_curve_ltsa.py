#!/usr/bin/env python3
"""Plot a sampled S-curve manifold and its two-dimensional LTSA embedding."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.datasets import make_s_curve
from sklearn.manifold import LocallyLinearEmbedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a 3D S-curve beside its 2D LTSA unwrapping."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2_500,
        help="Number of points to sample (default: 2500).",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=30,
        help="Number of neighbors used by LTSA (default: 30).",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.03,
        help="Gaussian noise added to the S-curve (default: 0.03).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional output image path, for example figures/s_curve_isomap.png.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (useful together with --save).",
    )
    args = parser.parse_args()

    if args.samples < 3:
        parser.error("--samples must be at least 3")
    if not 1 <= args.neighbors < args.samples:
        parser.error("--neighbors must be at least 1 and smaller than --samples")
    if args.noise < 0:
        parser.error("--noise cannot be negative")
    if args.no_show and args.save is None:
        parser.error("--no-show requires --save")

    return args


def main() -> None:
    args = parse_args()

    points_3d, position_along_curve = make_s_curve(
        n_samples=args.samples,
        noise=args.noise,
        random_state=args.seed,
    )
    points_2d = LocallyLinearEmbedding(
        n_neighbors=args.neighbors,
        n_components=2,
        method="ltsa",
        eigen_solver="arpack",
    ).fit_transform(points_3d)

    color_norm = Normalize(
        vmin=position_along_curve.min(), vmax=position_along_curve.max()
    )
    scatter_options = {
        "c": position_along_curve,
        "cmap": "viridis",
        "norm": color_norm,
        "s": 22,
        "alpha": 0.8,
        "edgecolors": "none",
    }

    figure = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.0))

    axis_3d = figure.add_subplot(grid[0], projection="3d")
    axis_3d.scatter(
        points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], **scatter_options
    )
    axis_3d.set(
        title="Original S-curve manifold (3D)",
        xlabel="x",
        ylabel="y",
        zlabel="z",
    )
    axis_3d.view_init(elev=18, azim=-68)
    axis_3d.set_box_aspect((1, 1, 1))

    axis_2d = figure.add_subplot(grid[1])
    scatter = axis_2d.scatter(points_2d[:, 0], points_2d[:, 1], **scatter_options)
    axis_2d.set(
        title=f"LTSA unwrapping (2D, k={args.neighbors})",
        xlabel="LTSA component 1",
        ylabel="LTSA component 2",
    )
    axis_2d.set_box_aspect(1)
    axis_2d.margins(x=0.04, y=0.08)
    axis_2d.grid(alpha=0.2)

    colorbar = figure.colorbar(scatter, ax=(axis_3d, axis_2d), shrink=0.82, pad=0.04)
    colorbar.set_label("Position along the S-curve")
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {args.save}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
