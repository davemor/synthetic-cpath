#!/usr/bin/env python3
"""Plot two one-dimensional Gaussian distributions and their FID."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REAL_COLOR = "#2468b4"
GENERATED_COLOR = "#dc3c3c"


def gaussian_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Return the probability density of N(mean, std^2)."""
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))


def fid_1d(mean_1: float, std_1: float, mean_2: float, std_2: float) -> float:
    """Compute FID between two univariate Gaussian distributions.

    Starting from the general Gaussian FID formula,

        ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2(C_1 C_2)^(1/2)),

    the scalar covariances C_i = std_i^2 reduce it to
    (mean_1 - mean_2)^2 + (std_1 - std_2)^2.
    """
    return (mean_1 - mean_2) ** 2 + (std_1 - std_2) ** 2


def make_figure(
    real_mean: float = -0.7,
    real_std: float = 1.0,
    generated_mean: float = 1.0,
    generated_std: float = 1.35,
) -> plt.Figure:
    """Create the Gaussian comparison figure."""
    if real_std <= 0 or generated_std <= 0:
        raise ValueError("Standard deviations must be positive")

    lower = min(real_mean - 4 * real_std, generated_mean - 4 * generated_std)
    upper = max(real_mean + 4 * real_std, generated_mean + 4 * generated_std)
    x = np.linspace(lower, upper, 1000)
    real_density = gaussian_pdf(x, real_mean, real_std)
    generated_density = gaussian_pdf(x, generated_mean, generated_std)
    distance = fid_1d(real_mean, real_std, generated_mean, generated_std)

    fig, ax = plt.subplots(figsize=(10, 4.35), constrained_layout=True)

    ax.fill_between(x, real_density, color=REAL_COLOR, alpha=0.18)
    ax.fill_between(x, generated_density, color=GENERATED_COLOR, alpha=0.18)
    ax.plot(
        x,
        real_density,
        color=REAL_COLOR,
        lw=2.8,
        label=rf"Real: $\mathcal{{N}}({real_mean:g},\,{real_std**2:g})$",
    )
    ax.plot(
        x,
        generated_density,
        color=GENERATED_COLOR,
        lw=2.8,
        label=rf"Generated: $\mathcal{{N}}({generated_mean:g},\,{generated_std**2:g})$",
    )

    # Mark the two means and show their separation.
    real_peak = gaussian_pdf(np.asarray([real_mean]), real_mean, real_std)[0]
    generated_peak = gaussian_pdf(np.asarray([generated_mean]), generated_mean, generated_std)[0]
    ax.vlines(real_mean, 0, real_peak, color=REAL_COLOR, lw=1.4, ls="--", alpha=0.8)
    ax.vlines(generated_mean, 0, generated_peak, color=GENERATED_COLOR, lw=1.4, ls="--", alpha=0.8)

    arrow_y = max(real_peak, generated_peak) * 0.56
    ax.annotate(
        "",
        xy=(generated_mean, arrow_y),
        xytext=(real_mean, arrow_y),
        arrowprops={"arrowstyle": "<->", "color": "#444444", "lw": 1.5},
    )
    ax.text(
        (real_mean + generated_mean) / 2,
        arrow_y * 1.06,
        rf"$|\mu_r-\mu_g|={abs(real_mean-generated_mean):g}$",
        ha="center",
        va="bottom",
        color="#333333",
        fontsize=11,
    )

    formula = (
        r"$\mathrm{FID}=(\mu_r-\mu_g)^2+"
        r"(\sigma_r-\sigma_g)^2$"
        "\n"
        rf"$=({real_mean:g}-{generated_mean:g})^2+"
        rf"({real_std:g}-{generated_std:g})^2={distance:.3f}$"
    )
    ax.text(
        0.98,
        0.95,
        formula,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.95},
    )

    ax.set_title("Fréchet Distance Between Two 1D Gaussian Distributions", fontsize=16, pad=14)
    ax.set_xlabel("Feature value", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.8, alpha=0.75)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-mean", type=float, default=-0.7)
    parser.add_argument("--real-std", type=float, default=1.0)
    parser.add_argument("--generated-mean", type=float, default=1.0)
    parser.add_argument("--generated-std", type=float, default=1.35)
    parser.add_argument("-o", "--output", type=Path, default=Path("gaussian-fid-1d.png"))
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    figure = make_figure(
        real_mean=args.real_mean,
        real_std=args.real_std,
        generated_mean=args.generated_mean,
        generated_std=args.generated_std,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.output}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
