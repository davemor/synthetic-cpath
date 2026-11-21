#!/usr/bin/env python3
"""Create a combined accuracy/ROC-AUC vs dimensions figure with one legend.

This script intentionally avoids matplotlib so it can run on a plain Python
install. It writes SVG directly, then uses rsvg-convert when available to make
a PNG copy.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


PROJECTOR_ORDER = ["grp", "pca", "umap"]
PROJECTOR_LABELS = {
    "grp": "GRP",
    "pca": "PCA",
    "umap": "UMAP",
}
PROJECTOR_COLOURS = {
    "grp": "#1f77b4",
    "pca": "#ff7f0e",
    "umap": "#2ca02c",
}

METRICS = [
    {
        "key": "accuracy",
        "title": "logistic-regression: accuracy vs n_components",
        "ylabel": "accuracy",
    },
    {
        "key": "roc_auc",
        "title": "logistic-regression: roc_auc vs n_components",
        "ylabel": "roc_auc",
    },
]


def parse_args() -> argparse.Namespace:
    run_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=run_root / "curve" / "summaries" / "logistic-regression_dataset_summary.csv",
        help="Input curve summary CSV.",
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=run_root / "curve" / "plots" / "logistic-regression_accuracy_roc_auc.svg",
        help="Output SVG path.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=run_root / "curve" / "plots" / "logistic-regression_accuracy_roc_auc.png",
        help="Output PNG path. Use an empty string to skip PNG conversion.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["projector"]].append(row)
    for projector_rows in grouped.values():
        projector_rows.sort(key=lambda row: float(row["n_components"]))
    return dict(grouped)


def nice_ticks(min_value: float, max_value: float, count: int = 6) -> list[float]:
    if math.isclose(min_value, max_value):
        return [min_value]
    span = max_value - min_value
    raw_step = span / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(raw_step))
    step = min([1, 2, 5, 10], key=lambda m: abs(raw_step - m * power)) * power
    start = math.floor(min_value / step) * step
    ticks = []
    value = start
    while value <= max_value + step * 0.5:
        if value >= min_value - step * 0.5:
            ticks.append(value)
        value += step
    return ticks


def metric_extent(rows: list[dict[str, str]], metric: str) -> tuple[float, float]:
    y_col = f"{metric}_seed_mean_dataset_mean"
    err_col = f"{metric}_seed_mean_dataset_std"
    values = []
    for row in rows:
        y = float(row[y_col])
        err = float(row[err_col])
        if math.isfinite(y) and math.isfinite(err):
            values.extend([y - err, y + err])
    low = max(0.0, min(values))
    high = min(1.0, max(values))
    padding = max((high - low) * 0.08, 0.02)
    return max(0.0, low - padding), min(1.0, high + padding)


def fmt_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.parts = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            ),
            '<rect width="100%" height="100%" fill="white"/>',
            (
                '<style>'
                'text{font-family:Arial,Helvetica,sans-serif;fill:#222}'
                '.title{font-size:18px;font-weight:600}'
                '.axis{font-size:14px}'
                '.tick{font-size:12px;fill:#444}'
                '.legend{font-size:14px}'
                '</style>'
            ),
        ]

    def add(self, text: str) -> None:
        self.parts.append(text)

    def text(
        self,
        x: float,
        y: float,
        value: str,
        cls: str = "",
        anchor: str = "middle",
        rotate: float | None = None,
    ) -> None:
        attrs = f'x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}"'
        if cls:
            attrs += f' class="{cls}"'
        if rotate is not None:
            attrs += f' transform="rotate({rotate:.2f} {x:.2f} {y:.2f})"'
        self.add(f"<text {attrs}>{escape(value)}</text>")

    def finish(self) -> str:
        return "\n".join(self.parts + ["</svg>", ""])


def draw_panel(
    svg: Svg,
    rows_by_projector: dict[str, list[dict[str, str]]],
    metric: dict[str, str],
    panel: tuple[int, int, int, int],
    x_extent: tuple[float, float],
) -> None:
    left, top, width, height = panel
    plot_left = left + 78
    plot_top = top + 48
    plot_width = width - 108
    plot_height = height - 112
    plot_bottom = plot_top + plot_height
    plot_right = plot_left + plot_width

    key = metric["key"]
    y_col = f"{key}_seed_mean_dataset_mean"
    err_col = f"{key}_seed_mean_dataset_std"
    all_rows = [row for rows in rows_by_projector.values() for row in rows]
    y_min, y_max = metric_extent(all_rows, key)
    x_min, x_max = x_extent

    def sx(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * plot_height

    svg.text(left + width / 2, top + 24, metric["title"], cls="title")

    # Grid and axes
    for tick in nice_ticks(y_min, y_max):
        y = sy(tick)
        svg.add(
            f'<line x1="{plot_left:.2f}" y1="{y:.2f}" x2="{plot_right:.2f}" '
            'y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        svg.text(plot_left - 10, y + 4, fmt_tick(tick), cls="tick", anchor="end")

    x_ticks = [1, 200, 400, 600, 800, 1000]
    for tick in x_ticks:
        if x_min <= tick <= x_max:
            x = sx(tick)
            svg.add(
                f'<line x1="{x:.2f}" y1="{plot_bottom:.2f}" '
                f'x2="{x:.2f}" y2="{plot_bottom + 5:.2f}" '
                'stroke="#222" stroke-width="1"/>'
            )
            svg.text(x, plot_bottom + 22, str(tick), cls="tick")

    svg.add(
        f'<rect x="{plot_left:.2f}" y="{plot_top:.2f}" width="{plot_width:.2f}" '
        f'height="{plot_height:.2f}" fill="none" stroke="#222" stroke-width="1"/>'
    )
    svg.text(left + width / 2, top + height - 18, "n_components", cls="axis")
    svg.text(left + 18, top + height / 2, metric["ylabel"], cls="axis", rotate=-90)

    for projector in PROJECTOR_ORDER:
        rows = rows_by_projector.get(projector, [])
        if not rows:
            continue
        colour = PROJECTOR_COLOURS[projector]
        points = []
        for row in rows:
            x_value = float(row["n_components"])
            y_value = float(row[y_col])
            err_value = float(row[err_col])
            if not math.isfinite(y_value):
                continue
            x = sx(x_value)
            y = sy(y_value)
            points.append((x, y))
            if math.isfinite(err_value):
                y_low = sy(max(y_min, y_value - err_value))
                y_high = sy(min(y_max, y_value + err_value))
                svg.add(
                    f'<line x1="{x:.2f}" y1="{y_low:.2f}" '
                    f'x2="{x:.2f}" y2="{y_high:.2f}" '
                    f'stroke="{colour}" stroke-width="1.2" opacity="0.55"/>'
                )
                svg.add(
                    f'<line x1="{x - 3:.2f}" y1="{y_low:.2f}" '
                    f'x2="{x + 3:.2f}" y2="{y_low:.2f}" '
                    f'stroke="{colour}" stroke-width="1.2" opacity="0.55"/>'
                )
                svg.add(
                    f'<line x1="{x - 3:.2f}" y1="{y_high:.2f}" '
                    f'x2="{x + 3:.2f}" y2="{y_high:.2f}" '
                    f'stroke="{colour}" stroke-width="1.2" opacity="0.55"/>'
                )
            svg.add(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" '
                f'fill="{colour}" stroke="white" stroke-width="0.7"/>'
            )

        path_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        svg.add(
            f'<polyline points="{path_points}" fill="none" stroke="{colour}" '
            'stroke-width="2.0" stroke-linejoin="round" stroke-linecap="round"/>'
        )


def draw_legend(svg: Svg, x: int, y: int) -> None:
    spacing = 110
    total_width = spacing * len(PROJECTOR_ORDER)
    start_x = x - total_width / 2 + 20
    for idx, projector in enumerate(PROJECTOR_ORDER):
        item_x = start_x + idx * spacing
        colour = PROJECTOR_COLOURS[projector]
        label = PROJECTOR_LABELS[projector]
        svg.add(
            f'<line x1="{item_x:.2f}" y1="{y:.2f}" '
            f'x2="{item_x + 28:.2f}" y2="{y:.2f}" '
            f'stroke="{colour}" stroke-width="2"/>'
        )
        svg.add(
            f'<circle cx="{item_x + 14:.2f}" cy="{y:.2f}" r="3.2" '
            f'fill="{colour}" stroke="white" stroke-width="0.7"/>'
        )
        svg.text(item_x + 42, y + 5, label, cls="legend", anchor="start")


def render_svg(rows: list[dict[str, str]]) -> str:
    width = 1400
    height = 620
    svg = Svg(width, height)
    rows_by_projector = group_rows(rows)
    x_values = [float(row["n_components"]) for row in rows]
    x_extent = (min(x_values), max(x_values))

    draw_panel(svg, rows_by_projector, METRICS[0], (30, 34, 660, 500), x_extent)
    draw_panel(svg, rows_by_projector, METRICS[1], (700, 34, 660, 500), x_extent)
    draw_legend(svg, width // 2, 580)
    return svg.finish()


def maybe_write_png(svg_path: Path, png_path: Path | None) -> None:
    if png_path is None:
        return
    converter = shutil.which("rsvg-convert")
    if converter is None:
        print("Skipping PNG: rsvg-convert is not available.")
        return
    png_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [converter, "-w", "2800", "-h", "1240", "-o", str(png_path), str(svg_path)],
        check=True,
    )
    print(f"Wrote {png_path}")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    svg_text = render_svg(rows)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(svg_text)
    print(f"Wrote {args.output_svg}")

    output_png = None if str(args.output_png) == "" else args.output_png
    maybe_write_png(args.output_svg, output_png)


if __name__ == "__main__":
    main()
