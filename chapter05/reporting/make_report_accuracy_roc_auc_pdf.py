#!/usr/bin/env python3
"""Create separate report metric dimension plots as PDFs.

The original report plots are one metric per PDF, each with a full-dimension
panel and a lower-dimension detail panel. This reproduces that layout for
accuracy, ROC AUC, and trustworthiness with consistent styling.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


PROJECTOR_ORDER = ["umap", "pca", "grp"]
PROJECTOR_LABELS = {"umap": "UMAP", "pca": "PCA", "grp": "GRP"}
PROJECTOR_COLOURS = {"umap": "#1f77b4", "pca": "#ff7f0e", "grp": "#2ca02c"}
PLOT_ASPECT = 1.2

@dataclass(frozen=True)
class MetricSpec:
    name: str
    y_label: str
    title: str
    filename_stem: str
    min_dimension: float = 1.0
    clamp_y_unit_interval: bool = True


METRICS = [
    MetricSpec("accuracy", "Acc", "Acc vs Dimensions", "accuracy_dimension_curves"),
    MetricSpec("roc_auc", "ROC AUC", "ROC AUC vs Dimensions", "roc_auc_dimension_curves"),
    MetricSpec(
        "trustworthiness",
        "Trustworthiness",
        "Trustworthiness vs Dimensions",
        "trustworthiness_dimension_curves",
    ),
    MetricSpec(
        "silhouette-score",
        "Silhouette Score",
        "Silhouette Score vs Dimensions",
        "silhouette_score_dimension_curves",
        min_dimension=2.0,
        clamp_y_unit_interval=False,
    ),
]


def parse_args() -> argparse.Namespace:
    run_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=run_root / "curve" / "summaries" / "logistic-regression_dataset_summary.csv",
        help="Input classifier dataset-summary CSV.",
    )
    parser.add_argument(
        "--detail-dims",
        type=int,
        default=50,
        help="Maximum dimension shown in the lower-dimensions panels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=run_root / "report" / "plots",
        help="Output directory for metric-specific PDF/SVG files.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def group_rows(
    rows: list[dict[str, str]],
    detail_dims: int | None = None,
    min_dimension: float = 1.0,
):
    grouped = defaultdict(list)
    for row in rows:
        n_components = float(row["n_components"])
        if n_components < min_dimension:
            continue
        if detail_dims is not None and n_components > detail_dims:
            continue
        grouped[row["projector"]].append(row)
    for projector_rows in grouped.values():
        projector_rows.sort(key=lambda row: float(row["n_components"]))
    return grouped


def metric_extent(
    rows: list[dict[str, str]],
    metric: str,
    clamp_unit_interval: bool = True,
) -> tuple[float, float]:
    mean_col = f"{metric}_seed_mean_dataset_mean"
    std_col = f"{metric}_seed_mean_dataset_std"
    values = []
    for row in rows:
        mean = float(row[mean_col])
        std = float(row[std_col])
        if math.isfinite(mean) and math.isfinite(std):
            values.extend([mean - std, mean + std])
    low = min(values)
    high = max(values)
    padding = max((high - low) * 0.06, 0.015)
    low -= padding
    high += padding
    if clamp_unit_interval:
        low = max(0.0, low)
        high = min(1.0, high)
    return low, high


def nice_ticks(min_value: float, max_value: float, count: int = 5) -> list[float]:
    if math.isclose(min_value, max_value):
        return [min_value]
    span = max_value - min_value
    raw_step = span / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(raw_step))
    step = min([1, 2, 5, 10], key=lambda m: abs(raw_step - m * power)) * power
    start = math.ceil(min_value / step) * step
    ticks = []
    value = start
    while value <= max_value + step * 0.5:
        ticks.append(value)
        value += step
    return ticks


def axis_ticks(min_value: float, max_value: float, middle_ticks: list[float]) -> list[float]:
    ticks = [min_value, *middle_ticks, max_value]
    out = []
    for tick in sorted(ticks):
        if min_value <= tick <= max_value and (not out or abs(tick - out[-1]) > 0.5):
            out.append(tick)
    return out


def fmt_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}" if abs(value * 10 - round(value * 10)) < 1e-8 else f"{value:.2f}"


def fmt_dimension_tick(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-8):
        return f"{round(value):.0f}"
    return f"{value:g}"


class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.parts = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}">'
            ),
            '<rect width="100%" height="100%" fill="white"/>',
            (
                "<style>"
                "text{font-family:Arial,Helvetica,sans-serif;fill:#222}"
                ".suptitle{font-size:62px;font-weight:700}"
                ".title{font-size:48px;font-weight:500}"
                ".axis{font-size:46px}"
                ".tick{font-size:36px;fill:#444}"
                ".legend{font-size:42px}"
                ".legend-title{font-size:42px;font-weight:600}"
                "</style>"
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
    all_rows: list[dict[str, str]],
    metric_spec: MetricSpec,
    title: str,
    panel: tuple[int, int, int, int],
    x_extent: tuple[float, float],
    detail_dims: int | None,
) -> None:
    left, top, width, height = panel
    plot_left = left + 132
    plot_top = top + 82
    plot_width = width - 166
    plot_height = plot_width / PLOT_ASPECT
    plot_right = plot_left + plot_width
    plot_bottom = plot_top + plot_height

    rows_by_projector = group_rows(
        all_rows,
        detail_dims=detail_dims,
        min_dimension=metric_spec.min_dimension,
    )
    visible_rows = [row for rows in rows_by_projector.values() for row in rows]
    y_min, y_max = metric_extent(
        visible_rows,
        metric_spec.name,
        clamp_unit_interval=metric_spec.clamp_y_unit_interval,
    )
    x_min, x_max = x_extent

    mean_col = f"{metric_spec.name}_seed_mean_dataset_mean"
    std_col = f"{metric_spec.name}_seed_mean_dataset_std"

    def sx(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * plot_height

    svg.text(left + width / 2, top + 40, title, cls="title")

    for tick in nice_ticks(y_min, y_max):
        y = sy(tick)
        svg.add(
            f'<line x1="{plot_left:.2f}" y1="{y:.2f}" x2="{plot_right:.2f}" '
            f'y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        svg.text(plot_left - 12, y + 6, fmt_tick(tick), cls="tick", anchor="end")

    if x_max <= 50:
        x_ticks = axis_ticks(x_min, x_max, [10, 20, 30, 40])
    else:
        x_ticks = axis_ticks(x_min, x_max, [200, 400, 600, 800])
    for tick in x_ticks:
        if x_min <= tick <= x_max:
            x = sx(tick)
            svg.add(
                f'<line x1="{x:.2f}" y1="{plot_bottom:.2f}" '
                f'x2="{x:.2f}" y2="{plot_bottom + 5:.2f}" '
                'stroke="#222" stroke-width="1"/>'
            )
            svg.text(x, plot_bottom + 42, fmt_dimension_tick(tick), cls="tick")

    svg.add(
        f'<rect x="{plot_left:.2f}" y="{plot_top:.2f}" width="{plot_width:.2f}" '
        f'height="{plot_height:.2f}" fill="none" stroke="#222" stroke-width="1"/>'
    )
    svg.text(left + width / 2, plot_bottom + 102, "Dimensions", cls="axis")
    svg.text(
        left + 40,
        plot_top + plot_height / 2,
        metric_spec.y_label,
        cls="axis",
        rotate=-90,
    )

    for projector in PROJECTOR_ORDER:
        rows = rows_by_projector.get(projector, [])
        if not rows:
            continue
        colour = PROJECTOR_COLOURS[projector]
        upper = []
        lower = []
        line = []
        for row in rows:
            x_value = float(row["n_components"])
            mean = float(row[mean_col])
            std = float(row[std_col])
            x = sx(x_value)
            y = sy(mean)
            line.append((x, y))
            upper.append((x, sy(min(y_max, mean + std))))
            lower.append((x, sy(max(y_min, mean - std))))

        fill_points = " ".join(
            f"{x:.2f},{y:.2f}" for x, y in upper + list(reversed(lower))
        )
        svg.add(
            f'<polygon points="{fill_points}" fill="{colour}" opacity="0.2" '
            'stroke="none"/>'
        )
        line_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in line)
        svg.add(
            f'<polyline points="{line_points}" fill="none" stroke="{colour}" '
            'stroke-width="4.2" stroke-linejoin="round" stroke-linecap="round"/>'
        )


def draw_legend(svg: Svg, x: int, y: int) -> None:
    box_width = 820
    box_height = 132
    box_x = x - box_width / 2
    box_y = y - box_height / 2
    svg.add(
        f'<rect x="{box_x:.2f}" y="{box_y:.2f}" width="{box_width:.2f}" '
        f'height="{box_height:.2f}" rx="0" ry="0" fill="white" '
        f'stroke="#222" stroke-width="1.2"/>'
    )
    title_y = box_y + 42
    item_y = box_y + 92
    svg.text(x, title_y, "Projector", cls="legend-title")
    item_width = 210
    total_width = item_width * len(PROJECTOR_ORDER)
    start_x = x - total_width / 2
    for idx, projector in enumerate(PROJECTOR_ORDER):
        item_x = start_x + idx * item_width
        colour = PROJECTOR_COLOURS[projector]
        label = PROJECTOR_LABELS[projector]
        swatch_x = item_x + 12
        label_x = swatch_x + 76
        svg.add(
            f'<line x1="{swatch_x:.2f}" y1="{item_y:.2f}" '
            f'x2="{swatch_x + 56:.2f}" y2="{item_y:.2f}" '
            f'stroke="{colour}" stroke-width="4.2"/>'
        )
        svg.add(
            f'<rect x="{swatch_x:.2f}" y="{item_y - 13:.2f}" width="56" height="26" '
            f'fill="{colour}" opacity="0.2" stroke="none"/>'
        )
        svg.text(label_x, item_y + 11, label, cls="legend", anchor="start")


def render_metric_svg(
    rows: list[dict[str, str]],
    detail_dims: int,
    metric_spec: MetricSpec,
) -> str:
    width = 2200
    height = 1320
    svg = Svg(width, height)
    panel_w = 980
    panel_h = 1010
    left_x = 70
    right_x = 1150
    top_y = 150

    all_x = [float(row["n_components"]) for row in rows]
    full_extent = (metric_spec.min_dimension, max(all_x))
    detail_extent = (metric_spec.min_dimension, float(detail_dims))

    svg.text(width / 2, top_y - 54, metric_spec.title, cls="suptitle")
    draw_panel(
        svg,
        rows,
        metric_spec,
        "All Dimensions",
        (left_x, top_y, panel_w, panel_h),
        full_extent,
        detail_dims=None,
    )
    draw_panel(
        svg,
        rows,
        metric_spec,
        "Lower Dimensions",
        (right_x, top_y, panel_w, panel_h),
        detail_extent,
        detail_dims=detail_dims,
    )

    draw_legend(svg, width // 2, 1190)
    return svg.finish()


def convert_svg_to_pdf(svg_path: Path, pdf_path: Path) -> None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise SystemExit("rsvg-convert is required to create the PDF output.")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [converter, "-f", "pdf", "-o", str(pdf_path), str(svg_path)],
        check=True,
    )


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric_spec in METRICS:
        svg_path = args.output_dir / f"logistic-regression_{metric_spec.filename_stem}.svg"
        pdf_path = args.output_dir / f"logistic-regression_{metric_spec.filename_stem}.pdf"
        svg = render_metric_svg(rows, args.detail_dims, metric_spec)
        svg_path.write_text(svg)
        convert_svg_to_pdf(svg_path, pdf_path)
        print(f"Wrote {svg_path}")
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
