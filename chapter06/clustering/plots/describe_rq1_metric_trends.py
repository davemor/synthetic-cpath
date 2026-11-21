from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_exp.dimension_plots import (
    select_best_raw_configs,
    summarise_raw_baselines,
)
from cluster_exp.table1 import METHOD_LABELS, METHOD_ORDER


METRICS = [
    ("ami", "AMI", "higher", "External label alignment"),
    ("ari", "ARI", "higher", "External label alignment"),
    ("v_measure", "V-measure", "higher", "External label alignment"),
    ("silhouette", "Silhouette", "higher", "Internal validation"),
    ("stability", "Stability", "higher", "Stability"),
    ("sd_index", "SD-index", "lower", "Internal validation"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/Users/dm236/Documents/Research/AllResults/Chapter6-clustering/res"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    trends_path = args.run_dir / "clustering-results-rq1-dimension-trends-plotted-values.csv"
    stability_path = args.run_dir / "clustering_stability.csv"
    trends = pd.read_csv(trends_path)
    stability = pd.read_csv(stability_path)
    raw_baselines = summarise_raw_baselines(select_best_raw_configs(stability))

    rows = []
    for projector in ["pca", "umap"]:
        projector_trends = trends[trends["projector"] == projector]
        for clusterer in METHOD_ORDER:
            method_trends = projector_trends[projector_trends["clusterer"] == clusterer]
            if method_trends.empty:
                continue
            method_trends = method_trends.sort_values("dimension")
            baseline_row = raw_baselines[raw_baselines["clusterer"] == clusterer]
            for metric_key, metric_label, direction, category in METRICS:
                value_col = f"{metric_key}_mean"
                if value_col not in method_trends.columns:
                    continue
                values = pd.to_numeric(method_trends[value_col], errors="coerce")
                dims = pd.to_numeric(method_trends["dimension"], errors="coerce").astype(int)
                valid = values.notna()
                if not valid.any():
                    continue
                dims = dims[valid].to_numpy()
                values = values[valid].to_numpy(dtype=float)
                baseline = np.nan
                if not baseline_row.empty and value_col in baseline_row.columns:
                    baseline = float(baseline_row.iloc[0][value_col])
                dim1_value = _value_at(dims, values, 1)
                dim50_value = _value_at(dims, values, 50)
                rows.append(
                    {
                        "projector": projector,
                        "clusterer": clusterer,
                        "method": METHOD_LABELS.get(clusterer, clusterer),
                        "metric": metric_label,
                        "category": category,
                        "direction": direction,
                        "baseline": baseline,
                        "dim1": dim1_value,
                        "dim50": dim50_value,
                        "delta_50_minus_1": dim50_value - dim1_value,
                        "trend": _trend(dim1_value, dim50_value, direction),
                        "quality_change": _quality_change(dim1_value, dim50_value, direction),
                        "baseline_relation": _baseline_relation(dims, values, baseline),
                        "first_cross_dimension": _first_cross_dimension(dims, values, baseline),
                    }
                )

    output = pd.DataFrame(rows)
    if args.output is not None:
        output.to_csv(args.output, index=False)
    print(output.to_csv(index=False, float_format="%.6f"))


def _value_at(dims: np.ndarray, values: np.ndarray, dimension: int) -> float:
    matches = np.where(dims == dimension)[0]
    if len(matches) == 0:
        return float("nan")
    return float(values[matches[0]])


def _trend(dim1_value: float, dim50_value: float, direction: str) -> str:
    if not np.isfinite(dim1_value) or not np.isfinite(dim50_value):
        return "not available"
    delta = dim50_value - dim1_value
    if np.isclose(delta, 0.0):
        return "flat"
    if delta > 0:
        return "increased"
    return "decreased"


def _quality_change(dim1_value: float, dim50_value: float, direction: str) -> str:
    if not np.isfinite(dim1_value) or not np.isfinite(dim50_value):
        return "not available"
    delta = dim50_value - dim1_value
    if np.isclose(delta, 0.0):
        return "unchanged"
    if direction == "lower":
        return "improves" if delta < 0 else "deteriorates"
    return "improves" if delta > 0 else "deteriorates"


def _baseline_relation(dims: np.ndarray, values: np.ndarray, baseline: float) -> str:
    if not np.isfinite(baseline):
        return "no baseline"
    tolerance = 1e-12
    above = values > baseline + tolerance
    below = values < baseline - tolerance
    if above.all():
        return "always above baseline"
    if below.all():
        return "always below baseline"
    if (~above & ~below).all():
        return "equal to baseline"
    start = values[0] - baseline
    end = values[-1] - baseline
    if start < -tolerance and end > tolerance:
        return "crossed from below to above baseline"
    if start > tolerance and end < -tolerance:
        return "crossed from above to below baseline"
    return "mixed relative to baseline"


def _first_cross_dimension(dims: np.ndarray, values: np.ndarray, baseline: float) -> str:
    if not np.isfinite(baseline):
        return ""
    tolerance = 1e-12
    diffs = values - baseline
    start_sign = _sign(diffs[0], tolerance)
    if start_sign == 0:
        return str(int(dims[0]))
    for dim, diff in zip(dims[1:], diffs[1:]):
        sign = _sign(diff, tolerance)
        if sign == 0 or sign != start_sign:
            return str(int(dim))
    return ""


def _sign(value: float, tolerance: float) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


if __name__ == "__main__":
    main()
