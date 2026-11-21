#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path(
    "/raid/experiments/lowgen_v2/output/"
    "lowgen-v2-cluster-guided-full-embedding/"
    "cluster_guided_full_embedding_gmm4_full_run"
)

DEFAULT_GENERATOR = "cluster_guided_full_diag_gmm4_probe"
GROUP_COLUMNS = ["dataset", "encoder"]
QUALITY_COLUMNS = ["fid", "kid", "precision", "recall"]
LOWER_IS_BETTER = {"fid", "kid"}
DEFAULT_EXCLUDED_CLUSTERERS = ["singleton", "leiden"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select stable cluster-conditioning settings using completed synthetic "
            "quality metrics."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--selected", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--generator", default=DEFAULT_GENERATOR)
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=GROUP_COLUMNS,
        help="Columns within which stability thresholds and ranks are computed.",
    )
    parser.add_argument(
        "--stable-quantile",
        type=float,
        default=0.50,
        help=(
            "Keep rows at or above this within-group stability quantile before "
            "quality ranking. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--min-stability",
        type=float,
        default=None,
        help="Optional hard minimum for the stability metric.",
    )
    parser.add_argument(
        "--min-effective-support",
        type=float,
        default=750.0,
        help="Optional hard minimum effective samples per non-noise cluster.",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=16,
        help="Optional hard minimum number of non-noise clusters.",
    )
    parser.add_argument(
        "--max-noise",
        type=float,
        default=0.10,
        help="Optional hard maximum noise fraction.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=1e-12,
        help=(
            "Optional hard minimum synthetic precision. The default excludes "
            "zero-precision solutions."
        ),
    )
    parser.add_argument(
        "--exclude-clusterers",
        nargs="*",
        default=DEFAULT_EXCLUDED_CLUSTERERS,
        help=(
            "Clusterer names or substrings to exclude before ranking. Defaults "
            "exclude singleton-style and Leiden overfragmented solutions."
        ),
    )
    parser.add_argument("--stability-weight", type=float, default=0.35)
    parser.add_argument("--fid-weight", type=float, default=0.25)
    parser.add_argument("--kid-weight", type=float, default=0.25)
    parser.add_argument("--precision-weight", type=float, default=0.25)
    parser.add_argument("--recall-weight", type=float, default=0.25)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    outputs = run_selection(args)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))

    best = pd.read_csv(outputs["best_by_group"])
    print()
    print(_format_best(best))


def run_selection(args: argparse.Namespace) -> dict[str, Path]:
    run_dir = args.run_dir
    results_path = args.results or _existing(
        run_dir / "cluster_guided_full_embedding_results_with_sd_stability.csv",
        run_dir / "cluster_guided_full_embedding_results.csv",
    )
    selected_path = args.selected or _existing(
        run_dir / "selected_cluster_configs_with_sd_stability.csv",
        run_dir / "selected_cluster_configs.csv",
    )
    output_dir = args.output_dir or run_dir / "stable_synthetic_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(results_path)
    selected = pd.read_csv(selected_path)
    ranked = rank_candidates(
        results,
        selected,
        generator=args.generator,
        group_columns=list(args.group_columns),
        stable_quantile=float(args.stable_quantile),
        min_stability=args.min_stability,
        min_effective_support=float(args.min_effective_support),
        min_clusters=int(args.min_clusters),
        max_noise=args.max_noise,
        min_precision=float(args.min_precision),
        excluded_clusterers=list(args.exclude_clusterers),
        stability_weight=float(args.stability_weight),
        quality_weights={
            "fid": float(args.fid_weight),
            "kid": float(args.kid_weight),
            "precision": float(args.precision_weight),
            "recall": float(args.recall_weight),
        },
    )

    best = (
        ranked[ranked["eligible"]]
        .sort_values(list(args.group_columns) + ["final_score"], ascending=[True] * len(args.group_columns) + [False])
        .groupby(list(args.group_columns), dropna=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    top = (
        ranked[ranked["eligible"]]
        .sort_values(list(args.group_columns) + ["final_score"], ascending=[True] * len(args.group_columns) + [False])
        .groupby(list(args.group_columns), dropna=False, as_index=False)
        .head(int(args.top_n))
        .reset_index(drop=True)
    )

    ranked_path = output_dir / "stable_synthetic_ranked_candidates.csv"
    best_path = output_dir / "stable_synthetic_best_by_group.csv"
    top_path = output_dir / "stable_synthetic_top_by_group.csv"
    metadata_path = output_dir / "stable_synthetic_selection_metadata.json"
    ranked.to_csv(ranked_path, index=False)
    best.to_csv(best_path, index=False)
    top.to_csv(top_path, index=False)

    metadata = {
        "results": str(results_path),
        "selected": str(selected_path),
        "generator": args.generator,
        "group_columns": list(args.group_columns),
        "stable_quantile": float(args.stable_quantile),
        "min_stability": args.min_stability,
        "min_effective_support": float(args.min_effective_support),
        "min_clusters": int(args.min_clusters),
        "max_noise": args.max_noise,
        "min_precision": float(args.min_precision),
        "excluded_clusterers": list(args.exclude_clusterers),
        "stability_weight": float(args.stability_weight),
        "quality_weights": {
            "fid": float(args.fid_weight),
            "kid": float(args.kid_weight),
            "precision": float(args.precision_weight),
            "recall": float(args.recall_weight),
        },
        "n_input_rows": int(len(results)),
        "n_ranked_rows": int(len(ranked)),
        "n_eligible_rows": int(ranked["eligible"].sum()),
        "n_best_rows": int(len(best)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {
        "ranked_candidates": ranked_path,
        "best_by_group": best_path,
        "top_by_group": top_path,
        "metadata": metadata_path,
    }


def rank_candidates(
    results: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    generator: str,
    group_columns: list[str],
    stable_quantile: float,
    min_stability: float | None,
    min_effective_support: float,
    min_clusters: int,
    max_noise: float | None,
    min_precision: float,
    excluded_clusterers: list[str],
    stability_weight: float,
    quality_weights: dict[str, float],
) -> pd.DataFrame:
    frame = results.copy()
    if "generator" in frame.columns:
        frame = frame[
            frame["generator"].astype(str).str.strip() == str(generator).strip()
        ].copy()
    if "status" in frame.columns:
        status = frame["status"].astype(str).str.strip().str.lower()
        if status.eq("success").any():
            frame = frame[status.eq("success")].copy()
        else:
            frame = frame[~status.isin({"failed", "failure", "error", "skipped"})].copy()
    if frame.empty:
        raise ValueError(
            "No rows left after generator/status filtering. "
            f"generator={generator!r}; available="
            f"{sorted(results.get('generator', pd.Series(dtype=str)).astype(str).str.strip().unique())}"
        )

    frame = _attach_selected_columns(frame, selected)
    for column in QUALITY_COLUMNS + ["mean_pairwise_ari", "mean_cluster_persistence", "n_clusters", "noise_fraction", "n_samples"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    stability_column = _stability_column(frame)
    frame["stability_metric_name"] = stability_column
    frame["stability_metric"] = pd.to_numeric(frame[stability_column], errors="coerce")
    frame["effective_support"] = _effective_support(frame)

    required = QUALITY_COLUMNS + ["stability_metric"]
    valid = np.ones(len(frame), dtype=bool)
    for column in required:
        valid &= np.isfinite(pd.to_numeric(frame[column], errors="coerce").to_numpy())
    if min_stability is not None:
        valid &= frame["stability_metric"].to_numpy() >= float(min_stability)
    if min_effective_support > 0:
        valid &= frame["effective_support"].fillna(-np.inf).to_numpy() >= min_effective_support
    if min_clusters > 1:
        valid &= frame["n_clusters"].fillna(-np.inf).to_numpy() >= int(min_clusters)
    if max_noise is not None:
        valid &= frame["noise_fraction"].fillna(np.inf).to_numpy() <= float(max_noise)
    if min_precision > 0:
        valid &= frame["precision"].fillna(-np.inf).to_numpy() >= float(min_precision)
    if excluded_clusterers:
        valid &= ~_excluded_clusterer_mask(frame, excluded_clusterers).to_numpy()
    frame["valid_for_selection"] = valid

    frame["stability_threshold"] = np.nan
    if stable_quantile > 0:
        thresholds = (
            frame[frame["valid_for_selection"]]
            .groupby(group_columns, dropna=False)["stability_metric"]
            .quantile(float(stable_quantile))
            .rename("stability_threshold")
            .reset_index()
        )
        frame = frame.merge(thresholds, on=group_columns, how="left", suffixes=("", "_computed"))
        if "stability_threshold_computed" in frame.columns:
            frame["stability_threshold"] = frame["stability_threshold_computed"]
            frame = frame.drop(columns=["stability_threshold_computed"])
        stable_mask = frame["stability_metric"] >= frame["stability_threshold"]
    else:
        stable_mask = np.ones(len(frame), dtype=bool)
    frame["passes_stability_gate"] = frame["valid_for_selection"] & stable_mask.fillna(False)

    score_frame = frame[frame["valid_for_selection"]].copy()
    rank_columns = group_columns
    frame["stability_score"] = _rank_in_groups(score_frame, rank_columns, "stability_metric", higher_is_better=True).reindex(frame.index)
    for column in QUALITY_COLUMNS:
        frame[f"{column}_score"] = _rank_in_groups(
            score_frame,
            rank_columns,
            column,
            higher_is_better=column not in LOWER_IS_BETTER,
        ).reindex(frame.index)

    quality_weight_sum = sum(quality_weights.values())
    if quality_weight_sum <= 0:
        raise ValueError("At least one synthetic quality weight must be positive.")
    quality_score = np.zeros(len(frame), dtype=float)
    for column, weight in quality_weights.items():
        quality_score += (float(weight) / quality_weight_sum) * frame[f"{column}_score"].fillna(0).to_numpy()
    frame["synthetic_quality_score"] = quality_score

    stability_weight = float(stability_weight)
    if stability_weight < 0 or stability_weight > 1:
        raise ValueError("--stability-weight must be between 0 and 1.")
    frame["final_score"] = (
        stability_weight * frame["stability_score"].fillna(0)
        + (1.0 - stability_weight) * frame["synthetic_quality_score"].fillna(0)
    )
    frame["eligible"] = frame["passes_stability_gate"]
    frame.loc[~frame["eligible"], "final_score"] = np.nan

    sort_columns = group_columns + ["eligible", "final_score"]
    ascending = [True] * len(group_columns) + [False, False]
    return frame.sort_values(sort_columns, ascending=ascending, kind="mergesort").reset_index(drop=True)


def _attach_selected_columns(frame: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if "row_number" not in frame.columns:
        return frame
    selected = selected.copy()
    selected["row_number"] = np.arange(1, len(selected) + 1)
    keep = [
        "row_number",
        "n_samples",
        "mean_cluster_persistence",
        "label_purity_non_noise",
        "label_coverage",
        "valid_cluster_count",
    ]
    keep = [column for column in keep if column in selected.columns and column not in frame.columns]
    if not keep:
        return frame
    return frame.merge(selected[["row_number"] + keep], on="row_number", how="left")


def _effective_support(frame: pd.DataFrame) -> pd.Series:
    n_samples = pd.to_numeric(frame.get("n_samples"), errors="coerce")
    n_clusters = pd.to_numeric(frame.get("n_clusters"), errors="coerce")
    noise = pd.to_numeric(frame.get("noise_fraction"), errors="coerce").fillna(0)
    support = n_samples * (1.0 - noise) / n_clusters
    return support.replace([np.inf, -np.inf], np.nan)


def _stability_column(frame: pd.DataFrame) -> str:
    for column in ["mean_pairwise_ari", "mean_cluster_persistence"]:
        if column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().any():
            return column
    raise ValueError("No usable stability column found; expected mean_pairwise_ari or mean_cluster_persistence.")


def _excluded_clusterer_mask(frame: pd.DataFrame, excluded: list[str]) -> pd.Series:
    if "clusterer" not in frame.columns:
        return pd.Series(False, index=frame.index)
    clusterers = frame["clusterer"].astype(str).str.strip().str.lower()
    mask = pd.Series(False, index=frame.index)
    for value in excluded:
        term = str(value).strip().lower()
        if not term:
            continue
        mask |= clusterers.eq(term) | clusterers.str.contains(term, regex=False)
    return mask


def _rank_in_groups(
    frame: pd.DataFrame,
    group_columns: list[str],
    column: str,
    *,
    higher_is_better: bool,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    ranks = values.groupby([frame[col] for col in group_columns], dropna=False).rank(
        pct=True,
        ascending=not higher_is_better,
        method="average",
    )
    return ranks.fillna(0.0)


def _existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exist: " + ", ".join(str(path) for path in paths))


def _format_best(best: pd.DataFrame) -> str:
    columns = [
        "dataset",
        "encoder",
        "projector",
        "dimension",
        "target_name",
        "clusterer",
        "cluster_params_json",
        "n_clusters",
        "effective_support",
        "stability_metric",
        "fid",
        "kid",
        "precision",
        "recall",
        "final_score",
    ]
    columns = [column for column in columns if column in best.columns]
    return best[columns].round(4).to_string(index=False)


if __name__ == "__main__":
    main()
