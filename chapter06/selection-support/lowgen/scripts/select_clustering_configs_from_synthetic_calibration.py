#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_SYNTHETIC_RESULTS = Path(
    "/raid/experiments/lowgen_v2/output/"
    "lowgen-v2-cluster-guided-full-embedding/"
    "cluster_guided_full_embedding_gmm4_full_run/"
    "cluster_guided_full_embedding_results_with_sd_stability.csv"
)
DEFAULT_CLUSTERING_STABILITY = Path(
    "/sharedscratch/dm236/histology_clustering/output/"
    "clustering-curve-targets/clustering_stability.csv"
)
DEFAULT_CLUSTERING_RESULTS = Path(
    "/sharedscratch/dm236/histology_clustering/output/"
    "clustering-curve-targets/clustering_results.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/raid/experiments/lowgen_v2/output/calibrated_clustering_selection_gmm4"
)

DEFAULT_GENERATOR = "cluster_guided_full_diag_gmm4_probe"
DEFAULT_EXCLUDED_CLUSTERERS = ["singleton", "leiden"]
DEFAULT_GROUP_COLUMNS = ["dataset", "encoder"]
QUALITY_DIRECTIONS = {"fid": -1.0, "kid": -1.0, "precision": 1.0, "recall": 1.0}
GROUP_KEY_COLUMNS = [
    "dataset",
    "encoder",
    "projector",
    "dimension",
    "target_name",
    "source_metric",
    "condition",
    "clusterer",
    "cluster_params_json",
]
ROW_KEY_COLUMNS = ["task_id", "config_id", "clusterer", "cluster_seed"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    synthetic_column: str
    clustering_column: str
    requires_labels: bool = False


METRIC_SPECS = [
    MetricSpec("mean_pairwise_ari", "mean_pairwise_ari", "mean_pairwise_ari"),
    MetricSpec("sd_index", "sd_index", "sd_index"),
    MetricSpec("silhouette", "silhouette", "mean_silhouette"),
    MetricSpec("noise_fraction", "noise_fraction", "mean_noise_fraction"),
    MetricSpec("n_clusters", "n_clusters", "mean_n_clusters"),
    MetricSpec(
        "cluster_persistence",
        "mean_cluster_persistence",
        "mean_cluster_persistence",
    ),
    MetricSpec("label_ari", "label_ari", "mean_label_ari", True),
    MetricSpec("label_ami", "label_ami", "mean_label_ami", True),
    MetricSpec("label_nmi", "label_nmi", "mean_label_nmi", True),
    MetricSpec(
        "label_homogeneity",
        "label_homogeneity",
        "mean_label_homogeneity",
        True,
    ),
    MetricSpec(
        "label_completeness",
        "label_completeness",
        "mean_label_completeness",
        True,
    ),
    MetricSpec("label_v_measure", "label_v_measure", "mean_label_v_measure", True),
    MetricSpec("label_purity", "label_purity", "mean_label_purity", True),
    MetricSpec("label_coverage", "label_coverage", "mean_label_coverage", True),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use synthetic-quality calibration to select clustering configs from "
            "the full clustering experiment."
        )
    )
    parser.add_argument("--synthetic-results", type=Path, default=DEFAULT_SYNTHETIC_RESULTS)
    parser.add_argument(
        "--clustering-stability",
        type=Path,
        default=DEFAULT_CLUSTERING_STABILITY,
    )
    parser.add_argument(
        "--clustering-results",
        type=Path,
        default=DEFAULT_CLUSTERING_RESULTS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generator", default=DEFAULT_GENERATOR)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--encoder", action="append", default=None)
    parser.add_argument("--projector", action="append", default=None)
    parser.add_argument("--group-columns", nargs="+", default=DEFAULT_GROUP_COLUMNS)
    parser.add_argument("--quality-metrics", nargs="+", default=list(QUALITY_DIRECTIONS))
    parser.add_argument("--quality-weights", nargs="+", type=float, default=None)
    parser.add_argument("--max-noise", type=float, default=0.10)
    parser.add_argument("--min-effective-support", type=float, default=750.0)
    parser.add_argument("--min-clusters", type=float, default=16.0)
    parser.add_argument("--valid-cluster-count-rate", type=float, default=1.0)
    parser.add_argument("--min-precision", type=float, default=1e-12)
    parser.add_argument(
        "--exclude-clusterers",
        nargs="*",
        default=DEFAULT_EXCLUDED_CLUSTERERS,
    )
    parser.add_argument("--min-correlation-n", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    quality_weights = _quality_weights(args.quality_metrics, args.quality_weights)
    outputs = run_selection(
        synthetic_results_path=args.synthetic_results,
        clustering_stability_path=args.clustering_stability,
        clustering_results_path=args.clustering_results,
        output_dir=args.output_dir,
        generator=args.generator,
        dataset_filter=args.dataset,
        encoder_filter=args.encoder,
        projector_filter=args.projector,
        group_columns=list(args.group_columns),
        quality_metrics=list(args.quality_metrics),
        quality_weights=quality_weights,
        max_noise=float(args.max_noise),
        min_effective_support=float(args.min_effective_support),
        min_clusters=float(args.min_clusters),
        valid_cluster_count_rate=float(args.valid_cluster_count_rate),
        min_precision=float(args.min_precision),
        excluded_clusterers=list(args.exclude_clusterers),
        min_correlation_n=int(args.min_correlation_n),
        top_n=int(args.top_n),
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    best = pd.read_csv(outputs["best_by_dataset_encoder"])
    print()
    print(_format_best(best))


def run_selection(
    *,
    synthetic_results_path: Path,
    clustering_stability_path: Path,
    clustering_results_path: Path,
    output_dir: Path,
    generator: str,
    dataset_filter: Optional[list[str]],
    encoder_filter: Optional[list[str]],
    projector_filter: Optional[list[str]],
    group_columns: list[str],
    quality_metrics: list[str],
    quality_weights: dict[str, float],
    max_noise: float,
    min_effective_support: float,
    min_clusters: float,
    valid_cluster_count_rate: float,
    min_precision: float,
    excluded_clusterers: list[str],
    min_correlation_n: int,
    top_n: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = pd.read_csv(synthetic_results_path)
    clustering_stability = pd.read_csv(clustering_stability_path)
    clustering_results = pd.read_csv(clustering_results_path)

    synthetic = attach_n_samples(synthetic, clustering_results)
    clustering_stability = attach_n_samples(clustering_stability, clustering_results)

    synthetic = apply_optional_filters(
        synthetic,
        dataset_filter=dataset_filter,
        encoder_filter=encoder_filter,
        projector_filter=projector_filter,
    )
    clustering_stability = apply_optional_filters(
        clustering_stability,
        dataset_filter=dataset_filter,
        encoder_filter=encoder_filter,
        projector_filter=projector_filter,
    )

    calibration_input, calibration_excluded = filter_synthetic_calibration_rows(
        synthetic,
        generator=generator,
        max_noise=max_noise,
        min_effective_support=min_effective_support,
        min_clusters=min_clusters,
        min_precision=min_precision,
        excluded_clusterers=excluded_clusterers,
        quality_metrics=quality_metrics,
    )
    calibration = build_metric_calibration(
        calibration_input,
        metric_specs=METRIC_SPECS,
        quality_metrics=quality_metrics,
        quality_weights=quality_weights,
        min_n=min_correlation_n,
    )
    ranked, full_excluded = rank_clustering_experiment_configs(
        clustering_stability,
        calibration,
        group_columns=group_columns,
        max_noise=max_noise,
        min_effective_support=min_effective_support,
        min_clusters=min_clusters,
        valid_cluster_count_rate=valid_cluster_count_rate,
        excluded_clusterers=excluded_clusterers,
    )

    best_by_dataset_encoder = best_by_group(
        ranked,
        ["scenario", "dataset", "encoder"],
    )
    best_by_dataset_encoder_projector = best_by_group(
        ranked,
        ["scenario", "dataset", "encoder", "projector"],
    )
    top_by_dataset_encoder = top_by_group(
        ranked,
        ["scenario", "dataset", "encoder"],
        top_n=top_n,
    )
    excluded = pd.concat(
        [
            calibration_excluded.assign(source="synthetic_calibration"),
            full_excluded.assign(source="clustering_experiment"),
        ],
        ignore_index=True,
        sort=False,
    )

    paths = {
        "synthetic_metric_calibration": output_dir / "synthetic_metric_calibration.csv",
        "clustering_experiment_ranked_configs": output_dir
        / "clustering_experiment_ranked_configs.csv",
        "best_by_dataset_encoder": output_dir / "best_configs_by_dataset_encoder.csv",
        "best_by_dataset_encoder_projector": output_dir
        / "best_configs_by_dataset_encoder_projector.csv",
        "top_by_dataset_encoder": output_dir / "top_configs_by_dataset_encoder.csv",
        "excluded_rows": output_dir / "excluded_rows.csv",
        "metadata": output_dir / "metadata.json",
    }
    calibration.to_csv(paths["synthetic_metric_calibration"], index=False)
    ranked.to_csv(paths["clustering_experiment_ranked_configs"], index=False)
    best_by_dataset_encoder.to_csv(paths["best_by_dataset_encoder"], index=False)
    best_by_dataset_encoder_projector.to_csv(
        paths["best_by_dataset_encoder_projector"],
        index=False,
    )
    top_by_dataset_encoder.to_csv(paths["top_by_dataset_encoder"], index=False)
    excluded.to_csv(paths["excluded_rows"], index=False)
    metadata = {
        "synthetic_results": str(synthetic_results_path),
        "clustering_stability": str(clustering_stability_path),
        "clustering_results": str(clustering_results_path),
        "generator": generator,
        "dataset_filter": dataset_filter,
        "encoder_filter": encoder_filter,
        "projector_filter": projector_filter,
        "group_columns": group_columns,
        "quality_metrics": quality_metrics,
        "quality_weights": quality_weights,
        "max_noise": max_noise,
        "min_effective_support": min_effective_support,
        "min_clusters": min_clusters,
        "valid_cluster_count_rate": valid_cluster_count_rate,
        "min_precision": min_precision,
        "excluded_clusterers": excluded_clusterers,
        "min_correlation_n": min_correlation_n,
        "synthetic_input_rows": int(len(synthetic)),
        "synthetic_calibration_rows": int(len(calibration_input)),
        "clustering_input_rows": int(len(clustering_stability)),
        "clustering_ranked_rows": int(len(ranked)),
        "clustering_eligible_rows": int(ranked["eligible"].sum()),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return paths


def filter_synthetic_calibration_rows(
    frame: pd.DataFrame,
    *,
    generator: str,
    max_noise: float,
    min_effective_support: float,
    min_clusters: float,
    min_precision: float,
    excluded_clusterers: list[str],
    quality_metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    out["effective_support"] = effective_support(
        out,
        n_samples_column="n_samples",
        noise_column="noise_fraction",
        clusters_column="n_clusters",
    )
    reasons = invalid_reasons(
        out,
        checks=[
            (
                "wrong_generator",
                out.get("generator", pd.Series("", index=out.index))
                .astype(str)
                .str.strip()
                .ne(str(generator).strip()),
            ),
            ("failed_status", failed_status_mask(out)),
            ("high_noise", numeric(out, "noise_fraction") > max_noise),
            ("low_effective_support", numeric(out, "effective_support") < min_effective_support),
            ("too_few_clusters", numeric(out, "n_clusters") < min_clusters),
            ("zero_precision", numeric(out, "precision") < min_precision),
            ("excluded_clusterer", excluded_clusterer_mask(out, excluded_clusterers)),
        ]
        + [
            (f"missing_{metric}", ~np.isfinite(numeric(out, metric)))
            for metric in quality_metrics
        ],
    )
    valid = reasons.eq("")
    kept = out.loc[valid].reset_index(drop=True)
    excluded = out.loc[~valid].copy()
    excluded["exclude_reason"] = reasons.loc[~valid].to_numpy()
    return kept, excluded.reset_index(drop=True)


def build_metric_calibration(
    synthetic: pd.DataFrame,
    *,
    metric_specs: Iterable[MetricSpec],
    quality_metrics: list[str],
    quality_weights: dict[str, float],
    min_n: int,
) -> pd.DataFrame:
    rows = []
    for scenario in ["no_label", "label_aware"]:
        for spec in metric_specs:
            if scenario == "no_label" and spec.requires_labels:
                continue
            if spec.synthetic_column not in synthetic.columns:
                continue
            desired_effects = []
            quality_rows = []
            for quality_metric in quality_metrics:
                if quality_metric not in synthetic.columns:
                    continue
                metric_values = numeric(synthetic, spec.synthetic_column)
                quality_values = numeric(synthetic, quality_metric)
                mask = np.isfinite(metric_values) & np.isfinite(quality_values)
                n = int(mask.sum())
                if n < min_n:
                    rho = float("nan")
                    desired = float("nan")
                else:
                    rho = spearman_correlation(metric_values[mask], quality_values[mask])
                    desired = float(QUALITY_DIRECTIONS[quality_metric] * rho)
                    if np.isfinite(desired):
                        desired_effects.append(
                            desired * float(quality_weights[quality_metric])
                        )
                quality_rows.append(
                    {
                        "quality_metric": quality_metric,
                        "n": n,
                        "spearman": rho,
                        "desired_association": desired,
                    }
                )
            weight_sum = sum(
                float(quality_weights[row["quality_metric"]])
                for row in quality_rows
                if np.isfinite(row["desired_association"])
            )
            effect = float(np.sum(desired_effects) / weight_sum) if weight_sum else float("nan")
            rows.append(
                {
                    "scenario": scenario,
                    "metric": spec.name,
                    "synthetic_column": spec.synthetic_column,
                    "clustering_column": spec.clustering_column,
                    "requires_labels": spec.requires_labels,
                    "effect": effect,
                    "weight": abs(effect) if np.isfinite(effect) else 0.0,
                    "higher_is_better": bool(effect >= 0) if np.isfinite(effect) else True,
                    "quality_details_json": json.dumps(quality_rows, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def rank_clustering_experiment_configs(
    stability: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    group_columns: list[str],
    max_noise: float,
    min_effective_support: float,
    min_clusters: float,
    valid_cluster_count_rate: float,
    excluded_clusterers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = stability.copy()
    base["effective_support"] = effective_support(
        base,
        n_samples_column="n_samples",
        noise_column="mean_noise_fraction",
        clusters_column="mean_n_clusters",
    )
    reasons = invalid_reasons(
        base,
        checks=[
            ("high_noise", numeric(base, "mean_noise_fraction") > max_noise),
            ("low_effective_support", numeric(base, "effective_support") < min_effective_support),
            ("too_few_clusters", numeric(base, "mean_n_clusters") < min_clusters),
            (
                "invalid_cluster_count_rate",
                numeric(base, "valid_cluster_count_rate") < valid_cluster_count_rate,
            ),
            ("excluded_clusterer", excluded_clusterer_mask(base, excluded_clusterers)),
        ],
    )
    base["eligible_base"] = reasons.eq("")
    excluded = base.loc[~base["eligible_base"]].copy()
    excluded["exclude_reason"] = reasons.loc[~base["eligible_base"]].to_numpy()

    scenario_frames = []
    for scenario, scenario_calibration in calibration.groupby("scenario", dropna=False):
        usable = scenario_calibration[
            (scenario_calibration["weight"] > 0)
            & scenario_calibration["clustering_column"].isin(base.columns)
        ].copy()
        frame = base.copy()
        frame["scenario"] = scenario
        frame["eligible"] = frame["eligible_base"] & (len(usable) > 0)
        frame["calibrated_synthetic_quality_score"] = np.nan
        frame["rank"] = np.nan
        if len(usable) > 0:
            score = np.zeros(len(frame), dtype=float)
            weight_sum = float(usable["weight"].sum())
            for row in usable.to_dict("records"):
                column = str(row["clustering_column"])
                ranks = percentile_rank(
                    frame,
                    group_columns,
                    column,
                    higher_is_better=bool(row["higher_is_better"]),
                )
                score += float(row["weight"]) * ranks.fillna(0.0).to_numpy()
            frame["calibrated_synthetic_quality_score"] = score / weight_sum
            frame.loc[~frame["eligible"], "calibrated_synthetic_quality_score"] = np.nan
            frame["rank"] = (
                frame[frame["eligible"]]
                .groupby(group_columns, dropna=False)[
                    "calibrated_synthetic_quality_score"
                ]
                .rank(method="first", ascending=False)
            )
        scenario_frames.append(frame.drop(columns=["eligible_base"]))
    ranked = pd.concat(scenario_frames, ignore_index=True, sort=False)
    ranked = ranked.sort_values(
        ["scenario", *group_columns, "eligible", "calibrated_synthetic_quality_score"],
        ascending=[True, *([True] * len(group_columns)), False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    return ranked, excluded.reset_index(drop=True)


def attach_n_samples(frame: pd.DataFrame, clustering_results: pd.DataFrame) -> pd.DataFrame:
    if "n_samples" in frame.columns and numeric(frame, "n_samples").notna().any():
        return frame
    out = frame.copy()
    if all(column in out.columns for column in ROW_KEY_COLUMNS) and all(
        column in clustering_results.columns for column in ROW_KEY_COLUMNS
    ):
        row_counts = clustering_results[ROW_KEY_COLUMNS + ["n_samples"]].drop_duplicates()
        out = out.merge(row_counts, on=ROW_KEY_COLUMNS, how="left")
        if numeric(out, "n_samples").notna().all():
            return out
        out = out.drop(columns=["n_samples"])
    key_columns = [column for column in GROUP_KEY_COLUMNS if column in out.columns]
    result_keys = [column for column in key_columns if column in clustering_results.columns]
    counts = clustering_results[result_keys + ["n_samples"]].copy()
    counts = normalise_key_columns(counts, result_keys)
    counts = (
        counts.groupby(result_keys, dropna=False, as_index=False)
        .agg(n_samples=("n_samples", "median"))
    )
    out = normalise_key_columns(out, result_keys)
    return out.merge(counts, on=result_keys, how="left")


def apply_optional_filters(
    frame: pd.DataFrame,
    *,
    dataset_filter: Optional[list[str]],
    encoder_filter: Optional[list[str]],
    projector_filter: Optional[list[str]],
) -> pd.DataFrame:
    out = frame.copy()
    for column, values in [
        ("dataset", dataset_filter),
        ("encoder", encoder_filter),
        ("projector", projector_filter),
    ]:
        if values:
            out = out[out[column].astype(str).isin(set(map(str, values)))]
    return out.reset_index(drop=True)


def effective_support(
    frame: pd.DataFrame,
    *,
    n_samples_column: str,
    noise_column: str,
    clusters_column: str,
) -> pd.Series:
    support = (
        numeric(frame, n_samples_column)
        * (1.0 - numeric(frame, noise_column).fillna(0.0))
        / numeric(frame, clusters_column)
    )
    return support.replace([np.inf, -np.inf], np.nan)


def best_by_group(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        frame[frame["eligible"]]
        .sort_values(
            columns + ["calibrated_synthetic_quality_score"],
            ascending=[True] * len(columns) + [False],
            kind="mergesort",
        )
        .groupby(columns, dropna=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def top_by_group(frame: pd.DataFrame, columns: list[str], *, top_n: int) -> pd.DataFrame:
    return (
        frame[frame["eligible"]]
        .sort_values(
            columns + ["calibrated_synthetic_quality_score"],
            ascending=[True] * len(columns) + [False],
            kind="mergesort",
        )
        .groupby(columns, dropna=False, as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def percentile_rank(
    frame: pd.DataFrame,
    group_columns: list[str],
    column: str,
    *,
    higher_is_better: bool,
) -> pd.Series:
    values = numeric(frame, column)
    ranks = values.groupby([frame[group] for group in group_columns], dropna=False).rank(
        pct=True,
        ascending=higher_is_better,
        method="average",
    )
    return ranks


def spearman_correlation(left: pd.Series, right: pd.Series) -> float:
    left_rank = left.rank(method="average")
    right_rank = right.rank(method="average")
    value = left_rank.corr(right_rank, method="pearson")
    return float(value) if np.isfinite(value) else float("nan")


def invalid_reasons(
    frame: pd.DataFrame,
    *,
    checks: list[tuple[str, pd.Series]],
) -> pd.Series:
    reasons = pd.Series("", index=frame.index, dtype=object)
    for name, mask in checks:
        mask = mask.fillna(True)
        reasons.loc[mask] = reasons.loc[mask].map(
            lambda value, reason=name: reason if value == "" else f"{value};{reason}"
        )
    return reasons


def failed_status_mask(frame: pd.DataFrame) -> pd.Series:
    if "status" not in frame.columns:
        return pd.Series(False, index=frame.index)
    status = frame["status"].astype(str).str.strip().str.lower()
    return status.isin({"failed", "failure", "error", "skipped"})


def excluded_clusterer_mask(frame: pd.DataFrame, excluded: list[str]) -> pd.Series:
    if "clusterer" not in frame.columns:
        return pd.Series(False, index=frame.index)
    clusterers = frame["clusterer"].astype(str).str.strip().str.lower()
    mask = pd.Series(False, index=frame.index)
    for value in excluded:
        term = str(value).strip().lower()
        if term:
            mask |= clusterers.eq(term) | clusterers.str.contains(term, regex=False)
    return mask


def normalise_key_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column == "dimension":
            out[column] = out[column].map(normalise_dimension_value)
        else:
            out[column] = out[column].fillna("").astype(str)
    return out


def normalise_dimension_value(value: object) -> str:
    if pd.isna(value) or str(value).strip().lower() in {"", "nan", "none"}:
        return ""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return str(numeric_value)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _quality_weights(
    quality_metrics: list[str],
    values: Optional[list[float]],
) -> dict[str, float]:
    if values is None:
        values = [1.0] * len(quality_metrics)
    if len(values) != len(quality_metrics):
        raise ValueError("--quality-weights must match --quality-metrics length.")
    total = float(sum(values))
    if total <= 0:
        raise ValueError("--quality-weights must sum to a positive value.")
    out = {}
    for metric, value in zip(quality_metrics, values):
        if metric not in QUALITY_DIRECTIONS:
            raise ValueError(f"Unsupported quality metric: {metric}")
        out[metric] = float(value) / total
    return out


def _format_best(best: pd.DataFrame) -> str:
    columns = [
        "scenario",
        "dataset",
        "encoder",
        "projector",
        "dimension",
        "target_name",
        "clusterer",
        "cluster_params_json",
        "mean_n_clusters",
        "effective_support",
        "mean_noise_fraction",
        "mean_pairwise_ari",
        "calibrated_synthetic_quality_score",
    ]
    columns = [column for column in columns if column in best.columns]
    return best[columns].round(4).to_string(index=False)


if __name__ == "__main__":
    main()
