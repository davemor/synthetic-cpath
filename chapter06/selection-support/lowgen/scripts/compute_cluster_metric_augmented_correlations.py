from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from lowgen.dgx.cluster_conditioning import (
    _match_task,
    _task_index,
    load_cluster_split_labels,
    metric_correlations,
)
from lowgen.dgx.config import ExperimentConfig
from lowgen.dgx.data import load_dataset
from lowgen.dgx.projections import build_representation


GROUP_COLS = [
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

CLUSTER_METRICS = [
    "silhouette",
    "sd_index",
    "mean_pairwise_ari",
    "label_ari",
    "label_ami",
    "label_nmi",
    "label_homogeneity",
    "label_completeness",
    "label_v_measure",
    "label_purity",
    "noise_fraction",
    "n_clusters",
]

QUALITY_METRICS = ["fid", "kid", "precision", "recall"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--metric-subsample", type=int, default=10_000)
    parser.add_argument("--sd-alpha", type=float, default=1.0)
    parser.add_argument(
        "--row-metrics",
        type=Path,
        default=None,
        help="Optional precomputed row_number, sd_index, and mean_pairwise_ari CSV.",
    )
    parser.add_argument(
        "--generator",
        default="cluster_guided_full_diag_gmm4_probe",
        help="Generator to print as a compact matrix.",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    run_dir = args.run_dir
    selected_path = run_dir / "selected_cluster_configs.csv"
    results_path = run_dir / "cluster_guided_full_embedding_results.csv"
    if not selected_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Missing completed run outputs under {run_dir}")

    selected = pd.read_csv(selected_path)
    results = pd.read_csv(results_path)
    if args.row_metrics is not None:
        row_metric_frame = pd.read_csv(args.row_metrics)
        row_metrics = {
            int(row["row_number"]): {
                "sd_index": row.get("sd_index", np.nan),
                "sd_scat": row.get("sd_scat", np.nan),
                "sd_dis": row.get("sd_dis", np.nan),
                "mean_pairwise_ari": row.get("mean_pairwise_ari", np.nan),
            }
            for row in row_metric_frame.to_dict("records")
        }
    else:
        task_index = _task_index(cfg, args.manifest)
        row_metrics = _compute_row_metrics(
            cfg,
            selected,
            task_index,
            metric_subsample=int(args.metric_subsample),
            sd_alpha=float(args.sd_alpha),
        )
    augmented_selected = selected.copy()
    for column in ["sd_index", "sd_scat", "sd_dis", "mean_pairwise_ari"]:
        augmented_selected[column] = [
            row_metrics[int(row_number)].get(column, np.nan)
            for row_number in range(1, len(augmented_selected) + 1)
        ]

    augmented_results = results.copy()
    for column in ["sd_index", "sd_scat", "sd_dis", "mean_pairwise_ari"]:
        augmented_results[column] = [
            row_metrics[int(row_number)].get(column, np.nan)
            for row_number in augmented_results["row_number"].to_numpy()
        ]

    correlations = metric_correlations(augmented_results)
    augmented_results_path = run_dir / "cluster_guided_full_embedding_results_with_sd_stability.csv"
    augmented_selected_path = run_dir / "selected_cluster_configs_with_sd_stability.csv"
    correlations_path = run_dir / "cluster_metric_quality_correlations_with_sd_stability.csv"
    matrix_path = run_dir / f"{args.generator}_spearman_matrix_with_sd_stability.csv"

    augmented_results.to_csv(augmented_results_path, index=False)
    augmented_selected.to_csv(augmented_selected_path, index=False)
    correlations.to_csv(correlations_path, index=False)
    matrix = _matrix(correlations, args.generator, method="spearman")
    matrix.to_csv(matrix_path)

    summary = {
        "rows": int(len(selected)),
        "sd_index_non_missing": int(augmented_selected["sd_index"].notna().sum()),
        "mean_pairwise_ari_non_missing": int(
            augmented_selected["mean_pairwise_ari"].notna().sum()
        ),
        "augmented_results": str(augmented_results_path),
        "augmented_selected": str(augmented_selected_path),
        "augmented_correlations": str(correlations_path),
        "printed_matrix": str(matrix_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print()
    print(matrix.round(3).to_string())


def _compute_row_metrics(
    cfg: ExperimentConfig,
    selected: pd.DataFrame,
    task_index: dict[tuple[str, str, str, str, int], dict[str, Any]],
    *,
    metric_subsample: int,
    sd_alpha: float,
) -> dict[int, dict[str, float]]:
    data_cache: dict[tuple[str, str], Any] = {}
    rep_cache: dict[str, Any] = {}
    labels_cache: dict[tuple[int, str], np.ndarray] = {}
    row_metrics: dict[int, dict[str, float]] = {}

    for row_number, row in enumerate(selected.to_dict("records"), start=1):
        if row_number == 1 or row_number % 25 == 0 or row_number == len(selected):
            print(f"sd-index rows: {row_number}/{len(selected)}", flush=True)
        task = _match_task(pd.Series(row), task_index)
        data_key = (str(task["dataset"]), str(task["encoder"]))
        if data_key not in data_cache:
            data_cache[data_key] = load_dataset(cfg, data_key[0], data_key[1])
        data = data_cache[data_key]

        rep_key = str(task["task_id"])
        if rep_key not in rep_cache:
            dimension = None if task["dimension"] == "" else int(float(task["dimension"]))
            rep_cache[rep_key] = build_representation(
                cfg,
                str(task["projector"]),
                dimension,
                int(task["seed"]),
                data.train.features,
                data.test.features,
                data.valid.features if data.valid is not None else None,
                str(task["projection_cache_key"]),
                force=False,
            )
        rep = rep_cache[rep_key]

        labels_path = Path(str(row["labels_path"]))
        train_labels, _ = load_cluster_split_labels(
            labels_path,
            data.train.sample_ids,
            data.test.sample_ids,
        )
        labels_cache[(row_number, str(labels_path))] = train_labels
        metric_indices = deterministic_sample_indices(
            len(train_labels),
            min(metric_subsample, len(train_labels)),
            0,
        )
        sd = sd_index(rep.train[metric_indices], train_labels[metric_indices], alpha=sd_alpha)
        row_metrics[row_number] = {
            "sd_index": sd["sd_index"],
            "sd_scat": sd["sd_scat"],
            "sd_dis": sd["sd_dis"],
        }

    group_cols = [column for column in GROUP_COLS if column in selected.columns]
    grouped = selected.assign(_row_number=np.arange(1, len(selected) + 1)).groupby(
        group_cols,
        dropna=False,
    )
    for _, group in grouped:
        labels = []
        for row in group.to_dict("records"):
            row_number = int(row["_row_number"])
            labels_path = str(row["labels_path"])
            if (row_number, labels_path) in labels_cache:
                labels.append(labels_cache[(row_number, labels_path)])
        ari_values = [
            float(adjusted_rand_score(left, right))
            for left, right in combinations(labels, 2)
            if len(left) == len(right)
        ]
        mean_ari = float(np.mean(ari_values)) if ari_values else float("nan")
        for row_number in group["_row_number"].to_numpy():
            row_metrics[int(row_number)]["mean_pairwise_ari"] = mean_ari
    print(f"stability groups: {len(grouped)}", flush=True)

    return row_metrics


def deterministic_sample_indices(n_samples: int, sample_size: int, seed: int) -> np.ndarray:
    sample_size = min(int(sample_size), int(n_samples))
    rng = np.random.default_rng(int(seed))
    if sample_size >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    return np.sort(rng.choice(n_samples, size=sample_size, replace=False)).astype(np.int64)


def sd_index(features: np.ndarray, labels: np.ndarray, *, alpha: float = 1.0) -> dict[str, float]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    mask = labels != -1
    features = features[mask]
    labels = labels[mask]
    unique_labels, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if features.shape[0] < 3 or unique_labels.shape[0] < 2:
        return _nan_sd()

    if len(unique_labels) < 2:
        return _nan_sd()
    sums = np.zeros((len(unique_labels), features.shape[1]), dtype=np.float64)
    sums_sq = np.zeros_like(sums)
    np.add.at(sums, inverse, features)
    np.add.at(sums_sq, inverse, features * features)
    count_matrix = counts.astype(np.float64)[:, None]
    centroids_array = sums / count_matrix
    variances = np.maximum(sums_sq / count_matrix - centroids_array * centroids_array, 0.0)
    cluster_std_norms = np.linalg.norm(np.sqrt(variances), axis=1)

    data_std_norm = float(np.linalg.norm(np.std(features, axis=0)))
    if not np.isfinite(data_std_norm) or data_std_norm <= 0:
        return _nan_sd()

    scat = float(np.mean(cluster_std_norms) / data_std_norm)
    distances = _pairwise_distances(centroids_array)
    positive = distances[distances > 0]
    if positive.size == 0:
        return _nan_sd(scat=scat)
    d_min = float(np.min(positive))
    d_max = float(np.max(positive))
    row_sums = distances.sum(axis=1)
    if d_min <= 0 or not np.all(row_sums > 0):
        return _nan_sd(scat=scat)
    dis = float((d_max / d_min) * np.sum(1.0 / row_sums))
    return {"sd_index": float(alpha * scat + dis), "sd_scat": scat, "sd_dis": dis}


def _pairwise_distances(features: np.ndarray) -> np.ndarray:
    diffs = features[:, None, :] - features[None, :, :]
    return np.linalg.norm(diffs, axis=2)


def _nan_sd(*, scat: float = float("nan")) -> dict[str, float]:
    return {"sd_index": float("nan"), "sd_scat": scat, "sd_dis": float("nan")}


def _matrix(correlations: pd.DataFrame, generator: str, *, method: str) -> pd.DataFrame:
    rows = correlations[correlations["generator"] == generator]
    matrix = pd.DataFrame(index=CLUSTER_METRICS, columns=QUALITY_METRICS, dtype=float)
    for _, row in rows.iterrows():
        cluster_metric = str(row["cluster_metric"])
        quality_metric = str(row["quality_metric"])
        if cluster_metric in matrix.index and quality_metric in matrix.columns:
            matrix.loc[cluster_metric, quality_metric] = pd.to_numeric(
                row[method],
                errors="coerce",
            )
    return matrix


if __name__ == "__main__":
    main()
