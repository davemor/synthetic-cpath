from __future__ import annotations

import os
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.metrics.cluster import contingency_matrix

from .config import ExperimentConfig
from .embeddings import load_embedding_labels


_WORKER_EMBEDDINGS_ROOT: Path | None = None
_WORKER_TRUE_LABELS: dict[str, np.ndarray | None] = {}


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


EXTERNAL_METRIC_COLS = [
    "label_ari",
    "label_ami",
    "label_nmi",
    "label_homogeneity",
    "label_completeness",
    "label_v_measure",
    "label_fowlkes_mallows",
    "label_purity",
    "label_coverage",
    "label_ari_non_noise",
    "label_ami_non_noise",
    "label_nmi_non_noise",
    "label_purity_non_noise",
]


def summarise_run(
    cfg: ExperimentConfig,
    run_dir: Path | None = None,
    workers: int = 1,
) -> dict[str, Path]:
    run_dir = run_dir or cfg.paths.run_dir
    task_dir = run_dir / "tasks"
    rows = []
    for path in task_dir.glob("*/result.json"):
        with path.open() as handle:
            loaded = json.load(handle)
        items = loaded if isinstance(loaded, list) else [loaded]
        for item in items:
            item["cluster_params_json"] = json.dumps(item["cluster_params"], sort_keys=True)
            rows.append(item)
    if not rows:
        raise FileNotFoundError(f"No task result JSON files found under {task_dir}")

    rows = _add_external_label_metrics_to_rows(cfg, rows, workers=workers)
    results = pd.DataFrame(rows)
    results_path = run_dir / "clustering_results.csv"
    results.to_csv(results_path, index=False)

    stability = _compute_stability(results, workers=workers)
    stability_path = run_dir / "clustering_stability.csv"
    stability.to_csv(stability_path, index=False)

    best = _rank_best(stability)
    best_path = run_dir / "best_methods.csv"
    best.to_csv(best_path, index=False)
    kmeans_stability = _filter_clusterer(stability, "kmeans")
    kmeans_stability_path = run_dir / "kmeans_stability.csv"
    kmeans_stability.to_csv(kmeans_stability_path, index=False)
    best_kmeans = _rank_best_kmeans(kmeans_stability)
    best_kmeans_path = run_dir / "best_kmeans_methods.csv"
    best_kmeans.to_csv(best_kmeans_path, index=False)

    spherical_kmeans_stability = _filter_clusterer(stability, "spherical_kmeans")
    spherical_kmeans_stability_path = run_dir / "spherical_kmeans_stability.csv"
    spherical_kmeans_stability.to_csv(spherical_kmeans_stability_path, index=False)
    best_spherical_kmeans = _rank_best_kmeans(spherical_kmeans_stability)
    best_spherical_kmeans_path = run_dir / "best_spherical_kmeans_methods.csv"
    best_spherical_kmeans.to_csv(best_spherical_kmeans_path, index=False)

    hdbscan_stability = _filter_clusterer(stability, "hdbscan")
    hdbscan_stability_path = run_dir / "hdbscan_stability.csv"
    hdbscan_stability.to_csv(hdbscan_stability_path, index=False)
    best_hdbscan = _rank_best_hdbscan(hdbscan_stability)
    best_hdbscan_path = run_dir / "best_hdbscan_methods.csv"
    best_hdbscan.to_csv(best_hdbscan_path, index=False)
    return {
        "results": results_path,
        "stability": stability_path,
        "best": best_path,
        "kmeans_stability": kmeans_stability_path,
        "best_kmeans": best_kmeans_path,
        "spherical_kmeans_stability": spherical_kmeans_stability_path,
        "best_spherical_kmeans": best_spherical_kmeans_path,
        "hdbscan_stability": hdbscan_stability_path,
        "best_hdbscan": best_hdbscan_path,
    }


def _add_external_label_metrics_to_rows(
    cfg: ExperimentConfig,
    rows: list[dict[str, Any]],
    workers: int = 1,
) -> list[dict[str, Any]]:
    workers = _normalise_workers(workers)
    if workers == 1:
        true_labels_cache: dict[str, np.ndarray | None] = {}
        for index, item in enumerate(rows, start=1):
            _add_external_label_metrics(cfg, item, true_labels_cache)
            _print_progress("label metrics", index, len(rows))
        return rows

    print(f"Computing label metrics with {workers} workers for {len(rows)} rows.", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=_process_context(),
        initializer=_initialise_metric_worker,
        initargs=(str(cfg.paths.embeddings_root),),
    ) as executor:
        mapped = executor.map(_external_metric_worker, rows, chunksize=20)
        return _collect_with_progress(mapped, len(rows), "label metrics")


def _compute_stability(results: pd.DataFrame, workers: int = 1) -> pd.DataFrame:
    group_cols = [col for col in GROUP_COLS if col in results.columns]
    group_records = []
    for group_key, group in results.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = dict(zip(group_cols, group_key))
        group_records.append((base, group.to_dict("records")))

    workers = _normalise_workers(workers)
    if workers == 1:
        rows = []
        for index, group_record in enumerate(group_records, start=1):
            rows.append(_stability_group_worker(group_record))
            _print_progress("stability", index, len(group_records))
        return pd.DataFrame(rows)

    print(f"Computing stability with {workers} workers for {len(group_records)} groups.", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=_process_context()) as executor:
        mapped = executor.map(_stability_group_worker, group_records, chunksize=20)
        return pd.DataFrame(_collect_with_progress(mapped, len(group_records), "stability"))


def _initialise_metric_worker(embeddings_root: str) -> None:
    global _WORKER_EMBEDDINGS_ROOT, _WORKER_TRUE_LABELS
    _WORKER_EMBEDDINGS_ROOT = Path(embeddings_root)
    _WORKER_TRUE_LABELS = {}


def _external_metric_worker(item: dict[str, Any]) -> dict[str, Any]:
    item = dict(item)
    labels_path = Path(str(item["labels_path"]))
    if not labels_path.is_file():
        return item
    true_labels = _load_worker_true_labels(str(item["dataset"]), str(item["encoder"]))
    if true_labels is None:
        return item

    with np.load(labels_path, allow_pickle=True) as data:
        pred_labels = np.asarray(data["labels"])
    if len(true_labels) != len(pred_labels):
        return item

    item.update(_external_label_metrics(true_labels, pred_labels))
    return item


def _load_worker_true_labels(dataset: str, encoder: str) -> np.ndarray | None:
    if _WORKER_EMBEDDINGS_ROOT is None:
        raise RuntimeError("Metric worker has not been initialised with an embeddings root.")
    cache_key = f"{dataset}:{encoder}"
    if cache_key not in _WORKER_TRUE_LABELS:
        _WORKER_TRUE_LABELS[cache_key] = load_embedding_labels(
            _WORKER_EMBEDDINGS_ROOT,
            dataset,
            encoder,
        )
    return _WORKER_TRUE_LABELS[cache_key]


def _stability_group_worker(group_record: tuple[dict[str, Any], list[dict[str, Any]]]) -> dict[str, Any]:
    base, records = group_record
    base = dict(base)
    labels_by_seed = []
    for item in records:
        labels_path = Path(str(item["labels_path"]))
        if labels_path.exists():
            with np.load(labels_path, allow_pickle=True) as data:
                labels_by_seed.append((int(item["cluster_seed"]), np.asarray(data["labels"])))
    ari_values = [
        adjusted_rand_score(labels_a, labels_b)
        for (_, labels_a), (_, labels_b) in combinations(labels_by_seed, 2)
    ]
    base.update(
        {
            "n_seed_runs": len(labels_by_seed),
            "mean_pairwise_ari": float(np.mean(ari_values)) if ari_values else float("nan"),
            "mean_silhouette": _record_mean(records, "silhouette"),
            "mean_sd_index": _record_mean(records, "sd_index"),
            "mean_sd_scat": _record_mean(records, "sd_scat"),
            "mean_sd_dis": _record_mean(records, "sd_dis"),
            "mean_noise_fraction": _record_mean(records, "noise_fraction"),
            "mean_n_clusters": _record_mean(records, "n_clusters"),
            "mean_runtime_seconds": _record_mean(records, "runtime_seconds"),
            "valid_cluster_count_rate": _record_mean(records, "valid_cluster_count"),
        }
    )
    if any("mean_cluster_persistence" in record for record in records):
        base["mean_cluster_persistence"] = _record_mean(records, "mean_cluster_persistence")
    for col in EXTERNAL_METRIC_COLS:
        if any(col in record for record in records):
            base[f"mean_{col}"] = _record_mean(records, col)
    return base


def _record_mean(records: list[dict[str, Any]], key: str) -> float:
    values = [record.get(key) for record in records]
    numeric = [float(value) for value in values if value is not None and not pd.isna(value)]
    return float(np.mean(numeric)) if numeric else float("nan")


def _normalise_workers(workers: int) -> int:
    if workers < 1:
        return max(1, os.cpu_count() or 1)
    return workers


def _process_context() -> mp.context.BaseContext | None:
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return None


def _collect_with_progress(items: Any, total: int, label: str) -> list[Any]:
    collected = []
    for index, item in enumerate(items, start=1):
        collected.append(item)
        _print_progress(label, index, total)
    return collected


def _print_progress(label: str, index: int, total: int) -> None:
    if index == 1 or index == total or index % 1000 == 0:
        print(f"{label}: {index}/{total}", flush=True)


def _add_external_label_metrics(
    cfg: ExperimentConfig, item: dict[str, Any], true_labels_cache: dict[str, np.ndarray | None]
) -> None:
    labels_path = Path(item["labels_path"])
    if not labels_path.is_file():
        return
    true_labels = _load_true_labels_for_item(cfg, item, true_labels_cache)
    if true_labels is None:
        return

    with np.load(labels_path, allow_pickle=True) as data:
        pred_labels = np.asarray(data["labels"])
    if len(true_labels) != len(pred_labels):
        return

    item.update(_external_label_metrics(true_labels, pred_labels))


def _external_label_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, float]:
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    metrics = {
        "label_ari": float(adjusted_rand_score(true_labels, pred_labels)),
        "label_ami": float(adjusted_mutual_info_score(true_labels, pred_labels)),
        "label_nmi": float(normalized_mutual_info_score(true_labels, pred_labels)),
        "label_homogeneity": float(homogeneity_score(true_labels, pred_labels)),
        "label_completeness": float(completeness_score(true_labels, pred_labels)),
        "label_v_measure": float(v_measure_score(true_labels, pred_labels)),
        "label_fowlkes_mallows": float(fowlkes_mallows_score(true_labels, pred_labels)),
        "label_purity": _purity_score(true_labels, pred_labels),
    }
    non_noise = pred_labels != -1
    metrics["label_coverage"] = float(np.mean(non_noise))
    if np.count_nonzero(non_noise) > 1 and np.unique(pred_labels[non_noise]).shape[0] > 1:
        metrics["label_ari_non_noise"] = float(
            adjusted_rand_score(true_labels[non_noise], pred_labels[non_noise])
        )
        metrics["label_ami_non_noise"] = float(
            adjusted_mutual_info_score(true_labels[non_noise], pred_labels[non_noise])
        )
        metrics["label_nmi_non_noise"] = float(
            normalized_mutual_info_score(true_labels[non_noise], pred_labels[non_noise])
        )
        metrics["label_purity_non_noise"] = _purity_score(
            true_labels[non_noise], pred_labels[non_noise]
        )
    else:
        metrics["label_ari_non_noise"] = float("nan")
        metrics["label_ami_non_noise"] = float("nan")
        metrics["label_nmi_non_noise"] = float("nan")
        metrics["label_purity_non_noise"] = float("nan")
    return metrics


def _load_true_labels_for_item(
    cfg: ExperimentConfig, item: dict[str, Any], true_labels_cache: dict[str, np.ndarray | None]
) -> np.ndarray | None:
    projection_path = Path(str(item.get("projection_path", "")))
    if projection_path.is_file():
        cache_key = f"projection:{projection_path}"
        if cache_key not in true_labels_cache:
            with np.load(projection_path, allow_pickle=True) as data:
                true_labels_cache[cache_key] = (
                    np.asarray(data["labels"]) if "labels" in data.files else None
                )
        return true_labels_cache[cache_key]

    cache_key = f"raw:{item['dataset']}:{item['encoder']}"
    if cache_key not in true_labels_cache:
        true_labels_cache[cache_key] = load_embedding_labels(
            cfg.paths.embeddings_root, str(item["dataset"]), str(item["encoder"])
        )
    return true_labels_cache[cache_key]


def _purity_score(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    matrix = contingency_matrix(true_labels, pred_labels)
    if matrix.size == 0 or matrix.sum() == 0:
        return float("nan")
    return float(np.sum(np.max(matrix, axis=0)) / np.sum(matrix))


def _rank_best(stability: pd.DataFrame) -> pd.DataFrame:
    if stability.empty:
        return stability
    ranked = stability.copy()
    ranked["noise_sort"] = ranked["mean_noise_fraction"].fillna(1.0)
    ranked["silhouette_sort"] = ranked["mean_silhouette"].fillna(-np.inf)
    ranked["stability_sort"] = ranked["mean_pairwise_ari"].fillna(-np.inf)
    ranked = ranked.sort_values(
        [
            "dataset",
            "encoder",
            "stability_sort",
            "valid_cluster_count_rate",
            "noise_sort",
            "silhouette_sort",
            "mean_runtime_seconds",
        ],
        ascending=[True, True, False, False, True, False, True],
    )
    return ranked.groupby(["dataset", "encoder"], dropna=False).head(1).drop(
        columns=["noise_sort", "silhouette_sort", "stability_sort"]
    )


def _filter_clusterer(stability: pd.DataFrame, clusterer: str) -> pd.DataFrame:
    if stability.empty or "clusterer" not in stability.columns:
        return stability.iloc[0:0].copy()
    return stability[stability["clusterer"] == clusterer].copy()


def _rank_best_kmeans(stability: pd.DataFrame) -> pd.DataFrame:
    if stability.empty:
        return stability
    ranked = stability.copy()
    ranked["stability_sort"] = ranked["mean_pairwise_ari"].fillna(-np.inf)
    ranked["silhouette_sort"] = ranked["mean_silhouette"].fillna(-np.inf)
    ranked = ranked.sort_values(
        [
            "dataset",
            "encoder",
            "stability_sort",
            "silhouette_sort",
            "mean_runtime_seconds",
        ],
        ascending=[True, True, False, False, True],
    )
    return ranked.groupby(["dataset", "encoder"], dropna=False).head(1).drop(
        columns=["stability_sort", "silhouette_sort"]
    )


def _rank_best_hdbscan(stability: pd.DataFrame) -> pd.DataFrame:
    if stability.empty:
        return stability
    ranked = stability.copy()
    ranked["noise_sort"] = ranked["mean_noise_fraction"].fillna(1.0)
    ranked["silhouette_sort"] = ranked["mean_silhouette"].fillna(-np.inf)
    ranked["stability_sort"] = ranked["mean_pairwise_ari"].fillna(-np.inf)
    ranked["persistence_sort"] = ranked.get(
        "mean_cluster_persistence", pd.Series(np.nan, index=ranked.index)
    ).fillna(-np.inf)
    ranked = ranked.sort_values(
        [
            "dataset",
            "encoder",
            "valid_cluster_count_rate",
            "noise_sort",
            "persistence_sort",
            "stability_sort",
            "silhouette_sort",
            "mean_runtime_seconds",
        ],
        ascending=[True, True, False, True, False, False, False, True],
    )
    return ranked.groupby(["dataset", "encoder"], dropna=False).head(1).drop(
        columns=["noise_sort", "silhouette_sort", "stability_sort", "persistence_sort"]
    )
