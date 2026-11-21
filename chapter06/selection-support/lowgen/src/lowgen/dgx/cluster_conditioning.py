from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import load_dataset
from .generators import fit_generator
from .manifest import build_task_manifest, manifest_path
from .metrics import fidelity_metrics, lowdim_fidelity_metrics, utility_metrics
from .projections import build_representation, inverse_transform
from .utils import atomic_write_json, read_jsonl, stable_hash


FULL_DIAG_PROBE = "cluster_guided_full_diag_gaussian_probe"
FULL_DIAG_GMM4_PROBE = "cluster_guided_full_diag_gmm4_probe"
FULL_DIAG_GMM4_TRUE_LABEL_PROBE = "label_guided_full_diag_gmm4_probe"
FULL_DIAG_GMM4_LABEL_PROBE = "cluster_label_guided_full_diag_gmm4_probe"
LOWDIM_ISOTROPIC_PROBE = "cluster_guided_gaussian_probe"
FULL_EMBEDDING_PROBES = {
    FULL_DIAG_PROBE,
    FULL_DIAG_GMM4_PROBE,
    FULL_DIAG_GMM4_TRUE_LABEL_PROBE,
    FULL_DIAG_GMM4_LABEL_PROBE,
}


CLUSTER_METRIC_COLUMNS = [
    "silhouette",
    "sd_index",
    "sd_scat",
    "sd_dis",
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

QUALITY_METRIC_COLUMNS = [
    "lowdim_fid",
    "lowdim_kid",
    "lowdim_precision",
    "lowdim_recall",
    "lowdim_centroid_distance",
    "lowdim_covariance_frobenius",
    "fid",
    "kid",
    "precision",
    "recall",
    "real_clusters",
    "synthetic_clusters",
    "tstr_accuracy",
    "tstr_macro_roc_auc_ovr",
    "aug_accuracy",
    "aug_macro_roc_auc_ovr",
]


def run_cluster_conditioning_audit(
    cfg: ExperimentConfig,
    clustering_results_path: Path | None = None,
    output_dir: Path | None = None,
    manifest: Path | None = None,
    max_rows: int | None = None,
    max_cluster_configs: int | None = None,
    force: bool = False,
    generator_kinds: tuple[str, ...] | list[str] | None = None,
    include_noise_cluster: bool | None = None,
    dataset: str | None = None,
    encoder: str | None = None,
    projector: str | None = None,
    target_name: str | None = None,
    seed: int | None = None,
    synthetic_total: int | None = None,
    n_workers: int | None = None,
    dask_chunk_size: int | None = None,
) -> dict[str, Path]:
    """Evaluate generators conditioned on saved clustering labels.

    This intentionally reuses existing projection targets, cached representations,
    and cluster label artifacts. It writes one tidy row per
    clustering-configuration/generator pair plus a compact metric-correlation table.
    """

    clustering_results_path = (
        clustering_results_path
        or cfg.cluster_conditioning.clustering_results_path
    )
    if clustering_results_path is None:
        raise ValueError(
            "Provide a clustering_results_path or set cluster_conditioning.clustering_results_path."
        )
    clustering_results_path = Path(clustering_results_path)
    output_dir = output_dir or cfg.paths.run_dir / "cluster_conditioning_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "cluster_guided_full_embedding_results.csv"
    correlation_path = output_dir / "cluster_metric_quality_correlations.csv"
    selected_path = output_dir / "selected_cluster_configs.csv"
    skipped_path = output_dir / "skipped_rows.csv"
    metadata_path = output_dir / "metadata.json"
    if result_path.exists() and correlation_path.exists() and not force:
        return {
            "results": result_path,
            "correlations": correlation_path,
            "selected_cluster_configs": selected_path,
            "skipped": skipped_path,
            "metadata": metadata_path,
        }

    task_index = _task_index(cfg, manifest)
    cluster_rows = _load_cluster_rows(
        clustering_results_path,
        max_rows=max_rows if max_rows is not None else cfg.cluster_conditioning.max_rows,
        max_cluster_configs=max_cluster_configs,
        min_clusters=cfg.cluster_conditioning.min_clusters,
        max_noise_fraction=cfg.cluster_conditioning.max_noise_fraction,
        dataset=dataset,
        encoder=encoder,
        projector=projector,
        target_name=target_name,
        seed=seed,
    )
    kinds = list(generator_kinds or cfg.cluster_conditioning.generator_kinds)
    sample_cap = (
        int(synthetic_total)
        if synthetic_total is not None
        else cfg.cluster_conditioning.synthetic_total
    )
    include_noise = (
        cfg.cluster_conditioning.include_noise_cluster
        if include_noise_cluster is None
        else bool(include_noise_cluster)
    )

    row_dicts = [
        row.to_dict() | {"_row_number": row_number}
        for row_number, row in enumerate(cluster_rows, start=1)
    ]
    if n_workers is not None and int(n_workers) > 1 and len(row_dicts) > 1:
        rows, skipped, execution_backend, actual_chunk_size = _run_cluster_rows_dask(
            cfg,
            row_dicts,
            task_index,
            kinds,
            sample_cap,
            include_noise,
            int(n_workers),
            dask_chunk_size,
        )
    else:
        rows, skipped = _run_cluster_rows_chunk(
            cfg,
            row_dicts,
            task_index,
            kinds,
            sample_cap,
            include_noise,
        )
        execution_backend = "serial"
        actual_chunk_size = None

    rows = sorted(rows, key=lambda value: int(value.get("row_number", 0)))
    skipped = sorted(skipped, key=lambda value: int(value.get("row_number", 0)))
    results = pd.DataFrame(rows)
    correlations = metric_correlations(results)
    selected_df = pd.DataFrame([row.to_dict() for row in cluster_rows])
    pd.DataFrame(rows).to_csv(result_path, index=False)
    correlations.to_csv(correlation_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    pd.DataFrame(skipped, columns=_skipped_columns(skipped)).to_csv(
        skipped_path, index=False
    )
    atomic_write_json(
        metadata_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "clustering_results_path": str(clustering_results_path),
            "manifest_path": str(manifest or manifest_path(cfg)),
            "output_dir": str(output_dir),
            "generator_kinds": kinds,
            "include_noise_cluster": include_noise,
            "filters": {
                "dataset": dataset,
                "encoder": encoder,
                "projector": projector,
                "target_name": target_name,
                "seed": seed,
            },
            "synthetic_total": sample_cap,
            "max_cluster_configs": max_cluster_configs,
            "execution_backend": execution_backend,
            "n_workers": n_workers,
            "dask_chunk_size": actual_chunk_size,
            "correlations": "exploratory_small_exemplar",
            "n_cluster_rows": len(cluster_rows),
            "n_result_rows": len(rows),
            "n_skipped_rows": len(skipped),
        },
    )
    return {
        "results": result_path,
        "correlations": correlation_path,
        "selected_cluster_configs": selected_path,
        "skipped": skipped_path,
        "metadata": metadata_path,
    }


def _run_cluster_rows_dask(
    cfg: ExperimentConfig,
    cluster_rows: list[dict[str, Any]],
    task_index: dict[tuple[str, str, str, str, int], dict[str, Any]],
    kinds: list[str],
    sample_cap: int,
    include_noise: bool,
    n_workers: int,
    dask_chunk_size: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, int]:
    from dask.distributed import as_completed

    from .runner import _make_client

    chunk_size = _resolve_dask_chunk_size(
        len(cluster_rows),
        n_workers,
        dask_chunk_size,
    )
    chunks = list(_chunked(cluster_rows, chunk_size))
    client = _make_client(cfg, n_workers)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    try:
        client.wait_for_workers(min(n_workers, len(chunks)), timeout="120s")
        futures = [
            client.submit(
                _run_cluster_rows_chunk,
                cfg,
                chunk,
                task_index,
                kinds,
                sample_cap,
                include_noise,
                pure=False,
            )
            for chunk in chunks
        ]
        for future in as_completed(futures):
            chunk_rows, chunk_skipped = future.result()
            rows.extend(chunk_rows)
            skipped.extend(chunk_skipped)
    finally:
        client.close()
    return rows, skipped, "dask", chunk_size


def _run_cluster_rows_chunk(
    cfg: ExperimentConfig,
    cluster_rows: list[dict[str, Any]],
    task_index: dict[tuple[str, str, str, str, int], dict[str, Any]],
    kinds: list[str],
    sample_cap: int,
    include_noise: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    data_cache: dict[tuple[str, str], Any] = {}
    rep_cache: dict[str, Any] = {}
    generator_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    for row_offset, row_dict in enumerate(cluster_rows, start=1):
        cluster_row = pd.Series(row_dict)
        row_number = int(row_dict.get("_row_number", row_offset))
        started = perf_counter()
        try:
            task = _match_task(cluster_row, task_index)
            data_key = (str(task["dataset"]), str(task["encoder"]))
            if data_key not in data_cache:
                data_cache[data_key] = load_dataset(cfg, data_key[0], data_key[1])
            data = data_cache[data_key]
            rep = None
            if any(kind not in FULL_EMBEDDING_PROBES for kind in kinds):
                rep_key = str(task["task_id"])
                if rep_key not in rep_cache:
                    dimension = (
                        None if task["dimension"] == "" else int(float(task["dimension"]))
                    )
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

            train_clusters, test_clusters = load_cluster_split_labels(
                Path(str(cluster_row["labels_path"])),
                data.train.sample_ids,
                data.test.sample_ids,
            )
            usable = (
                np.ones_like(train_clusters, dtype=bool)
                if include_noise
                else train_clusters >= 0
            )
            if (
                np.unique(train_clusters[usable]).shape[0]
                < cfg.cluster_conditioning.min_clusters
            ):
                raise ValueError(
                    "Cluster labels have too few non-noise clusters in the train split."
                )

            base = _base_row(cluster_row, task, row_number)
            base["train_cluster_coverage"] = float(np.mean(usable))
            base["test_cluster_coverage"] = float(np.mean(test_clusters >= 0))
            base["runtime_seconds"] = perf_counter() - started
            base["matched_lowgen_task_id"] = task["task_id"]
            cluster_counts = _label_counts(train_clusters[usable])

            for kind in kinds:
                rows.append(
                    _run_cluster_generator(
                        cfg,
                        base,
                        data,
                        rep,
                        kind,
                        train_clusters[usable],
                        cluster_counts,
                        int(task["seed"]),
                        usable,
                        synthetic_total=sample_cap,
                        generator_cache=generator_cache,
                    )
                )
        except Exception as exc:
            skipped.append(
                _cluster_identity(cluster_row)
                | {
                    "row_number": row_number,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return rows, skipped


def _resolve_dask_chunk_size(
    n_rows: int,
    n_workers: int,
    requested: int | None,
) -> int:
    if requested is not None and int(requested) > 0:
        return int(requested)
    # A few chunks per worker gives Dask enough scheduling flexibility while
    # still letting each task reuse loaded embedding arrays across rows.
    target_chunks = max(1, int(n_workers) * 4)
    return max(1, int(np.ceil(max(1, n_rows) / target_chunks)))


def _chunked(items: list[dict[str, Any]], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def load_cluster_split_labels(
    labels_path: Path,
    train_sample_ids: np.ndarray,
    test_sample_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(labels_path, allow_pickle=True) as data:
        labels = np.asarray(data["labels"], dtype=np.int64)
        if "sample_ids" not in data.files:
            train_n = len(train_sample_ids)
            test_n = len(test_sample_ids)
            if len(labels) < train_n + test_n:
                raise ValueError(
                    f"Labels at {labels_path} do not contain enough rows for train/test splits."
                )
            return labels[:train_n], labels[train_n : train_n + test_n]
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
    lookup: dict[str, int] = {}
    for sample_id, label in zip(sample_ids, labels):
        for alias in _sample_id_aliases(sample_id):
            lookup.setdefault(alias, int(label))
    return (
        _align_labels(lookup, train_sample_ids, "train", labels_path),
        _align_labels(lookup, test_sample_ids, "test", labels_path),
    )


def majority_label_map(cluster_labels: np.ndarray, true_labels: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for cluster in np.unique(cluster_labels):
        mask = cluster_labels == cluster
        labels, counts = np.unique(true_labels[mask], return_counts=True)
        mapping[int(cluster)] = int(labels[np.argmax(counts)])
    return mapping


def metric_correlations(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(
            columns=[
                "generator",
                "cluster_metric",
                "quality_metric",
                "n",
                "pearson",
                "spearman",
            ]
        )
    rows: list[dict[str, Any]] = []
    for generator, group in results.groupby("generator", dropna=False):
        for cluster_metric in CLUSTER_METRIC_COLUMNS:
            if cluster_metric not in group.columns:
                continue
            x = pd.to_numeric(group[cluster_metric], errors="coerce")
            for quality_metric in QUALITY_METRIC_COLUMNS:
                if quality_metric not in group.columns:
                    continue
                y = pd.to_numeric(group[quality_metric], errors="coerce")
                mask = x.notna() & y.notna()
                n = int(mask.sum())
                if n == 0:
                    continue
                rows.append(
                    {
                        "generator": generator,
                        "cluster_metric": cluster_metric,
                        "quality_metric": quality_metric,
                        "n": n,
                        "pearson": (
                            float(x[mask].corr(y[mask], method="pearson"))
                            if n >= 3
                            else float("nan")
                        ),
                        "spearman": (
                            float(x[mask].corr(y[mask], method="spearman"))
                            if n >= 3
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _run_cluster_generator(
    cfg: ExperimentConfig,
    base: dict[str, Any],
    data: Any,
    rep: Any,
    kind: str,
    train_clusters: np.ndarray,
    cluster_counts: dict[int, int],
    seed: int,
    usable_train_mask: np.ndarray,
    synthetic_total: int,
    generator_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rng_seed = seed + int(stable_hash([kind, "cluster"], length=8), 16) % 100000
    rng = np.random.default_rng(rng_seed)
    true_label_baseline = kind == FULL_DIAG_GMM4_TRUE_LABEL_PROBE
    cluster_labelled_branch = kind == FULL_DIAG_GMM4_LABEL_PROBE
    usable_true_labels = data.train.labels[usable_train_mask]
    if true_label_baseline:
        counts_for_metadata = _label_counts(data.train.labels)
        conditioning = "true_label"
        count_source = "train_true_labels"
    elif cluster_labelled_branch:
        counts_for_metadata = _cluster_label_counts(train_clusters, usable_true_labels)
        conditioning = "cluster_true_label"
        count_source = "train_cluster_label_cells"
    else:
        counts_for_metadata = cluster_counts
        conditioning = "cluster"
        count_source = "train_clusters"
    row = {
        **base,
        "generator": kind,
        "conditioning": conditioning,
        "synthetic_count_source": count_source,
        "synthetic_target_counts": json.dumps(counts_for_metadata, sort_keys=True),
        "status": "ok",
    }
    cache_key = None
    if true_label_baseline and generator_cache is not None:
        cache_key = (
            kind,
            str(base["dataset"]),
            str(base["encoder"]),
            int(seed),
            int(synthetic_total),
            int(cfg.cluster_conditioning.min_cluster_size),
            float(cfg.cluster_conditioning.min_variance),
            int(cfg.cluster_conditioning.gmm_components),
            int(cfg.cluster_conditioning.gmm_fit_subsample),
            int(cfg.cluster_conditioning.gmm_max_iter),
            float(cfg.cluster_conditioning.gmm_reg_covar),
        )
        cached = generator_cache.get(cache_key)
        if cached is not None:
            row.update(cached)
            row["baseline_cache_hit"] = True
            row["generator_runtime_seconds"] = 0.0
            return row
    started = perf_counter()
    try:
        if kind in FULL_EMBEDDING_PROBES:
            if kind == FULL_DIAG_PROBE:
                x_synth, generator_metadata = cluster_guided_full_diag_gaussian_probe(
                    data.train.features[usable_train_mask],
                    train_clusters,
                    synthetic_total,
                    cfg.cluster_conditioning.min_cluster_size,
                    cfg.cluster_conditioning.min_variance,
                    rng,
                )
                y_synth = None
            elif kind == FULL_DIAG_GMM4_PROBE:
                x_synth, generator_metadata = cluster_guided_full_diag_gmm4_probe(
                    data.train.features[usable_train_mask],
                    train_clusters,
                    synthetic_total,
                    cfg.cluster_conditioning.min_cluster_size,
                    cfg.cluster_conditioning.min_variance,
                    rng,
                    n_components=cfg.cluster_conditioning.gmm_components,
                    fit_subsample=cfg.cluster_conditioning.gmm_fit_subsample,
                    max_iter=cfg.cluster_conditioning.gmm_max_iter,
                    reg_covar=cfg.cluster_conditioning.gmm_reg_covar,
                )
                y_synth = None
            elif kind == FULL_DIAG_GMM4_TRUE_LABEL_PROBE:
                x_synth, y_synth, generator_metadata = (
                    label_guided_full_diag_gmm4_probe(
                        data.train.features,
                        data.train.labels,
                        synthetic_total,
                        cfg.cluster_conditioning.min_cluster_size,
                        cfg.cluster_conditioning.min_variance,
                        rng,
                        n_components=cfg.cluster_conditioning.gmm_components,
                        fit_subsample=cfg.cluster_conditioning.gmm_fit_subsample,
                        max_iter=cfg.cluster_conditioning.gmm_max_iter,
                        reg_covar=cfg.cluster_conditioning.gmm_reg_covar,
                    )
                )
            elif kind == FULL_DIAG_GMM4_LABEL_PROBE:
                x_synth, y_synth, generator_metadata = (
                    cluster_label_guided_full_diag_gmm4_probe(
                        data.train.features[usable_train_mask],
                        train_clusters,
                        usable_true_labels,
                        synthetic_total,
                        cfg.cluster_conditioning.min_cluster_size,
                        cfg.cluster_conditioning.min_variance,
                        rng,
                        n_components=cfg.cluster_conditioning.gmm_components,
                        fit_subsample=cfg.cluster_conditioning.gmm_fit_subsample,
                        max_iter=cfg.cluster_conditioning.gmm_max_iter,
                        reg_covar=cfg.cluster_conditioning.gmm_reg_covar,
                    )
                )
            else:
                raise ValueError(f"Unsupported full-embedding generator kind {kind!r}.")
            fid = fidelity_metrics(
                data.test.features,
                x_synth,
                cfg.eval,
                seed,
                str(base["encoder"]),
            )
            utility = (
                utility_metrics(
                    x_synth,
                    y_synth,
                    data.train.features,
                    data.train.labels,
                    data.test.features,
                    data.test.labels,
                    cfg.eval,
                    data.class_names,
                )
                if y_synth is not None
                else {}
            )
            payload = _flatten_generator_metrics({}, fid, {}, utility)
            payload["synthetic_n_samples"] = int(len(x_synth))
            payload["synthetic_dimension"] = int(x_synth.shape[1])
            payload["generation_space"] = "full_embedding"
            if y_synth is not None:
                payload["synthetic_label_counts"] = json.dumps(
                    _label_counts(y_synth),
                    sort_keys=True,
                )
            payload["generator_metadata"] = json.dumps(
                generator_metadata,
                sort_keys=True,
            )
            payload["baseline_cache_hit"] = False
            row.update(payload)
            if cache_key is not None and generator_cache is not None:
                generator_cache[cache_key] = dict(payload)
        else:
            if rep is None:
                raise RuntimeError(f"Reduced representation is required for {kind!r}.")
            z_synth, _cluster_synth, generator_metadata = _sample_legacy_generator(
                cfg,
                kind,
                rep.train[usable_train_mask],
                train_clusters,
                cluster_counts,
                seed,
                rng,
                rep.valid,
            )
            x_synth = (
                z_synth
                if base["projector"] == "raw_highdim"
                else inverse_transform(
                    rep.model,
                    z_synth,
                    reference=(rep.train, data.train.features),
                )
            )
            lowdim_fid = lowdim_fidelity_metrics(rep.test, z_synth, cfg.eval, seed)
            fid = fidelity_metrics(
                data.test.features,
                x_synth,
                cfg.eval,
                seed,
                str(base["encoder"]),
            )
            row.update(_flatten_generator_metrics(lowdim_fid, fid, {}, {}))
            row["synthetic_n_samples"] = int(len(z_synth))
            row["synthetic_dimension"] = int(x_synth.shape[1])
            row["generation_space"] = "reduced_then_inverse"
            row["generator_metadata"] = json.dumps(generator_metadata, sort_keys=True)
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    row["generator_runtime_seconds"] = perf_counter() - started
    return row


def cluster_guided_full_diag_gaussian_probe(
    x_train: np.ndarray,
    cluster_labels: np.ndarray,
    synthetic_total: int,
    min_cluster_size: int,
    min_variance: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample full-dimensional points from cluster-local diagonal Gaussians."""

    x_train = np.asarray(x_train, dtype=np.float32)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    clusters, counts = np.unique(cluster_labels, return_counts=True)
    eligible = counts >= int(min_cluster_size)
    clusters = clusters[eligible]
    counts = counts[eligible]
    if len(clusters) == 0:
        raise ValueError("No clusters met the minimum size for Gaussian probing.")

    allocations = _allocate_synthetic_counts(counts, int(synthetic_total))
    parts = []
    cluster_metadata = []
    floored_variance_dimensions = 0
    for cluster, real_count, synthetic_count in zip(clusters, counts, allocations):
        if synthetic_count <= 0:
            continue
        x_cluster = x_train[cluster_labels == cluster]
        mean = x_cluster.mean(axis=0)
        variance = np.asarray(x_cluster.var(axis=0), dtype=np.float32)
        floor_mask = variance < float(min_variance)
        floored_variance_dimensions += int(np.count_nonzero(floor_mask))
        variance = np.maximum(variance, float(min_variance))
        samples = rng.normal(
            loc=mean,
            scale=np.sqrt(variance),
            size=(int(synthetic_count), x_train.shape[1]),
        ).astype(np.float32)
        parts.append(samples)
        cluster_metadata.append(
            {
                "cluster": int(cluster),
                "real_count": int(real_count),
                "synthetic_count": int(synthetic_count),
                "mean_variance": float(np.mean(variance)),
                "floored_variance_dimensions": int(np.count_nonzero(floor_mask)),
            }
        )

    if not parts:
        raise ValueError("Synthetic allocation produced no samples.")
    metadata = {
        "kind": FULL_DIAG_PROBE,
        "distribution": "cluster_mean_full_embedding_diagonal_gaussian",
        "generation_space": "full_embedding",
        "synthetic_total_requested": int(synthetic_total),
        "synthetic_total": int(sum(len(part) for part in parts)),
        "min_cluster_size": int(min_cluster_size),
        "min_variance": float(min_variance),
        "n_eligible_clusters": int(len(clusters)),
        "synthetic_dimension": int(x_train.shape[1]),
        "floored_variance_dimensions": int(floored_variance_dimensions),
        "clusters": cluster_metadata,
    }
    return np.concatenate(parts, axis=0), metadata


def cluster_guided_full_diag_gmm4_probe(
    x_train: np.ndarray,
    cluster_labels: np.ndarray,
    synthetic_total: int,
    min_cluster_size: int,
    min_variance: float,
    rng: np.random.Generator,
    *,
    n_components: int = 4,
    fit_subsample: int = 10000,
    max_iter: int = 100,
    reg_covar: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample full-dimensional points from cluster-local diagonal GMMs."""

    from sklearn.mixture import GaussianMixture

    x_train = np.asarray(x_train, dtype=np.float32)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    clusters, counts = np.unique(cluster_labels, return_counts=True)
    eligible = counts >= int(min_cluster_size)
    clusters = clusters[eligible]
    counts = counts[eligible]
    if len(clusters) == 0:
        raise ValueError("No clusters met the minimum size for GMM probing.")

    requested_components = max(1, int(n_components))
    fit_cap = max(0, int(fit_subsample))
    allocations = _allocate_synthetic_counts(counts, int(synthetic_total))
    parts = []
    cluster_metadata = []
    total_components = 0
    converged_models = 0
    total_iterations = 0
    total_fit_samples = 0
    for cluster, real_count, synthetic_count in zip(clusters, counts, allocations):
        if synthetic_count <= 0:
            continue
        x_cluster = x_train[cluster_labels == cluster]
        fit_x = x_cluster
        if fit_cap > 0 and len(x_cluster) > fit_cap:
            fit_indices = rng.choice(len(x_cluster), size=fit_cap, replace=False)
            fit_x = x_cluster[fit_indices]
        fitted_components = min(requested_components, len(fit_x))
        model_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        model = GaussianMixture(
            n_components=int(fitted_components),
            covariance_type="diag",
            reg_covar=max(float(reg_covar), float(min_variance)),
            max_iter=int(max_iter),
            n_init=1,
            init_params="kmeans",
            random_state=model_seed,
        ).fit(fit_x)
        samples, component_labels = model.sample(int(synthetic_count))
        samples = np.asarray(samples, dtype=np.float32)
        parts.append(samples)
        total_components += int(fitted_components)
        converged_models += int(bool(model.converged_))
        total_iterations += int(model.n_iter_)
        total_fit_samples += int(len(fit_x))
        cluster_metadata.append(
            {
                "cluster": int(cluster),
                "real_count": int(real_count),
                "fit_count": int(len(fit_x)),
                "synthetic_count": int(synthetic_count),
                "n_components_requested": int(requested_components),
                "n_components_fitted": int(fitted_components),
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
                "mean_variance": float(np.mean(model.covariances_)),
                "component_counts": {
                    str(component): int(count)
                    for component, count in zip(
                        *np.unique(component_labels, return_counts=True)
                    )
                },
            }
        )

    if not parts:
        raise ValueError("Synthetic allocation produced no samples.")
    metadata = {
        "kind": FULL_DIAG_GMM4_PROBE,
        "distribution": "cluster_local_full_embedding_diagonal_gmm",
        "generation_space": "full_embedding",
        "synthetic_total_requested": int(synthetic_total),
        "synthetic_total": int(sum(len(part) for part in parts)),
        "min_cluster_size": int(min_cluster_size),
        "min_variance": float(min_variance),
        "gmm_components_requested": int(requested_components),
        "gmm_fit_subsample": int(fit_cap),
        "gmm_max_iter": int(max_iter),
        "gmm_reg_covar": float(max(float(reg_covar), float(min_variance))),
        "n_eligible_clusters": int(len(clusters)),
        "total_components_fitted": int(total_components),
        "converged_models": int(converged_models),
        "mean_n_iter": float(total_iterations / max(1, len(cluster_metadata))),
        "total_fit_samples": int(total_fit_samples),
        "synthetic_dimension": int(x_train.shape[1]),
        "clusters": cluster_metadata,
    }
    return np.concatenate(parts, axis=0), metadata


def label_guided_full_diag_gmm4_probe(
    x_train: np.ndarray,
    true_labels: np.ndarray,
    synthetic_total: int,
    min_cluster_size: int,
    min_variance: float,
    rng: np.random.Generator,
    *,
    n_components: int = 4,
    fit_subsample: int = 10000,
    max_iter: int = 100,
    reg_covar: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample full-dimensional points from true-label-local diagonal GMMs."""

    from sklearn.mixture import GaussianMixture

    x_train = np.asarray(x_train, dtype=np.float32)
    true_labels = np.asarray(true_labels, dtype=np.int64)
    if len(x_train) != len(true_labels):
        raise ValueError("Features and true labels must align.")

    labels, counts = np.unique(true_labels, return_counts=True)
    eligible = counts >= int(min_cluster_size)
    labels = labels[eligible]
    counts = counts[eligible]
    if len(labels) == 0:
        raise ValueError("No true-label cells met the minimum size for GMM probing.")

    requested_components = max(1, int(n_components))
    fit_cap = max(0, int(fit_subsample))
    allocations = _allocate_synthetic_counts(counts, int(synthetic_total))
    x_parts = []
    y_parts = []
    label_metadata = []
    total_components = 0
    converged_models = 0
    total_iterations = 0
    total_fit_samples = 0
    for label, real_count, synthetic_count in zip(labels, counts, allocations):
        if synthetic_count <= 0:
            continue
        x_label = x_train[true_labels == label]
        fit_x = x_label
        if fit_cap > 0 and len(x_label) > fit_cap:
            fit_indices = rng.choice(len(x_label), size=fit_cap, replace=False)
            fit_x = x_label[fit_indices]
        fitted_components = min(requested_components, len(fit_x))
        model_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        model = GaussianMixture(
            n_components=int(fitted_components),
            covariance_type="diag",
            reg_covar=max(float(reg_covar), float(min_variance)),
            max_iter=int(max_iter),
            n_init=1,
            init_params="kmeans",
            random_state=model_seed,
        ).fit(fit_x)
        samples, component_labels = model.sample(int(synthetic_count))
        samples = np.asarray(samples, dtype=np.float32)
        x_parts.append(samples)
        y_parts.append(np.full(int(synthetic_count), int(label), dtype=np.int64))
        total_components += int(fitted_components)
        converged_models += int(bool(model.converged_))
        total_iterations += int(model.n_iter_)
        total_fit_samples += int(len(fit_x))
        label_metadata.append(
            {
                "label": int(label),
                "real_count": int(real_count),
                "fit_count": int(len(fit_x)),
                "synthetic_count": int(synthetic_count),
                "n_components_requested": int(requested_components),
                "n_components_fitted": int(fitted_components),
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
                "mean_variance": float(np.mean(model.covariances_)),
                "component_counts": {
                    str(component): int(count)
                    for component, count in zip(
                        *np.unique(component_labels, return_counts=True)
                    )
                },
            }
        )

    if not x_parts:
        raise ValueError("Synthetic allocation produced no samples.")
    y_synth = np.concatenate(y_parts)
    metadata = {
        "kind": FULL_DIAG_GMM4_TRUE_LABEL_PROBE,
        "distribution": "label_local_full_embedding_diagonal_gmm",
        "generation_space": "full_embedding",
        "conditioning": "true_label",
        "synthetic_total_requested": int(synthetic_total),
        "synthetic_total": int(sum(len(part) for part in x_parts)),
        "min_cluster_size": int(min_cluster_size),
        "min_variance": float(min_variance),
        "gmm_components_requested": int(requested_components),
        "gmm_fit_subsample": int(fit_cap),
        "gmm_max_iter": int(max_iter),
        "gmm_reg_covar": float(max(float(reg_covar), float(min_variance))),
        "n_eligible_labels": int(len(labels)),
        "n_synthetic_labels": int(np.unique(y_synth).shape[0]),
        "synthetic_label_counts": _label_counts(y_synth),
        "total_components_fitted": int(total_components),
        "converged_models": int(converged_models),
        "mean_n_iter": float(total_iterations / max(1, len(label_metadata))),
        "total_fit_samples": int(total_fit_samples),
        "synthetic_dimension": int(x_train.shape[1]),
        "labels": label_metadata,
    }
    return np.concatenate(x_parts, axis=0), y_synth, metadata


def cluster_label_guided_full_diag_gmm4_probe(
    x_train: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    synthetic_total: int,
    min_cluster_size: int,
    min_variance: float,
    rng: np.random.Generator,
    *,
    n_components: int = 4,
    fit_subsample: int = 10000,
    max_iter: int = 100,
    reg_covar: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample full-dimensional points from cluster-and-label-local diagonal GMMs."""

    from sklearn.mixture import GaussianMixture

    x_train = np.asarray(x_train, dtype=np.float32)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    true_labels = np.asarray(true_labels, dtype=np.int64)
    if len(x_train) != len(cluster_labels) or len(x_train) != len(true_labels):
        raise ValueError("Features, cluster labels, and true labels must align.")

    cells, counts = _cluster_label_cells(cluster_labels, true_labels)
    eligible = counts >= int(min_cluster_size)
    cells = cells[eligible]
    counts = counts[eligible]
    if len(cells) == 0:
        raise ValueError("No cluster-label cells met the minimum size for GMM probing.")

    requested_components = max(1, int(n_components))
    fit_cap = max(0, int(fit_subsample))
    allocations = _allocate_synthetic_counts(counts, int(synthetic_total))
    x_parts = []
    y_parts = []
    cell_metadata = []
    total_components = 0
    converged_models = 0
    total_iterations = 0
    total_fit_samples = 0
    for (cluster, label), real_count, synthetic_count in zip(cells, counts, allocations):
        if synthetic_count <= 0:
            continue
        mask = (cluster_labels == cluster) & (true_labels == label)
        x_cell = x_train[mask]
        fit_x = x_cell
        if fit_cap > 0 and len(x_cell) > fit_cap:
            fit_indices = rng.choice(len(x_cell), size=fit_cap, replace=False)
            fit_x = x_cell[fit_indices]
        fitted_components = min(requested_components, len(fit_x))
        model_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        model = GaussianMixture(
            n_components=int(fitted_components),
            covariance_type="diag",
            reg_covar=max(float(reg_covar), float(min_variance)),
            max_iter=int(max_iter),
            n_init=1,
            init_params="kmeans",
            random_state=model_seed,
        ).fit(fit_x)
        samples, component_labels = model.sample(int(synthetic_count))
        samples = np.asarray(samples, dtype=np.float32)
        x_parts.append(samples)
        y_parts.append(np.full(int(synthetic_count), int(label), dtype=np.int64))
        total_components += int(fitted_components)
        converged_models += int(bool(model.converged_))
        total_iterations += int(model.n_iter_)
        total_fit_samples += int(len(fit_x))
        cell_metadata.append(
            {
                "cluster": int(cluster),
                "label": int(label),
                "real_count": int(real_count),
                "fit_count": int(len(fit_x)),
                "synthetic_count": int(synthetic_count),
                "n_components_requested": int(requested_components),
                "n_components_fitted": int(fitted_components),
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
                "mean_variance": float(np.mean(model.covariances_)),
                "component_counts": {
                    str(component): int(count)
                    for component, count in zip(
                        *np.unique(component_labels, return_counts=True)
                    )
                },
            }
        )

    if not x_parts:
        raise ValueError("Synthetic allocation produced no samples.")
    y_synth = np.concatenate(y_parts)
    metadata = {
        "kind": FULL_DIAG_GMM4_LABEL_PROBE,
        "distribution": "cluster_label_local_full_embedding_diagonal_gmm",
        "generation_space": "full_embedding",
        "conditioning": "cluster_true_label",
        "synthetic_total_requested": int(synthetic_total),
        "synthetic_total": int(sum(len(part) for part in x_parts)),
        "min_cluster_size": int(min_cluster_size),
        "min_variance": float(min_variance),
        "gmm_components_requested": int(requested_components),
        "gmm_fit_subsample": int(fit_cap),
        "gmm_max_iter": int(max_iter),
        "gmm_reg_covar": float(max(float(reg_covar), float(min_variance))),
        "n_eligible_cluster_label_cells": int(len(cells)),
        "n_synthetic_labels": int(np.unique(y_synth).shape[0]),
        "synthetic_label_counts": _label_counts(y_synth),
        "total_components_fitted": int(total_components),
        "converged_models": int(converged_models),
        "mean_n_iter": float(total_iterations / max(1, len(cell_metadata))),
        "total_fit_samples": int(total_fit_samples),
        "synthetic_dimension": int(x_train.shape[1]),
        "cells": cell_metadata,
    }
    return np.concatenate(x_parts, axis=0), y_synth, metadata


def cluster_guided_gaussian_probe(
    z_train: np.ndarray,
    cluster_labels: np.ndarray,
    synthetic_total: int,
    min_cluster_size: int,
    min_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample low-dimensional points from cluster-local isotropic Gaussians."""

    z_train = np.asarray(z_train, dtype=np.float32)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    clusters, counts = np.unique(cluster_labels, return_counts=True)
    eligible = counts >= int(min_cluster_size)
    clusters = clusters[eligible]
    counts = counts[eligible]
    if len(clusters) == 0:
        raise ValueError("No clusters met the minimum size for Gaussian probing.")

    allocations = _allocate_synthetic_counts(counts, int(synthetic_total))
    parts = []
    cluster_metadata = []
    for cluster, real_count, synthetic_count in zip(clusters, counts, allocations):
        if synthetic_count <= 0:
            continue
        z_cluster = z_train[cluster_labels == cluster]
        mean = z_cluster.mean(axis=0)
        sigma = _cluster_isotropic_sigma(z_cluster, mean, float(min_sigma))
        samples = rng.normal(
            loc=mean,
            scale=sigma,
            size=(int(synthetic_count), z_train.shape[1]),
        ).astype(np.float32)
        parts.append(samples)
        cluster_metadata.append(
            {
                "cluster": int(cluster),
                "real_count": int(real_count),
                "synthetic_count": int(synthetic_count),
                "sigma": float(sigma),
            }
        )

    if not parts:
        raise ValueError("Synthetic allocation produced no samples.")
    metadata = {
        "kind": LOWDIM_ISOTROPIC_PROBE,
        "distribution": "cluster_mean_isotropic_gaussian",
        "synthetic_total_requested": int(synthetic_total),
        "synthetic_total": int(sum(len(part) for part in parts)),
        "min_cluster_size": int(min_cluster_size),
        "min_sigma": float(min_sigma),
        "n_eligible_clusters": int(len(clusters)),
        "clusters": cluster_metadata,
    }
    return np.concatenate(parts, axis=0), metadata


def _sample_legacy_generator(
    cfg: ExperimentConfig,
    kind: str,
    z_train: np.ndarray,
    train_clusters: np.ndarray,
    cluster_counts: dict[int, int],
    seed: int,
    rng: np.random.Generator,
    z_valid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fitted = fit_generator(
        kind,
        "true_label",
        z_train,
        train_clusters,
        seed,
        cfg.generator.gmm_components,
        cfg.generator.timeout_seconds,
        cfg.generator.gmm_fit_subsample,
        cfg.generator.gmm_max_iter,
        cfg.generator.gmm_select_k,
        cfg.generator.gmm_candidate_components,
        cfg.generator.gmm_selection_metric,
        cfg.generator.gmm_min_samples_per_component,
        x_valid=z_valid,
    )
    z_synth, cluster_synth = fitted.sample_counts(cluster_counts, rng)
    return z_synth, cluster_synth, fitted.metadata


def _cluster_isotropic_sigma(
    z_cluster: np.ndarray,
    mean: np.ndarray,
    min_sigma: float,
) -> float:
    if len(z_cluster) < 2:
        return float(min_sigma)
    squared_distances = np.sum((z_cluster - mean) ** 2, axis=1)
    sigma = np.sqrt(float(np.mean(squared_distances)) / max(1, z_cluster.shape[1]))
    return max(float(sigma), float(min_sigma))


def _allocate_synthetic_counts(counts: np.ndarray, synthetic_total: int) -> np.ndarray:
    total = max(1, min(int(synthetic_total), int(np.sum(counts))))
    proportions = counts.astype(np.float64) / float(np.sum(counts))
    raw = proportions * total
    allocations = np.floor(raw).astype(int)
    remainders = raw - allocations
    while allocations.sum() < total:
        idx = int(np.argmax(remainders))
        allocations[idx] += 1
        remainders[idx] = 0.0
    while allocations.sum() > total:
        idx = int(np.argmax(allocations))
        allocations[idx] -= 1
    return allocations


def _flatten_generator_metrics(
    lowdim_fid: dict[str, Any],
    fid: dict[str, Any],
    lowdim_utility: dict[str, Any],
    utility: dict[str, Any],
) -> dict[str, Any]:
    lowdim_tstr = lowdim_utility.get("train_synthetic_test_real", {})
    lowdim_aug = lowdim_utility.get("real_plus_synthetic_test_real", {})
    tstr = utility.get("train_synthetic_test_real", {})
    aug = utility.get("real_plus_synthetic_test_real", {})
    return {
        "lowdim_fid": lowdim_fid.get("fid"),
        "lowdim_kid": lowdim_fid.get("kid"),
        "lowdim_precision": lowdim_fid.get("precision"),
        "lowdim_recall": lowdim_fid.get("recall"),
        "lowdim_centroid_distance": lowdim_fid.get("centroid_distance"),
        "lowdim_covariance_frobenius": lowdim_fid.get("covariance_frobenius"),
        "fid": fid.get("fid"),
        "kid": fid.get("kid"),
        "precision": fid.get("precision"),
        "recall": fid.get("recall"),
        "real_clusters": fid.get("real_clusters"),
        "synthetic_clusters": fid.get("synthetic_clusters"),
        "tstr_accuracy": tstr.get("accuracy"),
        "tstr_macro_roc_auc_ovr": tstr.get("macro_roc_auc_ovr"),
        "aug_accuracy": aug.get("accuracy"),
        "aug_macro_roc_auc_ovr": aug.get("macro_roc_auc_ovr"),
        "lowdim_tstr_accuracy": lowdim_tstr.get("accuracy"),
        "lowdim_tstr_macro_roc_auc_ovr": lowdim_tstr.get("macro_roc_auc_ovr"),
        "lowdim_aug_accuracy": lowdim_aug.get("accuracy"),
        "lowdim_aug_macro_roc_auc_ovr": lowdim_aug.get("macro_roc_auc_ovr"),
    }


def _task_index(
    cfg: ExperimentConfig,
    manifest: Path | None,
) -> dict[tuple[str, str, str, str, int], dict[str, Any]]:
    path = manifest or manifest_path(cfg)
    if path.exists():
        tasks = read_jsonl(path)
    else:
        tasks = build_task_manifest(cfg)
    index: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for task in tasks:
        key = _representation_key(task)
        index.setdefault(key, task)
    return index


def _match_task(
    cluster_row: pd.Series,
    task_index: dict[tuple[str, str, str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    key = _representation_key(
        {
            "dataset": cluster_row["dataset"],
            "encoder": cluster_row["encoder"],
            "projector": cluster_row["projector"],
            "dimension": cluster_row.get("dimension", ""),
            "seed": cluster_row.get("projection_seed", cluster_row.get("cluster_seed", 0)),
        }
    )
    if key not in task_index:
        raise KeyError(f"No lowgen manifest task matches clustering row key {key}.")
    return task_index[key]


def _representation_key(row: dict[str, Any] | pd.Series) -> tuple[str, str, str, str, int]:
    dimension = row.get("dimension", "")
    if pd.isna(dimension) or dimension is None:
        dimension_key = ""
    elif str(dimension) == "":
        dimension_key = ""
    else:
        dimension_key = str(int(float(dimension)))
    return (
        str(row["dataset"]),
        str(row["encoder"]),
        _normalise_projector(str(row["projector"])),
        dimension_key,
        int(float(row.get("seed", row.get("projection_seed", row.get("cluster_seed", 0))))),
    )


def _load_cluster_rows(
    path: Path,
    max_rows: int | None,
    max_cluster_configs: int | None,
    min_clusters: int,
    max_noise_fraction: float,
    dataset: str | None,
    encoder: str | None,
    projector: str | None,
    target_name: str | None,
    seed: int | None,
) -> list[pd.Series]:
    df = pd.read_csv(path)
    required = {"dataset", "encoder", "projector", "labels_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required clustering result columns: {sorted(missing)}")
    df = df.copy()
    if "valid_cluster_count" in df.columns:
        valid = df["valid_cluster_count"].astype(str).str.lower().isin(["true", "1"])
        df = df[valid]
    if "n_clusters" in df.columns:
        df = df[pd.to_numeric(df["n_clusters"], errors="coerce") >= min_clusters]
    if "noise_fraction" in df.columns:
        df = df[pd.to_numeric(df["noise_fraction"], errors="coerce") <= max_noise_fraction]
    df = df[df["labels_path"].astype(str).map(lambda value: Path(value).exists())]

    df = _filter_cluster_rows(
        df,
        dataset=dataset,
        encoder=encoder,
        projector=projector,
        target_name=target_name,
        seed=seed,
    )
    if target_name is not None and df.empty and target_name != "intrinsic_mle":
        df = pd.read_csv(path)
        if "valid_cluster_count" in df.columns:
            valid = df["valid_cluster_count"].astype(str).str.lower().isin(["true", "1"])
            df = df[valid]
        if "n_clusters" in df.columns:
            df = df[pd.to_numeric(df["n_clusters"], errors="coerce") >= min_clusters]
        if "noise_fraction" in df.columns:
            df = df[
                pd.to_numeric(df["noise_fraction"], errors="coerce")
                <= max_noise_fraction
            ]
        df = df[df["labels_path"].astype(str).map(lambda value: Path(value).exists())]
        df = _filter_cluster_rows(
            df,
            dataset=dataset,
            encoder=encoder,
            projector=projector,
            target_name="intrinsic_mle",
            seed=seed,
        )

    cap = max_cluster_configs if max_cluster_configs is not None else max_rows
    if cap is not None and cap > 0:
        df = select_metric_spread_cluster_rows(df, int(cap))
    elif "silhouette" in df.columns:
        df["_sort_silhouette"] = pd.to_numeric(df["silhouette"], errors="coerce")
        df = df.sort_values("_sort_silhouette", ascending=False, kind="mergesort")
    return [row for _, row in df.iterrows()]


def _filter_cluster_rows(
    df: pd.DataFrame,
    dataset: str | None,
    encoder: str | None,
    projector: str | None,
    target_name: str | None,
    seed: int | None,
) -> pd.DataFrame:
    out = df.copy()
    if dataset is not None:
        out = out[out["dataset"].astype(str) == str(dataset)]
    if encoder is not None:
        out = out[out["encoder"].astype(str) == str(encoder)]
    if projector is not None:
        out = out[
            out["projector"].astype(str).map(_normalise_projector)
            == _normalise_projector(str(projector))
        ]
    if target_name is not None and "target_name" in out.columns:
        out = out[out["target_name"].astype(str) == str(target_name)]
    if seed is not None:
        seed_col = "projection_seed" if "projection_seed" in out.columns else "cluster_seed"
        out = out[pd.to_numeric(out[seed_col], errors="coerce") == int(seed)]
    return out


def select_metric_spread_cluster_rows(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if df.empty or cap <= 0:
        return df.head(0).copy()

    selected_indices: list[Any] = []
    for metric, ascending in [
        ("silhouette", False),
        ("sd_index", True),
        ("mean_pairwise_ari", False),
    ]:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        valid = df.loc[values.notna()].copy()
        if valid.empty:
            continue
        valid["_metric_value"] = pd.to_numeric(valid[metric], errors="coerce")
        ordered = valid.sort_values(
            "_metric_value",
            ascending=ascending,
            kind="mergesort",
        )
        for idx in _top_middle_bottom_indices(ordered):
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= cap:
                return df.loc[selected_indices].reset_index(drop=True)

    if len(selected_indices) < cap:
        remainder = df.loc[[idx for idx in df.index if idx not in selected_indices]]
        if "silhouette" in remainder.columns:
            remainder = remainder.assign(
                _sort_silhouette=pd.to_numeric(
                    remainder["silhouette"],
                    errors="coerce",
                )
            ).sort_values("_sort_silhouette", ascending=False, kind="mergesort")
        for idx in remainder.index:
            selected_indices.append(idx)
            if len(selected_indices) >= cap:
                break
    return df.loc[selected_indices].reset_index(drop=True)


def _top_middle_bottom_indices(df: pd.DataFrame) -> list[Any]:
    if df.empty:
        return []
    positions = [0, len(df) // 2, len(df) - 1]
    indices = []
    for position in positions:
        idx = df.index[position]
        if idx not in indices:
            indices.append(idx)
    return indices


def _normalise_projector(projector: str) -> str:
    return "raw_highdim" if projector == "raw" else projector


def _align_labels(
    lookup: dict[str, int],
    sample_ids: np.ndarray,
    split: str,
    labels_path: Path,
) -> np.ndarray:
    labels = []
    missing = []
    for sample_id in sample_ids.astype(str):
        aliases = _sample_id_aliases(sample_id)
        matched = next((lookup[alias] for alias in aliases if alias in lookup), None)
        if matched is None:
            missing.append(sample_id)
        else:
            labels.append(matched)
    if missing:
        raise KeyError(
            f"{len(missing)} {split} sample IDs are missing from {labels_path}; "
            f"first={missing[0]!r}."
        )
    return np.asarray(labels, dtype=np.int64)


def _normalise_sample_id(sample_id: str) -> str:
    return _sample_id_aliases(sample_id)[0]


def _sample_id_aliases(sample_id: str) -> list[str]:
    value = str(sample_id)
    candidates = [value]
    if ":" in value:
        prefix, rest = value.split(":", 1)
        if prefix in {"train", "test", "valid", "validation"}:
            candidates.append(rest)
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        for key in ("sample_id", "patch_filename", "slide_id", "patch_path", "path"):
            if key in parsed and parsed[key] is not None:
                candidates.append(str(parsed[key]))
    aliases: list[str] = []
    for candidate in candidates:
        raw = str(candidate)
        for item in (raw, Path(raw).name, Path(raw).stem):
            if item and item not in aliases:
                aliases.append(item)
    return aliases


def _base_row(cluster_row: pd.Series, task: dict[str, Any], row_number: int) -> dict[str, Any]:
    base = _cluster_identity(cluster_row)
    for col in CLUSTER_METRIC_COLUMNS:
        if col in cluster_row:
            base[col] = cluster_row[col]
    base.update(
        {
            "row_number": row_number,
            "lowgen_target_name": task.get("target_name"),
            "lowgen_projection_cache_key": task.get("projection_cache_key"),
        }
    )
    return base


def _cluster_identity(cluster_row: pd.Series) -> dict[str, Any]:
    columns = [
        "task_id",
        "dataset",
        "encoder",
        "projector",
        "dimension",
        "target_name",
        "source_metric",
        "condition",
        "projection_seed",
        "cluster_seed",
        "clusterer",
        "config_id",
        "cluster_params_json",
        "labels_path",
    ]
    return {col: cluster_row.get(col) for col in columns if col in cluster_row}


def _skipped_columns(skipped: list[dict[str, Any]]) -> list[str]:
    base = [
        "task_id",
        "dataset",
        "encoder",
        "projector",
        "dimension",
        "target_name",
        "projection_seed",
        "cluster_seed",
        "clusterer",
        "config_id",
        "labels_path",
        "row_number",
        "error_type",
        "error",
    ]
    extra = sorted({key for row in skipped for key in row if key not in base})
    return base + extra


def _label_counts(labels: np.ndarray) -> dict[int, int]:
    keys, values = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(keys, values)}


def _cluster_label_cells(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cells, counts = np.unique(
        np.column_stack(
            [
                np.asarray(cluster_labels, dtype=np.int64),
                np.asarray(true_labels, dtype=np.int64),
            ]
        ),
        axis=0,
        return_counts=True,
    )
    return cells.astype(np.int64, copy=False), counts.astype(np.int64, copy=False)


def _cluster_label_counts(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
) -> dict[str, int]:
    cells, counts = _cluster_label_cells(cluster_labels, true_labels)
    return {
        f"{int(cluster)}:{int(label)}": int(count)
        for (cluster, label), count in zip(cells, counts)
    }
