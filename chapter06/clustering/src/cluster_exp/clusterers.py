from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .config import ExperimentConfig
from .embeddings import EmbeddingData
from .metrics import sd_index
from .projections import deterministic_sample_indices, load_projection_features
from .utils import atomic_save_npz, atomic_write_json, stable_hash


def run_clustering_task(
    cfg: ExperimentConfig,
    task: dict[str, Any],
    force: bool = False,
) -> list[dict[str, Any]]:
    task_dir = cfg.paths.run_dir / "tasks" / str(task["task_id"])
    result_path = task_dir / "result.json"
    if result_path.exists() and not force:
        import json

        with result_path.open() as handle:
            return json.load(handle)

    data = load_projection_features(cfg, task)
    fit_subsample_size = compute_fit_subsample_size(cfg, data.n_samples)
    fit_indices = deterministic_sample_indices(
        data.n_samples,
        fit_subsample_size,
        int(task["cluster_seed"]),
    )
    results: list[dict[str, Any]] = []
    for clusterer, params in clustering_param_grid(cfg):
        config_id = stable_hash({"clusterer": clusterer, "params": params}, length=12)
        labels_path = task_dir / f"labels_{config_id}.npz"
        config_task = {
            **task,
            "clusterer": clusterer,
            "cluster_params": params,
            "config_id": config_id,
        }
        started = perf_counter()
        if clusterer == "kmeans":
            labels, fit_labels, model_metadata = _run_kmeans(cfg, data.features, fit_indices, config_task)
        elif clusterer == "spherical_kmeans":
            labels, fit_labels, model_metadata = _run_spherical_kmeans(
                cfg,
                data.features,
                fit_indices,
                config_task,
            )
        elif clusterer == "hdbscan":
            labels, fit_labels, model_metadata = _run_hdbscan(cfg, data.features, fit_indices, config_task)
        elif clusterer == "leiden":
            labels, fit_labels, model_metadata = _run_leiden(cfg, data.features, fit_indices, config_task)
        else:
            raise ValueError(f"Unsupported clusterer {clusterer!r}.")

        runtime = perf_counter() - started
        metrics = _compute_cluster_metrics(cfg, data, labels, fit_indices)
        result = {
            **{key: value for key, value in config_task.items() if key != "cluster_params"},
            "cluster_params": params,
            "fit_subsample": int(len(fit_indices)),
            "runtime_seconds": float(runtime),
            **model_metadata,
            **metrics,
            "labels_path": str(labels_path),
        }
        atomic_save_npz(
            labels_path,
            labels=np.asarray(labels, dtype=np.int64),
            fit_indices=np.asarray(fit_indices, dtype=np.int64),
            fit_labels=np.asarray(fit_labels, dtype=np.int64),
            sample_ids=data.sample_ids.astype(str),
            splits=data.splits.astype(str),
        )
        results.append(result)
    atomic_write_json(result_path, results)
    return results


def clustering_param_grid(cfg: ExperimentConfig) -> list[tuple[str, dict[str, Any]]]:
    configs: list[tuple[str, dict[str, Any]]] = []
    for clusterer in cfg.clustering.clusterers:
        if clusterer == "kmeans":
            for k in cfg.clustering.kmeans.k:
                configs.append(
                    (
                        "kmeans",
                        {
                            "n_clusters": int(k),
                            "init": cfg.clustering.kmeans.init,
                            "n_init": int(cfg.clustering.kmeans.n_init),
                            "max_iter": int(cfg.clustering.kmeans.max_iter),
                        },
                    )
                )
        elif clusterer == "spherical_kmeans":
            for k in cfg.clustering.spherical_kmeans.k:
                configs.append(
                    (
                        "spherical_kmeans",
                        {
                            "n_clusters": int(k),
                            "init": cfg.clustering.spherical_kmeans.init,
                            "n_init": int(cfg.clustering.spherical_kmeans.n_init),
                            "max_iter": int(cfg.clustering.spherical_kmeans.max_iter),
                            "normalization": "l2",
                            "geometry": "cosine",
                        },
                    )
                )
        elif clusterer == "hdbscan":
            for min_cluster_size in cfg.clustering.hdbscan.min_cluster_size:
                for min_samples in cfg.clustering.hdbscan.min_samples:
                    for method in cfg.clustering.hdbscan.cluster_selection_method:
                        configs.append(
                            (
                                "hdbscan",
                                {
                                    "min_cluster_size": int(min_cluster_size),
                                    "min_samples": int(min_samples),
                                    "cluster_selection_method": str(method),
                                    "alpha": float(cfg.clustering.hdbscan.alpha),
                                },
                            )
                        )
        elif clusterer == "leiden":
            for n_neighbors in cfg.clustering.leiden.n_neighbors:
                for resolution in cfg.clustering.leiden.resolution:
                    configs.append(
                        (
                            "leiden",
                            {
                                "n_neighbors": int(n_neighbors),
                                "resolution": float(resolution),
                                "metric": cfg.clustering.leiden.metric,
                                "max_iter": int(cfg.clustering.leiden.max_iter),
                                "theta": float(cfg.clustering.leiden.theta),
                            },
                        )
                    )
        else:
            raise ValueError(f"Unsupported clusterer {clusterer!r}.")
    return configs


def compute_fit_subsample_size(cfg: ExperimentConfig, n_samples: int) -> int:
    if cfg.clustering.fit_subsample is not None:
        return min(int(cfg.clustering.fit_subsample), int(n_samples))
    fraction_size = int(round(float(cfg.clustering.fit_subsample_fraction) * int(n_samples)))
    size = max(int(cfg.clustering.fit_subsample_min), fraction_size)
    size = min(int(cfg.clustering.fit_subsample_max), size)
    return min(size, int(n_samples))


def _run_kmeans(
    cfg: ExperimentConfig, features: np.ndarray, fit_indices: np.ndarray, task: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    params = dict(task["cluster_params"])
    seed = int(task["cluster_seed"])
    if _use_cuml(cfg):
        import cupy as cp
        from cuml.cluster import KMeans

        gpu_features = cp.asarray(features)
        model = KMeans(
            n_clusters=int(params["n_clusters"]),
            init=params.get("init", "k-means++"),
            n_init=int(params.get("n_init", 10)),
            max_iter=int(params.get("max_iter", 300)),
            random_state=seed,
        )
        fit_labels = model.fit_predict(gpu_features[fit_indices])
        labels = model.predict(gpu_features)
        inertia = float(getattr(model, "inertia_", np.nan))
        return _to_numpy(labels), _to_numpy(fit_labels), {"backend": "cuml", "inertia": inertia}

    from sklearn.cluster import KMeans

    model = KMeans(
        n_clusters=int(params["n_clusters"]),
        init=params.get("init", "k-means++"),
        n_init=int(params.get("n_init", 10)),
        max_iter=int(params.get("max_iter", 300)),
        random_state=seed,
    )
    fit_labels = model.fit_predict(features[fit_indices])
    labels = model.predict(features)
    return labels, fit_labels, {"backend": "sklearn", "inertia": float(model.inertia_)}


def _run_spherical_kmeans(
    cfg: ExperimentConfig, features: np.ndarray, fit_indices: np.ndarray, task: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    params = dict(task["cluster_params"])
    seed = int(task["cluster_seed"])
    if _use_cuml(cfg):
        import cupy as cp
        from cuml.cluster import KMeans

        gpu_features = _l2_normalize_cupy(cp.asarray(features))
        model = KMeans(
            n_clusters=int(params["n_clusters"]),
            init=params.get("init", "k-means++"),
            n_init=int(params.get("n_init", 10)),
            max_iter=int(params.get("max_iter", 300)),
            random_state=seed,
        )
        fit_labels = model.fit_predict(gpu_features[fit_indices])
        labels = model.predict(gpu_features)
        inertia = float(getattr(model, "inertia_", np.nan))
        return (
            _to_numpy(labels),
            _to_numpy(fit_labels),
            {
                "backend": "cuml",
                "inertia": inertia,
                "feature_normalization": "l2",
                "geometry": "cosine",
            },
        )

    from sklearn.cluster import KMeans

    normalized = _l2_normalize_numpy(features)
    model = KMeans(
        n_clusters=int(params["n_clusters"]),
        init=params.get("init", "k-means++"),
        n_init=int(params.get("n_init", 10)),
        max_iter=int(params.get("max_iter", 300)),
        random_state=seed,
    )
    fit_labels = model.fit_predict(normalized[fit_indices])
    labels = model.predict(normalized)
    return (
        labels,
        fit_labels,
        {
            "backend": "sklearn",
            "inertia": float(model.inertia_),
            "feature_normalization": "l2",
            "geometry": "cosine",
        },
    )


def _run_hdbscan(
    cfg: ExperimentConfig, features: np.ndarray, fit_indices: np.ndarray, task: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    params = dict(task["cluster_params"])
    if _use_cuml(cfg):
        from cuml.cluster.hdbscan import HDBSCAN, approximate_predict

        model = HDBSCAN(
            min_cluster_size=int(params["min_cluster_size"]),
            min_samples=int(params["min_samples"]),
            cluster_selection_method=str(params["cluster_selection_method"]),
            alpha=float(params["alpha"]),
            metric="euclidean",
            prediction_data=True,
        )
        fit_labels = model.fit_predict(features[fit_indices])
        labels, _strength = approximate_predict(model, features)
        persistence = getattr(model, "cluster_persistence_", None)
        persistence_np = _to_numpy(persistence) if persistence is not None else np.asarray([])
        return (
            _to_numpy(labels),
            _to_numpy(fit_labels),
            {
                "backend": "cuml",
                "mean_cluster_persistence": float(np.mean(persistence_np))
                if persistence_np.size
                else float("nan"),
            },
        )

    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise RuntimeError("CPU fallback for HDBSCAN requires sklearn.cluster.HDBSCAN.") from exc
    model = HDBSCAN(
        min_cluster_size=int(params["min_cluster_size"]),
        min_samples=int(params["min_samples"]),
        cluster_selection_method=str(params["cluster_selection_method"]),
        alpha=float(params["alpha"]),
    )
    fit_labels = model.fit_predict(features[fit_indices])
    labels = np.full(features.shape[0], -1, dtype=np.int64)
    labels[fit_indices] = fit_labels
    return labels, fit_labels, {"backend": "sklearn_no_approximate_predict"}


def _run_leiden(
    cfg: ExperimentConfig, features: np.ndarray, fit_indices: np.ndarray, task: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not _use_cuml(cfg):
        raise RuntimeError("Leiden clustering requires RAPIDS cuGraph in the production runtime.")

    params = dict(task["cluster_params"])
    seed = int(task["cluster_seed"])
    n_neighbors = int(params["n_neighbors"])
    if n_neighbors < 1:
        raise ValueError(f"Leiden n_neighbors must be >= 1, got {n_neighbors}.")

    import cudf
    import cugraph
    import cupy as cp
    from cuml.neighbors import NearestNeighbors

    metric = str(params.get("metric", "cosine"))
    gpu_features = cp.asarray(features)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nn.fit(gpu_features)
    distances, indices = nn.kneighbors(gpu_features)
    distances = cp.asarray(distances)
    indices = cp.asarray(indices)

    vertices = cp.arange(features.shape[0], dtype=cp.int64)[:, None]
    keep = indices != vertices
    src = cp.broadcast_to(vertices, indices.shape)[keep].astype(cp.int64)
    dst = indices[keep].astype(cp.int64)
    dist = distances[keep].astype(cp.float32)
    if src.size == 0:
        raise RuntimeError("Leiden kNN graph has no edges.")

    # Convert distances to positive similarities for weighted community detection.
    if metric == "cosine":
        weights = cp.maximum(1.0 - dist, 1e-6).astype(cp.float32)
    else:
        weights = (1.0 / (1.0 + cp.maximum(dist, 0.0))).astype(cp.float32)

    edge_df = cudf.DataFrame({"src": src, "dst": dst, "weight": weights})
    graph = cugraph.Graph(directed=False)
    graph.from_cudf_edgelist(edge_df, source="src", destination="dst", edge_attr="weight")
    parts, modularity_score = cugraph.leiden(
        graph,
        max_iter=int(params.get("max_iter", 100)),
        resolution=float(params.get("resolution", 1.0)),
        random_state=seed,
        theta=float(params.get("theta", 1.0)),
    )
    parts = parts.sort_values("vertex")
    labels = parts["partition"].to_numpy()
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape[0] != features.shape[0]:
        full_labels = np.full(features.shape[0], -1, dtype=np.int64)
        vertices_np = np.asarray(parts["vertex"].to_numpy(), dtype=np.int64)
        full_labels[vertices_np] = labels
        labels = full_labels
    return labels, labels[fit_indices], {"backend": "cugraph", "modularity_score": float(modularity_score)}


def _compute_cluster_metrics(
    cfg: ExperimentConfig, data: EmbeddingData, labels: np.ndarray, fit_indices: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    non_noise = labels != -1
    n_clusters = int(np.unique(labels[non_noise]).shape[0]) if np.any(non_noise) else 0
    noise_fraction = float(np.mean(~non_noise))
    metric_indices = deterministic_sample_indices(
        data.n_samples,
        min(cfg.clustering.metric_subsample, data.n_samples),
        0,
    )
    metric_features = data.features[metric_indices]
    metric_labels = labels[metric_indices]
    silhouette = _safe_silhouette(metric_features, metric_labels)
    sd_metrics = sd_index(
        metric_features,
        metric_labels,
        alpha=float(cfg.clustering.sd_index_alpha),
    )
    label_ari = float("nan")
    if data.labels is not None and len(data.labels) == len(labels):
        label_ari = float(adjusted_rand_score(data.labels, labels))
    return {
        "n_clusters": n_clusters,
        "noise_fraction": noise_fraction,
        "silhouette": silhouette,
        "sd_index": sd_metrics.sd_index,
        "sd_scat": sd_metrics.sd_scat,
        "sd_dis": sd_metrics.sd_dis,
        "label_ari": label_ari,
        "valid_cluster_count": bool(n_clusters >= 2),
        "fit_noise_fraction": float(np.mean(np.asarray(labels[fit_indices]) == -1)),
    }


def _safe_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    mask = labels != -1
    labels = labels[mask]
    if labels.shape[0] < 3 or np.unique(labels).shape[0] < 2:
        return float("nan")
    try:
        return float(silhouette_score(features[mask], labels, metric="euclidean"))
    except ValueError:
        return float("nan")


def _use_cuml(cfg: ExperimentConfig) -> bool:
    if cfg.runtime.allow_cpu_fallback:
        try:
            import cuml  # noqa: F401

            return True
        except ImportError:
            return False
    return bool(cfg.runtime.require_cuml)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)


def _l2_normalize_numpy(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return features / norms


def _l2_normalize_cupy(features: Any) -> Any:
    import cupy as cp

    features = features.astype(cp.float32, copy=False)
    norms = cp.linalg.norm(features, axis=1, keepdims=True)
    norms = cp.where(norms > 0, norms, 1.0)
    return features / norms
