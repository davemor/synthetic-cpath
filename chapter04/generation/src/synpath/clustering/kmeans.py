from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score
from sklearn.metrics import silhouette_score

from synpath.artifacts import utc_now, write_json, write_yaml
from synpath.embeddings.uni import feature_matrix, load_embedding_frame
from synpath.manifests import write_manifest
from synpath.paths import SynpathPaths


@dataclass(frozen=True)
class KSelection:
    k: int
    silhouette: float
    davies_bouldin: float
    stability: float
    score: float


def run_clustering(
    dataset: str,
    paths: SynpathPaths,
    cfg: dict,
) -> list[Path]:
    frame = load_embedding_frame(
        paths.cache_root
        / "embeddings"
        / dataset
        / "cluster_pool_uni_224.parquet"
    )
    x = feature_matrix(frame)
    cluster_cfg = cfg.get("pseudo_labels", {})
    seeds = cfg.get("seeds", {}).get("clustering", [0])
    min_cluster_size = int(cluster_cfg.get("min_cluster_size", 50))

    root = paths.cache_root / "clusters" / dataset
    root.mkdir(parents=True, exist_ok=True)

    global_candidates = [
        k
        for k in cluster_cfg.get("k_candidates", [4, 8, 16])
        if k > 1 and k < len(frame)
    ]
    global_selection = select_k(x, global_candidates, seeds)
    global_model = fit_model(x, global_selection.k, seeds[0])
    global_labels = global_model.labels_
    pseudo_rows = [
        {"image_path": row.image_path, "cluster_id": int(label)}
        for row, label in zip(frame.itertuples(), global_labels, strict=True)
    ]
    pseudo_path = root / "pseudo_labels.csv"
    write_manifest(pseudo_path, pseudo_rows, ["image_path", "cluster_id"])

    local_path = root / "label_local_cluster_labels.csv"
    local_rows, local_models = local_cluster_rows(
        frame, x, cluster_cfg, seeds, min_cluster_size
    )
    write_manifest(
        local_path,
        local_rows,
        ["image_path", "cluster_id"],
    )

    global_model_path = root / "global_kmeans_model.joblib"
    local_model_path = root / "label_local_kmeans_models.joblib"
    joblib.dump(
        {
            "kind": "global",
            "projector": "identity",
            "models": global_model,
            "training_assignment_agreement": 1.0,
        },
        global_model_path,
    )
    joblib.dump(
        {
            "kind": "local",
            "projector": "identity",
            "models": local_models,
            "training_assignment_agreement": 1.0,
        },
        local_model_path,
    )

    selected_path = root / "selected_global_k.yaml"
    write_yaml(
        selected_path,
        {
            "global": global_selection.__dict__,
            "seeds": seeds,
            "k_candidates": global_candidates,
        },
    )
    metadata_path = root / "full_pool_complete.json"
    write_json(
        metadata_path,
        {
            "created_at": utc_now(),
            "dataset": dataset,
            "embedding_path": str(
                paths.cache_root
                / "embeddings"
                / dataset
                / "cluster_pool_uni_224.parquet"
            ),
            "n_samples": len(frame),
            "clustering_population": "complete_train_split",
            "held_out_splits_excluded": True,
        },
    )
    return [
        selected_path,
        pseudo_path,
        local_path,
        global_model_path,
        local_model_path,
        metadata_path,
    ]


def select_k(
    x: np.ndarray, candidates: list[int], seeds: list[int]
) -> KSelection:
    if not candidates:
        return KSelection(
            k=1,
            silhouette=0.0,
            davies_bouldin=0.0,
            stability=1.0,
            score=0.0,
        )
    sample_x, sample_idx = metric_sample(x)
    selections = []
    for k in candidates:
        labels = fit_labels(x, k, seeds[0])
        sample_labels = labels[sample_idx]
        silhouette = safe_silhouette(sample_x, sample_labels)
        db = safe_davies(sample_x, sample_labels)
        stability = clustering_stability(x, k, seeds)
        score = silhouette - normalise_db(db) + stability
        selections.append(KSelection(k, silhouette, db, stability, score))
    return max(selections, key=lambda item: item.score)


def metric_sample(
    x: np.ndarray, max_rows: int = 10000
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_rows:
        idx = np.arange(len(x))
        return x, idx
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), size=max_rows, replace=False)
    return x[idx], idx


def fit_labels(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    return fit_model(x, k, seed).labels_


def fit_model(x: np.ndarray, k: int, seed: int) -> KMeans:
    if k <= 1:
        k = 1
    model = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    return model.fit(x)


def safe_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return 0.0
    return float(silhouette_score(x, labels))


def safe_davies(x: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return 0.0
    return float(davies_bouldin_score(x, labels))


def normalise_db(value: float) -> float:
    if value <= 0:
        return 0.0
    return value / (1.0 + value)


def clustering_stability(x: np.ndarray, k: int, seeds: list[int]) -> float:
    if len(seeds) < 2 or k <= 1:
        return 1.0
    base = fit_labels(x, k, seeds[0])
    scores = [
        adjusted_rand_score(base, fit_labels(x, k, seed)) for seed in seeds[1:]
    ]
    return float(np.mean(scores)) if scores else 1.0


def local_cluster_rows(
    frame: pd.DataFrame,
    x: np.ndarray,
    cluster_cfg: dict,
    seeds: list[int],
    min_cluster_size: int,
) -> tuple[list[dict[str, object]], dict[str, KMeans]]:
    candidates = cluster_cfg.get("label_local_cluster_k_candidates", [2, 4, 8])
    rows = []
    models = {}
    labels = frame["label"].astype(str).to_numpy()
    for label in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == label)
        max_k = max(1, len(idx) // min_cluster_size)
        valid = [k for k in candidates if 1 < k <= max_k and k < len(idx)]
        k = select_k(x[idx], valid, seeds).k if valid else 1
        model = fit_model(x[idx], k, seeds[0])
        models[label] = model
        cluster_ids = model.labels_
        for row_idx, cluster_id in zip(idx, cluster_ids, strict=True):
            rows.append(
                {
                    "image_path": frame.iloc[row_idx]["image_path"],
                    "cluster_id": int(cluster_id),
                }
            )
    return rows, models
