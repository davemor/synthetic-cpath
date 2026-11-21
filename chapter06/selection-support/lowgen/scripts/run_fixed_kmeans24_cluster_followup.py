#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lowgen.dgx.cluster_conditioning import run_cluster_conditioning_audit
from lowgen.dgx.config import ExperimentConfig
from lowgen.dgx.data import load_dataset
from lowgen.dgx.manifest import build_task_manifest, manifest_path
from lowgen.dgx.projections import build_representation, representation_cache_dir
from lowgen.dgx.utils import atomic_write_json, stable_hash


DEFAULT_CONFIG = REPO_ROOT / "config" / "cluster_kmeans24_followup.yml"
KMEANS_PARAMS = {
    "n_clusters": 24,
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 0,
}


@dataclass(frozen=True)
class FixedClusterSpec:
    dataset: str
    display_dataset: str
    encoder: str
    projector: str
    dimension: int
    clusterer: str
    scope: str


def fixed_cluster_specs() -> list[FixedClusterSpec]:
    return [
        FixedClusterSpec(
            dataset="icaird100k",
            display_dataset="iCAIRD Cervical 100K",
            encoder="uni2",
            projector="pca",
            dimension=42,
            clusterer="kmeans",
            scope="global",
        ),
        FixedClusterSpec(
            dataset="kather100k",
            display_dataset="Kather / PathMNIST224",
            encoder="uni2",
            projector="pca",
            dimension=50,
            clusterer="kmeans",
            scope="global",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed KMeans-24 cluster-conditioning follow-up."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force-inputs", action="store_true")
    parser.add_argument("--run-audit", action="store_true")
    parser.add_argument("--force-audit", action="store_true")
    parser.add_argument("--synthetic-total", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--dask-chunk-size", type=int, default=1)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    input_dir = args.input_dir or cfg.paths.run_dir / "fixed_kmeans24_inputs"
    output_dir = args.output_dir or cfg.paths.run_dir / "fixed_kmeans24_cluster_conditioning"
    clustering_results = generate_fixed_kmeans_inputs(
        cfg,
        input_dir=input_dir,
        force=args.force_inputs,
    )
    outputs: dict[str, str] = {"clustering_results": str(clustering_results)}
    if args.run_audit:
        audit_outputs = run_cluster_conditioning_audit(
            cfg,
            clustering_results_path=clustering_results,
            output_dir=output_dir,
            manifest=manifest_path(cfg),
            force=args.force_audit,
            synthetic_total=args.synthetic_total,
            n_workers=args.n_workers,
            dask_chunk_size=args.dask_chunk_size,
        )
        outputs.update({name: str(path) for name, path in audit_outputs.items()})
    print(json.dumps(outputs, indent=2, sort_keys=True))


def generate_fixed_kmeans_inputs(
    cfg: ExperimentConfig,
    input_dir: Path,
    force: bool = False,
) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = input_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    clustering_results_path = input_dir / "clustering_results_fixed_kmeans24.csv"
    metadata_path = input_dir / "metadata.json"
    if clustering_results_path.exists() and not force:
        return clustering_results_path

    tasks = build_task_manifest(cfg)
    task_index = {
        (task["dataset"], task["encoder"], task["projector"], str(task["dimension"]), int(task["seed"])): task
        for task in tasks
    }
    rows = []
    for spec in fixed_cluster_specs():
        key = (spec.dataset, spec.encoder, spec.projector, str(spec.dimension), 0)
        if key not in task_index:
            raise KeyError(f"No manifest task for fixed cluster spec {key}")
        rows.append(_cluster_spec_row(cfg, spec, task_index[key], labels_dir))

    pd.DataFrame(rows).to_csv(clustering_results_path, index=False)
    atomic_write_json(
        metadata_path,
        {
            "description": "Fixed global KMeans-24 cluster-conditioning follow-up inputs.",
            "kmeans_params": KMEANS_PARAMS,
            "specs": [spec.__dict__ for spec in fixed_cluster_specs()],
            "clustering_results": str(clustering_results_path),
            "manifest": str(manifest_path(cfg)),
        },
    )
    return clustering_results_path


def _cluster_spec_row(
    cfg: ExperimentConfig,
    spec: FixedClusterSpec,
    task: dict[str, Any],
    labels_dir: Path,
) -> dict[str, Any]:
    started = perf_counter()
    data = load_dataset(cfg, spec.dataset, spec.encoder)
    rep = build_representation(
        cfg,
        spec.projector,
        spec.dimension,
        int(task["seed"]),
        data.train.features,
        data.test.features,
        data.valid.features if data.valid is not None else None,
        str(task["projection_cache_key"]),
        force=False,
    )
    train_labels, test_labels, kmeans_metadata = fit_predict_kmeans24(rep.train, rep.test)
    labels_path = labels_dir / f"{task['task_id']}_kmeans24_labels.npz"
    np.savez_compressed(
        labels_path,
        labels=np.concatenate([train_labels, test_labels]).astype(np.int64),
        sample_ids=np.concatenate([data.train.sample_ids, data.test.sample_ids]).astype(str),
    )
    arrays_path = representation_cache_dir(cfg) / str(task["projection_cache_key"]) / "arrays.npz"
    cluster_params_json = json.dumps(KMEANS_PARAMS, sort_keys=True)
    return {
        "backend": kmeans_metadata["backend"],
        "cluster_grid_size": 1,
        "cluster_params": str(KMEANS_PARAMS),
        "cluster_params_json": cluster_params_json,
        "cluster_seed": KMEANS_PARAMS["random_state"],
        "clusterer": spec.clusterer,
        "condition": spec.scope,
        "config_id": stable_hash(KMEANS_PARAMS, length=12),
        "dataset": spec.dataset,
        "display_dataset": spec.display_dataset,
        "dimension": int(spec.dimension),
        "encoder": spec.encoder,
        "fit_noise_fraction": 0.0,
        "fit_subsample": int(len(rep.train)),
        "inertia": kmeans_metadata.get("inertia"),
        "label_ari": adjusted_rand(rep.train, data.train.labels, train_labels),
        "labels_path": str(labels_path),
        "n_clusters": int(np.unique(train_labels).shape[0]),
        "n_samples": int(len(rep.train)),
        "noise_fraction": 0.0,
        "projection_cache_key": task["projection_cache_key"],
        "projection_path": str(arrays_path),
        "projection_seed": int(task["seed"]),
        "projector": spec.projector,
        "runtime_seconds": float(perf_counter() - started),
        "silhouette": silhouette_sample(rep.train, train_labels),
        "source_dimension": int(task["source_dimension"]),
        "source_metric": "fixed_kmeans24_followup",
        "target_name": task["target_name"],
        "task_id": task["task_id"],
        "valid_cluster_count": True,
    }


def fit_predict_kmeans24(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import cupy as cp
        from cuml.cluster import KMeans

        model = KMeans(**KMEANS_PARAMS)
        train_gpu = cp.asarray(train)
        test_gpu = cp.asarray(test)
        train_labels = cp.asnumpy(model.fit_predict(train_gpu)).astype(np.int64)
        test_labels = cp.asnumpy(model.predict(test_gpu)).astype(np.int64)
        inertia = float(cp.asnumpy(cp.asarray(model.inertia_)))
        return train_labels, test_labels, {"backend": "cuml", "inertia": inertia}
    except Exception:
        from sklearn.cluster import KMeans

        model = KMeans(**KMEANS_PARAMS)
        train_labels = model.fit_predict(train).astype(np.int64)
        test_labels = model.predict(test).astype(np.int64)
        return train_labels, test_labels, {"backend": "sklearn", "inertia": float(model.inertia_)}


def adjusted_rand(features: np.ndarray, true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    del features
    try:
        from sklearn.metrics import adjusted_rand_score

        return float(adjusted_rand_score(true_labels, cluster_labels))
    except Exception:
        return float("nan")


def silhouette_sample(features: np.ndarray, cluster_labels: np.ndarray, max_samples: int = 5000) -> float:
    try:
        from sklearn.metrics import silhouette_score

        n = min(int(max_samples), len(features))
        rng = np.random.default_rng(0)
        indices = np.sort(rng.choice(len(features), size=n, replace=False)) if n < len(features) else np.arange(n)
        sampled_labels = cluster_labels[indices]
        if np.unique(sampled_labels).shape[0] < 2:
            return float("nan")
        return float(silhouette_score(features[indices], sampled_labels, metric="euclidean"))
    except Exception:
        return float("nan")


if __name__ == "__main__":
    main()
