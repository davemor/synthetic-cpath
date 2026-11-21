from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import ExperimentConfig
from .dimensions import ProjectionTarget, load_projection_targets
from .embeddings import EmbeddingData, load_embedding_dataset, source_fingerprint
from .utils import atomic_save_npz, atomic_write_json, atomic_write_jsonl, stable_hash


def projection_cache_dir(cfg: ExperimentConfig) -> Path:
    return cfg.paths.run_dir / "projection_cache"


def projection_cache_key(
    cfg: ExperimentConfig,
    target: ProjectionTarget,
    embedding_fingerprint: dict[str, Any] | None = None,
) -> str:
    payload = {
        "version": 1,
        "target": asdict(target),
        "umap": cfg.projection.umap.params() | {"fit_subsample": cfg.projection.umap.fit_subsample},
        "fingerprint": embedding_fingerprint
        or source_fingerprint(cfg.paths.embeddings_root, target.dataset, target.encoder),
    }
    return stable_hash(payload, length=20)


def projection_cache_path(cfg: ExperimentConfig, cache_key: str) -> Path:
    return projection_cache_dir(cfg) / f"{cache_key}.npz"


def prepare_projection_cache(cfg: ExperimentConfig, force: bool = False) -> pd.DataFrame:
    manifest = build_projection_manifest(cfg)
    task_records = [
        row
        for row in manifest.to_dict(orient="records")
        if row["projector"] != "raw"
    ]
    for record in tqdm(task_records, desc="Projection tasks"):
        run_projection_task(cfg, record, force=force)
    manifest = build_projection_manifest(cfg)
    return manifest


def build_projection_manifest(
    cfg: ExperimentConfig, task_manifest_path: Path | None = None
) -> pd.DataFrame:
    targets = load_projection_targets(cfg)
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[ProjectionTarget]] = {}
    for target in targets:
        grouped.setdefault((target.dataset, target.encoder), []).append(target)

    for (dataset, encoder), dataset_targets in tqdm(grouped.items(), desc="Projection manifest"):
        embeddings = load_embedding_dataset(cfg.paths.embeddings_root, dataset, encoder)
        fingerprint = source_fingerprint(cfg.paths.embeddings_root, dataset, encoder)
        for target in dataset_targets:
            cache_key = projection_cache_key(cfg, target, fingerprint)
            path = projection_cache_path(cfg, cache_key)
            row = _manifest_row(target, cache_key, path, embeddings, fingerprint)
            if target.projector == "raw":
                row["cache_status"] = "logical_raw"
            elif path.exists():
                row["cache_status"] = "hit"
            else:
                row["cache_status"] = "pending"
            rows.append(row)

    manifest = pd.DataFrame(rows)
    projection_cache_dir(cfg).mkdir(parents=True, exist_ok=True)
    manifest.to_csv(projection_cache_dir(cfg) / "manifest.csv", index=False)
    atomic_write_json(
        projection_cache_dir(cfg) / "targets.json",
        {"targets": [asdict(target) for target in targets]},
    )
    if task_manifest_path is not None:
        task_records = [
            _projection_task_record(row)
            for row in manifest.to_dict(orient="records")
            if row["projector"] != "raw"
        ]
        atomic_write_jsonl(task_manifest_path, task_records)
    return manifest


def run_projection_task(
    cfg: ExperimentConfig, record: dict[str, Any], force: bool = False
) -> dict[str, Any]:
    target = _target_from_record(record)
    cache_key = str(record.get("projection_cache_key") or projection_cache_key(cfg, target))
    path = projection_cache_path(cfg, cache_key)
    if path.exists() and not force:
        return {
            **record,
            "projection_cache_key": cache_key,
            "projection_path": str(path),
            "cache_status": "hit",
        }
    embeddings = load_embedding_dataset(cfg.paths.embeddings_root, target.dataset, target.encoder)
    fingerprint = source_fingerprint(cfg.paths.embeddings_root, target.dataset, target.encoder)
    started = perf_counter()
    projected, fit_metadata = fit_transform_projection(cfg, embeddings, target)
    atomic_save_npz(
        path,
        features=projected.astype(np.float32, copy=False),
        labels=_labels_for_npz(embeddings),
        sample_ids=embeddings.sample_ids.astype(str),
        splits=embeddings.splits.astype(str),
        source_paths=embeddings.source_paths.astype(str),
    )
    runtime = perf_counter() - started
    atomic_write_json(
        path.with_suffix(".json"),
        {
            "cache_key": cache_key,
            "target": asdict(target),
            "runtime_seconds": runtime,
            "fit_metadata": fit_metadata,
            "source_fingerprint": fingerprint,
        },
    )
    return {
        **record,
        "projection_cache_key": cache_key,
        "projection_path": str(path),
        "cache_status": "written",
        "runtime_seconds": runtime,
    }


def fit_transform_projection(
    cfg: ExperimentConfig, embeddings: EmbeddingData, target: ProjectionTarget
) -> tuple[np.ndarray, dict[str, Any]]:
    if target.dimension is None:
        raise ValueError("Raw projection target does not need fit_transform_projection.")
    if target.dimension < 1:
        raise ValueError(f"Projection dimension must be >= 1, got {target.dimension}.")
    if target.dimension > embeddings.n_dimensions:
        raise ValueError(
            f"Projection dimension {target.dimension} exceeds source dimension {embeddings.n_dimensions}."
        )
    seed = int(target.seed or 0)
    if target.projector == "pca":
        return _fit_transform_pca(cfg, embeddings.features, target.dimension, seed)
    if target.projector == "umap":
        return _fit_transform_umap(cfg, embeddings.features, target.dimension, seed)
    raise ValueError(f"Unsupported projector {target.projector!r}.")


def load_projection_features(cfg: ExperimentConfig, record: dict[str, Any]) -> EmbeddingData:
    if record["projector"] == "raw":
        return load_embedding_dataset(cfg.paths.embeddings_root, record["dataset"], record["encoder"])
    path = Path(record["projection_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Projection cache artifact missing for task {record.get('task_id')}: {path}"
        )
    with np.load(path, allow_pickle=True) as data:
        labels = np.asarray(data["labels"]) if "labels" in data.files else None
        return EmbeddingData(
            features=np.asarray(data["features"], dtype=np.float32),
            labels=labels,
            sample_ids=np.asarray(data["sample_ids"]).astype(str),
            splits=np.asarray(data["splits"]).astype(str),
            source_paths=np.asarray(data["source_paths"]).astype(str)
            if "source_paths" in data.files
            else np.asarray([""] * len(data["features"])),
            metadata={"cache_key": record["projection_cache_key"]},
        )


def read_projection_manifest(cfg: ExperimentConfig) -> pd.DataFrame:
    path = projection_cache_dir(cfg) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Projection manifest not found at {path}. Run prepare-projections first."
        )
    return pd.read_csv(path)


def _fit_transform_pca(
    cfg: ExperimentConfig, features: np.ndarray, n_components: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    if _use_cuml(cfg):
        import cupy as cp
        from cuml import PCA

        model = PCA(n_components=n_components)
        projected = model.fit_transform(cp.asarray(features))
        return _to_numpy(projected), {
            "backend": "cuml",
            "fit_samples": int(features.shape[0]),
            "seed": int(seed),
        }

    from sklearn.decomposition import PCA

    model = PCA(n_components=n_components, svd_solver="auto")
    projected = model.fit_transform(features)
    return projected, {
        "backend": "sklearn",
        "fit_samples": int(features.shape[0]),
        "seed": int(seed),
    }


def _fit_transform_umap(
    cfg: ExperimentConfig, features: np.ndarray, n_components: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    fit_indices = deterministic_sample_indices(
        len(features), min(cfg.projection.umap.fit_subsample, len(features)), seed
    )
    if _use_cuml(cfg):
        import cupy as cp
        from cuml.manifold import UMAP

        model = UMAP(
            n_components=n_components,
            random_state=seed,
            **cfg.projection.umap.params(),
        )
        gpu_features = cp.asarray(features)
        model.fit(gpu_features[fit_indices])
        projected = model.transform(gpu_features)
        return _to_numpy(projected), {
            "backend": "cuml",
            "fit_samples": int(len(fit_indices)),
            "fit_subsample": int(cfg.projection.umap.fit_subsample),
            "seed": int(seed),
        }

    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("UMAP projection requires cuML or umap-learn.") from exc
    model = umap.UMAP(n_components=n_components, random_state=seed, **cfg.projection.umap.params())
    model.fit(features[fit_indices])
    return model.transform(features), {
        "backend": "umap-learn",
        "fit_samples": int(len(fit_indices)),
        "fit_subsample": int(cfg.projection.umap.fit_subsample),
        "seed": int(seed),
    }


def deterministic_sample_indices(n_samples: int, sample_size: int, seed: int) -> np.ndarray:
    sample_size = min(int(sample_size), int(n_samples))
    rng = np.random.default_rng(int(seed))
    if sample_size >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    return np.sort(rng.choice(n_samples, size=sample_size, replace=False)).astype(np.int64)


def _manifest_row(
    target: ProjectionTarget,
    cache_key: str,
    path: Path,
    embeddings: EmbeddingData,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": target.dataset,
        "encoder": target.encoder,
        "projector": target.projector,
        "dimension": "" if target.dimension is None else target.dimension,
        "target_name": target.target_name,
        "projection_seed": "" if target.seed is None else int(target.seed),
        "source_metric": target.source_metric or "",
        "condition": "" if target.condition is None else target.condition,
        "projection_cache_key": cache_key,
        "projection_path": "" if target.projector == "raw" else str(path),
        "n_samples": embeddings.n_samples,
        "source_dimension": embeddings.n_dimensions,
        "source_fingerprint_hash": stable_hash(fingerprint),
    }


def _projection_task_record(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "dataset": row["dataset"],
        "encoder": row["encoder"],
        "projector": row["projector"],
        "dimension": int(row["dimension"]),
        "target_name": row["target_name"],
        "projection_seed": int(row["projection_seed"]),
        "source_metric": row.get("source_metric", ""),
        "condition": row.get("condition", ""),
        "projection_cache_key": row["projection_cache_key"],
        "projection_path": row["projection_path"],
    }
    payload["task_id"] = stable_hash(payload, length=24)
    return payload


def _target_from_record(record: dict[str, Any]) -> ProjectionTarget:
    condition = record.get("condition", None)
    if condition == "" or condition is None or pd.isna(condition):
        condition_value = None
    else:
        condition_value = float(condition)
    source_metric = record.get("source_metric", None)
    if source_metric == "" or source_metric is None or pd.isna(source_metric):
        source_metric = None
    return ProjectionTarget(
        dataset=str(record["dataset"]),
        encoder=str(record["encoder"]),
        projector=str(record["projector"]),
        dimension=int(float(record["dimension"])),
        target_name=str(record["target_name"]),
        seed=int(float(record["projection_seed"])),
        source_metric=source_metric,
        condition=condition_value,
    )


def _labels_for_npz(embeddings: EmbeddingData) -> np.ndarray:
    if embeddings.labels is None:
        return np.asarray([], dtype=np.int64)
    return embeddings.labels


def _use_cuml(cfg: ExperimentConfig) -> bool:
    if cfg.runtime.allow_cpu_fallback:
        try:
            import cuml  # noqa: F401

            return True
        except ImportError:
            return False
    if cfg.runtime.require_cuml:
        return True
    return False


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)
