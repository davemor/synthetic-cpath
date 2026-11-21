from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import ExperimentConfig


@dataclass
class Representation:
    train: np.ndarray
    test: np.ndarray
    valid: np.ndarray | None
    reconstructed_test: np.ndarray
    model: Any
    metadata: dict[str, Any]


def representation_cache_dir(cfg: ExperimentConfig) -> Path:
    return cfg.paths.run_dir / "projection_cache"


def build_representation(
    cfg: ExperimentConfig,
    projector: str,
    dimension: int | None,
    seed: int,
    x_train: np.ndarray,
    x_test: np.ndarray,
    x_valid: np.ndarray | None,
    cache_key: str,
    force: bool = False,
) -> Representation:
    if projector == "raw_highdim":
        return Representation(
            train=x_train,
            test=x_test,
            valid=x_valid,
            reconstructed_test=x_test,
            model=None,
            metadata={"backend": "identity", "projector": projector},
        )

    cache_dir = representation_cache_dir(cfg) / cache_key
    arrays_path = cache_dir / "arrays.npz"
    model_path = cache_dir / "model.pkl"
    if arrays_path.exists() and model_path.exists() and not force:
        with np.load(arrays_path) as data:
            valid = data["valid"] if "valid" in data.files else None
            return Representation(
                train=data["train"],
                test=data["test"],
                valid=valid,
                reconstructed_test=data["reconstructed_test"],
                model=_load_pickle(model_path),
                metadata={"cache_hit": True, "projector": projector},
            )

    if dimension is None:
        raise ValueError(f"{projector} requires a dimension")
    if projector == "pca":
        rep = _fit_pca(int(dimension), seed, x_train, x_test, x_valid)
    elif projector == "umap":
        rep = _fit_umap(cfg, int(dimension), seed, x_train, x_test, x_valid)
    else:
        raise ValueError(f"Unsupported projector {projector!r}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "train": rep.train.astype(np.float32, copy=False),
        "test": rep.test.astype(np.float32, copy=False),
        "reconstructed_test": rep.reconstructed_test.astype(np.float32, copy=False),
    }
    if rep.valid is not None:
        arrays["valid"] = rep.valid.astype(np.float32, copy=False)
    np.savez_compressed(arrays_path, **arrays)
    _save_pickle(model_path, rep.model)
    return rep


def inverse_transform(model: Any, z: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    if model is None:
        return z.astype(np.float32, copy=False)
    if hasattr(model, "inverse_transform"):
        return np.asarray(model.inverse_transform(z), dtype=np.float32)
    if isinstance(model, dict) and model.get("kind") == "cuml_pca":
        return (z @ model["components"] + model["mean"]).astype(np.float32)
    if isinstance(model, dict) and model.get("kind") == "knn_decoder":
        if reference is None or not isinstance(reference, tuple):
            raise RuntimeError("kNN decoder requires reference=(z_train, x_train)")
        z_train, x_train = reference
        return _knn_decode(z, z_train, x_train, int(model.get("n_neighbors", 15)))
    raise RuntimeError(f"Model of type {type(model).__name__} has no inverse_transform")


def _fit_pca(
    dimension: int,
    seed: int,
    x_train: np.ndarray,
    x_test: np.ndarray,
    x_valid: np.ndarray | None,
) -> Representation:
    try:
        import cupy as cp
        from cuml import PCA

        model = PCA(n_components=dimension)
        z_train = model.fit_transform(cp.asarray(x_train))
        z_test = model.transform(cp.asarray(x_test))
        z_valid = model.transform(cp.asarray(x_valid)) if x_valid is not None else None
        pca_model = {
            "kind": "cuml_pca",
            "components": cp.asnumpy(cp.asarray(model.components_)).astype(np.float32),
            "mean": cp.asnumpy(cp.asarray(model.mean_)).astype(np.float32),
        }
        z_train_np = cp.asnumpy(z_train).astype(np.float32)
        z_test_np = cp.asnumpy(z_test).astype(np.float32)
        z_valid_np = cp.asnumpy(z_valid).astype(np.float32) if z_valid is not None else None
        x_recon = inverse_transform(pca_model, z_test_np)
        return Representation(
            z_train_np,
            z_test_np,
            z_valid_np,
            x_recon,
            pca_model,
            {"backend": "cuml", "seed": seed, "projector": "pca"},
        )
    except Exception:
        from sklearn.decomposition import PCA

        model = PCA(n_components=dimension, random_state=seed)
        z_train = model.fit_transform(x_train)
        z_test = model.transform(x_test)
        z_valid = model.transform(x_valid) if x_valid is not None else None
        x_recon = model.inverse_transform(z_test)
        return Representation(
            z_train.astype(np.float32),
            z_test.astype(np.float32),
            z_valid.astype(np.float32) if z_valid is not None else None,
            x_recon.astype(np.float32),
            model,
            {"backend": "sklearn", "seed": seed, "projector": "pca"},
        )


def _fit_umap(
    cfg: ExperimentConfig,
    dimension: int,
    seed: int,
    x_train: np.ndarray,
    x_test: np.ndarray,
    x_valid: np.ndarray | None,
) -> Representation:
    kwargs = {
        "n_components": dimension,
        "n_neighbors": cfg.projection.umap_n_neighbors,
        "min_dist": cfg.projection.umap_min_dist,
        "metric": cfg.projection.umap_metric,
        "random_state": seed,
    }
    try:
        import cupy as cp
        from cuml.manifold import UMAP

        model = UMAP(**kwargs)
        model.fit(cp.asarray(x_train))
        z_train = cp.asnumpy(model.transform(cp.asarray(x_train))).astype(np.float32)
        z_test = cp.asnumpy(model.transform(cp.asarray(x_test))).astype(np.float32)
        z_valid = (
            cp.asnumpy(model.transform(cp.asarray(x_valid))).astype(np.float32) if x_valid is not None else None
        )
        decoder = {"kind": "knn_decoder", "n_neighbors": min(15, len(z_train))}
        x_recon = _knn_decode(z_test, z_train, x_train, decoder["n_neighbors"])
        return Representation(z_train, z_test, z_valid, x_recon, decoder, {"backend": "cuml", **kwargs})
    except Exception as exc:
        try:
            import umap
        except ImportError as import_exc:
            if cfg.runtime.require_cuml:
                raise RuntimeError("UMAP requires cuML in this configuration.") from exc
            raise RuntimeError("UMAP requires cuML or umap-learn.") from import_exc
        model = umap.UMAP(**kwargs)
        model.fit(x_train)
        z_train = model.transform(x_train).astype(np.float32)
        z_test = model.transform(x_test).astype(np.float32)
        z_valid = model.transform(x_valid).astype(np.float32) if x_valid is not None else None
        decoder = {"kind": "knn_decoder", "n_neighbors": min(15, len(z_train))}
        x_recon = _knn_decode(z_test, z_train, x_train, decoder["n_neighbors"])
        return Representation(z_train, z_test, z_valid, x_recon, decoder, {"backend": "umap-learn", **kwargs})


def _knn_decode(
    z_query: np.ndarray,
    z_train: np.ndarray,
    x_train: np.ndarray,
    n_neighbors: int = 15,
    chunk_size: int = 10000,
) -> np.ndarray:
    k = min(int(n_neighbors), len(z_train))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(z_train)
    parts = []
    for start in range(0, len(z_query), chunk_size):
        stop = min(start + chunk_size, len(z_query))
        distances, indices = nn.kneighbors(z_query[start:stop], return_distance=True)
        weights = 1.0 / np.maximum(distances, 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)
        decoded = np.einsum("bk,bkd->bd", weights, x_train[indices])
        parts.append(decoded.astype(np.float32))
    return np.concatenate(parts, axis=0)


def _save_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)
