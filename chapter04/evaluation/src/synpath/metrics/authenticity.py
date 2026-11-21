from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from synpath.artifacts import file_sha256, write_json


UNI_MODEL = "hf-hub:MahmoodLab/UNI"


@dataclass(frozen=True)
class AuthenticityInputs:
    real_manifest: Path
    real_embeddings: Path
    synthetic_manifest: Path
    synthetic_embeddings: Path
    output: Path
    dataset: str
    model: str
    conditioning: str
    seed: int
    family: str


def require_rapids() -> tuple[Any, Any]:
    try:
        import cupy as cp
        from cuml.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError(
            "UNI Authenticity requires CUDA, CuPy and RAPIDS cuML. "
            "Install SynPath with the 'rapids' extra."
        ) from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("UNI Authenticity requires a visible CUDA GPU")
    return cp, NearestNeighbors


def select_gpu(device_id: int) -> None:
    cp, _ = require_rapids()
    cp.cuda.Device(device_id).use()


def authenticity(
    real: np.ndarray,
    synthetic: np.ndarray,
    batch_size: int = 1024,
    real_neighbour_distances: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Calculate exact Authenticity in Euclidean feature space with cuML."""
    real = _validate_matrix(real, "real")
    synthetic = _validate_matrix(synthetic, "synthetic")
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError("Real and synthetic feature dimensions do not match")
    if len(real) < 2:
        raise ValueError("Authenticity requires at least two real samples")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    cp, NearestNeighbors = require_rapids()
    real_gpu = cp.asarray(real, dtype=cp.float32)
    index = NearestNeighbors(n_neighbors=2, metric="euclidean")
    index.fit(real_gpu)

    if real_neighbour_distances is None:
        d_real = _nearest_other_real(
            index, real_gpu, len(real), batch_size, cp
        )
    else:
        d_real = np.asarray(real_neighbour_distances, dtype=np.float32)
        if d_real.shape != (len(real),) or not np.isfinite(d_real).all():
            raise ValueError("Invalid cached real-neighbour distances")

    authentic_count = 0
    synthetic_count = len(synthetic)
    for start in range(0, synthetic_count, batch_size):
        stop = min(start + batch_size, synthetic_count)
        query = cp.asarray(synthetic[start:stop], dtype=cp.float32)
        distances, indices = index.kneighbors(
            query, n_neighbors=1, return_distance=True
        )
        d_synth = cp.asnumpy(distances[:, 0])
        nearest_real = cp.asnumpy(indices[:, 0]).astype(np.int64)
        authentic_count += int((d_synth > d_real[nearest_real]).sum())

    suspicious_count = synthetic_count - authentic_count
    score = authentic_count / synthetic_count
    return {
        "authenticity": float(score),
        "memorisation_risk": float(1.0 - score),
        "authentic_count": authentic_count,
        "suspicious_count": suspicious_count,
        "real_samples": len(real),
        "synthetic_samples": synthetic_count,
    }


def real_neighbour_distances(
    real: np.ndarray, batch_size: int = 1024
) -> np.ndarray:
    real = _validate_matrix(real, "real")
    if len(real) < 2:
        raise ValueError("Authenticity requires at least two real samples")
    cp, NearestNeighbors = require_rapids()
    real_gpu = cp.asarray(real, dtype=cp.float32)
    index = NearestNeighbors(n_neighbors=2, metric="euclidean")
    index.fit(real_gpu)
    return _nearest_other_real(index, real_gpu, len(real), batch_size, cp)


def _nearest_other_real(index, real_gpu, count: int, batch_size: int, cp):
    output = np.empty(count, dtype=np.float32)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        distances, indices = index.kneighbors(
            real_gpu[start:stop], n_neighbors=2, return_distance=True
        )
        distances = cp.asnumpy(distances)
        indices = cp.asnumpy(indices).astype(np.int64)
        row_indices = np.arange(start, stop)[:, None]
        is_other = indices != row_indices
        if not is_other.any(axis=1).all():
            raise RuntimeError("cuML did not return an other-real neighbour")
        positions = is_other.argmax(axis=1)
        output[start:stop] = distances[
            np.arange(stop - start), positions
        ]
    return output


def evaluate_authenticity(
    inputs: AuthenticityInputs,
    cache_root: Path,
    batch_size: int = 1024,
) -> Path:
    real_frame = validate_embedding_cache(
        inputs.real_manifest, inputs.real_embeddings
    )
    synthetic_frame = validate_embedding_cache(
        inputs.synthetic_manifest, inputs.synthetic_embeddings
    )
    real = _feature_matrix(real_frame)
    synthetic = _feature_matrix(synthetic_frame)
    neighbour_cache, d_real, cache_reused = prepare_neighbour_cache(
        real_frame, cache_root, batch_size
    )

    result = authenticity(
        real,
        synthetic,
        batch_size=batch_size,
        real_neighbour_distances=d_real,
    )
    result.update(
        {
            "backend": "rapids_cuml",
            "batch_size": batch_size,
            "conditioning": inputs.conditioning,
            "dataset": inputs.dataset,
            "distance_metric": "euclidean",
            "family": inputs.family,
            "feature_dimension": real.shape[1],
            "model": inputs.model,
            "nearest_other_real_cache": str(neighbour_cache),
            "nearest_other_real_cache_reused": cache_reused,
            "real_embedding_sha256": file_sha256(inputs.real_embeddings),
            "real_manifest_sha256": file_sha256(inputs.real_manifest),
            "seed": inputs.seed,
            "synthetic_embedding_sha256": file_sha256(
                inputs.synthetic_embeddings
            ),
            "synthetic_manifest_sha256": file_sha256(
                inputs.synthetic_manifest
            ),
            "uni_model": UNI_MODEL,
        }
    )
    write_json(inputs.output, result)
    return inputs.output


def prepare_neighbour_cache(
    real_frame: pd.DataFrame,
    cache_root: Path,
    batch_size: int,
) -> tuple[Path, np.ndarray, bool]:
    population_hash = ordered_population_hash(real_frame)
    path = cache_root / f"{population_hash}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path) as payload:
            distances = payload["distances"]
        if distances.shape != (len(real_frame),):
            raise RuntimeError(f"Invalid real-neighbour cache: {path}")
        return path, distances, True
    distances = real_neighbour_distances(
        _feature_matrix(real_frame), batch_size=batch_size
    )
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, distances=distances)
    temporary.replace(path)
    return path, distances, False


def validate_embedding_cache(
    manifest_path: Path, embedding_path: Path
) -> pd.DataFrame:
    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest: {manifest_path}")
    if not embedding_path.exists():
        raise RuntimeError(f"Missing UNI embeddings: {embedding_path}")
    manifest = pd.read_csv(manifest_path)
    embeddings = pd.read_parquet(embedding_path)
    for name, frame in (("manifest", manifest), ("embeddings", embeddings)):
        if "image_path" not in frame:
            raise RuntimeError(f"{name} has no image_path column")
        if frame["image_path"].duplicated().any():
            raise RuntimeError(f"{name} contains duplicate image paths")
    manifest_paths = manifest["image_path"].astype(str).tolist()
    embedding_paths = embeddings["image_path"].astype(str).tolist()
    if set(manifest_paths) != set(embedding_paths):
        missing = len(set(manifest_paths) - set(embedding_paths))
        extra = len(set(embedding_paths) - set(manifest_paths))
        raise RuntimeError(
            f"Embedding/manifest mismatch: {missing} missing, {extra} extra"
        )
    indexed = embeddings.set_index(embeddings["image_path"].astype(str))
    aligned = indexed.loc[manifest_paths].reset_index(drop=True)
    feature_columns = [col for col in aligned if col.startswith("f")]
    if not feature_columns:
        raise RuntimeError("Embedding cache contains no feature columns")
    features = aligned[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError("Embedding cache contains non-finite values")
    return aligned


def ordered_population_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for path in frame["image_path"].astype(str):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
    digest.update(str(len([c for c in frame if c.startswith("f")])).encode())
    return digest.hexdigest()


def _validate_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = [column for column in frame if column.startswith("f")]
    return frame[columns].to_numpy(dtype=np.float32)


def rows_to_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))
