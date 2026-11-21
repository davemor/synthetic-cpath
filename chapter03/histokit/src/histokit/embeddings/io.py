"""Embedding result IO and PatchSet discovery."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .encoder import EmbeddingResult

FEATURES_NPZ = "features.npz"
LEGACY_FEATURES_NPY = "features.npy"
INDEX_PARQUET = "index.parquet"
MANIFEST_JSON = "manifest.json"
ALIGNMENT_COLUMNS = ("frame_index", "row", "column", "context_id")


def is_patchset_dir(path: Path | str) -> bool:
    path = Path(path)
    return (path / "frame.parquet").exists() and (
        path / "manifest.json"
    ).exists()


def discover_patchset_dirs(path: Path | str) -> list[Path]:
    """Return one PatchSet dir, or immediate child ``*.patchset`` dirs."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"PatchSet path not found: {root}")
    if is_patchset_dir(root):
        return [root]
    if not root.is_dir():
        raise ValueError(f"PatchSet input is not a directory: {root}")

    patchsets = [
        child
        for child in sorted(root.iterdir(), key=lambda p: p.name)
        if child.is_dir() and is_patchset_dir(child)
    ]
    if not patchsets:
        raise ValueError(f"No PatchSet directories found in {root}")
    return patchsets


def write_embedding_result(
    result: EmbeddingResult,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> None:
    """Write embeddings as a small sidecar directory."""
    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Embedding output already exists: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    path.mkdir(parents=True, exist_ok=True)
    arrays = _embedding_npz_arrays(result)
    np.savez_compressed(path / FEATURES_NPZ, **arrays)
    result.index.to_parquet(path / INDEX_PARQUET, index=False)

    manifest = dict(result.manifest)
    manifest["storage"] = {
        "format": "npz+parquet",
        "features": FEATURES_NPZ,
        "index": INDEX_PARQUET,
        "dtype": "float32",
        "alignment_arrays": [
            column for column in ALIGNMENT_COLUMNS if column in arrays
        ],
    }
    with open(path / MANIFEST_JSON, "w") as file:
        json.dump(manifest, file, indent=2)


def read_embedding_result(path: Path | str) -> EmbeddingResult:
    """Read an embedding result directory.

    The standard format is ``features.npz`` plus parquet/json sidecars. Legacy
    ``features.npy`` directories remain readable for older outputs.
    """
    path = Path(path)
    index_path = path / INDEX_PARQUET
    manifest_path = path / MANIFEST_JSON
    npz_path = path / FEATURES_NPZ
    legacy_npy_path = path / LEGACY_FEATURES_NPY

    if not index_path.exists():
        raise FileNotFoundError(f"Embedding index not found: {index_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Embedding manifest not found: {manifest_path}")

    if npz_path.exists():
        with np.load(npz_path) as archive:
            if "features" not in archive:
                raise KeyError(f"{npz_path} does not contain a 'features' array")
            features = np.asarray(archive["features"])
    elif legacy_npy_path.exists():
        features = np.load(legacy_npy_path)
    else:
        raise FileNotFoundError(
            f"Embedding features not found: expected {npz_path} or "
            f"{legacy_npy_path}"
        )

    index = pd.read_parquet(index_path)
    with open(manifest_path) as file:
        manifest = json.load(file)
    return EmbeddingResult(features=features, index=index, manifest=manifest)


def _embedding_npz_arrays(result: EmbeddingResult) -> dict[str, np.ndarray]:
    features = np.asarray(result.features, dtype=np.float32)
    arrays: dict[str, np.ndarray] = {"features": features}

    for column in ALIGNMENT_COLUMNS:
        if column not in result.index.columns:
            continue
        series = result.index[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        arrays[column] = series.to_numpy()
    return arrays


def safe_model_name(model: str) -> str:
    """Make a model reference safe for filenames."""
    safe = model.replace("/", "_").replace(":", "_")
    safe = safe.replace(" ", "_")
    return "".join(ch for ch in safe if ch.isalnum() or ch in "._-")
