from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors

from synpath.metrics.authenticity import (
    authenticity,
    ordered_population_hash,
    validate_embedding_cache,
)


def sklearn_authenticity(real: np.ndarray, synthetic: np.ndarray) -> float:
    real_nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(real)
    d_real = real_nn.kneighbors(real)[0][:, 1]
    synth_nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(real)
    d_synth, nearest_real = synth_nn.kneighbors(synthetic)
    return float((d_synth[:, 0] > d_real[nearest_real[:, 0]]).mean())


@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 3, 32])
def test_cuml_matches_sklearn_reference(batch_size: int) -> None:
    rng = np.random.default_rng(7)
    real = rng.normal(size=(19, 8)).astype(np.float32)
    synthetic = rng.normal(size=(11, 8)).astype(np.float32)
    result = authenticity(real, synthetic, batch_size=batch_size)
    assert result["authenticity"] == sklearn_authenticity(real, synthetic)


@pytest.mark.gpu
def test_exact_copies_are_suspicious() -> None:
    real = np.array([[0.0], [2.0], [10.0]], dtype=np.float32)
    synthetic = np.array([[0.0], [100.0]], dtype=np.float32)
    result = authenticity(real, synthetic, batch_size=1)
    assert result["authenticity"] == 0.5
    assert result["suspicious_count"] == 1


@pytest.mark.gpu
def test_duplicate_real_embeddings_exclude_self() -> None:
    real = np.array([[0.0], [0.0], [4.0]], dtype=np.float32)
    synthetic = np.array([[0.0], [2.0]], dtype=np.float32)
    result = authenticity(real, synthetic)
    assert result["authenticity"] == 0.5


@pytest.mark.gpu
def test_strict_boundary_is_not_authentic() -> None:
    real = np.array([[0.0], [2.0]], dtype=np.float32)
    synthetic = np.array([[4.0]], dtype=np.float32)
    assert authenticity(real, synthetic)["authenticity"] == 0.0


def test_embedding_cache_is_aligned_to_manifest(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    embeddings = tmp_path / "embeddings.parquet"
    pd.DataFrame({"image_path": ["b.png", "a.png"]}).to_csv(
        manifest, index=False
    )
    pd.DataFrame(
        {"image_path": ["a.png", "b.png"], "f0": [1.0, 2.0]}
    ).to_parquet(embeddings, index=False)
    frame = validate_embedding_cache(manifest, embeddings)
    assert frame["f0"].tolist() == [2.0, 1.0]


def test_embedding_cache_rejects_membership_mismatch(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    embeddings = tmp_path / "embeddings.parquet"
    pd.DataFrame({"image_path": ["a.png"]}).to_csv(manifest, index=False)
    pd.DataFrame({"image_path": ["b.png"], "f0": [1.0]}).to_parquet(
        embeddings, index=False
    )
    with pytest.raises(RuntimeError, match="mismatch"):
        validate_embedding_cache(manifest, embeddings)


def test_population_hash_depends_on_order() -> None:
    first = pd.DataFrame({"image_path": ["a", "b"], "f0": [0.0, 1.0]})
    second = first.iloc[::-1].reset_index(drop=True)
    assert ordered_population_hash(first) != ordered_population_hash(second)
