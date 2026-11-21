from __future__ import annotations

from pathlib import Path

import numpy as np

from lowgen.data.feature_space import FeatureSpace
from .base import Reducer

# Use cuML's GPU-accelerated UMAP when available; fall back to umap-learn.
try:
    from cuml.manifold import UMAP as _UMAP
    _CUML_UMAP = True
except ImportError:
    import umap as _umap_module  # type: ignore[import]
    _CUML_UMAP = False


class UMAPReducer(Reducer):
    """UMAP-based dimensionality reducer.

    Uses cuML's GPU-accelerated UMAP when available, otherwise falls back to
    umap-learn (CPU).  CuPy ↔ NumPy conversion is handled internally so that
    callers always receive the same device-type as the input FeatureSpace.

    The inverse_transform uses UMAP's built-in approximate inverse; it should
    be validated empirically rather than assumed exact.
    """

    def __init__(
        self,
        n_components: int,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self._model = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, space: FeatureSpace) -> "UMAPReducer":
        kwargs = dict(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )
        if _CUML_UMAP:
            self._model = _UMAP(**kwargs)
            self._model.fit(space.features)
        else:
            X = self._to_numpy(space.features).astype(np.float32)
            self._model = _umap_module.UMAP(**kwargs)
            self._model.fit(X)
        return self

    def transform(self, space: FeatureSpace) -> FeatureSpace:
        if _CUML_UMAP:
            Z = self._model.transform(space.features)
            Z_out = self._to_device(np.asarray(Z).astype(np.float32), space.features)
        else:
            X = self._to_numpy(space.features).astype(np.float32)
            Z = self._model.transform(X).astype(np.float32)
            Z_out = self._to_device(Z, space.features)
        return FeatureSpace(Z_out, space.labels)

    def inverse_transform(self, space: FeatureSpace) -> FeatureSpace:
        if _CUML_UMAP:
            X_recon = self._model.inverse_transform(space.features)
            X_out = self._to_device(np.asarray(X_recon).astype(np.float32), space.features)
        else:
            Z = self._to_numpy(space.features).astype(np.float32)
            X_recon = self._model.inverse_transform(Z).astype(np.float32)
            X_out = self._to_device(X_recon, space.features)
        return FeatureSpace(X_out, space.labels)

    def key(self) -> str:
        md_str = str(self.min_dist).replace(".", "")
        return f"umap_d{self.n_components}_nn{self.n_neighbors}_md{md_str}"
