from __future__ import annotations

import cupy as cp
import numpy as np
from cuml.decomposition import PCA as cumlPCA

from lowgen.data.feature_space import FeatureSpace
from .base import Reducer


class PCAReducer(Reducer):
    """PCA-based dimensionality reducer using cuML's GPU-accelerated PCA.

    The inverse transform is computed manually from the fitted components and
    mean, since cuML PCA does not expose an ``inverse_transform`` method.
    """

    def __init__(self, n_components: int, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten
        self._model: cumlPCA | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, space: FeatureSpace) -> "PCAReducer":
        self._model = cumlPCA(n_components=self.n_components, whiten=self.whiten)
        self._model.fit(space.features)
        return self

    def transform(self, space: FeatureSpace) -> FeatureSpace:
        Z = self._model.transform(space.features).astype(cp.float32)
        return FeatureSpace(Z, space.labels)

    def inverse_transform(self, space: FeatureSpace) -> FeatureSpace:
        # cuML PCA has no inverse_transform; reconstruct manually.
        Z = space.features.astype(cp.float32)
        components = cp.asarray(self._model.components_)
        mean = cp.asarray(self._model.mean_)
        if self.whiten:
            std = cp.sqrt(cp.asarray(self._model.explained_variance_))
            Z = Z * std
        X_recon = (Z @ components + mean).astype(cp.float32)
        return FeatureSpace(X_recon, space.labels)

    def key(self) -> str:
        tag = f"pca_d{self.n_components}"
        if self.whiten:
            tag += "_white"
        return tag
