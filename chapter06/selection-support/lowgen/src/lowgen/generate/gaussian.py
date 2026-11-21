from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal

from .base import LatentGenerator


class _GaussianBase(LatentGenerator):
    """Internal base: stores per-class or global Gaussian parameters."""

    def __init__(self):
        # global parameters (unconditional)
        self._mean: np.ndarray | None = None
        self._cov_or_std = None  # scalar (isotropic) or ndarray (full)
        # per-class parameters (conditional)
        self._per_class: dict[int, tuple] | None = None

    def _fit_single(self, Z: np.ndarray) -> tuple:
        raise NotImplementedError

    def _sample_single(
        self, params: tuple, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        raise NotImplementedError

    def _ll_single(self, Z: np.ndarray, params: tuple) -> float:
        raise NotImplementedError

    def fit(self, Z: np.ndarray, y: np.ndarray | None = None) -> "_GaussianBase":
        global_params = self._fit_single(Z)
        self._mean, self._cov_or_std = global_params
        if y is not None:
            self._per_class = {}
            for label in np.unique(y):
                self._per_class[int(label)] = self._fit_single(Z[y == label])
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._sample_single((self._mean, self._cov_or_std), n, rng)

    def sample_conditional(
        self, n_per_class: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._per_class is None:
            raise RuntimeError("Fit with labels to use sample_conditional.")
        parts_z, parts_y = [], []
        for label, params in self._per_class.items():
            parts_z.append(self._sample_single(params, n_per_class, rng))
            parts_y.append(np.full(n_per_class, label, dtype=np.int64))
        return np.concatenate(parts_z), np.concatenate(parts_y)

    def log_likelihood(self, Z: np.ndarray) -> float:
        return self._ll_single(Z, (self._mean, self._cov_or_std))


class IsotropicGaussian(_GaussianBase):
    """Single Gaussian with isotropic (scalar) covariance."""

    def _fit_single(self, Z: np.ndarray) -> tuple:
        mean = Z.mean(axis=0)
        std = float(Z.std())
        return mean, std

    def _sample_single(
        self, params: tuple, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        mean, std = params
        return rng.normal(loc=mean, scale=std, size=(n, len(mean))).astype(np.float32)

    def _ll_single(self, Z: np.ndarray, params: tuple) -> float:
        mean, std = params
        d = len(mean)
        cov = np.eye(d) * (std**2)
        lls = multivariate_normal.logpdf(Z, mean=mean, cov=cov, allow_singular=True)
        return float(np.mean(lls))

    def key(self) -> str:
        return "isotropic"


class FullGaussian(_GaussianBase):
    """Single Gaussian with full covariance matrix."""

    def _fit_single(self, Z: np.ndarray) -> tuple:
        mean = Z.mean(axis=0)
        cov = np.cov(Z, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        return mean, cov

    def _sample_single(
        self, params: tuple, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        mean, cov = params
        return rng.multivariate_normal(mean=mean, cov=cov, size=n).astype(np.float32)

    def _ll_single(self, Z: np.ndarray, params: tuple) -> float:
        mean, cov = params
        lls = multivariate_normal.logpdf(Z, mean=mean, cov=cov, allow_singular=True)
        return float(np.mean(lls))

    def key(self) -> str:
        return "gaussian"
