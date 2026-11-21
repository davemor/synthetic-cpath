from __future__ import annotations

import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from .base import LatentGenerator


class _MixtureBase(LatentGenerator):
    """Shared logic for GMM and BayesianGMM generators.

    Class-conditional sampling fits one mixture model per unique class label.
    """

    def __init__(self, n_components: int, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._model: GaussianMixture | BayesianGaussianMixture | None = None
        self._per_class: dict[int, GaussianMixture | BayesianGaussianMixture] | None = None

    def _build_model(self) -> GaussianMixture | BayesianGaussianMixture:
        raise NotImplementedError

    def fit(self, Z: np.ndarray, y: np.ndarray | None = None) -> "_MixtureBase":
        self._model = self._build_model()
        self._model.fit(Z)
        if y is not None:
            self._per_class = {}
            for label in np.unique(y):
                m = self._build_model()
                m.fit(Z[y == label])
                self._per_class[int(label)] = m
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # sklearn GaussianMixture.sample uses its own internal RandomState;
        # we seed it from the provided rng for reproducibility.
        seed = int(rng.integers(0, 2**31))
        self._model.random_state = seed
        Z, _ = self._model.sample(n)
        return Z.astype(np.float32)

    def sample_conditional(
        self, n_per_class: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._per_class is None:
            raise RuntimeError("Fit with labels to use sample_conditional.")
        parts_z, parts_y = [], []
        for label, model in self._per_class.items():
            seed = int(rng.integers(0, 2**31))
            model.random_state = seed
            Z, _ = model.sample(n_per_class)
            parts_z.append(Z.astype(np.float32))
            parts_y.append(np.full(n_per_class, label, dtype=np.int64))
        return np.concatenate(parts_z), np.concatenate(parts_y)

    def log_likelihood(self, Z: np.ndarray) -> float:
        return float(self._model.score(Z))  # mean log-likelihood

    def bic(self, Z: np.ndarray) -> float:
        return float(self._model.bic(Z))

    def aic(self, Z: np.ndarray) -> float:
        return float(self._model.aic(Z))


class GMMGenerator(_MixtureBase):
    """Gaussian Mixture Model using sklearn GaussianMixture."""

    def __init__(
        self,
        n_components: int,
        covariance_type: str = "full",
        reg_covar: float = 1e-4,
        n_init: int = 5,
        random_state: int = 42,
    ):
        super().__init__(n_components=n_components, random_state=random_state)
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.n_init = n_init

    def _build_model(self) -> GaussianMixture:
        return GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            n_init=self.n_init,
            random_state=self.random_state,
            max_iter=300,
        )

    def key(self) -> str:
        return f"gmm_k{self.n_components}_{self.covariance_type}"


class BayesianGMMGenerator(_MixtureBase):
    """Bayesian Gaussian Mixture Model using sklearn BayesianGaussianMixture."""

    def __init__(
        self,
        n_components: int,
        covariance_type: str = "full",
        reg_covar: float = 1e-4,
        n_init: int = 5,
        random_state: int = 42,
    ):
        super().__init__(n_components=n_components, random_state=random_state)
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.n_init = n_init

    def _build_model(self) -> BayesianGaussianMixture:
        return BayesianGaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            n_init=self.n_init,
            random_state=self.random_state,
            max_iter=300,
        )

    def key(self) -> str:
        return f"bgmm_k{self.n_components}_{self.covariance_type}"
