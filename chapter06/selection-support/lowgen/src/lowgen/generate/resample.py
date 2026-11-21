from __future__ import annotations

import numpy as np

from .base import LatentGenerator


class EmpiricalResampler(LatentGenerator):
    """Baseline generator: resamples with replacement from the training set.

    Class-conditional variant stores per-class sub-arrays and resamples each
    class independently.
    """

    def __init__(self):
        self._Z: np.ndarray | None = None
        self._per_class: dict[int, np.ndarray] | None = None

    def fit(self, Z: np.ndarray, y: np.ndarray | None = None) -> "EmpiricalResampler":
        self._Z = Z.copy()
        if y is not None:
            self._per_class = {}
            for label in np.unique(y):
                self._per_class[int(label)] = Z[y == label].copy()
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.integers(0, len(self._Z), size=n)
        return self._Z[idx]

    def sample_conditional(
        self, n_per_class: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._per_class is None:
            raise RuntimeError("Fit with labels to use sample_conditional.")
        parts_z, parts_y = [], []
        for label, Zc in self._per_class.items():
            idx = rng.integers(0, len(Zc), size=n_per_class)
            parts_z.append(Zc[idx])
            parts_y.append(np.full(n_per_class, label, dtype=np.int64))
        return np.concatenate(parts_z), np.concatenate(parts_y)

    def log_likelihood(self, Z: np.ndarray) -> float:
        # Not well-defined for a non-parametric resampler; return NaN.
        return float("nan")

    def key(self) -> str:
        return "resample"
