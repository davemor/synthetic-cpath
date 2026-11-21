from abc import ABC, abstractmethod
from pathlib import Path
import pickle

import numpy as np


class LatentGenerator(ABC):
    """Abstract base for latent-space density models.

    All generators work with plain NumPy arrays (latent coordinates after
    dimensionality reduction).
    """

    @abstractmethod
    def fit(self, Z: np.ndarray, y: np.ndarray | None = None) -> "LatentGenerator":
        """Fit the generator on latent points Z (and optional labels y)."""

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw *n* unconditional samples.  Returns array of shape (n, d)."""

    @abstractmethod
    def sample_conditional(
        self, n_per_class: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw *n_per_class* samples per class.

        Returns (Z_synth, y_synth) where y_synth contains integer class labels.
        Raises RuntimeError if the generator was not fit with labels.
        """

    @abstractmethod
    def log_likelihood(self, Z: np.ndarray) -> float:
        """Mean log-likelihood of Z under the fitted model."""

    @abstractmethod
    def key(self) -> str:
        """A short, filesystem-safe identifier for this generator and its config."""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "LatentGenerator":
        with open(path, "rb") as f:
            return pickle.load(f)
