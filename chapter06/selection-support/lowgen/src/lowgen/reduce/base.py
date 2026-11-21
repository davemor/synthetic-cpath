from abc import ABC, abstractmethod
from pathlib import Path
import pickle

import cupy as cp
import numpy as np
import torch

from lowgen.data.feature_space import FeatureSpace


class Reducer(ABC):
    """Abstract base for dimensionality reducers.

    All reducers operate on FeatureSpace objects.  Internally they convert
    CuPy arrays to NumPy (and back) as needed — callers never need to worry
    about the device.
    """

    @abstractmethod
    def fit(self, space: FeatureSpace) -> "Reducer":
        """Fit the reducer on the given feature space. Returns self."""

    @abstractmethod
    def transform(self, space: FeatureSpace) -> FeatureSpace:
        """Project space into the low-dimensional representation."""

    @abstractmethod
    def inverse_transform(self, space: FeatureSpace) -> FeatureSpace:
        """Map low-dimensional space back to the original dimensionality."""

    def fit_transform(self, space: FeatureSpace) -> FeatureSpace:
        self.fit(space)
        return self.transform(space)

    @abstractmethod
    def key(self) -> str:
        """A short, filesystem-safe identifier for this reducer and its config."""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "Reducer":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Helpers shared by subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(arr) -> np.ndarray:
        """Convert CuPy or Torch tensor to NumPy without an unnecessary copy."""
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    @staticmethod
    def _to_device(arr: np.ndarray, reference):
        """Return arr on the same device / array library as *reference*."""
        if isinstance(reference, cp.ndarray):
            return cp.asarray(arr)
        return arr
