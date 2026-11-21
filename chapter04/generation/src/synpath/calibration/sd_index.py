from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SdIndexComponents:
    scattering: float
    separation: float


def sd_index_components(
    features: np.ndarray,
    labels: np.ndarray,
) -> SdIndexComponents:
    """Return the Scat and Dis terms of the Halkidi SD validity index."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("labels must contain one value per feature row")

    cluster_ids = np.unique(y)
    if len(cluster_ids) < 2:
        raise ValueError("SD-index requires at least two clusters")

    data_variance_norm = float(np.linalg.norm(np.var(x, axis=0)))
    if not np.isfinite(data_variance_norm) or data_variance_norm <= 0:
        raise ValueError("features have no finite global dispersion")

    centroids = []
    cluster_variance_norms = []
    for cluster_id in cluster_ids:
        cluster = x[y == cluster_id]
        if not len(cluster):
            continue
        centroids.append(np.mean(cluster, axis=0))
        cluster_variance_norms.append(
            float(np.linalg.norm(np.var(cluster, axis=0)))
        )

    scattering = float(np.mean(cluster_variance_norms) / data_variance_norm)
    centroid_matrix = np.vstack(centroids)
    distances = pairwise_distances(centroid_matrix)
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError("cluster centroids are not distinct")

    minimum = float(np.min(positive))
    maximum = float(np.max(positive))
    row_sums = distances.sum(axis=1)
    if minimum <= 0 or np.any(row_sums <= 0):
        raise ValueError("cluster separation is not finite")
    separation = float((maximum / minimum) * np.sum(1.0 / row_sums))
    return SdIndexComponents(scattering, separation)


def sd_index(
    components: SdIndexComponents,
    alpha: float,
) -> float:
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")
    return float(alpha * components.scattering + components.separation)


def select_minimum(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    finite = [row for row in rows if np.isfinite(float(row["sd_index"]))]
    if not finite:
        raise ValueError("no finite SD-index values are available")
    return min(finite, key=lambda row: (float(row["sd_index"]), int(row["k"])))


def pairwise_distances(features: np.ndarray) -> np.ndarray:
    differences = features[:, None, :] - features[None, :, :]
    return np.linalg.norm(differences, axis=2)
