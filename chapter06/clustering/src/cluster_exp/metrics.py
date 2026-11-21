from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SdIndexResult:
    sd_index: float
    sd_scat: float
    sd_dis: float


def sd_index(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = 1.0,
) -> SdIndexResult:
    """Compute the SD internal validity index on non-noise labels.

    Lower values are better. Noise points labelled -1 are excluded so density
    clusterers are assessed on their assigned clusters.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    mask = labels != -1
    features = features[mask]
    labels = labels[mask]
    unique_labels = np.unique(labels)
    if features.shape[0] < 3 or unique_labels.shape[0] < 2:
        return _nan_result()

    centroids = []
    cluster_std_norms = []
    for label in unique_labels:
        cluster = features[labels == label]
        if cluster.shape[0] == 0:
            continue
        centroids.append(np.mean(cluster, axis=0))
        cluster_std_norms.append(float(np.linalg.norm(np.std(cluster, axis=0))))

    if len(centroids) < 2:
        return _nan_result()

    data_std_norm = float(np.linalg.norm(np.std(features, axis=0)))
    if not np.isfinite(data_std_norm) or data_std_norm <= 0:
        return _nan_result()

    scat = float(np.mean(cluster_std_norms) / data_std_norm)
    centroids_array = np.vstack(centroids)
    distances = _pairwise_distances(centroids_array)
    positive = distances[distances > 0]
    if positive.size == 0:
        return _nan_result(scat=scat)

    d_min = float(np.min(positive))
    d_max = float(np.max(positive))
    row_sums = distances.sum(axis=1)
    if d_min <= 0 or not np.all(row_sums > 0):
        return _nan_result(scat=scat)

    dis = float((d_max / d_min) * np.sum(1.0 / row_sums))
    return SdIndexResult(
        sd_index=float(alpha * scat + dis),
        sd_scat=scat,
        sd_dis=dis,
    )


def _pairwise_distances(features: np.ndarray) -> np.ndarray:
    diffs = features[:, None, :] - features[None, :, :]
    return np.linalg.norm(diffs, axis=2)


def _nan_result(*, scat: float = float("nan")) -> SdIndexResult:
    return SdIndexResult(sd_index=float("nan"), sd_scat=scat, sd_dis=float("nan"))
