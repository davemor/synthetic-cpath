from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def detect_elbow(
    k_values: Sequence[int],
    inertias: Sequence[float],
    sensitivity: float = 1.0,
) -> int | None:
    from kneed import KneeLocator

    k = np.asarray(k_values, dtype=int)
    values = np.asarray(inertias, dtype=float)
    if k.ndim != 1 or values.ndim != 1 or len(k) != len(values):
        raise ValueError("k values and inertias must be aligned vectors")
    if len(k) < 3 or not np.all(np.isfinite(values)):
        raise ValueError("elbow detection requires at least three finite values")
    if np.any(np.diff(k) <= 0):
        raise ValueError("k values must be strictly increasing")
    if np.any(np.diff(values) > 0):
        raise ValueError("K-means inertia must be non-increasing")

    locator = KneeLocator(
        k,
        values,
        S=float(sensitivity),
        curve="convex",
        direction="decreasing",
        online=False,
        interp_method="interp1d",
    )
    return None if locator.knee is None else int(locator.knee)


def sampled_silhouette_scores(
    features: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    seeds: Sequence[int],
    scorer: Callable[[np.ndarray, np.ndarray], float],
) -> list[float]:
    x = np.asarray(features)
    y = np.asarray(labels).reshape(-1)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("features and labels must contain aligned rows")
    if sample_size < 2:
        raise ValueError("silhouette sample size must be at least two")
    if not seeds:
        raise ValueError("at least one silhouette seed is required")

    size = min(int(sample_size), len(x))
    scores = []
    for seed in seeds:
        if size == len(x):
            indices = np.arange(len(x))
        else:
            rng = np.random.default_rng(int(seed))
            indices = np.sort(rng.choice(len(x), size=size, replace=False))
        sample_labels = y[indices]
        if len(np.unique(sample_labels)) < 2:
            raise RuntimeError(
                f"silhouette sample for seed {seed} contains fewer than two clusters"
            )
        value = float(scorer(np.asarray(x[indices]), sample_labels))
        if not np.isfinite(value):
            raise RuntimeError(f"non-finite silhouette score for seed {seed}")
        scores.append(value)
    return scores


def select_maximum(
    rows: Sequence[dict[str, object]], metric: str
) -> dict[str, object]:
    finite = [row for row in rows if np.isfinite(float(row[metric]))]
    if not finite:
        raise ValueError(f"no finite {metric} values are available")
    return max(finite, key=lambda row: (float(row[metric]), -int(row["k"])))
