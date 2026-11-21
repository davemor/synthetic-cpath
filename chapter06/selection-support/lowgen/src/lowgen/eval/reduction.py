"""Phase 1 metrics: how well does the low-D representation preserve task signal?"""

from __future__ import annotations

import numpy as np
from lowgen.util.compat import (
    KNeighborsClassifier,
    LogisticRegression,
    balanced_accuracy_score,
    trustworthiness,
)

from lowgen.util.config import EvalConfig


def eval_reduction_quality(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_test_orig: np.ndarray,
    cfg: EvalConfig,
) -> dict:
    """Evaluate a low-dimensional representation on held-out real data.

    Parameters
    ----------
    Z_train, Z_test : reduced embeddings (train / test splits)
    y_train, y_test : integer class labels
    X_test_orig     : original high-dimensional test embeddings
    cfg             : EvalConfig with knn_k, probe_max_iter

    Returns
    -------
    Dict with keys: linear_probe_balanced_acc, knn_balanced_acc,
    trustworthiness, pairwise_dist_correlation
    """
    results: dict = {}

    results["linear_probe_balanced_acc"] = _linear_probe(
        Z_train, Z_test, y_train, y_test, cfg
    )
    results["knn_balanced_acc"] = _knn_acc(
        Z_train, Z_test, y_train, y_test, cfg
    )
    results["trustworthiness"] = float(
        trustworthiness(X_test_orig, Z_test, n_neighbors=cfg.knn_k)
    )
    results["pairwise_dist_correlation"] = _pairwise_dist_corr(
        X_test_orig, Z_test, subsample=cfg.mmd_subsample
    )
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_probe(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cfg: EvalConfig,
) -> float:
    clf = LogisticRegression(max_iter=cfg.probe_max_iter)
    clf.fit(Z_train, y_train)
    y_pred = np.array(clf.predict(Z_test))
    return float(balanced_accuracy_score(y_test, y_pred))


def _knn_acc(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cfg: EvalConfig,
) -> float:
    clf = KNeighborsClassifier(n_neighbors=cfg.knn_k)
    clf.fit(Z_train, y_train)
    y_pred = np.array(clf.predict(Z_test))
    return float(balanced_accuracy_score(y_test, y_pred))


def _pairwise_dist_corr(
    X: np.ndarray, Z: np.ndarray, subsample: int = 1000
) -> float:
    """Pearson correlation between pairwise L2 distances in X and Z."""
    n = min(subsample, len(X))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=n, replace=False)
    Xs, Zs = X[idx], Z[idx]

    # flatten upper triangle of pairwise distance matrices
    from scipy.spatial.distance import pdist

    d_orig = pdist(Xs, metric="euclidean")
    d_red = pdist(Zs, metric="euclidean")

    if d_orig.std() < 1e-12 or d_red.std() < 1e-12:
        return float("nan")

    corr = float(np.corrcoef(d_orig, d_red)[0, 1])
    return corr
