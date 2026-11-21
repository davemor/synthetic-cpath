"""Phase 2 metrics: how well does the reducer's inverse map reconstruct originals?"""

from __future__ import annotations

import numpy as np
from lowgen.util.compat import (
    KNeighborsClassifier,
    LogisticRegression,
    StandardScaler,
    StratifiedShuffleSplit,
    balanced_accuracy_score,
    trustworthiness,
)

from lowgen.util.config import EvalConfig


def eval_reconstruction_quality(
    X_orig: np.ndarray,
    X_recon: np.ndarray,
    y: np.ndarray,
    cfg: EvalConfig,
) -> dict:
    """Measure round-trip reconstruction quality on held-out real embeddings.

    Parameters
    ----------
    X_orig  : original high-dimensional test embeddings
    X_recon : embeddings reconstructed by reducer.inverse_transform(reducer.transform(...))
    y       : class labels for the test set
    cfg     : EvalConfig

    Returns
    -------
    Dict with keys: mse, cosine_similarity, probe_balanced_acc_after_recon,
    knn_balanced_acc_after_recon, neighborhood_preservation
    """
    results: dict = {}

    results["mse"] = float(np.mean((X_orig - X_recon) ** 2))
    results["cosine_similarity"] = _mean_cosine_similarity(X_orig, X_recon)
    results["probe_balanced_acc_after_recon"] = _probe_self(X_recon, y, cfg)
    results["knn_balanced_acc_after_recon"] = _knn_self(X_recon, y, cfg)
    results["neighborhood_preservation"] = float(
        trustworthiness(X_orig, X_recon, n_neighbors=cfg.knn_k)
    )
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_cosine_similarity(A: np.ndarray, B: np.ndarray) -> float:
    norm_a = np.linalg.norm(A, axis=1, keepdims=True)
    norm_b = np.linalg.norm(B, axis=1, keepdims=True)
    # avoid division by zero
    denom = np.maximum(norm_a * norm_b, 1e-12)
    cos = np.sum(A * B, axis=1) / denom.squeeze()
    return float(np.mean(cos))


def _probe_self(X: np.ndarray, y: np.ndarray, cfg: EvalConfig) -> float:
    """Leave-20%-out linear probe on the reconstructed set itself."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr_idx, te_idx = next(sss.split(X, y))
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_te = scaler.transform(X[te_idx])
    clf = LogisticRegression(max_iter=cfg.probe_max_iter, class_weight="balanced")
    clf.fit(X_tr, y[tr_idx])
    return float(balanced_accuracy_score(y[te_idx], clf.predict(X_te)))


def _knn_self(X: np.ndarray, y: np.ndarray, cfg: EvalConfig) -> float:
    """Leave-20%-out k-NN on the reconstructed set itself."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr_idx, te_idx = next(sss.split(X, y))
    clf = KNeighborsClassifier(n_neighbors=cfg.knn_k)
    clf.fit(X[tr_idx], y[tr_idx])
    return float(balanced_accuracy_score(y[te_idx], clf.predict(X[te_idx])))
