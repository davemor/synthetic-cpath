"""Phase 3b metrics: downstream utility of synthetic embeddings."""

from __future__ import annotations

import numpy as np
from lowgen.util.compat import (
    LogisticRegression,
    StandardScaler,
    balanced_accuracy_score,
    confusion_matrix,
)

from lowgen.util.config import EvalConfig


def eval_utility(
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_real_train: np.ndarray,
    y_real_train: np.ndarray,
    cfg: EvalConfig,
) -> dict:
    """Measure downstream classification utility of synthetic embeddings.

    Parameters
    ----------
    X_synth, y_synth          : synthetic embeddings and labels
    X_test, y_test            : held-out *real* test set (never seen by generators)
    X_real_train, y_real_train: full real training set
    cfg                       : EvalConfig (probe_max_iter)

    Returns
    -------
    Dict with keys: tostr_balanced_acc, tstr_balanced_acc,
    real_only_balanced_acc, per_class_accuracy
    """
    results: dict = {}

    # Train-on-Synthetic Test-on-Real
    results["tostr_balanced_acc"] = _probe(
        X_synth, y_synth, X_test, y_test, cfg
    )

    # Train-on-Real+Synthetic Test-on-Real
    X_aug = np.concatenate([X_real_train, X_synth])
    y_aug = np.concatenate([y_real_train, y_synth])
    results["tstr_balanced_acc"] = _probe(X_aug, y_aug, X_test, y_test, cfg)

    # Matched-size Real-only baseline (same n as X_synth)
    n_matched = min(len(X_synth), len(X_real_train))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_real_train), n_matched, replace=False)
    results["real_only_balanced_acc"] = _probe(
        X_real_train[idx], y_real_train[idx], X_test, y_test, cfg
    )

    # Per-class accuracy from the TOSTR probe (refit to get per-class breakdown)
    results["per_class_accuracy"] = _per_class_acc(
        X_synth, y_synth, X_test, y_test, cfg
    )

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: EvalConfig,
) -> float:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = LogisticRegression(
        max_iter=cfg.probe_max_iter, class_weight="balanced"
    )
    clf.fit(X_tr, y_train)
    return float(balanced_accuracy_score(y_test, clf.predict(X_te)))


def _per_class_acc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: EvalConfig,
) -> dict[str, float]:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = LogisticRegression(
        max_iter=cfg.probe_max_iter, class_weight="balanced"
    )
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)

    classes = np.unique(y_test)
    per_class = {}
    for c in classes:
        mask = y_test == c
        if mask.sum() == 0:
            per_class[str(int(c))] = float("nan")
        else:
            # Per-class accuracy = recall for that class; balanced_accuracy_score
            # is undefined when y_true has a single label, so compute directly.
            per_class[str(int(c))] = float((y_pred[mask] == c).mean())
    return per_class
