"""cuML-or-sklearn compatibility shims.

For each symbol, cuML's GPU-accelerated implementation is tried first,
falling back silently to the equivalent scikit-learn class if cuML is not
installed or the import fails.

Symbols with no cuML equivalent are imported directly from scikit-learn and
are grouped at the bottom under a clear comment.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# LogisticRegression — cuml.linear_model / sklearn.linear_model
# ---------------------------------------------------------------------------
try:
    from cuml.linear_model import LogisticRegression
except ImportError:
    from sklearn.linear_model import LogisticRegression  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# StandardScaler — cuml.preprocessing / sklearn.preprocessing
# ---------------------------------------------------------------------------
try:
    from cuml.preprocessing import StandardScaler
except ImportError:
    from sklearn.preprocessing import StandardScaler  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# KNeighborsClassifier — cuml.neighbors / sklearn.neighbors
# ---------------------------------------------------------------------------
try:
    from cuml.neighbors import KNeighborsClassifier
except ImportError:
    from sklearn.neighbors import KNeighborsClassifier  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# NearestNeighbors — cuml.neighbors / sklearn.neighbors
# ---------------------------------------------------------------------------
try:
    from cuml.neighbors import NearestNeighbors
except ImportError:
    from sklearn.neighbors import NearestNeighbors  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# No cuML equivalent — always sklearn
# ---------------------------------------------------------------------------
from sklearn.manifold import trustworthiness  # noqa: F401
from sklearn.metrics import confusion_matrix  # noqa: F401
from sklearn.model_selection import StratifiedKFold  # noqa: F401
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: F401

import numpy as _np
from sklearn.metrics import recall_score as _recall_score


def balanced_accuracy_score(y_true, y_pred, **kwargs):
    """Drop-in for sklearn's balanced_accuracy_score that restricts scoring to
    labels present in y_true, silencing spurious warnings when a classifier
    predicts classes absent from a small test split."""
    labels = _np.unique(y_true)
    return float(_recall_score(y_true, y_pred, average="macro",
                               labels=labels, zero_division=0))
