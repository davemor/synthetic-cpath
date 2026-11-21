"""Phase 3a metrics: fidelity of synthetic embeddings vs real held-out embeddings."""

from __future__ import annotations

import numpy as np


def eval_fidelity(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    y_real: np.ndarray,
    y_synth: np.ndarray,
    subsample: int = 1000,
) -> dict:
    """Measure statistical fidelity of synthetic embeddings against real test data.

    Parameters
    ----------
    X_real, X_synth : reconstructed embeddings in the original high-D space
    y_real, y_synth : integer class labels
    subsample       : max points used per class for MMD to control compute cost

    Returns
    -------
    Dict with keys: mmd, class_mmd (per-class dict), centroid_l2,
    covariance_frobenius, nn_precision, nn_recall
    """
    results: dict = {}

    results["mmd"] = _mmd(X_real, X_synth, subsample=subsample)
    results["class_mmd"] = _class_mmd(X_real, X_synth, y_real, y_synth, subsample)
    results["centroid_l2"] = _centroid_l2(X_real, X_synth, y_real, y_synth)
    results["covariance_frobenius"] = _cov_frobenius(X_real, X_synth, y_real, y_synth)
    prec, rec = _nn_precision_recall(X_real, X_synth, k=5)
    results["nn_precision"] = prec
    results["nn_recall"] = rec
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rbf_kernel(A: np.ndarray, B: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Gaussian RBF kernel matrix K(A, B)."""
    # pairwise squared distances
    diff = A[:, None, :] - B[None, :, :]
    sq_dists = (diff ** 2).sum(axis=-1)
    if sigma is None:
        sigma = float(np.median(np.sqrt(sq_dists[sq_dists > 0]))) + 1e-8
    return np.exp(-sq_dists / (2 * sigma ** 2))


def _mmd(A: np.ndarray, B: np.ndarray, subsample: int = 1000) -> float:
    """Maximum Mean Discrepancy with RBF kernel (unbiased estimator)."""
    rng = np.random.default_rng(0)
    na = min(subsample, len(A))
    nb = min(subsample, len(B))
    A = A[rng.choice(len(A), na, replace=False)]
    B = B[rng.choice(len(B), nb, replace=False)]

    # median heuristic on the combined set
    AB = np.concatenate([A, B])
    from scipy.spatial.distance import pdist
    sigma = float(np.median(pdist(AB[:min(500, len(AB))], metric="euclidean"))) + 1e-8

    Kaa = _rbf_kernel(A, A, sigma)
    Kbb = _rbf_kernel(B, B, sigma)
    Kab = _rbf_kernel(A, B, sigma)

    na, nb = len(A), len(B)
    # unbiased MMD^2
    np.fill_diagonal(Kaa, 0.0)
    np.fill_diagonal(Kbb, 0.0)
    mmd2 = (
        Kaa.sum() / (na * (na - 1))
        + Kbb.sum() / (nb * (nb - 1))
        - 2 * Kab.mean()
    )
    return float(np.sqrt(max(mmd2, 0.0)))


def _class_mmd(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    y_real: np.ndarray,
    y_synth: np.ndarray,
    subsample: int,
) -> dict[str, float]:
    classes = np.unique(y_real)
    out = {}
    for c in classes:
        Xr = X_real[y_real == c]
        Xs = X_synth[y_synth == c] if (y_synth == c).any() else X_synth[:1]
        out[str(int(c))] = _mmd(Xr, Xs, subsample=subsample)
    return out


def _centroid_l2(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    y_real: np.ndarray,
    y_synth: np.ndarray,
) -> float:
    classes = np.unique(y_real)
    dists = []
    for c in classes:
        cr = X_real[y_real == c].mean(axis=0)
        mask_s = y_synth == c
        if not mask_s.any():
            continue
        cs = X_synth[mask_s].mean(axis=0)
        dists.append(float(np.linalg.norm(cr - cs)))
    return float(np.mean(dists)) if dists else float("nan")


def _cov_frobenius(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    y_real: np.ndarray,
    y_synth: np.ndarray,
) -> float:
    classes = np.unique(y_real)
    frobs = []
    for c in classes:
        Xr = X_real[y_real == c]
        mask_s = y_synth == c
        if not mask_s.any() or len(Xr) < 2:
            continue
        Xs = X_synth[mask_s]
        if len(Xs) < 2:
            continue
        cov_r = np.cov(Xr, rowvar=False)
        cov_s = np.cov(Xs, rowvar=False)
        frobs.append(float(np.linalg.norm(cov_r - cov_s, ord="fro")))
    return float(np.mean(frobs)) if frobs else float("nan")


def _nn_precision_recall(
    X_real: np.ndarray, X_synth: np.ndarray, k: int = 5
) -> tuple[float, float]:
    """1-NN based precision and recall (Kynkäänniemi et al. style).

    Precision: fraction of synthetic samples whose nearest real neighbour
               is within the k-NN manifold of the real data.
    Recall   : fraction of real samples whose nearest synthetic neighbour
               is within the k-NN manifold of the synthetic data.
    """
    from lowgen.util.compat import NearestNeighbors

    # Limit size for efficiency
    rng = np.random.default_rng(0)
    max_n = 2000
    r = X_real[rng.choice(len(X_real), min(max_n, len(X_real)), replace=False)]
    s = X_synth[rng.choice(len(X_synth), min(max_n, len(X_synth)), replace=False)]

    # Build NN index on real data
    nn_real = NearestNeighbors(n_neighbors=k + 1).fit(r)
    real_radii, _ = nn_real.kneighbors(r)
    real_radii = real_radii[:, -1]  # k-th neighbour distance

    # Precision: is each synth point within any real point's radius?
    dists_s2r, _ = nn_real.kneighbors(s, n_neighbors=k)
    precision = float(np.mean(dists_s2r.min(axis=1) <= real_radii.mean()))

    # Build NN index on synth data
    nn_synth = NearestNeighbors(n_neighbors=k + 1).fit(s)
    synth_radii, _ = nn_synth.kneighbors(s)
    synth_radii = synth_radii[:, -1]

    # Recall: is each real point within any synth point's radius?
    dists_r2s, _ = nn_synth.kneighbors(r, n_neighbors=k)
    recall = float(np.mean(dists_r2s.min(axis=1) <= synth_radii.mean()))

    return precision, recall
