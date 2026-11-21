from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import EvalConfig


def classifier_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: EvalConfig,
    kind: str = "logistic_regression",
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    if kind == "knn":
        model = KNeighborsClassifier(n_neighbors=min(cfg.knn_k, len(x_train)))
        x_train_scaled, x_test_scaled = x_train, x_test
    else:
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model = LogisticRegression(max_iter=cfg.classifier_max_iter)
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    y_score = model.predict_proba(x_test_scaled) if hasattr(model, "predict_proba") else None
    labels = sorted(int(x) for x in np.unique(np.concatenate([y_train, y_test])))
    names = [class_names.get(label, str(label)) if class_names else str(label) for label in labels]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_roc_auc_ovr": safe_roc_auc(y_test, y_score, labels),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=names,
            output_dict=True,
            zero_division=0,
        ),
        "labels": labels,
        "label_names": names,
    }


def reduction_metrics(
    x_train_rep: np.ndarray,
    y_train: np.ndarray,
    x_test_rep: np.ndarray,
    y_test: np.ndarray,
    x_test_orig: np.ndarray,
    cfg: EvalConfig,
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    return {
        "linear_probe": classifier_metrics(
            x_train_rep, y_train, x_test_rep, y_test, cfg, "logistic_regression", class_names
        ),
        "knn_probe": classifier_metrics(x_train_rep, y_train, x_test_rep, y_test, cfg, "knn", class_names),
        "trustworthiness": safe_trustworthiness(x_test_orig, x_test_rep, cfg.knn_k),
        "pairwise_dist_correlation": pairwise_distance_correlation(
            x_test_orig, x_test_rep, cfg.pairwise_subsample
        ),
    }


def reconstruction_metrics(x_orig: np.ndarray, x_recon: np.ndarray) -> dict[str, float]:
    denom = np.maximum(np.linalg.norm(x_orig, axis=1) * np.linalg.norm(x_recon, axis=1), 1e-12)
    cosine = np.sum(x_orig * x_recon, axis=1) / denom
    return {"mse": float(np.mean((x_orig - x_recon) ** 2)), "cosine_similarity": float(np.mean(cosine))}


def fidelity_metrics(
    x_real: np.ndarray,
    x_synth: np.ndarray,
    cfg: EvalConfig,
    seed: int,
    encoder: str,
) -> dict[str, float]:
    real = subsample(x_real, cfg.fid_subsample, seed)
    synth = subsample(x_synth, cfg.fid_subsample, seed + 1)
    fid = frechet_distance(real, synth)
    kid_real = subsample(x_real, cfg.kid_subsample, seed + 2)
    kid_synth = subsample(x_synth, cfg.kid_subsample, seed + 3)
    precision, recall = nn_precision_recall(
        subsample(x_real, cfg.nn_subsample, seed + 4),
        subsample(x_synth, cfg.nn_subsample, seed + 5),
    )
    clusters_real = cluster_count(real, cfg.cluster_count_k, seed)
    clusters_synth = cluster_count(synth, cfg.cluster_count_k, seed)
    output = {
        "fid": fid,
        "kid": polynomial_mmd(kid_real, kid_synth),
        "precision": precision,
        "recall": recall,
        "real_clusters": float(clusters_real),
        "synthetic_clusters": float(clusters_synth),
    }
    output["uni_fid"] = fid if encoder == "uni" else float("nan")
    output["uni_kid"] = output["kid"] if encoder == "uni" else float("nan")
    return output


def lowdim_fidelity_metrics(
    z_real: np.ndarray,
    z_synth: np.ndarray,
    cfg: EvalConfig,
    seed: int,
) -> dict[str, float]:
    real = subsample(z_real, cfg.fid_subsample, seed)
    synth = subsample(z_synth, cfg.fid_subsample, seed + 1)
    kid_real = subsample(z_real, cfg.kid_subsample, seed + 2)
    kid_synth = subsample(z_synth, cfg.kid_subsample, seed + 3)
    precision, recall = nn_precision_recall(
        subsample(z_real, cfg.nn_subsample, seed + 4),
        subsample(z_synth, cfg.nn_subsample, seed + 5),
    )
    centroid_distance, covariance_frobenius = centroid_covariance_distances(real, synth)
    return {
        "fid": frechet_distance(real, synth),
        "kid": polynomial_mmd(kid_real, kid_synth),
        "precision": precision,
        "recall": recall,
        "real_clusters": float(cluster_count(real, cfg.cluster_count_k, seed)),
        "synthetic_clusters": float(cluster_count(synth, cfg.cluster_count_k, seed)),
        "centroid_distance": centroid_distance,
        "covariance_frobenius": covariance_frobenius,
    }


def utility_metrics(
    x_synth: np.ndarray,
    y_synth: np.ndarray,
    x_real_train: np.ndarray,
    y_real_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: EvalConfig,
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    return {
        "train_synthetic_test_real": classifier_metrics(
            x_synth, y_synth, x_test, y_test, cfg, "logistic_regression", class_names
        ),
        "real_plus_synthetic_test_real": classifier_metrics(
            np.concatenate([x_real_train, x_synth]),
            np.concatenate([y_real_train, y_synth]),
            x_test,
            y_test,
            cfg,
            "logistic_regression",
            class_names,
        ),
    }


def centroid_covariance_distances(x_real: np.ndarray, x_synth: np.ndarray) -> tuple[float, float]:
    centroid_distance = float(np.linalg.norm(x_real.mean(axis=0) - x_synth.mean(axis=0)))
    cov_real = np.atleast_2d(np.cov(x_real, rowvar=False))
    cov_synth = np.atleast_2d(np.cov(x_synth, rowvar=False))
    covariance_frobenius = float(np.linalg.norm(cov_real - cov_synth, ord="fro"))
    return centroid_distance, covariance_frobenius


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray | None, labels: list[int]) -> float:
    if y_score is None:
        return float("nan")
    try:
        if y_score.shape[1] == 2:
            return float(roc_auc_score(y_true, y_score[:, 1]))
        return float(roc_auc_score(y_true, y_score, labels=labels, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def safe_trustworthiness(x_orig: np.ndarray, x_rep: np.ndarray, n_neighbors: int) -> float:
    try:
        k = min(int(n_neighbors), max(1, len(x_orig) // 2 - 1))
        return float(trustworthiness(x_orig, x_rep, n_neighbors=k))
    except Exception:
        return float("nan")


def pairwise_distance_correlation(x_orig: np.ndarray, x_rep: np.ndarray, subsample_size: int) -> float:
    n = min(int(subsample_size), len(x_orig))
    if n < 3:
        return float("nan")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x_orig), size=n, replace=False)
    d_orig = pdist(x_orig[idx], metric="euclidean")
    d_rep = pdist(x_rep[idx], metric="euclidean")
    if d_orig.std() < 1e-12 or d_rep.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(d_orig, d_rep)[0, 1])


def frechet_distance(x_real: np.ndarray, x_synth: np.ndarray) -> float:
    mu1, mu2 = x_real.mean(axis=0), x_synth.mean(axis=0)
    cov1, cov2 = np.cov(x_real, rowvar=False), np.cov(x_synth, rowvar=False)
    covmean = linalg.sqrtm(cov1 @ cov2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = np.sum((mu1 - mu2) ** 2) + np.trace(cov1 + cov2 - 2 * covmean)
    return float(np.real(value))


def polynomial_mmd(x: np.ndarray, y: np.ndarray) -> float:
    gamma = 1.0 / max(1, x.shape[1])
    kxx = (gamma * x @ x.T + 1.0) ** 3
    kyy = (gamma * y @ y.T + 1.0) ** 3
    kxy = (gamma * x @ y.T + 1.0) ** 3
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def nn_precision_recall(x_real: np.ndarray, x_synth: np.ndarray, k: int = 5) -> tuple[float, float]:
    if len(x_real) <= k or len(x_synth) <= k:
        return float("nan"), float("nan")
    real_nn = NearestNeighbors(n_neighbors=k + 1).fit(x_real)
    synth_nn = NearestNeighbors(n_neighbors=k + 1).fit(x_synth)
    real_radii = real_nn.kneighbors(x_real, return_distance=True)[0][:, -1]
    synth_radii = synth_nn.kneighbors(x_synth, return_distance=True)[0][:, -1]
    s2r = NearestNeighbors(n_neighbors=1).fit(x_real).kneighbors(x_synth, return_distance=True)[0][:, 0]
    r2s = NearestNeighbors(n_neighbors=1).fit(x_synth).kneighbors(x_real, return_distance=True)[0][:, 0]
    return float(np.mean(s2r <= real_radii.mean())), float(np.mean(r2s <= synth_radii.mean()))


def cluster_count(x: np.ndarray, k: int, seed: int) -> int:
    if len(x) < 2:
        return 0
    n_clusters = min(int(k), len(x))
    labels = KMeans(n_clusters=n_clusters, n_init=5, random_state=seed).fit_predict(x)
    return int(np.unique(labels).shape[0])


def subsample(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    n = min(int(n), len(x))
    if n >= len(x):
        return x
    rng = np.random.default_rng(seed)
    return x[np.sort(rng.choice(len(x), size=n, replace=False))]
