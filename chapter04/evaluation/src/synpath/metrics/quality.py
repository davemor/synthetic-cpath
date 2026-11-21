from __future__ import annotations

import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from synpath.artifacts import write_json
from synpath.embeddings.uni import feature_matrix, load_embedding_frame
from synpath.manifests import read_manifest
from synpath.paths import SynpathPaths


def evaluate_quality(
    dataset: str,
    model: str,
    conditioning: str,
    seed: int,
    paths: SynpathPaths,
    cfg: dict,
) -> list[Path]:
    root = (
        paths.outputs_root
        / "generators"
        / dataset
        / model
        / conditioning
        / f"seed{seed}"
        / "metrics"
    )
    root.mkdir(parents=True, exist_ok=True)
    synthetic_manifest = (
        paths.outputs_root
        / "generators"
        / dataset
        / model
        / conditioning
        / f"seed{seed}"
        / "synthetic_manifest.csv"
    )
    if not synthetic_manifest.exists():
        raise RuntimeError(f"Missing synthetic manifest: {synthetic_manifest}")

    real_embedding_path = (
        paths.cache_root / "embeddings" / dataset / "train_uni_224.parquet"
    )
    synthetic_embedding_path = root / "synthetic_uni_224.parquet"
    real_frame = load_embedding_frame(real_embedding_path)
    real_frame = attach_real_clusters(real_frame, dataset, conditioning, paths)
    synthetic_frame = load_or_extract_synthetic_embeddings(
        synthetic_manifest,
        synthetic_embedding_path,
    )
    outputs = []
    if "uni_authenticity" in cfg.get("metrics", {}).get("quality", []):
        from synpath.metrics.authenticity import (
            AuthenticityInputs,
            evaluate_authenticity,
        )

        real_manifest = (
            paths.outputs_root
            / "conditions"
            / dataset
            / conditioning
            / "condition_manifest.csv"
        )
        outputs.append(
            evaluate_authenticity(
                AuthenticityInputs(
                    real_manifest=real_manifest,
                    real_embeddings=real_embedding_path,
                    synthetic_manifest=synthetic_manifest,
                    synthetic_embeddings=synthetic_embedding_path,
                    output=root / "uni_authenticity.json",
                    dataset=dataset,
                    model=model,
                    conditioning=conditioning,
                    seed=seed,
                    family="main",
                ),
                paths.cache_root / "authenticity" / "real_neighbours",
                batch_size=int(
                    cfg.get("metrics", {}).get(
                        "authenticity_batch_size", 1024
                    )
                ),
            )
        )
    real_frame, synthetic_frame = metric_sample(real_frame, synthetic_frame, cfg)
    real_x = feature_matrix(real_frame)
    synth_x = feature_matrix(synthetic_frame)

    metrics = {
        "uni_fid": frechet_distance(real_x, synth_x),
        "uni_kid": polynomial_mmd(real_x, synth_x),
    }
    fid_path = root / "uni_fid_kid.json"
    write_json(fid_path, metrics)
    outputs.append(fid_path)

    pr = precision_recall(real_x, synth_x)
    pr_path = root / "uni_precision_recall.json"
    write_json(pr_path, pr)
    outputs.append(pr_path)

    coverage = cluster_coverage(real_frame, synthetic_frame)
    coverage_path = root / "uni_cluster_coverage.json"
    write_json(coverage_path, coverage)
    outputs.append(coverage_path)

    privacy = privacy_nearest_neighbor(real_x, synth_x)
    privacy_path = root / "privacy_nearest_neighbor.json"
    write_json(privacy_path, privacy)
    outputs.append(privacy_path)

    inception_path = root / "inception_fid.json"
    write_json(
        inception_path,
        inception_fid(
            real_frame,
            synthetic_frame,
            root / "inception_real",
            root / "inception_synthetic",
        ),
    )
    outputs.append(inception_path)
    return outputs


def attach_real_clusters(
    frame: pd.DataFrame,
    dataset: str,
    conditioning: str,
    paths: SynpathPaths,
) -> pd.DataFrame:
    if conditioning == "true_label":
        frame = frame.copy()
        frame["cluster_id"] = frame["label"]
        return frame
    if conditioning in {"pseudo_label", "label_global_cluster"}:
        cache = paths.cache_root / "clusters" / dataset / "pseudo_labels.csv"
    elif conditioning == "label_local_cluster":
        cache = paths.cache_root / "clusters" / dataset / "label_local_cluster_labels.csv"
    else:
        return frame
    if not cache.exists():
        return frame
    clusters = pd.read_csv(cache)
    return frame.merge(clusters, on="image_path", how="left")


def load_or_extract_synthetic_embeddings(
    synthetic_manifest: Path,
    output: Path,
) -> pd.DataFrame:
    if output.exists():
        return pd.read_parquet(output)

    from synpath.embeddings.uni import embed_rows, load_uni_model
    from torchvision import transforms
    import torch

    rows = read_manifest(synthetic_manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_uni_model(device)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    frame = embed_rows(rows, model, transform, device, batch_size=128, num_workers=8)
    frame.to_parquet(output, index=False)
    return frame


def metric_sample(
    real_frame: pd.DataFrame,
    synthetic_frame: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_cfg = cfg.get("metrics", {})
    rng = np.random.default_rng(int(metrics_cfg.get("seed", 0)))
    max_real = int(metrics_cfg.get("max_real", len(real_frame)))
    max_synth = int(metrics_cfg.get("max_synthetic", len(synthetic_frame)))
    return (
        sample_frame(real_frame, max_real, rng),
        sample_frame(synthetic_frame, max_synth, rng),
    )


def sample_frame(frame: pd.DataFrame, max_rows: int, rng) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    idx = rng.choice(len(frame), size=max_rows, replace=False)
    return frame.iloc[idx].reset_index(drop=True)


def frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    cov_x, cov_y = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    covmean = linalg.sqrtm(cov_x.dot(cov_y))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = np.sum((mu_x - mu_y) ** 2) + np.trace(cov_x + cov_y - 2 * covmean)
    return float(value)


def polynomial_mmd(x: np.ndarray, y: np.ndarray) -> float:
    gamma = 1.0 / x.shape[1]
    k_xx = (gamma * x.dot(x.T) + 1.0) ** 3
    k_yy = (gamma * y.dot(y.T) + 1.0) ** 3
    k_xy = (gamma * x.dot(y.T) + 1.0) ** 3
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def precision_recall(real_x: np.ndarray, synth_x: np.ndarray) -> dict[str, float]:
    real_dist = pairwise_distances(real_x, real_x)
    synth_dist = pairwise_distances(synth_x, synth_x)
    np.fill_diagonal(real_dist, np.inf)
    np.fill_diagonal(synth_dist, np.inf)
    real_radius = np.median(real_dist.min(axis=1))
    synth_radius = np.median(synth_dist.min(axis=1))
    synth_to_real = NearestNeighbors(n_neighbors=1).fit(real_x).kneighbors(synth_x)[0]
    real_to_synth = NearestNeighbors(n_neighbors=1).fit(synth_x).kneighbors(real_x)[0]
    return {
        "precision": float((synth_to_real[:, 0] <= real_radius).mean()),
        "recall": float((real_to_synth[:, 0] <= synth_radius).mean()),
        "real_radius": float(real_radius),
        "synthetic_radius": float(synth_radius),
    }


def cluster_coverage(real_frame: pd.DataFrame, synth_frame: pd.DataFrame) -> dict:
    if "cluster_id" not in synth_frame.columns:
        return {"status": "not_available"}
    real_clusters = set(real_frame.get("cluster_id", pd.Series(dtype=str)).dropna())
    synth_clusters = set(synth_frame["cluster_id"].dropna())
    if not real_clusters:
        return {"status": "not_available"}
    return {
        "real_clusters": len(real_clusters),
        "synthetic_clusters": len(synth_clusters),
        "coverage": len(real_clusters & synth_clusters) / len(real_clusters),
    }


def inception_fid(
    real_frame: pd.DataFrame,
    synthetic_frame: pd.DataFrame,
    real_dir: Path,
    synthetic_dir: Path,
) -> dict:
    try:
        from cleanfid import fid
    except ImportError:
        return {"status": "not_computed", "reason": "clean-fid is not installed"}

    link_image_dirs(real_frame["image_path"].tolist(), real_dir)
    link_image_dirs(synthetic_frame["image_path"].tolist(), synthetic_dir)
    value = fid.compute_fid(str(real_dir), str(synthetic_dir), mode="clean")
    return {"status": "computed", "fid": float(value)}


def link_image_dirs(paths: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(paths):
        dst = output_dir / f"{idx:08d}{Path(src).suffix or '.png'}"
        if dst.exists():
            continue
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def privacy_nearest_neighbor(real_x: np.ndarray, synth_x: np.ndarray) -> dict[str, float]:
    distances = NearestNeighbors(n_neighbors=1).fit(real_x).kneighbors(synth_x)[0][:, 0]
    return {
        "min_distance": float(distances.min()),
        "median_distance": float(np.median(distances)),
        "p01_distance": float(np.quantile(distances, 0.01)),
        "p05_distance": float(np.quantile(distances, 0.05)),
    }
