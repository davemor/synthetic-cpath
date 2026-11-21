from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from .config import ClusterPseudoLabelConfig


FEATURE_KEY = "features"


def prepare_cluster_pseudolabels(config: ClusterPseudoLabelConfig) -> dict[str, Path]:
    """Fit PCA/KMeans on iCAIRD 100K train embeddings and label patch/bag roots."""
    _validate_config(config)
    assert config.source_embedding_dir is not None
    assert config.source_bag_root is not None

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.output_embedding_dir.mkdir(parents=True, exist_ok=True)
    config.output_bag_root.mkdir(parents=True, exist_ok=True)

    features, index = _load_patch_embedding_root(config.source_embedding_dir)
    split_mask = index[config.split_column].astype(str).to_numpy() == config.fit_split
    if not np.any(split_mask):
        raise ValueError(f"No rows with {config.split_column}={config.fit_split!r}.")

    pca = PCA(n_components=config.pca_components, random_state=config.random_state, svd_solver="auto")
    train_projected = pca.fit_transform(features[split_mask]).astype(np.float32, copy=False)
    kmeans = KMeans(
        n_clusters=config.kmeans_clusters,
        init=config.kmeans_init,
        n_init=config.kmeans_n_init,
        max_iter=config.kmeans_max_iter,
        random_state=config.random_state,
    )
    kmeans.fit(train_projected)

    with (config.artifact_dir / "pca.pkl").open("wb") as handle:
        pickle.dump(pca, handle)
    with (config.artifact_dir / "kmeans.pkl").open("wb") as handle:
        pickle.dump(kmeans, handle)

    patch_cluster_ids = _predict_clusters(features, pca, kmeans, config.transform_batch_size)
    patch_index = index.copy()
    patch_index[config.label_column] = _cluster_names(patch_cluster_ids, config.kmeans_clusters)
    _write_derived_patch_root(config, patch_index)

    bag_summary = _write_derived_bag_root(config, pca, kmeans)
    patch_summary = _write_assignment_summaries(config, patch_index, patch_cluster_ids, bag_summary)
    _write_manifest(config, pca, kmeans, patch_summary, bag_summary)

    return {
        "artifact_dir": config.artifact_dir,
        "output_embedding_dir": config.output_embedding_dir,
        "output_bag_root": config.output_bag_root,
    }


def _load_patch_embedding_root(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    features_path = path / "features.npz"
    index_path = path / "index.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features at {features_path}.")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index at {index_path}.")
    with np.load(features_path) as data:
        if FEATURE_KEY not in data:
            raise KeyError(f"{features_path} does not contain {FEATURE_KEY!r}.")
        features = np.asarray(data[FEATURE_KEY], dtype=np.float32)
    index = pd.read_parquet(index_path)
    if len(index) != len(features):
        raise ValueError(f"{index_path} has {len(index)} rows but features has {len(features)} rows.")
    return features, index


def _predict_clusters(
    features: np.ndarray,
    pca: PCA,
    kmeans: KMeans,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("transform_batch_size must be positive.")
    labels = np.empty(len(features), dtype=np.int64)
    for start in tqdm(range(0, len(features), batch_size), desc="predict clusters", unit="batch"):
        stop = min(start + batch_size, len(features))
        projected = pca.transform(features[start:stop])
        labels[start:stop] = kmeans.predict(projected).astype(np.int64, copy=False)
    return labels


def _write_derived_patch_root(config: ClusterPseudoLabelConfig, index: pd.DataFrame) -> None:
    assert config.source_embedding_dir is not None
    _link_or_copy(config.source_embedding_dir / "features.npz", config.output_embedding_dir / "features.npz")
    index.to_parquet(config.output_embedding_dir / "index.parquet", index=False)
    (config.output_embedding_dir / "cluster_pseudolabel_manifest.json").write_text(
        json.dumps(
            {
                "source_embedding_dir": str(config.source_embedding_dir),
                "artifact_dir": str(config.artifact_dir),
                "label_column": config.label_column,
                "n_rows": int(len(index)),
            },
            indent=2,
        )
    )


def _write_derived_bag_root(
    config: ClusterPseudoLabelConfig,
    pca: PCA,
    kmeans: KMeans,
) -> pd.DataFrame:
    assert config.source_bag_root is not None
    source_index_path = config.source_bag_root / "dataset_index_ometiff.csv"
    if not source_index_path.exists():
        raise FileNotFoundError(f"Missing bag index at {source_index_path}.")
    slide_index = pd.read_csv(source_index_path)
    slide_index.to_csv(config.output_bag_root / "dataset_index_ometiff.csv", index=False)

    rows = []
    for slide in tqdm(slide_index.itertuples(index=False), total=len(slide_index), desc="label bag slides", unit="slide"):
        slide_data = slide._asdict()
        slide_id = str(slide_data["id"])
        source_slide_dir = _slide_embedding_dir(config.source_bag_root, config.embedding_name, slide_id)
        output_slide_dir = _slide_embedding_dir(config.output_bag_root, config.embedding_name, slide_id)
        output_slide_dir.mkdir(parents=True, exist_ok=True)
        features_path = source_slide_dir / "features.npz"
        index_path = source_slide_dir / "index.parquet"
        if not features_path.exists() or not index_path.exists():
            raise FileNotFoundError(f"Missing features/index for slide {slide_id}: {source_slide_dir}")
        _link_or_copy(features_path, output_slide_dir / "features.npz")
        with np.load(features_path) as data:
            features = np.asarray(data[FEATURE_KEY], dtype=np.float32)
        patch_index = pd.read_parquet(index_path)
        if len(patch_index) != len(features):
            raise ValueError(f"{index_path} has {len(patch_index)} rows but features has {len(features)} rows.")
        cluster_ids = _predict_clusters(features, pca, kmeans, config.transform_batch_size)
        patch_index = patch_index.copy()
        patch_index[config.label_column] = _cluster_names(cluster_ids, config.kmeans_clusters)
        patch_index.to_parquet(output_slide_dir / "index.parquet", index=False)
        _copy_if_exists(source_slide_dir / "manifest.json", output_slide_dir / "manifest.json")
        counts = np.bincount(cluster_ids, minlength=config.kmeans_clusters)
        rows.append(
            {
                "slide_id": slide_id,
                "split": str(slide_data.get("split", "")),
                "category": str(slide_data.get("category", "")),
                "subcategory": str(slide_data.get("subcategory", "")),
                "patch_count": int(len(cluster_ids)),
                **{f"cluster_{idx:02d}": int(counts[idx]) for idx in range(config.kmeans_clusters)},
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(config.artifact_dir / "bag_cluster_counts_by_slide.csv", index=False)
    return summary


def _write_assignment_summaries(
    config: ClusterPseudoLabelConfig,
    patch_index: pd.DataFrame,
    patch_cluster_ids: np.ndarray,
    bag_summary: pd.DataFrame,
) -> dict[str, Path]:
    patch_counts = (
        pd.Series(_cluster_names(patch_cluster_ids, config.kmeans_clusters), name=config.label_column)
        .value_counts()
        .rename_axis(config.label_column)
        .reset_index(name="count")
        .sort_values(config.label_column)
    )
    patch_counts.to_csv(config.artifact_dir / "icaird100k_cluster_counts.csv", index=False)
    split_counts = (
        patch_index.groupby([config.split_column, config.label_column], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values([config.split_column, config.label_column])
    )
    split_counts.to_csv(config.artifact_dir / "icaird100k_cluster_counts_by_split.csv", index=False)
    bag_totals = (
        bag_summary[[f"cluster_{idx:02d}" for idx in range(config.kmeans_clusters)]]
        .sum(axis=0)
        .rename_axis(config.label_column)
        .reset_index(name="count")
    )
    bag_totals.to_csv(config.artifact_dir / "bag_cluster_counts.csv", index=False)
    return {
        "icaird100k_cluster_counts": config.artifact_dir / "icaird100k_cluster_counts.csv",
        "icaird100k_cluster_counts_by_split": config.artifact_dir / "icaird100k_cluster_counts_by_split.csv",
        "bag_cluster_counts": config.artifact_dir / "bag_cluster_counts.csv",
        "bag_cluster_counts_by_slide": config.artifact_dir / "bag_cluster_counts_by_slide.csv",
    }


def _write_manifest(
    config: ClusterPseudoLabelConfig,
    pca: PCA,
    kmeans: KMeans,
    patch_summary: dict[str, Path],
    bag_summary: pd.DataFrame,
) -> None:
    (config.artifact_dir / "cluster_label_mapping.json").write_text(
        json.dumps(
            {
                "label_column": config.label_column,
                "class_names": [f"cluster_{idx:02d}" for idx in range(config.kmeans_clusters)],
                "class_to_id": {f"cluster_{idx:02d}": idx for idx in range(config.kmeans_clusters)},
            },
            indent=2,
        )
    )
    (config.artifact_dir / "config_snapshot.json").write_text(
        json.dumps(_jsonable_config(config), indent=2, sort_keys=True)
    )
    (config.artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "description": "iCAIRD UNI2-h PCA/KMeans cluster pseudo-labels.",
                "source_embedding_dir": str(config.source_embedding_dir),
                "source_bag_root": str(config.source_bag_root),
                "output_embedding_dir": str(config.output_embedding_dir),
                "output_bag_root": str(config.output_bag_root),
                "pca": {
                    "n_components": int(config.pca_components),
                    "random_state": int(config.random_state),
                    "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
                },
                "kmeans": {
                    "n_clusters": int(config.kmeans_clusters),
                    "init": config.kmeans_init,
                    "n_init": int(config.kmeans_n_init),
                    "max_iter": int(config.kmeans_max_iter),
                    "random_state": int(config.random_state),
                    "inertia": float(kmeans.inertia_),
                },
                "label_column": config.label_column,
                "n_bag_slides": int(len(bag_summary)),
                "summaries": {name: str(path) for name, path in patch_summary.items()},
            },
            indent=2,
        )
    )


def _cluster_names(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    width = max(2, len(str(max(0, n_clusters - 1))))
    return np.asarray([f"cluster_{int(label):0{width}d}" for label in labels], dtype=object)


def _slide_embedding_dir(root: Path, embedding_name: str, slide_id: str) -> Path:
    return root / "embeddings" / embedding_name / "slides" / f"{slide_id}.{embedding_name}.embedding"


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists() and not destination.exists():
        shutil.copy2(source, destination)


def _validate_config(config: ClusterPseudoLabelConfig) -> None:
    if config.source_embedding_dir is None:
        raise ValueError("cluster_pseudolabels.source_embedding_dir is required.")
    if config.source_bag_root is None:
        raise ValueError("cluster_pseudolabels.source_bag_root is required.")
    if config.pca_components <= 0:
        raise ValueError("cluster_pseudolabels.pca_components must be positive.")
    if config.kmeans_clusters <= 1:
        raise ValueError("cluster_pseudolabels.kmeans_clusters must be greater than one.")


def _jsonable_config(config: ClusterPseudoLabelConfig) -> dict[str, object]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data
