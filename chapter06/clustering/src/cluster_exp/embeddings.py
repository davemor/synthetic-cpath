from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import path_fingerprint


@dataclass
class EmbeddingData:
    features: np.ndarray
    labels: np.ndarray | None
    sample_ids: np.ndarray
    splits: np.ndarray
    source_paths: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_dimensions(self) -> int:
        return int(self.features.shape[1])


def embedding_root(embeddings_root: Path, dataset: str, encoder: str) -> Path:
    return Path(embeddings_root) / f"cache-{dataset}-{encoder}"


def load_embedding_dataset(embeddings_root: Path, dataset: str, encoder: str) -> EmbeddingData:
    root = embedding_root(embeddings_root, dataset, encoder)
    if root.is_file() and root.suffix == ".npz":
        return _load_npz(root, dataset, encoder)
    if (root / "features.npz").exists():
        return _load_npz(root / "features.npz", dataset, encoder)
    return _load_hproj_split_root(root, dataset, encoder)


def load_embedding_labels(embeddings_root: Path, dataset: str, encoder: str) -> np.ndarray | None:
    root = embedding_root(embeddings_root, dataset, encoder)
    if root.is_file() and root.suffix == ".npz":
        return _load_npz_labels(root)
    if (root / "features.npz").exists():
        return _load_npz_labels(root / "features.npz")
    return _load_hproj_split_labels(root)


def source_fingerprint(embeddings_root: Path, dataset: str, encoder: str) -> dict[str, Any]:
    return path_fingerprint(embedding_root(embeddings_root, dataset, encoder))


def _load_npz_labels(path: Path) -> np.ndarray | None:
    with np.load(path, allow_pickle=True) as data:
        return np.asarray(data["labels"]) if "labels" in data.files else None


def _load_npz(path: Path, dataset: str, encoder: str) -> EmbeddingData:
    with np.load(path, allow_pickle=True) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"]) if "labels" in data.files else None
        if "sample_ids" in data.files:
            sample_ids = np.asarray(data["sample_ids"]).astype(str)
        else:
            sample_ids = np.asarray([f"{dataset}_{encoder}_{i}" for i in range(len(features))])
        if "splits" in data.files:
            splits = np.asarray(data["splits"]).astype(str)
        else:
            splits = np.asarray(["all"] * len(features))
        if "source_paths" in data.files:
            source_paths = np.asarray(data["source_paths"]).astype(str)
        else:
            source_paths = np.asarray([""] * len(features))
    return EmbeddingData(
        features=features,
        labels=labels,
        sample_ids=sample_ids,
        splits=splits,
        source_paths=source_paths,
        metadata={"dataset": dataset, "encoder": encoder, "source": str(path)},
    )


def _load_hproj_split_root(root: Path, dataset: str, encoder: str) -> EmbeddingData:
    features_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    sample_ids: list[str] = []
    splits: list[str] = []
    source_paths: list[str] = []

    for split in ("train", "test"):
        split_root = root / split
        if not split_root.exists():
            continue
        split_features, split_labels, split_paths = _load_hproj_split(split_root)
        features_list.append(split_features)
        if split_labels is not None:
            labels_list.append(split_labels)
        offset = len(sample_ids)
        n_rows = split_features.shape[0]
        splits.extend([split] * n_rows)
        source_paths.extend(split_paths or [""] * n_rows)
        sample_ids.extend(_make_sample_ids(dataset, encoder, split, split_paths, offset, n_rows))

    if not features_list:
        raise FileNotFoundError(
            f"No hproj embedding splits found at {root}. Expected train/test emb_*.pt shards."
        )

    labels: np.ndarray | None = None
    if labels_list and sum(len(x) for x in labels_list) == sum(len(x) for x in features_list):
        labels = np.concatenate(labels_list)

    return EmbeddingData(
        features=np.vstack(features_list).astype(np.float32, copy=False),
        labels=labels,
        sample_ids=np.asarray(sample_ids).astype(str),
        splits=np.asarray(splits).astype(str),
        source_paths=np.asarray(source_paths).astype(str),
        metadata={"dataset": dataset, "encoder": encoder, "source": str(root)},
    )


def _load_hproj_split_labels(root: Path) -> np.ndarray | None:
    torch = _import_torch()
    labels_list: list[np.ndarray] = []
    for split in ("train", "test"):
        split_root = root / split
        if not split_root.exists():
            continue
        label_paths = sorted(split_root.glob("labels_[0-9][0-9][0-9][0-9].pt"))
        labels_list.extend(_torch_to_numpy(torch.load(path, map_location="cpu")) for path in label_paths)
    if not labels_list:
        return None
    return np.concatenate(labels_list)


def _load_hproj_split(split_root: Path) -> tuple[np.ndarray, np.ndarray | None, list[str] | None]:
    torch = _import_torch()
    emb_paths = sorted(split_root.glob("emb_[0-9][0-9][0-9][0-9].pt"))
    label_paths = sorted(split_root.glob("labels_[0-9][0-9][0-9][0-9].pt"))
    paths_json = sorted(split_root.glob("paths_[0-9][0-9][0-9][0-9].json"))
    if not emb_paths:
        raise FileNotFoundError(f"No emb_*.pt shards found in {split_root}")

    features = [_torch_to_numpy(torch.load(path, map_location="cpu")) for path in emb_paths]
    labels = None
    if label_paths:
        labels = np.concatenate(
            [_torch_to_numpy(torch.load(path, map_location="cpu")) for path in label_paths]
        )
    source_paths = None
    if paths_json:
        source_paths = []
        for path in paths_json:
            with path.open() as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                loaded = list(loaded.values())
            source_paths.extend(str(item) for item in loaded)
    return np.vstack(features).astype(np.float32, copy=False), labels, source_paths


def _torch_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Loading hproj .pt embedding shards requires torch. Install the project "
            "in the Hypatia runtime or use an .npz fixture."
        ) from exc
    return torch


def _make_sample_ids(
    dataset: str,
    encoder: str,
    split: str,
    paths: list[str] | None,
    offset: int,
    n_rows: int,
) -> list[str]:
    if paths and len(paths) == n_rows:
        return [f"{split}:{Path(item).stem}" for item in paths]
    return [f"{dataset}:{encoder}:{split}:{offset + i}" for i in range(n_rows)]
