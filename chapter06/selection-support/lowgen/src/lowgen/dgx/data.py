from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .utils import stable_hash


@dataclass
class SplitData:
    features: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray


@dataclass
class DatasetData:
    dataset: str
    encoder: str
    root: Path
    train: SplitData
    test: SplitData
    valid: SplitData | None
    class_names: dict[int, str]
    metadata: dict[str, Any]

    @property
    def source_dimension(self) -> int:
        return int(self.train.features.shape[1])


def is_icaird(cfg: ExperimentConfig, dataset: str) -> bool:
    return dataset in set(cfg.icaird_dataset_names)


def resolve_embedding_root(cfg: ExperimentConfig, dataset: str, encoder: str) -> Path:
    if is_icaird(cfg, dataset):
        for alias in cfg.encoder_folder_aliases.get(encoder, [encoder]):
            candidate = cfg.paths.icaird_embeddings_root / alias
            if candidate.exists():
                return candidate
        return cfg.paths.icaird_embeddings_root / encoder
    return cfg.paths.non_icaird_embeddings_root / f"cache-{dataset}-{encoder}"


def inspect_embedding_metadata(cfg: ExperimentConfig, dataset: str, encoder: str) -> dict[str, Any]:
    root = resolve_embedding_root(cfg, dataset, encoder)
    if is_icaird(cfg, dataset):
        manifest = _read_json(root / "manifest.json")
        with np.load(root / "features.npz") as data:
            features = data["features"]
            n_samples, n_dimensions = features.shape
        return {
            "root": str(root),
            "layout": "icaird_npz",
            "n_samples": int(n_samples),
            "n_dimensions": int(n_dimensions),
            "manifest": manifest,
        }
    train_meta = _read_json(root / "train" / "meta.json")
    return {
        "root": str(root),
        "layout": "lowgen_cache",
        "n_samples": int(train_meta.get("num_samples", train_meta.get("n_samples", 0))),
        "n_dimensions": int(train_meta.get("embedding_dim", train_meta.get("num_dimensions", 0))),
    }


def source_fingerprint(cfg: ExperimentConfig, dataset: str, encoder: str) -> dict[str, Any]:
    root = resolve_embedding_root(cfg, dataset, encoder)
    files = []
    for pattern in ["manifest.json", "index.parquet", "features.npz", "train/meta.json", "test/meta.json"]:
        path = root / pattern
        if path.exists():
            stat = path.stat()
            files.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"dataset": dataset, "encoder": encoder, "root": str(root), "files": files}


def source_fingerprint_hash(cfg: ExperimentConfig, dataset: str, encoder: str) -> str:
    return stable_hash(source_fingerprint(cfg, dataset, encoder), length=24)


def load_dataset(cfg: ExperimentConfig, dataset: str, encoder: str) -> DatasetData:
    root = resolve_embedding_root(cfg, dataset, encoder)
    if is_icaird(cfg, dataset):
        return _load_icaird(dataset, encoder, root)
    return _load_lowgen_cache(dataset, encoder, root)


def _load_lowgen_cache(dataset: str, encoder: str, root: Path) -> DatasetData:
    train = _load_cache_split(root / "train")
    test = _load_cache_split(root / "test")
    class_names = {}
    label_map = root / "label_map.json"
    if label_map.exists():
        raw = json.loads(label_map.read_text())
        class_names = {int(k): str(v) for k, v in raw.items()}
    return DatasetData(
        dataset=dataset,
        encoder=encoder,
        root=root,
        train=train,
        valid=None,
        test=test,
        class_names=class_names,
        metadata={"layout": "lowgen_cache"},
    )


def _load_cache_split(root: Path) -> SplitData:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Loading lowgen cache embeddings requires torch.") from exc

    emb_paths = sorted(root.glob("emb_[0-9][0-9][0-9][0-9].pt"))
    label_paths = sorted(root.glob("labels_[0-9][0-9][0-9][0-9].pt"))
    if not emb_paths or not label_paths:
        raise FileNotFoundError(f"Missing embedding or label shards in {root}")
    features = np.concatenate([_torch_to_numpy(torch.load(p, weights_only=True)) for p in emb_paths])
    labels = np.concatenate([_torch_to_numpy(torch.load(p, weights_only=True)) for p in label_paths]).astype(
        np.int64
    )
    sample_ids: list[str] = []
    for path in sorted(root.glob("paths_[0-9][0-9][0-9][0-9].json")):
        sample_ids.extend([str(x) for x in json.loads(path.read_text())])
    if len(sample_ids) != len(labels):
        sample_ids = [f"{root.name}_{i}" for i in range(len(labels))]
    return SplitData(features.astype(np.float32, copy=False), labels, np.asarray(sample_ids, dtype=str))


def _load_icaird(dataset: str, encoder: str, root: Path) -> DatasetData:
    import pandas as pd

    with np.load(root / "features.npz") as data:
        features = np.asarray(data["features"], dtype=np.float32)
    index = pd.read_parquet(root / "index.parquet")
    if len(index) != len(features):
        raise ValueError(f"iCAIRD index/features row mismatch at {root}: {len(index)} != {len(features)}")
    labels = index["annotation_label"].to_numpy(dtype=np.int64)
    sample_ids = index["sample_id"].astype(str).to_numpy()
    split_values = index["split"].astype(str).to_numpy()
    class_names = (
        index[["annotation_label", "annotation_label_name"]]
        .drop_duplicates()
        .set_index("annotation_label")["annotation_label_name"]
        .astype(str)
        .to_dict()
    )
    class_names = {int(k): str(v) for k, v in class_names.items()}

    def split(name: str) -> SplitData:
        mask = split_values == name
        return SplitData(features[mask], labels[mask], sample_ids[mask])

    return DatasetData(
        dataset=dataset,
        encoder=encoder,
        root=root,
        train=split("train"),
        valid=split("valid") if np.any(split_values == "valid") else None,
        test=split("test"),
        class_names=class_names,
        metadata={"layout": "icaird_npz", "index_columns": list(index.columns)},
    )


def _torch_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())
