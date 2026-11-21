from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    class_name: str
    split: str
    sample_id: str


@dataclass
class ImageSplits:
    train: list[ImageRecord]
    valid: list[ImageRecord]
    test: list[ImageRecord]
    class_names: list[str]

    def for_split(self, split: str) -> list[ImageRecord]:
        return getattr(self, split)


def discover_image_splits(
    root: Path,
    label_index_path: Path | None = None,
    label_column: str | None = None,
) -> ImageSplits:
    missing = [split for split in SPLITS if not (root / split).is_dir()]
    if missing:
        raise ValueError(
            f"Expected dataset root to contain train/valid/test folders; missing {missing} under {root}."
        )
    if label_index_path is not None:
        return _discover_image_splits_from_label_index(root, label_index_path, label_column)
    class_names = sorted(
        {
            child.name
            for split in SPLITS
            for child in (root / split).iterdir()
            if child.is_dir()
        }
    )
    if not class_names:
        raise ValueError(f"No class folders found under {root}/train, {root}/valid, or {root}/test.")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    split_records: dict[str, list[ImageRecord]] = {}
    for split in SPLITS:
        records = []
        for class_name in class_names:
            class_dir = root / split / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    sample_id = str(path.relative_to(root))
                    records.append(
                        ImageRecord(
                            path=path,
                            label=class_to_idx[class_name],
                            class_name=class_name,
                            split=split,
                            sample_id=sample_id,
                        )
                    )
        split_records[split] = records
    _validate_split_classes(split_records, class_names)
    return ImageSplits(
        train=split_records["train"],
        valid=split_records["valid"],
        test=split_records["test"],
        class_names=class_names,
    )


def _discover_image_splits_from_label_index(
    root: Path,
    label_index_path: Path,
    label_column: str | None,
) -> ImageSplits:
    if not label_index_path.exists():
        raise ValueError(f"Configured image label index does not exist: {label_index_path}")
    index = pd.read_parquet(label_index_path)
    split_col = _first_existing_column(index, ["split", "dataset_split"])
    label_col = label_column or _first_existing_column(
        index,
        ["annotation_label_name", "subcategory", "class_name", "label", "category"],
    )
    if label_col not in index.columns:
        raise ValueError(
            f"Configured image label_column {label_col!r} was not found in {label_index_path}. "
            f"Available columns: {list(index.columns)}"
        )
    path_col = _first_existing_column(index, ["patch_png_path", "path", "image_path"])
    sample_col = _first_existing_column(index, ["sample_id", "id", "patch_png_path"])
    index_lookup = _image_label_lookup(index, root, split_col, label_col, path_col, sample_col)

    image_paths = {
        split: sorted(
            path
            for path in (root / split).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for split in SPLITS
    }
    matched_labels = set()
    for paths in image_paths.values():
        for path in paths:
            for key in _image_lookup_keys(path, root):
                if key in index_lookup:
                    matched_labels.add(str(index_lookup[key][1]))
                    break
    labels = sorted(matched_labels)
    if not labels:
        raise ValueError(
            f"No image paths under {root} matched rows in {label_index_path}; cannot assign patch labels."
        )
    class_to_idx = {name: idx for idx, name in enumerate(labels)}
    split_records: dict[str, list[ImageRecord]] = {}
    missing_examples: list[str] = []
    for split, paths in image_paths.items():
        records = []
        for path in paths:
            match = None
            for key in _image_lookup_keys(path, root):
                if key in index_lookup:
                    match = index_lookup[key]
                    break
            if match is None:
                if len(missing_examples) < 5:
                    missing_examples.append(str(path))
                continue
            metadata_split, class_name, sample_id = match
            if metadata_split != split:
                raise ValueError(
                    f"Image {path} is under split {split!r}, but {label_index_path} says {metadata_split!r}."
                )
            records.append(
                ImageRecord(
                    path=path,
                    label=class_to_idx[str(class_name)],
                    class_name=str(class_name),
                    split=split,
                    sample_id=str(sample_id),
                )
            )
        split_records[split] = records
    if missing_examples:
        raise ValueError(
            f"Some image files had no matching row in {label_index_path}. "
            f"Examples: {missing_examples}"
        )
    _validate_split_classes(split_records, labels)
    return ImageSplits(
        train=split_records["train"],
        valid=split_records["valid"],
        test=split_records["test"],
        class_names=labels,
    )


def _image_label_lookup(
    index: pd.DataFrame,
    root: Path,
    split_col: str,
    label_col: str,
    path_col: str,
    sample_col: str,
) -> dict[str, tuple[str, str, str]]:
    lookup = {}
    root = root.absolute()
    for row in index.itertuples(index=False):
        row_data = row._asdict()
        split = str(row_data[split_col])
        label = str(row_data[label_col])
        sample_id = str(row_data[sample_col])
        image_path = Path(str(row_data[path_col]))
        keys = {sample_id, image_path.name, image_path.stem, str(image_path)}
        if image_path.is_absolute():
            try:
                keys.add(str(image_path.absolute().relative_to(root)))
            except ValueError:
                pass
        for key in keys:
            lookup[key] = (split, label, sample_id)
    return lookup


def _image_lookup_keys(path: Path, root: Path) -> list[str]:
    return [
        str(path),
        str(path.absolute()),
        str(path.absolute().relative_to(root.absolute())),
        path.name,
        path.stem,
    ]


def _validate_split_classes(records: dict[str, list[ImageRecord]], class_names: list[str]) -> None:
    required = set(range(len(class_names)))
    for split, split_records in records.items():
        present = {record.label for record in split_records}
        if present != required:
            missing_names = [class_names[idx] for idx in sorted(required - present)]
            raise ValueError(
                f"Split {split!r} is missing classes {missing_names}; ACC/AUC would be misleading."
            )


def write_split_manifest(splits: ImageSplits, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in SPLITS:
        for record in splits.for_split(split):
            rows.append(
                {
                    "split": split,
                    "sample_id": record.sample_id,
                    "path": str(record.path),
                    "class_name": record.class_name,
                    "label": record.label,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "image_split_manifest.csv", index=False)


class PatchDataset:
    def __init__(self, records: list[ImageRecord], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        return self.transform(image), record.label, record.sample_id


@dataclass
class FeatureSplit:
    features: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]

    @property
    def dim(self) -> int:
        if self.features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {self.features.shape}.")
        return int(self.features.shape[1])


@dataclass
class FeatureSplits:
    train: FeatureSplit
    valid: FeatureSplit
    test: FeatureSplit
    class_names: list[str] | None = None


def load_feature_splits(path: Path, label_column: str | None = None) -> FeatureSplits:
    if path.is_dir():
        if (path / "features.npz").exists() and (path / "index.parquet").exists():
            return _load_npz_parquet_embedding(path, label_column=label_column)
        return FeatureSplits(
            train=_load_single_npz(path / "train.npz", "train"),
            valid=_load_single_npz(path / "valid.npz", "valid"),
            test=_load_single_npz(path / "test.npz", "test"),
        )
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        if {"train_features", "train_labels", "test_features", "test_labels"} <= keys:
            train = _feature_split_from_arrays(data, "train")
            test = _feature_split_from_arrays(data, "test")
            valid = _feature_split_from_arrays(data, "valid") if {"valid_features", "valid_labels"} <= keys else train
            return FeatureSplits(train=train, valid=valid, test=test)
        if {"features", "labels", "split"} <= keys:
            return _load_split_column_npz(data)
    raise ValueError(
        f"Unsupported embedding layout at {path}. Expected split directory, hproj-style arrays, or features/labels/split arrays."
    )


def _load_npz_parquet_embedding(path: Path, label_column: str | None = None) -> FeatureSplits:
    features_path = path / "features.npz"
    index_path = path / "index.parquet"
    with np.load(features_path, allow_pickle=True) as data:
        if "features" not in data.files:
            raise ValueError(f"{features_path} must contain a features array.")
        features = np.asarray(data["features"])
    index = pd.read_parquet(index_path)
    if len(index) != len(features):
        raise ValueError(
            f"{features_path} has {len(features)} rows but {index_path} has {len(index)} rows."
        )
    split_col = _first_existing_column(index, ["split", "dataset_split"])
    if label_column is not None:
        if label_column not in index.columns:
            raise ValueError(
                f"Configured label_column {label_column!r} was not found in {index_path}. "
                f"Available columns: {list(index.columns)}"
            )
        label_col = label_column
    else:
        label_col = _first_existing_column(
            index,
            ["subcategory", "class_name", "label", "annotation_label_name", "category"],
        )
    sample_col = _first_existing_column(index, ["sample_id", "id", "patch_png_path"])
    result = {}
    for split in SPLITS:
        mask = index[split_col].astype(str).to_numpy() == split
        if not mask.any():
            raise ValueError(f"{index_path} contains no rows for split {split!r}.")
        result[split] = FeatureSplit(
            features=features[mask],
            labels=index.loc[mask, label_col].astype(str).to_numpy(),
            sample_ids=index.loc[mask, sample_col].astype(str).tolist(),
        )
    return FeatureSplits(**result)


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"None of the expected columns {candidates} were found. Available columns: {list(df.columns)}"
    )


def _load_single_npz(path: Path, split: str) -> FeatureSplit:
    if not path.exists():
        raise ValueError(f"Expected embedding split file {path}.")
    with np.load(path, allow_pickle=True) as data:
        if not {"features", "labels"} <= set(data.files):
            raise ValueError(f"{path} must contain features and labels arrays.")
        return FeatureSplit(
            features=np.asarray(data["features"]),
            labels=np.asarray(data["labels"]),
            sample_ids=_sample_ids(data, split, len(data["labels"])),
        )


def _feature_split_from_arrays(data, split: str) -> FeatureSplit:
    labels = np.asarray(data[f"{split}_labels"])
    return FeatureSplit(
        features=np.asarray(data[f"{split}_features"]),
        labels=labels,
        sample_ids=_sample_ids(data, split, len(labels)),
    )


def _sample_ids(data, split: str, n: int) -> list[str]:
    for key in (f"{split}_sample_ids", f"{split}_paths", "sample_ids", "paths"):
        if key in data.files:
            values = data[key]
            if len(values) == n:
                return [str(value) for value in values]
    return [f"{split}_{idx}" for idx in range(n)]


def _load_split_column_npz(data) -> FeatureSplits:
    features = np.asarray(data["features"])
    labels = np.asarray(data["labels"])
    split_values = np.asarray(data["split"]).astype(str)
    sample_ids = _sample_ids(data, "all", len(labels))
    result = {}
    for split in SPLITS:
        mask = split_values == split
        if not mask.any():
            raise ValueError(f"Embedding split array contains no rows for {split!r}.")
        result[split] = FeatureSplit(
            features=features[mask],
            labels=labels[mask],
            sample_ids=[sample_ids[idx] for idx in np.flatnonzero(mask)],
        )
    return FeatureSplits(**result)


def validate_feature_splits(splits: FeatureSplits, expected_dim: int | None = None) -> int:
    dims = {splits.train.dim, splits.valid.dim, splits.test.dim}
    if len(dims) != 1:
        raise ValueError(f"Embedding feature dimensions differ across splits: {sorted(dims)}.")
    dim = dims.pop()
    if expected_dim is not None and dim != int(expected_dim):
        raise ValueError(f"Expected embedding dimension {expected_dim}, got {dim}.")
    classes = set(np.unique(splits.train.labels))
    for split_name, split in [("valid", splits.valid), ("test", splits.test)]:
        present = set(np.unique(split.labels))
        if present != classes:
            raise ValueError(
                f"Embedding split {split_name!r} has classes {sorted(present)}, expected {sorted(classes)}."
            )
    return dim
