from __future__ import annotations

import json
import gzip
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from synpath.config import load_resolved_yaml
from synpath.manifests import DATASET_COLUMNS, read_manifest, write_manifest
from synpath.paths import SynpathPaths

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SPIDER_REPOS = {
    "skin": "histai/SPIDER-skin",
    "colorectal": "histai/SPIDER-colorectal",
    "thorax": "histai/SPIDER-thorax",
    "breast": "histai/SPIDER-breast",
}


def safe_label(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def load_dataset_config(dataset: str) -> dict[str, Any]:
    path = Path("configs") / "datasets" / f"{dataset}.yaml"
    return load_resolved_yaml(path)


def build_manifest(
    dataset: str,
    paths: SynpathPaths,
    experiment_cfg: dict[str, Any],
) -> Path:
    full_path = build_full_manifest(dataset, paths, experiment_cfg)
    rows = read_manifest(full_path)
    return write_selected_manifest(dataset, rows, paths, experiment_cfg)


def build_full_manifest(
    dataset: str,
    paths: SynpathPaths,
    experiment_cfg: dict[str, Any],
) -> Path:
    dataset_cfg = load_dataset_config(dataset)
    kind = dataset_cfg.get("kind")
    if kind == "medmnist_plus":
        rows = build_pathmnist_manifest(dataset, dataset_cfg, paths)
    elif kind == "spider":
        rows = build_spider_manifest(
            dataset, dataset_cfg, paths, experiment_cfg
        )
    elif kind == "indexed_parquet":
        rows = build_indexed_parquet_manifest(
            dataset, dataset_cfg, experiment_cfg
        )
    elif kind == "imagefolder_with_metadata":
        rows = build_imagefolder_manifest(dataset, dataset_cfg)
    elif kind == "pcam_hdf5":
        rows = build_pcam_hdf5_manifest(
            dataset, dataset_cfg, paths, experiment_cfg
        )
    elif kind == "pcam_kaggle":
        rows = build_pcam_kaggle_manifest(dataset, dataset_cfg)
    else:
        raise ValueError(f"Unsupported dataset kind for {dataset}: {kind}")

    output = paths.data_root / "manifests" / f"{dataset}_full.csv"
    write_manifest(output, rows, DATASET_COLUMNS)
    return output


def write_selected_manifest(
    dataset: str,
    full_rows: list[dict[str, object]],
    paths: SynpathPaths,
    experiment_cfg: dict[str, Any],
) -> Path:
    rows = apply_limits(full_rows, experiment_cfg)
    output = paths.data_root / "manifests" / f"{dataset}.csv"
    write_manifest(output, rows, DATASET_COLUMNS)
    return output


def apply_limits(
    rows: list[dict[str, object]],
    experiment_cfg: dict[str, Any],
) -> list[dict[str, object]]:
    limits = experiment_cfg.get("limits", {})
    if not limits:
        return rows

    max_by_split = {
        "train": limits.get("max_train_images"),
        "val": limits.get("max_val_images"),
        "test": limits.get("max_test_images"),
    }
    limited = []
    for split, max_count in max_by_split.items():
        split_rows = [row for row in rows if row["split"] == split]
        if max_count is not None:
            strategy = limits.get(f"{split}_sampling", "head")
            split_rows = limited_rows(split_rows, int(max_count), str(strategy))
        limited.extend(split_rows)
    return limited


def limited_rows(
    rows: list[dict[str, object]],
    max_count: int,
    strategy: str,
) -> list[dict[str, object]]:
    if max_count >= len(rows):
        return rows
    if max_count < 0:
        raise ValueError("dataset limits must be non-negative")
    if strategy == "head":
        return rows[:max_count]
    if strategy != "balanced_labels":
        raise ValueError(f"unsupported subset sampling strategy: {strategy}")

    labels = sorted({str(row["label"]) for row in rows})
    base, remainder = divmod(max_count, len(labels))
    selected = []
    for offset, label in enumerate(labels):
        label_rows = [row for row in rows if str(row["label"]) == label]
        take = base + (1 if offset < remainder else 0)
        selected.extend(label_rows[:take])
    selected_paths = {str(row["image_path"]) for row in selected}
    return [row for row in rows if str(row["image_path"]) in selected_paths]


def build_pathmnist_manifest(
    dataset: str,
    dataset_cfg: dict[str, Any],
    paths: SynpathPaths,
) -> list[dict[str, object]]:
    try:
        from medmnist import INFO, PathMNIST
    except ImportError as exc:
        raise RuntimeError(
            "Install medmnist before preparing PathMNIST+."
        ) from exc

    raw_root = paths.data_root / "raw" / "medmnist"
    image_root = paths.data_root / "processed" / dataset / "images"
    raw_root.mkdir(parents=True, exist_ok=True)
    image_size = int(dataset_cfg.get("image_size", 224))
    label_map = {
        int(key): safe_label(value)
        for key, value in INFO["pathmnist"]["label"].items()
    }

    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        med_split = PathMNIST(
            split=split,
            download=True,
            size=image_size,
            root=str(raw_root),
        )
        imgs = med_split.imgs
        labels = med_split.labels.reshape(-1)
        for idx, (img, label_id) in enumerate(
            tqdm(
                zip(imgs, labels, strict=True),
                total=len(labels),
                desc=f"export {dataset} {split}",
            )
        ):
            label_int = int(label_id)
            label = label_map[label_int]
            out_path = image_root / split / label / f"{idx:08d}.png"
            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).convert("RGB").save(out_path)
            rows.append(
                {
                    "image_path": str(out_path),
                    "label": label,
                    "label_id": label_int,
                    "split": split,
                    "dataset": dataset,
                    "slide_id": "",
                    "patient_id": "",
                }
            )
    return rows


def build_spider_manifest(
    dataset: str,
    dataset_cfg: dict[str, Any],
    paths: SynpathPaths,
    experiment_cfg: dict[str, Any],
) -> list[dict[str, object]]:
    organ = dataset_cfg["organ"]
    root = paths.data_root / "raw" / "spider" / organ
    ensure_spider_downloaded(organ, root)
    extracted = find_spider_root(root)
    if extracted is None:
        raise RuntimeError(
            f"Could not find extracted SPIDER data under {root}. "
            "Expected metadata.json and an images/ directory."
        )

    metadata_path = extracted / "metadata.json"
    if metadata_path.exists():
        rows = rows_from_spider_metadata(dataset, extracted, metadata_path)
    else:
        rows = rows_from_image_tree(dataset, extracted)

    if not any(row["split"] == "val" for row in rows):
        rows = add_stratified_validation(rows, experiment_cfg)
    return rows


def ensure_spider_downloaded(organ: str, root: Path) -> None:
    if any(root.rglob("*")):
        extract_tar_shards(root)
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install huggingface-hub to download SPIDER."
        ) from exc
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=SPIDER_REPOS[organ],
        repo_type="dataset",
        local_dir=root,
        local_dir_use_symlinks=False,
    )
    extract_tar_shards(root)


def extract_tar_shards(root: Path) -> None:
    if any(
        (p.name == "metadata.json" or p.name == "images")
        for p in root.rglob("*")
    ):
        return
    shards = sorted(root.glob("*.tar.*"))
    if not shards:
        return
    pattern = str(root / "*.tar.*")
    subprocess.run(
        ["bash", "-lc", f"cat {pattern} | tar -xf - -C {root}"],
        check=True,
    )


def find_spider_root(root: Path) -> Path | None:
    candidates = [root] + [p.parent for p in root.rglob("metadata.json")]
    for candidate in candidates:
        if (candidate / "images").exists() or (
            candidate / "metadata.json"
        ).exists():
            return candidate
    return None


def rows_from_spider_metadata(
    dataset: str,
    root: Path,
    metadata_path: Path,
) -> list[dict[str, object]]:
    payload = json.loads(metadata_path.read_text())
    records = normalise_records(payload)
    rows = []
    label_ids: dict[str, int] = {}
    for idx, record in enumerate(records):
        image_path = find_value(
            record,
            ["image_path", "img_path", "path", "filename", "file", "image"],
        )
        label_value = find_value(
            record, ["label", "class", "class_name", "category"]
        )
        if image_path is None or label_value is None:
            continue
        resolved = resolve_image_path(root, str(image_path))
        if resolved is None:
            continue
        label = safe_label(label_value)
        label_ids.setdefault(label, len(label_ids))
        split = str(find_value(record, ["split", "subset"]) or "train").lower()
        if split == "valid":
            split = "val"
        rows.append(
            {
                "image_path": str(resolved),
                "label": label,
                "label_id": label_ids[label],
                "split": split,
                "dataset": dataset,
                "slide_id": find_value(record, ["slide_id", "slide", "wsi"])
                or "",
                "patient_id": find_value(record, ["patient_id", "case_id"])
                or "",
            }
        )
    if not rows:
        raise RuntimeError(f"No usable rows found in {metadata_path}")
    return rows


def normalise_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "items", "data", "annotations", "metadata"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [
            {"image_path": key, **value}
            for key, value in payload.items()
            if isinstance(value, dict)
        ]
    return []


def find_value(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def resolve_image_path(root: Path, image_path: str) -> Path | None:
    candidate = Path(image_path)
    candidates = [
        candidate,
        root / candidate,
        root / "images" / candidate,
        root / "images" / candidate.name,
    ]
    for item in candidates:
        if item.exists():
            return item.resolve()
    matches = list((root / "images").rglob(candidate.name))
    return matches[0].resolve() if matches else None


def rows_from_image_tree(dataset: str, root: Path) -> list[dict[str, object]]:
    rows = []
    label_ids: dict[str, int] = {}
    for image_path in sorted(root.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = image_path.relative_to(root).parts
        split = next(
            (part for part in parts if part in {"train", "val", "test"}), ""
        )
        if not split:
            continue
        split_index = parts.index(split)
        if len(parts) <= split_index + 1:
            continue
        label = safe_label(parts[split_index + 1])
        label_ids.setdefault(label, len(label_ids))
        slide_id = (
            parts[split_index + 2] if len(parts) > split_index + 3 else ""
        )
        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "label": label,
                "label_id": label_ids[label],
                "split": split,
                "dataset": dataset,
                "slide_id": slide_id,
                "patient_id": "",
            }
        )
    if not rows:
        raise RuntimeError(f"No image-tree rows found under {root}")
    return rows


def add_stratified_validation(
    rows: list[dict[str, object]],
    experiment_cfg: dict[str, Any],
) -> list[dict[str, object]]:
    train = [row for row in rows if row["split"] == "train"]
    other = [row for row in rows if row["split"] != "train"]
    if len(train) < 10:
        return rows
    val_fraction = float(experiment_cfg.get("validation_fraction", 0.1))
    labels = [row["label"] for row in train]
    try:
        train_rows, val_rows = train_test_split(
            train,
            test_size=val_fraction,
            random_state=42,
            stratify=labels,
        )
    except ValueError:
        train_rows, val_rows = train_test_split(
            train,
            test_size=val_fraction,
            random_state=42,
        )
    for row in train_rows:
        row["split"] = "train"
    for row in val_rows:
        row["split"] = "val"
    return train_rows + val_rows + other


def build_imagefolder_manifest(
    dataset: str,
    dataset_cfg: dict[str, Any],
) -> list[dict[str, object]]:
    metadata_csv = dataset_cfg.get("metadata_csv")
    if metadata_csv:
        import pandas as pd

        frame = pd.read_csv(metadata_csv)
        required = {"image_path", "label", "split"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"{metadata_csv} is missing columns: {', '.join(sorted(missing))}"
            )
        labels = {
            label: idx
            for idx, label in enumerate(sorted(frame["label"].unique()))
        }
        rows = []
        for _, row in frame.iterrows():
            label = safe_label(row["label"])
            rows.append(
                {
                    "image_path": row["image_path"],
                    "label": label,
                    "label_id": labels[row["label"]],
                    "split": row["split"],
                    "dataset": dataset,
                    "slide_id": row.get("slide_id", ""),
                    "patient_id": row.get("patient_id", ""),
                }
            )
        return rows
    raise RuntimeError(
        "iCAIRD/ImageFolder manifest generation requires metadata_csv in config."
    )


def build_pcam_hdf5_manifest(
    dataset: str,
    dataset_cfg: dict[str, Any],
    paths: SynpathPaths,
    experiment_cfg: dict[str, Any],
) -> list[dict[str, object]]:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Install h5py to prepare the PCam HDF5 dataset."
        ) from exc

    root = Path(
        dataset_cfg.get("root", paths.data_root / "raw" / "pcam")
    ).expanduser()
    image_root = Path(
        dataset_cfg.get(
            "image_root",
            paths.data_root / "processed" / dataset / "images",
        )
    ).expanduser()
    hdf5_key_images = str(dataset_cfg.get("image_key", "x"))
    hdf5_key_labels = str(dataset_cfg.get("label_key", "y"))
    label_names = {
        int(key): safe_label(value)
        for key, value in dataset_cfg.get(
            "label_names",
            {0: "non_metastatic", 1: "metastatic"},
        ).items()
    }
    split_mapping = dataset_cfg.get(
        "split_mapping",
        {"train": "train", "valid": "val", "test": "test"},
    )
    pcam_splits = dataset_cfg.get("pcam_splits", ["train", "valid", "test"])
    rows: list[dict[str, object]] = []
    for pcam_split in pcam_splits:
        split = normalise_split(pcam_split, split_mapping)
        x_path = ensure_pcam_h5(root, pcam_split, "x")
        y_path = ensure_pcam_h5(root, pcam_split, "y")
        with (
            h5py.File(x_path, "r") as x_handle,
            h5py.File(y_path, "r") as y_handle,
        ):
            if hdf5_key_images not in x_handle:
                raise RuntimeError(
                    f"{x_path} is missing HDF5 dataset {hdf5_key_images!r}"
                )
            if hdf5_key_labels not in y_handle:
                raise RuntimeError(
                    f"{y_path} is missing HDF5 dataset {hdf5_key_labels!r}"
                )

            images = x_handle[hdf5_key_images]
            labels = (
                np.asarray(y_handle[hdf5_key_labels]).reshape(-1).astype(int)
            )
            if len(images) != len(labels):
                raise RuntimeError(
                    f"PCam split {pcam_split} has {len(images)} images but "
                    f"{len(labels)} labels."
                )

            for idx in tqdm(
                range(len(labels)), desc=f"export {dataset} {split}"
            ):
                label_id = int(labels[idx])
                if label_id not in label_names:
                    raise RuntimeError(
                        f"PCam label id {label_id} in {y_path} is not configured."
                    )
                label = label_names[label_id]
                out_path = image_root / split / label / f"{idx:08d}.png"
                if not out_path.exists():
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(images[idx]).convert("RGB").save(out_path)
                rows.append(
                    {
                        "image_path": str(out_path),
                        "label": label,
                        "label_id": label_id,
                        "split": split,
                        "dataset": dataset,
                        "slide_id": "",
                        "patient_id": "",
                    }
                )

    if not rows:
        raise RuntimeError(f"No usable PCam rows found under {root}")
    return rows


def ensure_pcam_h5(root: Path, split: str, suffix: str) -> Path:
    path = root / f"camelyonpatch_level_2_split_{split}_{suffix}.h5"
    if path.exists():
        return path
    gz_path = path.with_suffix(path.suffix + ".gz")
    if gz_path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(gz_path, "rb") as source, path.open("wb") as target:
            shutil.copyfileobj(source, target)
        return path
    raise RuntimeError(
        f"Missing PCam file {path} or {gz_path}. Run "
        "`python scripts/setup/download_pcam.py --root <pcam-root>` or place the "
        "official HDF5 files there."
    )


def build_pcam_kaggle_manifest(
    dataset: str,
    dataset_cfg: dict[str, Any],
) -> list[dict[str, object]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install pandas to prepare Kaggle PCam.") from exc

    root = Path(dataset_cfg["root"]).expanduser()
    labels_path = root / "train_labels.csv"
    image_root = root / "train"
    expected = int(dataset_cfg.get("expected_train", 220_025))
    if not labels_path.is_file():
        raise RuntimeError(f"missing Kaggle labels: {labels_path}")
    if not image_root.is_dir():
        raise RuntimeError(f"missing Kaggle training images: {image_root}")

    labels = pd.read_csv(labels_path, dtype={"id": str, "label": int})
    if list(labels.columns) != ["id", "label"]:
        raise RuntimeError(
            f"unexpected columns in {labels_path}: {list(labels.columns)}"
        )
    if len(labels) != expected:
        raise RuntimeError(
            f"Kaggle labels contain {len(labels)} rows; expected {expected}"
        )
    if labels["id"].duplicated().any():
        raise RuntimeError("Kaggle labels contain duplicate image IDs")
    label_names = {0: "non_metastatic", 1: "metastatic"}
    observed_labels = set(labels["label"].unique())
    if observed_labels != set(label_names):
        raise RuntimeError(f"unexpected Kaggle labels: {observed_labels}")

    image_paths = labels["id"].map(lambda value: image_root / f"{value}.tif")
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"missing {len(missing)} labelled Kaggle images; first: {missing[0]}"
        )
    image_ids = {path.stem for path in image_root.glob("*.tif")}
    labelled_ids = set(labels["id"])
    if image_ids != labelled_ids:
        raise RuntimeError(
            "Kaggle train directory and train_labels.csv do not have identical IDs"
        )

    rows = []
    for image_path, label_id in zip(image_paths, labels["label"], strict=True):
        rows.append(
            {
                "image_path": str(image_path),
                "label": label_names[int(label_id)],
                "label_id": int(label_id),
                "split": "train",
                "dataset": dataset,
                "slide_id": "",
                "patient_id": "",
            }
        )
    return rows


def build_indexed_parquet_manifest(
    dataset: str,
    dataset_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
) -> list[dict[str, object]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Install pandas/pyarrow to read indexed parquet datasets."
        ) from exc

    index_path = Path(dataset_cfg["index_parquet"]).expanduser()
    root = Path(dataset_cfg.get("root", index_path.parent)).expanduser()
    image_root = Path(
        dataset_cfg.get("image_root", root / "images")
    ).expanduser()
    if not index_path.exists():
        raise RuntimeError(f"Indexed parquet does not exist: {index_path}")

    frame = pd.read_parquet(index_path)
    label_col = str(dataset_cfg.get("label_column", "annotation_label_name"))
    path_col = configured_or_first_existing(
        frame,
        dataset_cfg,
        "path_column",
        ["patch_png_path", "image_path", "path", "filename", "file"],
    )
    split_col = configured_or_first_existing(
        frame,
        dataset_cfg,
        "split_column",
        ["split", "dataset_split", "fold", "partition"],
        required=False,
    )
    slide_col = configured_or_first_existing(
        frame,
        dataset_cfg,
        "slide_column",
        ["slide_id", "slide", "wsi", "wsi_id", "case_id"],
        required=False,
    )
    patient_col = configured_or_first_existing(
        frame,
        dataset_cfg,
        "patient_column",
        ["patient_id", "patient", "case_id"],
        required=False,
    )
    id_col = configured_or_first_existing(
        frame,
        dataset_cfg,
        "id_column",
        ["sample_id", "id", "patch_id"],
        required=False,
    )
    if label_col not in frame.columns:
        raise RuntimeError(
            f"{index_path} is missing label column {label_col!r}; "
            f"available columns: {list(frame.columns)}"
        )

    labels = sorted(
        safe_label(value) for value in frame[label_col].dropna().unique()
    )
    label_ids = {label: idx for idx, label in enumerate(labels)}
    split_mapping = dataset_cfg.get(
        "split_mapping",
        {"valid": "val", "validation": "val", "dev": "val"},
    )

    rows: list[dict[str, object]] = []
    missing_paths: list[str] = []
    for idx, item in enumerate(frame.to_dict("records")):
        label_value = item.get(label_col)
        path_value = item.get(path_col)
        if label_value is None or path_value is None:
            continue
        resolved = resolve_indexed_image_path(root, image_root, str(path_value))
        if resolved is None:
            if len(missing_paths) < 10:
                missing_paths.append(str(path_value))
            continue
        label = safe_label(label_value)
        split = normalise_split(
            item.get(split_col)
            if split_col
            else dataset_cfg.get("default_split", "train"),
            split_mapping,
        )
        sample_id = item.get(id_col) if id_col else resolved.stem
        rows.append(
            {
                "image_path": str(resolved),
                "label": label,
                "label_id": label_ids[label],
                "split": split,
                "dataset": dataset,
                "slide_id": item.get(slide_col, "") if slide_col else sample_id,
                "patient_id": item.get(patient_col, "") if patient_col else "",
            }
        )

    if missing_paths:
        raise RuntimeError(
            f"{index_path} contains image paths that could not be resolved under "
            f"{root} or {image_root}. Examples: {missing_paths}"
        )
    if not rows:
        raise RuntimeError(
            f"No usable indexed parquet rows found in {index_path}"
        )
    if not any(row["split"] == "val" for row in rows):
        rows = add_stratified_validation(rows, experiment_cfg)
    return rows


def configured_or_first_existing(
    frame,
    dataset_cfg: dict[str, Any],
    config_key: str,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    configured = dataset_cfg.get(config_key)
    if configured:
        if configured not in frame.columns:
            raise RuntimeError(
                f"Configured {config_key}={configured!r} was not found; "
                f"available columns: {list(frame.columns)}"
            )
        return str(configured)
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise RuntimeError(
            f"None of the expected columns {candidates} were found; "
            f"available columns: {list(frame.columns)}"
        )
    return None


def resolve_indexed_image_path(
    root: Path, image_root: Path, image_path: str
) -> Path | None:
    candidate = Path(image_path)
    image_relative = relative_to_named_part(candidate, "images")
    candidates = [
        candidate,
        root / candidate,
        image_root / candidate,
        image_root / candidate.name,
    ]
    if image_relative is not None:
        candidates.insert(1, image_root / image_relative)
    for item in candidates:
        if item.exists():
            return item.resolve()
    if image_root.exists():
        matches = list(image_root.rglob(candidate.name))
        if matches:
            return matches[0].resolve()
    return None


def relative_to_named_part(path: Path, part_name: str) -> Path | None:
    parts = path.parts
    if part_name not in parts:
        return None
    index = parts.index(part_name)
    if index + 1 >= len(parts):
        return None
    return Path(*parts[index + 1 :])


def normalise_split(value: object, split_mapping: dict[str, str]) -> str:
    split = str(value).strip().lower()
    split = split_mapping.get(split, split)
    if split == "valid":
        split = "val"
    return split


def copy_or_resize_image(src: Path, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if size <= 0:
        shutil.copy2(src, dst)
        return
    image = Image.open(src).convert("RGB").resize((size, size))
    image.save(dst)
