from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
import sklearn
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from synpath.artifacts import file_sha256, utc_now, write_json
from synpath.calibration.common import calibration_root, load_feature_array
from synpath.calibration.curation import curated_manifest_for_dataset
from synpath.paths import SynpathPaths


@dataclass(frozen=True)
class FeatureArtifact:
    features_path: Path
    features_key: str | None
    index_path: Path
    metadata_path: Path


class ImageDataset(Dataset):
    def __init__(self, paths: list[str], transform: Callable):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def load_train_index(
    cfg: dict,
    dataset: str,
    paths: SynpathPaths,
) -> tuple[pd.DataFrame, str]:
    spec = cfg["datasets"][dataset]
    curated = curated_manifest_for_dataset(cfg, dataset, paths)
    manifest_path = curated[0] if curated else Path(spec["manifest"])
    frame = pd.read_csv(manifest_path)
    required = {"image_path", "label", "label_id", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"{manifest_path} is missing columns: {', '.join(sorted(missing))}"
        )
    train_mask = frame["split"].astype(str) == "train"
    train = frame.loc[train_mask].copy().reset_index(drop=True)
    train.insert(0, "manifest_row", np.arange(len(train), dtype=np.int64))
    train.insert(1, "sample_id", train["image_path"].map(sample_id))
    if train["sample_id"].duplicated().any():
        duplicate = train.loc[
            train["sample_id"].duplicated(), "sample_id"
        ].iloc[0]
        raise RuntimeError(
            f"duplicate train sample_id in {manifest_path}: {duplicate}"
        )
    expected = curated[1] if curated else int(spec["expected_train"])
    if len(train) != expected:
        raise RuntimeError(
            f"{dataset} has {len(train)} train rows; expected exactly {expected}"
        )
    missing_images = [
        path for path in train["image_path"] if not Path(path).is_file()
    ]
    if missing_images:
        raise RuntimeError(
            f"{dataset} has {len(missing_images)} missing train images; "
            f"first missing: {missing_images[0]}"
        )
    return train, index_fingerprint(train)


def sample_id(image_path: str) -> str:
    return Path(image_path).stem


def index_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False):
        payload = [
            str(row.sample_id),
            str(row.label),
            int(row.label_id),
            str(getattr(row, "split", "train")),
        ]
        digest.update(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_features(
    cfg: dict,
    dataset: str,
    encoder: str,
    paths: SynpathPaths,
    resume: bool,
) -> FeatureArtifact:
    train, fingerprint = load_train_index(cfg, dataset, paths)
    root = calibration_root(paths, cfg) / dataset / encoder
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "train_index.parquet"
    metadata_path = root / "features_metadata.json"
    encoder_cfg = cfg.get("encoders", {}).get(encoder, {})
    expected_dimension = encoder_cfg.get("dimension")
    if resume and metadata_path.is_file():
        artifact = artifact_from_metadata(metadata_path, index_path)
        validate_cached_artifact(
            artifact, fingerprint, len(train), expected_dimension
        )
    else:
        feature_reuse = cfg["datasets"][dataset].get("feature_reuse", {})
        reuse = feature_reuse.get(encoder)
        if reuse is None and encoder == "uni2":
            reuse = cfg["datasets"][dataset].get("uni2_reuse")
        if reuse:
            artifact = prepare_reused_features(
                reuse,
                train,
                fingerprint,
                root,
                index_path,
                metadata_path,
                dataset,
                encoder,
                encoder_cfg,
            )
        else:
            artifact = extract_features(
                cfg,
                train,
                fingerprint,
                root,
                index_path,
                metadata_path,
                dataset,
                encoder,
            )
        validate_cached_artifact(
            artifact, fingerprint, len(train), expected_dimension
        )
    fit_transform = (
        cfg.get("encoders", {}).get(encoder, {}).get("fit_transform")
    )
    if fit_transform:
        prepare_fit_transform(
            artifact,
            fit_transform,
            resume,
        )
    return artifact


def prepare_fit_transform(
    artifact: FeatureArtifact,
    transform: str,
    resume: bool,
) -> None:
    if transform != "standard_scaler":
        raise RuntimeError(
            f"unsupported calibration fit transform: {transform}"
        )
    root = artifact.metadata_path.parent
    transformed_path = root / "train_features_standardized.npy"
    scaler_path = root / "standard_scaler.npz"
    metadata = json.loads(artifact.metadata_path.read_text())
    if resume and transformed_path.is_file() and scaler_path.is_file():
        transformed = np.load(transformed_path, mmap_mode="r")
        source = load_feature_array(
            artifact.features_path, artifact.features_key
        )
        if transformed.shape != source.shape:
            raise RuntimeError(
                "cached standardized features have the wrong shape"
            )
        metadata.update(
            {
                "fit_feature_path": str(transformed_path),
                "fit_feature_key": None,
                "fit_transform": "sklearn.preprocessing.StandardScaler",
                "fit_transform_space": "global_training_population",
                "standard_scaler_path": str(scaler_path),
                "sklearn_version": sklearn.__version__,
            }
        )
        write_json(artifact.metadata_path, metadata)
        return

    source = load_feature_array(artifact.features_path, artifact.features_key)
    scaler = StandardScaler().fit(source)
    output = np.lib.format.open_memmap(
        transformed_path,
        mode="w+",
        dtype=np.float32,
        shape=source.shape,
    )
    for start in range(0, len(source), 4096):
        stop = min(start + 4096, len(source))
        output[start:stop] = scaler.transform(source[start:stop]).astype(
            np.float32, copy=False
        )
    output.flush()
    del output
    np.savez(
        scaler_path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        variance=scaler.var_,
        n_samples_seen=scaler.n_samples_seen_,
    )
    metadata.update(
        {
            "fit_feature_path": str(transformed_path),
            "fit_feature_key": None,
            "fit_transform": "sklearn.preprocessing.StandardScaler",
            "fit_transform_space": "global_training_population",
            "standard_scaler_path": str(scaler_path),
            "sklearn_version": sklearn.__version__,
        }
    )
    write_json(artifact.metadata_path, metadata)


def prepare_reused_features(
    reuse: dict,
    train: pd.DataFrame,
    fingerprint: str,
    root: Path,
    index_path: Path,
    metadata_path: Path,
    dataset: str,
    encoder: str,
    encoder_cfg: dict,
) -> FeatureArtifact:
    source_features = Path(reuse["features"])
    source_key = reuse.get("features_key", "features")
    if not source_features.is_file():
        raise RuntimeError(
            f"missing configured UNI2 feature source: {source_features}"
        )

    kind = reuse["kind"]
    output_features = source_features
    output_key = source_key
    if kind == "ordered_manifest":
        source = pd.read_csv(Path(reuse["manifest"]))
        source = (
            source.loc[source["split"] == "train"].copy().reset_index(drop=True)
        )
        source.insert(0, "sample_id", source["image_path"].map(sample_id))
        source.insert(0, "manifest_row", np.arange(len(source), dtype=np.int64))
        if index_fingerprint(source) != fingerprint:
            raise RuntimeError(
                f"configured UNI2 source manifest does not align with {dataset} train rows"
            )
        features = load_feature_array(source_features, source_key)
        if len(features) != len(train):
            raise RuntimeError(
                "ordered UNI2 feature source has the wrong row count"
            )
    elif kind == "indexed":
        source_index = pd.read_parquet(Path(reuse["index"]))
        source_id_column = reuse.get("sample_id_column", "sample_id")
        if source_id_column not in source_index:
            raise RuntimeError(
                f"UNI2 source index is missing {source_id_column!r}"
            )
        if source_index[source_id_column].duplicated().any():
            raise RuntimeError(
                "UNI2 source index contains duplicate sample IDs"
            )
        lookup = pd.Series(
            np.arange(len(source_index), dtype=np.int64),
            index=source_index[source_id_column].astype(str),
        )
        desired_ids = train["sample_id"].astype(str)
        missing = desired_ids.loc[~desired_ids.isin(lookup.index)]
        if len(missing):
            raise RuntimeError(
                f"UNI2 source index is missing {len(missing)} train IDs; "
                f"first missing: {missing.iloc[0]}"
            )
        positions = lookup.loc[desired_ids].to_numpy(dtype=np.int64)
        label_column = reuse.get("label_column")
        if label_column:
            source_labels = (
                source_index.iloc[positions][label_column]
                .astype(str)
                .to_numpy()
            )
            target_labels = train["label"].astype(str).to_numpy()
            if not np.array_equal(source_labels, target_labels):
                raise RuntimeError(
                    "UNI2 source labels do not align with train labels"
                )
        features = load_feature_array(source_features, source_key)
        output_features = root / "train_features.npy"
        aligned = np.lib.format.open_memmap(
            output_features,
            mode="w+",
            dtype=np.float32,
            shape=(len(train), features.shape[1]),
        )
        for start in range(0, len(train), 4096):
            stop = min(start + 4096, len(train))
            aligned[start:stop] = features[positions[start:stop]]
        aligned.flush()
        del aligned
        output_key = None
    else:
        raise RuntimeError(f"unsupported UNI2 reuse kind: {kind}")

    features = load_feature_array(output_features, output_key)
    expected_dimension = encoder_cfg.get("dimension")
    if expected_dimension and features.shape[1] != int(expected_dimension):
        raise RuntimeError(
            f"reused {encoder} features have dimension {features.shape[1]}; "
            f"expected {expected_dimension}"
        )
    source_metadata_path = reuse.get("source_metadata")
    source_metadata_sha256 = None
    if source_metadata_path:
        source_metadata_path = Path(source_metadata_path)
        if not source_metadata_path.is_file():
            raise RuntimeError(
                f"reused feature metadata is missing: {source_metadata_path}"
            )
        source_metadata = json.loads(source_metadata_path.read_text())
        expected = {
            "encoder": encoder,
            "model": encoder_cfg.get("model"),
            "preprocessing": encoder_cfg.get(
                "preprocessing", "model_creator_defaults"
            ),
        }
        for field, value in expected.items():
            if value is not None and source_metadata.get(field) != value:
                raise RuntimeError(
                    f"reused feature {field} does not match: "
                    f"{source_metadata.get(field)!r} != {value!r}"
                )
        source_metadata_sha256 = file_sha256(source_metadata_path)

    write_feature_index(train, index_path)
    write_json(
        metadata_path,
        {
            "created_at": utc_now(),
            "dataset": dataset,
            "encoder": encoder,
            "model": encoder_cfg.get("model"),
            "dimension": features.shape[1],
            "preprocessing": encoder_cfg.get(
                "preprocessing", "model_creator_defaults"
            ),
            "feature_path": str(output_features),
            "feature_key": output_key,
            "feature_source_path": str(source_features),
            "feature_source_sha256": file_sha256(source_features),
            "feature_source_metadata": (
                None if source_metadata_path is None else str(source_metadata_path)
            ),
            "feature_source_metadata_sha256": source_metadata_sha256,
            "index_path": str(index_path),
            "manifest_fingerprint": fingerprint,
            "n_samples": len(train),
            "reused": True,
        },
    )
    return FeatureArtifact(
        output_features, output_key, index_path, metadata_path
    )


def extract_features(
    cfg: dict,
    train: pd.DataFrame,
    fingerprint: str,
    root: Path,
    index_path: Path,
    metadata_path: Path,
    dataset: str,
    encoder: str,
) -> FeatureArtifact:
    if not torch.cuda.is_available():
        raise RuntimeError("calibration feature extraction requires CUDA")
    encoder_cfg = cfg["encoders"][encoder]
    device = torch.device("cuda")
    model, transform = load_encoder(encoder, encoder_cfg, device)
    data_parallel = (
        os.environ.get("SYNPATH_EMBED_DATA_PARALLEL", "0") == "1"
        and torch.cuda.device_count() > 1
    )
    if data_parallel:
        model = torch.nn.DataParallel(model)
    workers = int(cfg.get("resources", {}).get("embedding_workers", 12))
    per_device_batch = int(encoder_cfg["batch_size"])
    batch_size = per_device_batch * (
        torch.cuda.device_count() if data_parallel else 1
    )
    loader = DataLoader(
        ImageDataset(train["image_path"].tolist(), transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    expected_dim = int(encoder_cfg["dimension"])
    feature_path = root / "train_features.npy"
    output = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(train), expected_dim),
    )
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc=f"{dataset} {encoder}"):
            values = model(images.to(device, non_blocking=True))
            if encoder == "dino_vitb8":
                values = values.last_hidden_state[:, 0]
            values = values.detach().float().cpu().numpy()
            if values.shape[1] != expected_dim:
                raise RuntimeError(
                    f"{encoder} returned {values.shape[1]} features; expected {expected_dim}"
                )
            output[indices.numpy()] = values
    output.flush()
    del output
    write_feature_index(train, index_path)
    write_json(
        metadata_path,
        {
            "created_at": utc_now(),
            "dataset": dataset,
            "encoder": encoder,
            "model": encoder_cfg["model"],
            "feature_path": str(feature_path),
            "feature_key": None,
            "index_path": str(index_path),
            "manifest_fingerprint": fingerprint,
            "n_samples": len(train),
            "dimension": expected_dim,
            "preprocessing": encoder_cfg.get(
                "preprocessing", "model_creator_defaults"
            ),
            "projection": "none",
            "cuda_devices": torch.cuda.device_count(),
            "per_device_batch_size": per_device_batch,
            "effective_batch_size": batch_size,
            "data_parallel": data_parallel,
            "reused": False,
        },
    )
    return FeatureArtifact(feature_path, None, index_path, metadata_path)


def load_encoder(
    encoder: str,
    cfg: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, Callable]:
    if encoder == "dino_vitb8":
        try:
            from transformers import (
                AutoFeatureExtractor,
                AutoImageProcessor,
                AutoModel,
            )
        except ImportError as exc:
            raise RuntimeError(
                "install the synpath gpu extra for DINO"
            ) from exc
        preprocessing = cfg.get("preprocessing", "model_creator_defaults")
        if preprocessing == "model_creator_defaults":
            processor = AutoImageProcessor.from_pretrained(cfg["model"])
        elif preprocessing == "osorio_released_tensor_rescale":
            processor = AutoFeatureExtractor.from_pretrained(cfg["model"])
        else:
            raise RuntimeError(
                f"unsupported DINO preprocessing mode: {preprocessing}"
            )
        model = AutoModel.from_pretrained(cfg["model"]).eval().to(device)

        def transform(image: Image.Image) -> torch.Tensor:
            processor_input = dino_processor_input(image, preprocessing)
            return processor(images=processor_input, return_tensors="pt")[
                "pixel_values"
            ][0]

        return model, transform
    if encoder == "uni2":
        import timm

        kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        model = timm.create_model(cfg["model"], pretrained=True, **kwargs)
        model.eval().to(device)
        data_cfg = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_cfg, is_training=False)
        return model, transform
    raise RuntimeError(f"unsupported calibration encoder: {encoder}")


def dino_processor_input(
    image: Image.Image, preprocessing: str
) -> Image.Image | torch.Tensor:
    if preprocessing == "model_creator_defaults":
        return image
    if preprocessing == "osorio_released_tensor_rescale":
        from torchvision.transforms.functional import to_tensor

        # This intentionally reproduces the released Osorio implementation.
        # AutoFeatureExtractor rescales this already-scaled tensor a second time.
        return to_tensor(image)
    raise RuntimeError(f"unsupported DINO preprocessing mode: {preprocessing}")


def artifact_from_metadata(
    metadata_path: Path, index_path: Path
) -> FeatureArtifact:
    metadata = json.loads(metadata_path.read_text())
    return FeatureArtifact(
        Path(metadata["feature_path"]),
        metadata.get("feature_key"),
        index_path,
        metadata_path,
    )


def validate_cached_artifact(
    artifact: FeatureArtifact,
    fingerprint: str,
    expected_rows: int,
    expected_dimension: int | None = None,
) -> None:
    metadata = json.loads(artifact.metadata_path.read_text())
    if metadata["manifest_fingerprint"] != fingerprint:
        raise RuntimeError("cached feature manifest fingerprint does not match")
    if int(metadata["n_samples"]) != expected_rows:
        raise RuntimeError("cached feature row count does not match")
    if (
        not artifact.features_path.is_file()
        or not artifact.index_path.is_file()
    ):
        raise RuntimeError("cached feature artifact is incomplete")
    features = load_feature_array(artifact.features_path, artifact.features_key)
    index = pd.read_parquet(artifact.index_path)
    if len(features) != expected_rows or len(index) != expected_rows:
        raise RuntimeError("cached features and index do not align")
    if expected_dimension and features.shape[1] != int(expected_dimension):
        raise RuntimeError("cached feature dimension does not match")
    if index["sample_id"].duplicated().any():
        raise RuntimeError("cached feature index contains duplicate sample IDs")
    if "split" not in index or set(index["split"].astype(str)) != {"train"}:
        raise RuntimeError("cached feature index is not training-only")


def write_feature_index(train: pd.DataFrame, path: Path) -> None:
    if set(train["split"].astype(str)) != {"train"}:
        raise RuntimeError("feature indexes may contain training rows only")
    columns = [
        "manifest_row",
        "sample_id",
        "image_path",
        "label",
        "label_id",
        "split",
    ]
    train[columns].to_parquet(path, index=False)
