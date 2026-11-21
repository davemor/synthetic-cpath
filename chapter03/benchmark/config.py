from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    output_dir: Path = Path("runs/icaird_cervical")
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class DataConfig:
    dataset_root: Path
    image_size: int = 224
    num_workers: int = 4
    label_index_path: Path | None = None
    label_column: str | None = None


@dataclass
class ImageSearchConfig:
    method: str = "random"
    n_iter: int = 12
    learning_rate: list[float] = field(
        default_factory=lambda: [1e-5, 3e-5, 1e-4, 3e-4]
    )
    weight_decay: list[float] = field(default_factory=lambda: [0.0, 1e-4, 1e-2])


@dataclass
class ImageConfig:
    enabled: bool = True
    models: list[str] = field(
        default_factory=lambda: [
            "resnet18",
            "densenet121",
            "efficientnet_b4",
            "convnext_tiny",
            "swin_tiny_patch4_window7_224",
        ]
    )
    optional_models: list[str] = field(default_factory=lambda: ["vit_b_16"])
    epochs: int = 100
    batch_size: int = 64
    patience: int = 10
    pretrained: bool = True
    search: ImageSearchConfig = field(default_factory=ImageSearchConfig)


@dataclass
class SearchConfig:
    method: str = "grid"
    n_iter: int | None = None
    random_state: int = 0


@dataclass
class ProbeClassifierConfig:
    params: dict[str, list[Any]] = field(default_factory=dict)
    search: SearchConfig = field(default_factory=SearchConfig)


@dataclass
class EmbeddingConfig:
    path: Path
    citation: str | None = None
    expected_dim: int | None = None
    label_column: str | None = None


@dataclass
class ProbeConfig:
    enabled: bool = True
    cv_folds: int = 5
    embeddings: dict[str, EmbeddingConfig] = field(default_factory=dict)
    classifiers: dict[str, ProbeClassifierConfig] = field(default_factory=dict)


@dataclass
class MilConfig:
    enabled: bool = False
    bag_root: Path | None = None
    embedding_name: str = "uni2_h"
    embedding_dim: int = 1536
    tasks: list[str] = field(default_factory=lambda: ["malignant", "category", "subcategory"])
    epochs: int = 200
    batch_size: int = 1
    d_model: int = 512
    encoder_layers: int = 4
    heads: int = 4
    dim_feedforward: int = 2048
    activation: str = "gelu"
    dropout: float = 0.15
    lr: float = 1.0e-5
    weight_decay: float = 0.0
    balance_classes: bool = True
    use_amp: bool = True


@dataclass
class Config:
    run: RunConfig
    data: DataConfig
    images: ImageConfig
    probes: ProbeConfig
    mil: MilConfig


DEFAULT_PROBE_CLASSIFIERS = {
    "logistic-regression": {
        "params": {
            "C": [1, 100],
            "max_iter": [2000],
            "linesearch_max_iter": [300],
            "tol": [1.0e-5],
        }
    },
    "mlp": {
        "params": {
            "hidden_layer_sizes": [[100, 50]],
            "activation": ["relu"],
            "solver": ["adam"],
            "alpha": [0.001],
            "learning_rate": ["constant"],
            "learning_rate_init": [0.001],
            "max_iter": [200],
            "early_stopping": [True],
        }
    },
    "xgboost": {
        "search": {"method": "random", "n_iter": 12, "random_state": 0},
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 6],
            "min_child_weight": [1, 8],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "gamma": [0, 0.5],
            "reg_lambda": [1.0, 10.0],
            "reg_alpha": [0, 0.001],
        },
    },
}


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _run_config(data: dict[str, Any]) -> RunConfig:
    return RunConfig(
        output_dir=_as_path(data.get("output_dir", "runs/icaird_cervical")),
        seeds=[int(seed) for seed in data.get("seeds", [0, 1, 2])],
    )


def _data_config(data: dict[str, Any]) -> DataConfig:
    if "dataset_root" not in data:
        raise ValueError("config.data.dataset_root is required.")
    return DataConfig(
        dataset_root=_as_path(data["dataset_root"]),
        image_size=int(data.get("image_size", 224)),
        num_workers=int(data.get("num_workers", 4)),
        label_index_path=_as_path(data["label_index_path"]) if data.get("label_index_path") else None,
        label_column=data.get("label_column"),
    )


def _image_config(data: dict[str, Any]) -> ImageConfig:
    search_data = data.get("search", {})
    return ImageConfig(
        enabled=bool(data.get("enabled", True)),
        models=list(data.get("models", ImageConfig().models)),
        optional_models=list(data.get("optional_models", ImageConfig().optional_models)),
        epochs=int(data.get("epochs", 100)),
        batch_size=int(data.get("batch_size", 64)),
        patience=int(data.get("patience", 10)),
        pretrained=bool(data.get("pretrained", True)),
        search=ImageSearchConfig(
            method=search_data.get("method", "random"),
            n_iter=int(search_data.get("n_iter", 12)),
            learning_rate=[float(x) for x in search_data.get("learning_rate", [1e-5, 3e-5, 1e-4, 3e-4])],
            weight_decay=[float(x) for x in search_data.get("weight_decay", [0.0, 1e-4, 1e-2])],
        ),
    )


def _embedding_config(value: Any) -> EmbeddingConfig:
    if isinstance(value, dict):
        return EmbeddingConfig(
            path=_as_path(value["path"]),
            citation=value.get("citation"),
            expected_dim=value.get("expected_dim"),
            label_column=value.get("label_column"),
        )
    return EmbeddingConfig(path=_as_path(value))


def _search_config(data: dict[str, Any] | None) -> SearchConfig:
    data = data or {}
    return SearchConfig(
        method=data.get("method", "grid"),
        n_iter=data.get("n_iter"),
        random_state=int(data.get("random_state", 0)),
    )


def _probe_classifier_config(data: dict[str, Any]) -> ProbeClassifierConfig:
    return ProbeClassifierConfig(
        params=data.get("params", {}),
        search=_search_config(data.get("search")),
    )


def _probe_config(data: dict[str, Any]) -> ProbeConfig:
    classifier_data = data.get("classifiers") or DEFAULT_PROBE_CLASSIFIERS
    return ProbeConfig(
        enabled=bool(data.get("enabled", True)),
        cv_folds=int(data.get("cv_folds", 5)),
        embeddings={
            name: _embedding_config(value)
            for name, value in data.get("embeddings", {}).items()
        },
        classifiers={
            name: _probe_classifier_config(value)
            for name, value in classifier_data.items()
        },
    )


def _mil_config(data: dict[str, Any]) -> MilConfig:
    bag_root = data.get("bag_root")
    return MilConfig(
        enabled=bool(data.get("enabled", False)),
        bag_root=_as_path(bag_root) if bag_root else None,
        embedding_name=str(data.get("embedding_name", "uni2_h")),
        embedding_dim=int(data.get("embedding_dim", 1536)),
        tasks=list(data.get("tasks", ["malignant", "category", "subcategory"])),
        epochs=int(data.get("epochs", 200)),
        batch_size=int(data.get("batch_size", 1)),
        d_model=int(data.get("d_model", 512)),
        encoder_layers=int(data.get("encoder_layers", 4)),
        heads=int(data.get("heads", 4)),
        dim_feedforward=int(data.get("dim_feedforward", 2048)),
        activation=str(data.get("activation", "gelu")),
        dropout=float(data.get("dropout", 0.15)),
        lr=float(data.get("lr", 1.0e-5)),
        weight_decay=float(data.get("weight_decay", 0.0)),
        balance_classes=bool(data.get("balance_classes", True)),
        use_amp=bool(data.get("use_amp", True)),
    )


def load_config(path: str | Path) -> Config:
    with Path(path).open() as handle:
        raw = yaml.safe_load(handle) or {}
    return Config(
        run=_run_config(raw.get("run", {})),
        data=_data_config(raw.get("data", {})),
        images=_image_config(raw.get("images", {})),
        probes=_probe_config(raw.get("probes", {})),
        mil=_mil_config(raw.get("mil", {})),
    )
