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
    require_cuda: bool = False
    augmentation_suites: list[str] = field(default_factory=lambda: ["none"])
    augmentation_probability: float = 0.75
    hed_sigma: float = 0.03
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
class SyntheticEmbeddingConfig:
    enabled: bool = False
    embedding_dir: Path | None = None
    output_dir: Path = Path("/home/dm236/scratch/synthetic_embeddings/icaird100k_uni2h_vitdae_like")
    label_column: str = "annotation_label_name"
    split_column: str = "split"
    embedding_dim: int = 1536
    class_order: list[str] = field(
        default_factory=lambda: ["normal", "low_grade", "high_grade", "malignant"]
    )
    validation_fraction: float = 0.1
    model_mode: str = "conditional"
    sample_sizes: list[int] = field(default_factory=lambda: [10_000, 50_000, 100_000])
    seed: int = 0
    batch_size: int = 256
    num_workers: int = 4
    max_seen_samples: int = 10_000_000
    validation_interval_steps: int = 500
    validation_batches: int = 16
    patience_validations: int = 20
    lr: float = 1.0e-4
    weight_decay: float = 0.01
    timesteps: int = 1000
    beta: float = 0.008
    hidden_dim: int = 2048
    depth: int = 10
    dropout: float = 0.0
    ema_decay: float = 0.999
    use_amp: bool = True
    ddim_steps: int = 20
    edm_sigma_min: float = 0.002
    edm_sigma_max: float = 80.0
    edm_sigma_data: float = 0.5
    edm_rho: float = 7.0
    edm_class_dropout: float = 0.15
    guidance_scale: float = 1.5
    eval_max_per_class: int = 2000
    eval_knn_k: int = 5
    utility_max_iter: int = 2000


@dataclass
class SyntheticBagConfig:
    enabled: bool = False
    real_bag_root: Path | None = None
    output_dir: Path = Path("/home/dm236/scratch/synthetic_bags/icaird_cervical_conditional_edm")
    embedding_name: str = "uni2_h"
    embedding_dim: int = 1536
    patch_label_column: str = "annotation_label_name"
    patch_class_order: list[str] = field(
        default_factory=lambda: ["normal", "low_grade", "high_grade", "malignant"]
    )
    normal_label_aliases: list[str] = field(default_factory=lambda: ["normal_inflammation"])
    sample_multipliers: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    seed: int = 0
    count_model: str = "lognormal"
    composition_model: str = "multinomial"
    clamp_total_count_to_train_range: bool = True
    max_bags_per_multiplier: int | None = None
    materialize_synthetic_only: bool = True
    materialize_real_plus_synthetic: bool = True
    include_real_valid_test: bool = True


@dataclass
class ClusterPseudoLabelConfig:
    enabled: bool = False
    source_embedding_dir: Path | None = None
    source_bag_root: Path | None = None
    output_embedding_dir: Path = Path(
        "/home/dm236/scratch/icaird_cervical/cluster_pseudolabels/icaird100k_uni2h_kmeans24_pca42/embeddings/uni2_h"
    )
    output_bag_root: Path = Path(
        "/home/dm236/scratch/icaird_cervical/cluster_pseudolabels/icaird_bags_uni2h_kmeans24_pca42"
    )
    artifact_dir: Path = Path(
        "/home/dm236/scratch/icaird_cervical/cluster_pseudolabels/icaird100k_uni2h_kmeans24_pca42/artifacts"
    )
    embedding_name: str = "uni2_h"
    split_column: str = "split"
    fit_split: str = "train"
    label_column: str = "cluster_pseudo_label"
    pca_components: int = 42
    kmeans_clusters: int = 24
    kmeans_init: str = "k-means++"
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    random_state: int = 0
    transform_batch_size: int = 65536


@dataclass
class Config:
    run: RunConfig
    data: DataConfig
    images: ImageConfig
    probes: ProbeConfig
    mil: MilConfig
    synthetic_embeddings: SyntheticEmbeddingConfig
    synthetic_bags: SyntheticBagConfig
    cluster_pseudolabels: ClusterPseudoLabelConfig


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
    augmentation_suites = data.get(
        "augmentation_suites",
        data.get("augmentations", data.get("augmentation_suite", ["none"])),
    )
    if isinstance(augmentation_suites, str):
        augmentation_suites = [augmentation_suites]
    return ImageConfig(
        enabled=bool(data.get("enabled", True)),
        models=list(data.get("models", ImageConfig().models)),
        optional_models=list(data.get("optional_models", ImageConfig().optional_models)),
        epochs=int(data.get("epochs", 100)),
        batch_size=int(data.get("batch_size", 64)),
        patience=int(data.get("patience", 10)),
        pretrained=bool(data.get("pretrained", True)),
        require_cuda=bool(data.get("require_cuda", False)),
        augmentation_suites=[str(value) for value in augmentation_suites],
        augmentation_probability=float(data.get("augmentation_probability", 0.75)),
        hed_sigma=float(data.get("hed_sigma", 0.03)),
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


def _synthetic_embedding_config(data: dict[str, Any]) -> SyntheticEmbeddingConfig:
    embedding_dir = data.get("embedding_dir")
    return SyntheticEmbeddingConfig(
        enabled=bool(data.get("enabled", False)),
        embedding_dir=_as_path(embedding_dir) if embedding_dir else None,
        output_dir=_as_path(
            data.get(
                "output_dir",
                "/home/dm236/scratch/synthetic_embeddings/icaird100k_uni2h_vitdae_like",
            )
        ),
        label_column=str(data.get("label_column", "annotation_label_name")),
        split_column=str(data.get("split_column", "split")),
        embedding_dim=int(data.get("embedding_dim", 1536)),
        class_order=[str(value) for value in data.get(
            "class_order",
            ["normal", "low_grade", "high_grade", "malignant"],
        )],
        validation_fraction=float(data.get("validation_fraction", 0.1)),
        model_mode=str(data.get("model_mode", "conditional")),
        sample_sizes=[int(value) for value in data.get("sample_sizes", [10_000, 50_000, 100_000])],
        seed=int(data.get("seed", 0)),
        batch_size=int(data.get("batch_size", 256)),
        num_workers=int(data.get("num_workers", 4)),
        max_seen_samples=int(data.get("max_seen_samples", 10_000_000)),
        validation_interval_steps=int(data.get("validation_interval_steps", 500)),
        validation_batches=int(data.get("validation_batches", 16)),
        patience_validations=int(data.get("patience_validations", 20)),
        lr=float(data.get("lr", 1.0e-4)),
        weight_decay=float(data.get("weight_decay", 0.01)),
        timesteps=int(data.get("timesteps", 1000)),
        beta=float(data.get("beta", 0.008)),
        hidden_dim=int(data.get("hidden_dim", 2048)),
        depth=int(data.get("depth", 10)),
        dropout=float(data.get("dropout", 0.0)),
        ema_decay=float(data.get("ema_decay", 0.999)),
        use_amp=bool(data.get("use_amp", True)),
        ddim_steps=int(data.get("ddim_steps", 20)),
        edm_sigma_min=float(data.get("edm_sigma_min", 0.002)),
        edm_sigma_max=float(data.get("edm_sigma_max", 80.0)),
        edm_sigma_data=float(data.get("edm_sigma_data", 0.5)),
        edm_rho=float(data.get("edm_rho", 7.0)),
        edm_class_dropout=float(data.get("edm_class_dropout", 0.15)),
        guidance_scale=float(data.get("guidance_scale", 1.5)),
        eval_max_per_class=int(data.get("eval_max_per_class", 2000)),
        eval_knn_k=int(data.get("eval_knn_k", 5)),
        utility_max_iter=int(data.get("utility_max_iter", 2000)),
    )


def _synthetic_bag_config(data: dict[str, Any]) -> SyntheticBagConfig:
    real_bag_root = data.get("real_bag_root")
    max_bags = data.get("max_bags_per_multiplier")
    return SyntheticBagConfig(
        enabled=bool(data.get("enabled", False)),
        real_bag_root=_as_path(real_bag_root) if real_bag_root else None,
        output_dir=_as_path(
            data.get(
                "output_dir",
                "/home/dm236/scratch/synthetic_bags/icaird_cervical_conditional_edm",
            )
        ),
        embedding_name=str(data.get("embedding_name", "uni2_h")),
        embedding_dim=int(data.get("embedding_dim", 1536)),
        patch_label_column=str(data.get("patch_label_column", "annotation_label_name")),
        patch_class_order=[str(value) for value in data.get(
            "patch_class_order",
            ["normal", "low_grade", "high_grade", "malignant"],
        )],
        normal_label_aliases=[str(value) for value in data.get("normal_label_aliases", ["normal_inflammation"])],
        sample_multipliers=[float(value) for value in data.get("sample_multipliers", [0.5, 1.0, 2.0])],
        seed=int(data.get("seed", 0)),
        count_model=str(data.get("count_model", "lognormal")),
        composition_model=str(data.get("composition_model", "multinomial")),
        clamp_total_count_to_train_range=bool(data.get("clamp_total_count_to_train_range", True)),
        max_bags_per_multiplier=int(max_bags) if max_bags is not None else None,
        materialize_synthetic_only=bool(data.get("materialize_synthetic_only", True)),
        materialize_real_plus_synthetic=bool(data.get("materialize_real_plus_synthetic", True)),
        include_real_valid_test=bool(data.get("include_real_valid_test", True)),
    )


def _cluster_pseudolabel_config(data: dict[str, Any]) -> ClusterPseudoLabelConfig:
    source_embedding_dir = data.get("source_embedding_dir")
    source_bag_root = data.get("source_bag_root")
    return ClusterPseudoLabelConfig(
        enabled=bool(data.get("enabled", False)),
        source_embedding_dir=_as_path(source_embedding_dir) if source_embedding_dir else None,
        source_bag_root=_as_path(source_bag_root) if source_bag_root else None,
        output_embedding_dir=_as_path(
            data.get(
                "output_embedding_dir",
                "/home/dm236/scratch/icaird_cervical/cluster_pseudolabels/"
                "icaird100k_uni2h_kmeans24_pca42/embeddings/uni2_h",
            )
        ),
        output_bag_root=_as_path(
            data.get(
                "output_bag_root",
                "/home/dm236/scratch/icaird_cervical/cluster_pseudolabels/"
                "icaird_bags_uni2h_kmeans24_pca42",
            )
        ),
        artifact_dir=_as_path(
            data.get(
                "artifact_dir",
                "/home/dm236/scratch/icaird_cervical/cluster_pseudolabels/"
                "icaird100k_uni2h_kmeans24_pca42/artifacts",
            )
        ),
        embedding_name=str(data.get("embedding_name", "uni2_h")),
        split_column=str(data.get("split_column", "split")),
        fit_split=str(data.get("fit_split", "train")),
        label_column=str(data.get("label_column", "cluster_pseudo_label")),
        pca_components=int(data.get("pca_components", 42)),
        kmeans_clusters=int(data.get("kmeans_clusters", 24)),
        kmeans_init=str(data.get("kmeans_init", "k-means++")),
        kmeans_n_init=int(data.get("kmeans_n_init", 10)),
        kmeans_max_iter=int(data.get("kmeans_max_iter", 300)),
        random_state=int(data.get("random_state", 0)),
        transform_batch_size=int(data.get("transform_batch_size", 65536)),
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
        synthetic_embeddings=_synthetic_embedding_config(raw.get("synthetic_embeddings", {})),
        synthetic_bags=_synthetic_bag_config(raw.get("synthetic_bags", {})),
        cluster_pseudolabels=_cluster_pseudolabel_config(raw.get("cluster_pseudolabels", {})),
    )
