from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    embeddings_root: Path
    dimensionality_run_root: Path
    output_root: Path
    run_name: str = "histology_clustering"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PathsConfig":
        return cls(
            embeddings_root=Path(data["embeddings_root"]),
            dimensionality_run_root=Path(data["dimensionality_run_root"]),
            output_root=Path(data["output_root"]),
            run_name=str(data.get("run_name", "histology_clustering")),
        )

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name


@dataclass
class UmapConfig:
    n_neighbors: int = 10
    min_dist: float = 0.99
    metric: str = "cosine"
    fit_subsample: int = 50000

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "UmapConfig":
        return cls(**(data or {}))

    def params(self) -> dict[str, Any]:
        return {
            "n_neighbors": int(self.n_neighbors),
            "min_dist": float(self.min_dist),
            "metric": self.metric,
        }


@dataclass
class ProjectionConfig:
    projectors: list[str] = field(default_factory=lambda: ["pca", "umap"])
    include_raw_full: bool = True
    include_one_dim: bool = True
    explicit_dimensions: list[int] = field(default_factory=list)
    explicit_dimension_min: int | None = None
    explicit_dimension_max: int | None = None
    threshold_metrics: list[str] = field(default_factory=lambda: ["accuracy", "roc_auc"])
    threshold_conditions: list[float] = field(default_factory=lambda: [0.95, 0.99])
    threshold_classifier: str = "logistic-regression"
    umap: UmapConfig = field(default_factory=UmapConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProjectionConfig":
        data = dict(data or {})
        data["umap"] = UmapConfig.from_dict(data.get("umap"))
        return cls(**data)


@dataclass
class KMeansConfig:
    k: list[int] = field(default_factory=lambda: [16, 64, 128])
    init: str = "k-means++"
    n_init: int = 10
    max_iter: int = 300

    @classmethod
    def from_dict(cls, data: dict[str, Any] | "KMeansConfig" | None) -> "KMeansConfig":
        if isinstance(data, cls):
            return data
        return cls(**(data or {}))


@dataclass
class HdbscanConfig:
    min_cluster_size: list[int] = field(default_factory=lambda: [100, 500, 1000])
    min_samples: list[int] = field(default_factory=lambda: [25])
    cluster_selection_method: list[str] = field(default_factory=lambda: ["eom"])
    alpha: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HdbscanConfig":
        return cls(**(data or {}))


@dataclass
class LeidenConfig:
    n_neighbors: list[int] = field(default_factory=lambda: [15, 30])
    resolution: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    metric: str = "cosine"
    max_iter: int = 100
    theta: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LeidenConfig":
        return cls(**(data or {}))


@dataclass
class ClusteringConfig:
    fit_subsample: int | None = None
    fit_subsample_fraction: float = 0.10
    fit_subsample_min: int = 10000
    fit_subsample_max: int = 50000
    metric_subsample: int = 10000
    sd_index_alpha: float = 1.0
    clusterers: list[str] = field(default_factory=lambda: ["kmeans", "hdbscan"])
    kmeans: KMeansConfig = field(default_factory=KMeansConfig)
    spherical_kmeans: KMeansConfig = field(default_factory=KMeansConfig)
    hdbscan: HdbscanConfig = field(default_factory=HdbscanConfig)
    leiden: LeidenConfig = field(default_factory=LeidenConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ClusteringConfig":
        data = dict(data or {})
        kmeans_data = data.get("kmeans")
        data["kmeans"] = KMeansConfig.from_dict(data.get("kmeans"))
        data["spherical_kmeans"] = KMeansConfig.from_dict(
            data.get("spherical_kmeans") or kmeans_data
        )
        data["hdbscan"] = HdbscanConfig.from_dict(data.get("hdbscan"))
        data["leiden"] = LeidenConfig.from_dict(data.get("leiden"))
        return cls(**data)


@dataclass
class RuntimeConfig:
    require_cuml: bool = True
    allow_cpu_fallback: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RuntimeConfig":
        return cls(**(data or {}))


@dataclass
class ExperimentConfig:
    paths: PathsConfig
    datasets: list[str]
    encoders: list[str]
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        data = dict(data)
        data["paths"] = PathsConfig.from_dict(data["paths"])
        data["projection"] = ProjectionConfig.from_dict(data.get("projection"))
        data["clustering"] = ClusteringConfig.from_dict(data.get("clustering"))
        data["runtime"] = RuntimeConfig.from_dict(data.get("runtime"))
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        with Path(path).open() as handle:
            return cls.from_dict(yaml.safe_load(handle))

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": {
                "embeddings_root": str(self.paths.embeddings_root),
                "dimensionality_run_root": str(self.paths.dimensionality_run_root),
                "output_root": str(self.paths.output_root),
                "run_name": self.paths.run_name,
            },
            "datasets": self.datasets,
            "encoders": self.encoders,
            "seeds": self.seeds,
        }
