from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SeedConfig:
    base_seed: int = 0
    num_seeds: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedConfig":
        return cls(**data)


@dataclass
class ReducerConfig:
    kind: str  # "pca" | "umap"
    dimensions: list[int] = field(default_factory=lambda: [9])
    # PCA-only
    whiten: bool = False
    # UMAP-only
    n_neighbors: list[int] = field(default_factory=lambda: [15])
    min_dist: list[float] = field(default_factory=lambda: [0.1])
    metric: str = "euclidean"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReducerConfig":
        return cls(**data)


@dataclass
class GeneratorConfig:
    kinds: list[str] = field(
        default_factory=lambda: ["resample", "isotropic", "gaussian", "gmm", "bgmm"]
    )
    max_components: int = 20
    conditional: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratorConfig":
        return cls(**data)


@dataclass
class EvalConfig:
    knn_k: int = 15
    probe_max_iter: int = 1000
    mmd_subsample: int = 1000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalConfig":
        return cls(**data)


@dataclass
class Config:
    datasets: list[str] = field(default_factory=list)
    encoders: list[str] = field(default_factory=list)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    num_folds: int = 5
    reducers: list[ReducerConfig] = field(default_factory=list)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = dict(data)

        if "seeds" in data:
            data["seeds"] = SeedConfig.from_dict(data["seeds"])

        if "reducers" in data:
            data["reducers"] = [
                ReducerConfig.from_dict(r) for r in data["reducers"]
            ]

        if "generator" in data:
            data["generator"] = GeneratorConfig.from_dict(data["generator"])

        if "eval" in data:
            data["eval"] = EvalConfig.from_dict(data["eval"])

        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
