from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathConfig:
    hypatia_run_root: Path = Path("/raid/experiments/dimred_dgx_followup/hypatia_curve_source")
    non_icaird_embeddings_root: Path = Path("/raid/experiments/hdims/embeddings")
    icaird_embeddings_root: Path = Path(
        "/raid/datasets/icaird_cervical_120k_trainonly_slide_split_uni2_labelbalanced_kcenter/embeddings"
    )
    output_root: Path = Path("/raid/experiments/lowgen_v2/output")
    run_name: str = "lowgen-v2-unconditioned-true-label"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PathConfig":
        data = data or {}
        return cls(
            hypatia_run_root=Path(data.get("hypatia_run_root", cls.hypatia_run_root)),
            non_icaird_embeddings_root=Path(
                data.get("non_icaird_embeddings_root", cls.non_icaird_embeddings_root)
            ),
            icaird_embeddings_root=Path(
                data.get("icaird_embeddings_root", cls.icaird_embeddings_root)
            ),
            output_root=Path(data.get("output_root", cls.output_root)),
            run_name=str(data.get("run_name", cls.run_name)),
        )

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    @property
    def threshold_source(self) -> Path:
        return self.hypatia_run_root / "curve" / "thresholds_results.csv"

    @property
    def intrinsic_source(self) -> Path:
        return self.hypatia_run_root / "intrinsic_dims.csv"


@dataclass
class ProjectionConfig:
    projectors: list[str] = field(default_factory=lambda: ["pca", "umap"])
    fixed_dimensions: list[int] = field(default_factory=list)
    threshold_metrics: list[str] = field(default_factory=lambda: ["accuracy", "roc_auc"])
    threshold_conditions: list[float] = field(default_factory=lambda: [0.95, 0.99])
    threshold_classifier: str = "logistic-regression"
    include_intrinsic: bool = True
    include_raw_highdim: bool = True
    umap_n_neighbors: int = 10
    umap_min_dist: float = 0.99
    umap_metric: str = "cosine"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProjectionConfig":
        return cls(**(data or {}))


@dataclass
class GeneratorConfig:
    kinds: list[str] = field(
        default_factory=lambda: ["resample", "diagonal_gaussian", "diagonal_gmm"]
    )
    conditioning_modes: list[str] = field(default_factory=lambda: ["unconditioned", "true_label"])
    gmm_components: int = 5
    gmm_covariance_type: str = "diag"
    gmm_fit_subsample: int = 10000
    gmm_max_iter: int = 100
    gmm_select_k: bool = True
    gmm_candidate_components: list[int] = field(
        default_factory=lambda: [1, 2, 3, 5, 8, 10]
    )
    gmm_selection_metric: str = "bic"
    gmm_min_samples_per_component: int = 100
    timeout_seconds: int = 7200

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GeneratorConfig":
        return cls(**(data or {}))


@dataclass
class ClusterConditioningConfig:
    clustering_results_path: Path | None = None
    max_rows: int | None = None
    min_clusters: int = 2
    max_noise_fraction: float = 1.0
    include_noise_cluster: bool = False
    generator_kinds: list[str] = field(
        default_factory=lambda: ["cluster_guided_full_diag_gaussian_probe"]
    )
    synthetic_total: int = 5000
    min_cluster_size: int = 2
    min_variance: float = 1e-6
    gmm_components: int = 4
    gmm_fit_subsample: int = 10000
    gmm_max_iter: int = 100
    gmm_reg_covar: float = 1e-6

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ClusterConditioningConfig":
        data = data or {}
        results_path = data.get("clustering_results_path")
        return cls(
            clustering_results_path=Path(results_path) if results_path else None,
            max_rows=None if data.get("max_rows") is None else int(data["max_rows"]),
            min_clusters=int(data.get("min_clusters", cls.min_clusters)),
            max_noise_fraction=float(
                data.get("max_noise_fraction", cls.max_noise_fraction)
            ),
            include_noise_cluster=bool(
                data.get("include_noise_cluster", cls.include_noise_cluster)
            ),
            generator_kinds=list(
                data.get(
                    "generator_kinds",
                    ["cluster_guided_full_diag_gaussian_probe"],
                )
            ),
            synthetic_total=int(data.get("synthetic_total", cls.synthetic_total)),
            min_cluster_size=int(data.get("min_cluster_size", cls.min_cluster_size)),
            min_variance=float(data.get("min_variance", cls.min_variance)),
            gmm_components=int(data.get("gmm_components", cls.gmm_components)),
            gmm_fit_subsample=int(
                data.get("gmm_fit_subsample", cls.gmm_fit_subsample)
            ),
            gmm_max_iter=int(data.get("gmm_max_iter", cls.gmm_max_iter)),
            gmm_reg_covar=float(data.get("gmm_reg_covar", cls.gmm_reg_covar)),
        )


@dataclass
class EvalConfig:
    knn_k: int = 15
    classifier_max_iter: int = 2000
    fid_subsample: int = 5000
    kid_subsample: int = 2000
    nn_subsample: int = 5000
    pairwise_subsample: int = 2000
    cluster_count_k: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvalConfig":
        return cls(**(data or {}))


@dataclass
class RuntimeConfig:
    require_cuml: bool = False
    dask_dashboard_address: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RuntimeConfig":
        return cls(**(data or {}))


@dataclass
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    datasets: list[str] = field(
        default_factory=lambda: [
            "kather100k",
            "spider-colorectal",
            "spider-breast",
            "spider-skin",
            "spider-thorax",
            "icaird100k",
        ]
    )
    encoders: list[str] = field(default_factory=lambda: ["uni", "uni2", "hibou", "dinov2"])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    num_folds: int = 5
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    cluster_conditioning: ClusterConditioningConfig = field(
        default_factory=ClusterConditioningConfig
    )
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    icaird_dataset_names: list[str] = field(
        default_factory=lambda: ["icaird100k", "icaird_cervical"]
    )
    encoder_folder_aliases: dict[str, list[str]] = field(
        default_factory=lambda: {
            "uni": ["uni"],
            "uni2": ["uni2", "uni2_h", "uni2h"],
            "hibou": ["hibou", "hibou_l", "hibou-large"],
            "dinov2": ["dinov2", "dinov2_large", "dinov2-large"],
        }
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        data = dict(data)
        return cls(
            paths=PathConfig.from_dict(data.get("paths")),
            datasets=list(data.get("datasets", cls().datasets)),
            encoders=list(data.get("encoders", cls().encoders)),
            seeds=[int(x) for x in data.get("seeds", cls().seeds)],
            num_folds=int(data.get("num_folds", cls().num_folds)),
            projection=ProjectionConfig.from_dict(data.get("projection")),
            generator=GeneratorConfig.from_dict(data.get("generator")),
            cluster_conditioning=ClusterConditioningConfig.from_dict(
                data.get("cluster_conditioning")
            ),
            eval=EvalConfig.from_dict(data.get("eval")),
            runtime=RuntimeConfig.from_dict(data.get("runtime")),
            icaird_dataset_names=list(data.get("icaird_dataset_names", cls().icaird_dataset_names)),
            encoder_folder_aliases=dict(
                data.get("encoder_folder_aliases", cls().encoder_folder_aliases)
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open() as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def to_dict(self) -> dict[str, Any]:
        def convert(value):
            if isinstance(value, Path):
                return str(value)
            if hasattr(value, "__dataclass_fields__"):
                return {k: convert(getattr(value, k)) for k in value.__dataclass_fields__}
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            return value

        return convert(self)

    def write_run_config(self) -> Path:
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.paths.run_dir / "run_config.yml"
        with path.open("w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
        return path
