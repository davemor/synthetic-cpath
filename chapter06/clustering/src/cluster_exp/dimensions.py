from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from .config import ExperimentConfig
from .utils import ensure_int


@dataclass(frozen=True)
class ProjectionTarget:
    dataset: str
    encoder: str
    projector: str
    dimension: int | None
    target_name: str
    seed: int | None = None
    source_metric: str | None = None
    condition: float | None = None

    @property
    def representation(self) -> str:
        if self.projector == "raw":
            return "raw_full"
        return f"{self.projector}_{self.dimension}"


def load_projection_targets(cfg: ExperimentConfig) -> list[ProjectionTarget]:
    run_root = cfg.paths.dimensionality_run_root
    intrinsic = _load_intrinsic_dims(run_root / "intrinsic_dims.csv")
    thresholds = _load_threshold_dims(cfg)
    targets: list[ProjectionTarget] = []

    for dataset in cfg.datasets:
        for encoder in cfg.encoders:
            if cfg.projection.include_raw_full:
                for seed in cfg.seeds:
                    targets.append(
                        ProjectionTarget(
                            dataset=dataset,
                            encoder=encoder,
                            projector="raw",
                            dimension=None,
                            target_name="full",
                            seed=int(seed),
                        )
                    )
            d_id = intrinsic.get((dataset, encoder))
            for projector in cfg.projection.projectors:
                for seed in cfg.seeds:
                    dims: dict[int, ProjectionTarget] = {}
                    for dim in _explicit_dimensions(cfg):
                        dims[int(dim)] = ProjectionTarget(
                            dataset, encoder, projector, int(dim), f"dim_{int(dim)}", int(seed)
                        )
                    if cfg.projection.include_one_dim and 1 not in dims:
                        dims[1] = ProjectionTarget(
                            dataset, encoder, projector, 1, "one_dim", int(seed)
                        )
                    if d_id is not None:
                        dims[int(d_id)] = ProjectionTarget(
                            dataset, encoder, projector, int(d_id), "intrinsic_mle", int(seed)
                        )
                    for (ds, enc, proj, metric, condition), dim in thresholds.items():
                        if ds == dataset and enc == encoder and proj == projector and dim is not None:
                            name = f"{metric}_{int(float(condition) * 100)}"
                            dims[int(dim)] = ProjectionTarget(
                                dataset,
                                encoder,
                                projector,
                                int(dim),
                                name,
                                int(seed),
                                source_metric=metric,
                                condition=float(condition),
                            )
                    targets.extend(dims[dim] for dim in sorted(dims))
    return targets


def _explicit_dimensions(cfg: ExperimentConfig) -> list[int]:
    dims = {int(dim) for dim in cfg.projection.explicit_dimensions}
    if (
        cfg.projection.explicit_dimension_min is not None
        and cfg.projection.explicit_dimension_max is not None
    ):
        start = int(cfg.projection.explicit_dimension_min)
        stop = int(cfg.projection.explicit_dimension_max)
        if start < 1:
            raise ValueError(f"explicit_dimension_min must be >= 1, got {start}.")
        if stop < start:
            raise ValueError(
                f"explicit_dimension_max must be >= explicit_dimension_min, got {stop} < {start}."
            )
        dims.update(range(start, stop + 1))
    return sorted(dims)


def _load_intrinsic_dims(path: Path) -> dict[tuple[str, str], int]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    output: dict[tuple[str, str], int] = {}
    for _, row in df.iterrows():
        output[(str(row["dataset"]), str(row["encoder"]))] = ensure_int(row["intrinsic_dim_mle"])
    return output


def _load_threshold_dims(
    cfg: ExperimentConfig,
) -> dict[tuple[str, str, str, str, float], int | None]:
    threshold_path = cfg.paths.dimensionality_run_root / "curve" / "thresholds_results.csv"
    if threshold_path.exists():
        loaded = _load_thresholds_csv(cfg, threshold_path)
        if loaded:
            return loaded
    return _derive_thresholds_from_curve_tasks(cfg)


def _load_thresholds_csv(
    cfg: ExperimentConfig, path: Path
) -> dict[tuple[str, str, str, str, float], int | None]:
    df = pd.read_csv(path)
    required = {"dataset", "encoder", "projector", "metric", "condition", "achieved_dim"}
    if not required.issubset(df.columns):
        return {}
    output: dict[tuple[str, str, str, str, float], int | None] = {}
    for _, row in df.iterrows():
        if "classifier" in df.columns and row["classifier"] != cfg.projection.threshold_classifier:
            continue
        metric = str(row["metric"])
        condition = float(row["condition"])
        if metric not in cfg.projection.threshold_metrics:
            continue
        if condition not in [float(x) for x in cfg.projection.threshold_conditions]:
            continue
        dim = None if pd.isna(row["achieved_dim"]) else ensure_int(row["achieved_dim"])
        output[
            (
                str(row["dataset"]),
                str(row["encoder"]),
                str(row["projector"]),
                metric,
                condition,
            )
        ] = dim
    return output


def _derive_thresholds_from_curve_tasks(
    cfg: ExperimentConfig,
) -> dict[tuple[str, str, str, str, float], int | None]:
    baselines = _load_curve_baselines(cfg)
    curve_means = _load_curve_means(cfg)
    output: dict[tuple[str, str, str, str, float], int | None] = {}
    for dataset in cfg.datasets:
        for encoder in cfg.encoders:
            for projector in cfg.projection.projectors:
                for metric in cfg.projection.threshold_metrics:
                    baseline = baselines.get((dataset, encoder, projector, metric))
                    dims = sorted(
                        dim
                        for (ds, enc, proj, curve_metric, dim) in curve_means
                        if ds == dataset and enc == encoder and proj == projector and curve_metric == metric
                    )
                    for condition in cfg.projection.threshold_conditions:
                        achieved = None
                        if baseline is not None:
                            threshold = baseline * float(condition)
                            for dim in dims:
                                if curve_means[(dataset, encoder, projector, metric, dim)] >= threshold:
                                    achieved = dim
                                    break
                        output[(dataset, encoder, projector, metric, float(condition))] = achieved
    return output


def _load_curve_baselines(cfg: ExperimentConfig) -> dict[tuple[str, str, str, str], float]:
    path = cfg.paths.dimensionality_run_root / "curve" / "thresholds_results.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {"dataset", "encoder", "projector", "metric", "baseline"}.issubset(df.columns):
        return {}
    output: dict[tuple[str, str, str, str], float] = {}
    for _, row in df.iterrows():
        if "classifier" in df.columns and row["classifier"] != cfg.projection.threshold_classifier:
            continue
        output[
            (str(row["dataset"]), str(row["encoder"]), str(row["projector"]), str(row["metric"]))
        ] = float(row["baseline"])
    return output


def _load_curve_means(cfg: ExperimentConfig) -> dict[tuple[str, str, str, str, int], float]:
    values: dict[tuple[str, str, str, str, int], list[float]] = {}
    curve_dir = cfg.paths.dimensionality_run_root / "curve"
    for projector in cfg.projection.projectors:
        task_dir = curve_dir / f"{projector}_{cfg.projection.threshold_classifier}" / "tasks"
        for path in task_dir.glob("*.json"):
            with path.open() as handle:
                item: dict[str, Any] = json.load(handle)
            if item.get("projector") != projector:
                continue
            dataset, encoder = _parse_dataset_encoder(str(item["dataset"]))
            dim = int(item["n_components"])
            for metric in cfg.projection.threshold_metrics:
                if metric in item:
                    key = (dataset, encoder, projector, metric, dim)
                    values.setdefault(key, []).append(float(item[metric]))
    return {key: float(mean(metric_values)) for key, metric_values in values.items()}


def _parse_dataset_encoder(value: str) -> tuple[str, str]:
    return value.rsplit("_", 1)
