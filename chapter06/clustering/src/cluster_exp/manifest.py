from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig
from .projections import read_projection_manifest
from .utils import atomic_write_jsonl, stable_hash


def build_task_manifest(cfg: ExperimentConfig, output_path: Path) -> list[dict[str, Any]]:
    projection_manifest = read_projection_manifest(cfg)
    tasks: list[dict[str, Any]] = []
    for _, projection in projection_manifest.iterrows():
        projection_record = _projection_record(projection)
        tasks.append(_task_record(cfg, projection_record))
    atomic_write_jsonl(output_path, tasks)
    return tasks


def _projection_record(row: pd.Series) -> dict[str, Any]:
    dimension = row.get("dimension", "")
    return {
        "dataset": str(row["dataset"]),
        "encoder": str(row["encoder"]),
        "projector": str(row["projector"]),
        "dimension": None if pd.isna(dimension) or dimension == "" else int(float(dimension)),
        "target_name": str(row["target_name"]),
        "projection_seed": int(float(row.get("projection_seed", 0))),
        "source_metric": "" if pd.isna(row.get("source_metric", "")) else str(row.get("source_metric", "")),
        "condition": "" if pd.isna(row.get("condition", "")) else str(row.get("condition", "")),
        "projection_cache_key": str(row["projection_cache_key"]),
        "projection_path": "" if pd.isna(row.get("projection_path", "")) else str(row.get("projection_path", "")),
        "n_samples": int(float(row["n_samples"])),
        "source_dimension": int(float(row["source_dimension"])),
    }


def _task_record(cfg: ExperimentConfig, projection_record: dict[str, Any]) -> dict[str, Any]:
    task = {
        **projection_record,
        "cluster_grid_size": _cluster_grid_size(cfg),
        "cluster_seed": int(projection_record["projection_seed"]),
    }
    task["task_id"] = stable_hash(task, length=24)
    return task


def _cluster_grid_size(cfg: ExperimentConfig) -> int:
    total = 0
    for clusterer in cfg.clustering.clusterers:
        if clusterer == "kmeans":
            total += len(cfg.clustering.kmeans.k)
        elif clusterer == "spherical_kmeans":
            total += len(cfg.clustering.spherical_kmeans.k)
        elif clusterer == "hdbscan":
            total += (
                len(cfg.clustering.hdbscan.min_cluster_size)
                * len(cfg.clustering.hdbscan.min_samples)
                * len(cfg.clustering.hdbscan.cluster_selection_method)
            )
        elif clusterer == "leiden":
            total += len(cfg.clustering.leiden.n_neighbors) * len(cfg.clustering.leiden.resolution)
        else:
            raise ValueError(f"Unsupported clusterer {clusterer!r}.")
    return total
