from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig
from .data import inspect_embedding_metadata, resolve_embedding_root, source_fingerprint_hash
from .targets import Target, build_targets
from .utils import atomic_write_jsonl, stable_hash


def manifest_path(cfg: ExperimentConfig) -> Path:
    return cfg.paths.run_dir / "task_manifest.jsonl"


def audit_path(cfg: ExperimentConfig) -> Path:
    return cfg.paths.run_dir / "manifest_audit.csv"


def build_task_manifest(cfg: ExperimentConfig, output_path: Path | None = None) -> list[dict[str, Any]]:
    output_path = output_path or manifest_path(cfg)
    targets, audit = build_targets(cfg)
    context: dict[tuple[str, str], dict[str, Any]] = {}
    rows = [_task_row(cfg, target, context) for target in targets]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output_path, rows)
    pd.DataFrame(rows).to_csv(cfg.paths.run_dir / "task_manifest.csv", index=False)
    if audit.empty:
        audit = pd.DataFrame(columns=["dataset", "encoder", "projector", "metric", "condition", "reason"])
    audit.to_csv(audit_path(cfg), index=False)
    cfg.write_run_config()
    return rows


def projection_cache_key(cfg: ExperimentConfig, target: Target, fingerprint_hash: str | None = None) -> str:
    payload = {
        "target": asdict(target),
        "umap": {
            "n_neighbors": cfg.projection.umap_n_neighbors,
            "min_dist": cfg.projection.umap_min_dist,
            "metric": cfg.projection.umap_metric,
        },
        "source_fingerprint_hash": fingerprint_hash
        or source_fingerprint_hash(cfg, target.dataset, target.encoder),
    }
    return stable_hash(payload, length=24)


def _task_row(
    cfg: ExperimentConfig,
    target: Target,
    context: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    context = context if context is not None else {}
    key = (target.dataset, target.encoder)
    info = context.get(key)
    if info is None:
        info = {
            "metadata": inspect_embedding_metadata(cfg, target.dataset, target.encoder),
            "root": resolve_embedding_root(cfg, target.dataset, target.encoder),
            "fingerprint_hash": source_fingerprint_hash(cfg, target.dataset, target.encoder),
        }
        context[key] = info
    metadata = info["metadata"]
    row = {
        "dataset": target.dataset,
        "encoder": target.encoder,
        "projector": target.projector,
        "dimension": "" if target.dimension is None else int(target.dimension),
        "target_name": target.target_name,
        "seed": int(target.seed),
        "source_metric": target.source_metric,
        "condition": target.condition,
        "embedding_root": str(info["root"]),
        "source_fingerprint_hash": info["fingerprint_hash"],
        "source_dimension": int(metadata.get("n_dimensions", 0)),
        "n_samples": int(metadata.get("n_samples", 0)),
        "projection_cache_key": projection_cache_key(cfg, target, info["fingerprint_hash"]),
    }
    row["task_id"] = stable_hash(row, length=24)
    return row
