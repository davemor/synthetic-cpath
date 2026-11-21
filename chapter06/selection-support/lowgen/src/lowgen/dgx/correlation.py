from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import load_dataset
from .manifest import manifest_path
from .projections import representation_cache_dir
from .utils import atomic_write_json, read_jsonl, stable_hash


DEFAULT_PROJECTORS = ("raw_highdim", "pca", "umap")
CORRELATION_THRESHOLDS = (0.3, 0.5, 0.7)


def run_correlation_audit(
    cfg: ExperimentConfig,
    manifest: Path | None = None,
    output_dir: Path | None = None,
    top_n: int = 100,
    force: bool = False,
    include_projectors: tuple[str, ...] | list[str] | None = None,
    deduplicate: bool = True,
    max_rows: int | None = None,
    seed: int = 0,
    config_path: Path | None = None,
) -> dict[str, Path]:
    """Audit dimension correlations for cached DGX representations."""

    manifest = manifest or manifest_path(cfg)
    output_dir = output_dir or cfg.paths.run_dir / "correlation_audit"
    include = set(include_projectors or DEFAULT_PROJECTORS)
    _prepare_output_dir(output_dir, force)

    tasks = [
        task
        for task in read_jsonl(manifest)
        if str(task.get("projector", "")) in include
    ]
    if deduplicate:
        tasks = _deduplicate_tasks(tasks)

    summary_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    dataset_cache: dict[tuple[str, str], Any] = {}

    for task in tasks:
        base = _base_row(task)
        try:
            x_train, y_train = _load_train_representation(cfg, task, dataset_cache)
            if max_rows is not None and max_rows > 0:
                x_train = _subsample_rows(
                    x_train, y_train, int(max_rows), int(seed) + _task_seed_offset(task)
                )
            stats = correlation_statistics(x_train)
            matrix_name = _matrix_filename(task)
            matrix_path = output_dir / "matrices" / matrix_name
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                matrix_path,
                correlation=stats["correlation"].astype(np.float32, copy=False),
                zero_variance_mask=stats["zero_variance_mask"],
            )
            summary_rows.append(
                base
                | {
                    "n_samples": stats["n_samples"],
                    "n_dimensions": stats["n_dimensions"],
                    "n_zero_variance_dims": stats["n_zero_variance_dims"],
                    "mean_abs_corr": stats["mean_abs_corr"],
                    "median_abs_corr": stats["median_abs_corr"],
                    "p95_abs_corr": stats["p95_abs_corr"],
                    "p99_abs_corr": stats["p99_abs_corr"],
                    "max_abs_corr": stats["max_abs_corr"],
                    "fraction_abs_corr_ge_0_3": stats["fraction_abs_corr_ge_0_3"],
                    "fraction_abs_corr_ge_0_5": stats["fraction_abs_corr_ge_0_5"],
                    "fraction_abs_corr_ge_0_7": stats["fraction_abs_corr_ge_0_7"],
                    "effective_dimension": stats["effective_dimension"],
                    "matrix_path": str(matrix_path),
                }
            )
            pair_rows.extend(
                base | pair
                for pair in top_correlated_pairs(stats["correlation"], top_n)
            )
        except Exception as exc:
            audit_rows.append(
                base
                | {
                    "status": "skipped",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    outputs = {
        "summary": output_dir / "correlation_summary.csv",
        "top_pairs": output_dir / "top_correlated_pairs.csv",
        "audit": output_dir / "audit.csv",
        "metadata": output_dir / "metadata.json",
    }
    pd.DataFrame(summary_rows, columns=_summary_columns()).to_csv(
        outputs["summary"], index=False
    )
    pd.DataFrame(pair_rows, columns=_pair_columns()).to_csv(
        outputs["top_pairs"], index=False
    )
    pd.DataFrame(audit_rows, columns=_audit_columns()).to_csv(
        outputs["audit"], index=False
    )
    atomic_write_json(
        outputs["metadata"],
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path) if config_path is not None else None,
            "manifest_path": str(manifest),
            "output_dir": str(output_dir),
            "split": "train",
            "top_n": int(top_n),
            "include_projectors": sorted(include),
            "deduplicate": bool(deduplicate),
            "max_rows": max_rows,
            "seed": int(seed),
            "correlation": "pearson",
            "thresholds": list(CORRELATION_THRESHOLDS),
            "n_input_tasks": len(tasks),
            "n_success": len(summary_rows),
            "n_skipped": len(audit_rows),
        },
    )
    return outputs


def correlation_statistics(x: np.ndarray) -> dict[str, Any]:
    """Compute a Pearson dimension-correlation matrix and compact summaries."""

    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}")
    n_samples, n_dimensions = x.shape
    if n_samples < 2:
        raise ValueError("At least two samples are required to compute correlation")

    x64 = x.astype(np.float64, copy=False)
    std = x64.std(axis=0, ddof=1)
    valid = np.isfinite(std) & (std > 0)
    zero_variance_mask = ~valid
    corr = np.full((n_dimensions, n_dimensions), np.nan, dtype=np.float64)

    if valid.any():
        centered = x64[:, valid] - x64[:, valid].mean(axis=0, keepdims=True)
        scaled = centered / std[valid]
        corr_valid = (scaled.T @ scaled) / float(n_samples - 1)
        corr_valid = np.clip(corr_valid, -1.0, 1.0)
        valid_indices = np.flatnonzero(valid)
        corr[np.ix_(valid_indices, valid_indices)] = corr_valid

    valid_diag = np.flatnonzero(valid)
    corr[valid_diag, valid_diag] = 1.0

    off_diag = _finite_off_diagonal_abs(corr)
    summary = {
        "correlation": corr,
        "zero_variance_mask": zero_variance_mask,
        "n_samples": int(n_samples),
        "n_dimensions": int(n_dimensions),
        "n_zero_variance_dims": int(zero_variance_mask.sum()),
        "mean_abs_corr": _safe_stat(off_diag, np.mean),
        "median_abs_corr": _safe_stat(off_diag, np.median),
        "p95_abs_corr": _safe_percentile(off_diag, 95),
        "p99_abs_corr": _safe_percentile(off_diag, 99),
        "max_abs_corr": _safe_stat(off_diag, np.max),
        "effective_dimension": _effective_dimension(corr, valid),
    }
    for threshold in CORRELATION_THRESHOLDS:
        key = f"fraction_abs_corr_ge_{str(threshold).replace('.', '_')}"
        summary[key] = (
            float(np.mean(off_diag >= threshold)) if off_diag.size else float("nan")
        )
    return summary


def top_correlated_pairs(corr: np.ndarray, top_n: int) -> list[dict[str, Any]]:
    """Return the top absolute off-diagonal correlations from a matrix."""

    if top_n <= 0:
        return []
    corr = np.asarray(corr)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("Correlation matrix must be square")
    n_dimensions = corr.shape[0]
    if n_dimensions < 2:
        return []

    rows, cols = np.triu_indices(n_dimensions, k=1)
    values = corr[rows, cols]
    finite = np.isfinite(values)
    if not finite.any():
        return []

    rows, cols, values = rows[finite], cols[finite], values[finite]
    abs_values = np.abs(values)
    take = min(int(top_n), len(abs_values))
    selected = np.argpartition(-abs_values, take - 1)[:take]
    selected = selected[np.argsort(-abs_values[selected])]
    return [
        {
            "dim_i": int(rows[idx]),
            "dim_j": int(cols[idx]),
            "correlation": float(values[idx]),
            "abs_correlation": float(abs_values[idx]),
            "rank": int(rank),
        }
        for rank, idx in enumerate(selected, start=1)
    ]


def _load_train_representation(
    cfg: ExperimentConfig,
    task: dict[str, Any],
    dataset_cache: dict[tuple[str, str], Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    projector = str(task["projector"])
    data = None
    if projector == "raw_highdim":
        data = _cached_dataset(cfg, task, dataset_cache)
        return data.train.features, data.train.labels

    arrays_path = (
        representation_cache_dir(cfg) / str(task["projection_cache_key"]) / "arrays.npz"
    )
    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing projection cache arrays: {arrays_path}")
    with np.load(arrays_path) as arrays:
        x_train = np.asarray(arrays["train"], dtype=np.float32)
    return x_train, None


def _cached_dataset(
    cfg: ExperimentConfig,
    task: dict[str, Any],
    dataset_cache: dict[tuple[str, str], Any],
):
    key = (str(task["dataset"]), str(task["encoder"]))
    if key not in dataset_cache:
        dataset_cache[key] = load_dataset(cfg, key[0], key[1])
    return dataset_cache[key]


def _subsample_rows(
    x: np.ndarray,
    y: np.ndarray | None,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    if len(x) <= max_rows:
        return x
    rng = np.random.default_rng(seed)
    if y is None:
        idx = np.sort(rng.choice(len(x), size=max_rows, replace=False))
        return x[idx]

    labels, counts = np.unique(y, return_counts=True)
    quotas = np.floor(counts / counts.sum() * max_rows).astype(int)
    quotas = np.maximum(1, quotas)
    while quotas.sum() > max_rows:
        quotas[np.argmax(quotas)] -= 1
    while quotas.sum() < max_rows:
        quotas[np.argmax(counts - quotas)] += 1
    parts = []
    for label, quota in zip(labels, quotas):
        label_idx = np.flatnonzero(y == label)
        parts.append(
            rng.choice(label_idx, size=min(int(quota), len(label_idx)), replace=False)
        )
    idx = np.sort(np.concatenate(parts))
    return x[idx]


def _deduplicate_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    output = []
    for task in tasks:
        key = (
            task.get("dataset"),
            task.get("encoder"),
            task.get("projector"),
            task.get("dimension"),
            task.get("seed"),
            task.get("projection_cache_key"),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(task)
    return output


def _base_row(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "dataset": task.get("dataset"),
        "encoder": task.get("encoder"),
        "projector": task.get("projector"),
        "dimension": task.get("dimension"),
        "target_name": task.get("target_name"),
        "seed": task.get("seed"),
        "projection_cache_key": task.get("projection_cache_key"),
    }


def _matrix_filename(task: dict[str, Any]) -> str:
    task_id = task.get("task_id") or stable_hash(_base_row(task), length=24)
    projector = str(task.get("projector", "representation"))
    return f"{task_id}_{projector}_train_corr.npz"


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    expected = [
        output_dir / "correlation_summary.csv",
        output_dir / "top_correlated_pairs.csv",
        output_dir / "audit.csv",
        output_dir / "metadata.json",
    ]
    if not force and any(path.exists() for path in expected):
        raise FileExistsError(
            f"{output_dir} already contains audit outputs; pass --force to overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = output_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in matrix_dir.glob("*.npz"):
            path.unlink()


def _summary_columns() -> list[str]:
    return [
        *_base_columns(),
        "n_samples",
        "n_dimensions",
        "n_zero_variance_dims",
        "mean_abs_corr",
        "median_abs_corr",
        "p95_abs_corr",
        "p99_abs_corr",
        "max_abs_corr",
        "fraction_abs_corr_ge_0_3",
        "fraction_abs_corr_ge_0_5",
        "fraction_abs_corr_ge_0_7",
        "effective_dimension",
        "matrix_path",
    ]


def _pair_columns() -> list[str]:
    return [
        *_base_columns(),
        "dim_i",
        "dim_j",
        "correlation",
        "abs_correlation",
        "rank",
    ]


def _audit_columns() -> list[str]:
    return [*_base_columns(), "status", "error_type", "error"]


def _base_columns() -> list[str]:
    return [
        "task_id",
        "dataset",
        "encoder",
        "projector",
        "dimension",
        "target_name",
        "seed",
        "projection_cache_key",
    ]


def _finite_off_diagonal_abs(corr: np.ndarray) -> np.ndarray:
    n_dimensions = corr.shape[0]
    if n_dimensions < 2:
        return np.asarray([], dtype=np.float64)
    rows, cols = np.triu_indices(n_dimensions, k=1)
    values = np.abs(corr[rows, cols])
    return values[np.isfinite(values)]


def _safe_stat(values: np.ndarray, fn) -> float:
    if values.size == 0:
        return float("nan")
    return float(fn(values))


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def _effective_dimension(corr: np.ndarray, valid: np.ndarray) -> float:
    if not valid.any():
        return float("nan")
    corr_valid = corr[np.ix_(valid, valid)]
    if corr_valid.shape[0] == 1:
        return 1.0
    eigenvalues = np.linalg.eigvalsh(corr_valid)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    denom = float(np.sum(eigenvalues**2))
    if denom <= 0 or not math.isfinite(denom):
        return float("nan")
    return float(np.sum(eigenvalues) ** 2 / denom)


def _task_seed_offset(task: dict[str, Any]) -> int:
    return int(stable_hash(_base_row(task), length=8), 16) % 1_000_000
