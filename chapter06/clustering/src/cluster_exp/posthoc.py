from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .metrics import sd_index
from .projections import deterministic_sample_indices, load_projection_features
from .utils import atomic_write_json


def add_sd_index_to_run(
    cfg: ExperimentConfig,
    run_dir: Path | None = None,
    *,
    manifest_paths: list[Path] | None = None,
    workers: int = 1,
    force: bool = False,
) -> dict[str, int]:
    run_dir = run_dir or cfg.paths.run_dir
    result_paths = sorted((run_dir / "tasks").glob("*/result.json"))
    if manifest_paths:
        task_ids = _task_ids_from_manifests(manifest_paths)
        result_paths = [path for path in result_paths if path.parent.name in task_ids]
    if not result_paths:
        raise FileNotFoundError(f"No task result JSON files found under {run_dir / 'tasks'}")

    worker_count = _normalise_workers(workers)
    if worker_count == 1:
        stats = [_add_sd_index_to_result_file(cfg, path, force=force) for path in result_paths]
    else:
        print(
            f"Computing SD-index with {worker_count} workers for {len(result_paths)} task files.",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=_process_context()) as executor:
            mapped = executor.map(
                _sd_index_worker,
                [(cfg, path, force) for path in result_paths],
                chunksize=1,
            )
            stats = []
            for index, item in enumerate(mapped, start=1):
                stats.append(item)
                _print_progress("sd-index tasks", index, len(result_paths))

    return {
        "task_files": len(result_paths),
        "updated_task_files": int(sum(item["updated_task_file"] for item in stats)),
        "result_rows": int(sum(item["result_rows"] for item in stats)),
        "updated_result_rows": int(sum(item["updated_result_rows"] for item in stats)),
        "missing_label_files": int(sum(item["missing_label_files"] for item in stats)),
        "failed_task_files": int(sum(item["failed_task_file"] for item in stats)),
    }


def _task_ids_from_manifests(manifest_paths: list[Path]) -> set[str]:
    task_ids: set[str] = set()
    for manifest_path in manifest_paths:
        with Path(manifest_path).open() as handle:
            for line in handle:
                if line.strip():
                    task_ids.add(str(json.loads(line)["task_id"]))
    return task_ids


def _sd_index_worker(args: tuple[ExperimentConfig, Path, bool]) -> dict[str, int]:
    cfg, path, force = args
    return _add_sd_index_to_result_file(cfg, path, force=force)


def _add_sd_index_to_result_file(
    cfg: ExperimentConfig,
    path: Path,
    *,
    force: bool,
) -> dict[str, int]:
    with path.open() as handle:
        loaded = json.load(handle)
    was_list = isinstance(loaded, list)
    rows = loaded if was_list else [loaded]
    needs_update = [
        row
        for row in rows
        if force or not {"sd_index", "sd_scat", "sd_dis"}.issubset(row.keys())
    ]
    if not needs_update:
        return _stats(rows, updated_task_file=False)

    try:
        data = load_projection_features(cfg, rows[0])
        metric_indices = deterministic_sample_indices(
            data.n_samples,
            min(cfg.clustering.metric_subsample, data.n_samples),
            0,
        )
        metric_features = data.features[metric_indices]
    except Exception:
        return _stats(rows, updated_task_file=False, failed_task_file=True)

    updated_rows = 0
    missing_label_files = 0
    for row in rows:
        if row not in needs_update:
            continue
        labels_path = Path(str(row.get("labels_path", "")))
        if not labels_path.is_file():
            missing_label_files += 1
            continue
        with np.load(labels_path, allow_pickle=True) as labels_data:
            labels = np.asarray(labels_data["labels"])
        if labels.shape[0] != data.n_samples:
            continue
        result = sd_index(
            metric_features,
            labels[metric_indices],
            alpha=float(cfg.clustering.sd_index_alpha),
        )
        row["sd_index"] = result.sd_index
        row["sd_scat"] = result.sd_scat
        row["sd_dis"] = result.sd_dis
        updated_rows += 1

    if updated_rows:
        atomic_write_json(path, rows if was_list else rows[0])
    return _stats(
        rows,
        updated_task_file=bool(updated_rows),
        updated_result_rows=updated_rows,
        missing_label_files=missing_label_files,
    )


def _stats(
    rows: list[dict[str, Any]],
    *,
    updated_task_file: bool,
    updated_result_rows: int = 0,
    missing_label_files: int = 0,
    failed_task_file: bool = False,
) -> dict[str, int]:
    return {
        "updated_task_file": int(updated_task_file),
        "result_rows": len(rows),
        "updated_result_rows": int(updated_result_rows),
        "missing_label_files": int(missing_label_files),
        "failed_task_file": int(failed_task_file),
    }


def _normalise_workers(workers: int) -> int:
    if workers < 1:
        return max(1, os.cpu_count() or 1)
    return workers


def _process_context() -> mp.context.BaseContext | None:
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return None


def _print_progress(label: str, index: int, total: int) -> None:
    if index == 1 or index == total or index % 100 == 0:
        print(f"{label}: {index}/{total}", flush=True)
