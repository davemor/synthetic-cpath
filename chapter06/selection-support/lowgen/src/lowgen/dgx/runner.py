from __future__ import annotations

import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .data import load_dataset
from .generators import fit_generator
from .manifest import manifest_path
from .metrics import (
    fidelity_metrics,
    lowdim_fidelity_metrics,
    reconstruction_metrics,
    reduction_metrics,
    utility_metrics,
)
from .projections import build_representation, inverse_transform
from .utils import atomic_write_json, read_jsonl, stable_hash


def task_output_path(cfg: ExperimentConfig, task: dict[str, Any]) -> Path:
    return cfg.paths.run_dir / "tasks" / f"{task['task_id']}.json"


def run_task_by_index(
    cfg: ExperimentConfig, manifest: Path | None, task_index: int, force: bool = False
) -> dict[str, Any]:
    tasks = read_jsonl(manifest or manifest_path(cfg))
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(f"task-index {task_index} outside manifest range 0..{len(tasks) - 1}")
    return run_task(cfg, tasks[task_index], force=force)


def run_task(cfg: ExperimentConfig, task: dict[str, Any], force: bool = False) -> dict[str, Any]:
    output_path = task_output_path(cfg, task)
    if output_path.exists() and not force:
        import json

        with output_path.open() as handle:
            return json.load(handle)

    started = perf_counter()
    data = load_dataset(cfg, task["dataset"], task["encoder"])
    dimension = None if task["dimension"] == "" else int(float(task["dimension"]))
    rep = build_representation(
        cfg,
        task["projector"],
        dimension,
        int(task["seed"]),
        data.train.features,
        data.test.features,
        data.valid.features if data.valid is not None else None,
        str(task["projection_cache_key"]),
        force=force,
    )

    result: dict[str, Any] = {
        **task,
        "status": "ok",
        "class_names": data.class_names,
        "representation_metadata": rep.metadata,
        "reduction": reduction_metrics(
            rep.train,
            data.train.labels,
            rep.test,
            data.test.labels,
            data.test.features,
            cfg.eval,
            data.class_names,
        ),
        "reconstruction": reconstruction_metrics(data.test.features, rep.reconstructed_test),
        "generators": [],
    }

    for condition_mode in cfg.generator.conditioning_modes:
        for generator_kind in cfg.generator.kinds:
            result["generators"].append(
                _run_generator(cfg, task, data, rep, generator_kind, condition_mode, int(task["seed"]))
            )

    result["runtime_seconds"] = perf_counter() - started
    atomic_write_json(output_path, result)
    return result


def run_manifest_dask(
    cfg: ExperimentConfig,
    manifest: Path | None = None,
    force: bool = False,
    n_workers: int | None = None,
) -> list[dict[str, Any]]:
    from dask.distributed import Client, as_completed

    tasks = read_jsonl(manifest or manifest_path(cfg))
    client = _make_client(cfg, n_workers)
    futures = [client.submit(run_task, cfg, task, force, pure=False) for task in tasks]
    results: list[dict[str, Any]] = []
    try:
        for future in as_completed(futures):
            results.append(future.result())
    finally:
        client.close()
    return results


def _run_generator(
    cfg: ExperimentConfig,
    task: dict[str, Any],
    data,
    rep,
    generator_kind: str,
    condition_mode: str,
    seed: int,
) -> dict[str, Any]:
    started = perf_counter()
    rng_seed = (
        seed + int(stable_hash([generator_kind, condition_mode], length=8), 16) % 100000
    )
    rng = np.random.default_rng(rng_seed)
    train_counts = _label_counts(data.train.labels)
    row: dict[str, Any] = {
        "generator": generator_kind,
        "conditioning": condition_mode,
        "status": "ok",
        "synthetic_count_source": "train",
        "synthetic_target_counts": train_counts,
    }
    try:
        fitted = fit_generator(
            generator_kind,
            condition_mode,
            rep.train,
            data.train.labels,
            seed,
            cfg.generator.gmm_components,
            cfg.generator.timeout_seconds,
            cfg.generator.gmm_fit_subsample,
            cfg.generator.gmm_max_iter,
            cfg.generator.gmm_select_k,
            cfg.generator.gmm_candidate_components,
            cfg.generator.gmm_selection_metric,
            cfg.generator.gmm_min_samples_per_component,
            x_valid=rep.valid,
        )
        if condition_mode == "true_label":
            z_synth, y_synth = fitted.sample_counts(train_counts, rng)
        elif condition_mode == "unconditioned":
            z_synth, y_synth = fitted.sample(len(data.train.labels), rng)
        else:
            raise ValueError(f"Unsupported conditioning mode {condition_mode!r}")

        row["synthetic_n_samples"] = int(len(z_synth))
        row["synthetic_label_counts"] = (
            _label_counts(y_synth) if y_synth is not None else {}
        )
        x_synth = (
            z_synth
            if task["projector"] == "raw_highdim"
            else inverse_transform(rep.model, z_synth, reference=(rep.train, data.train.features))
        )
        row["generator_metadata"] = fitted.metadata
        row["lowdim_fidelity"] = lowdim_fidelity_metrics(rep.test, z_synth, cfg.eval, seed)
        row["fidelity"] = fidelity_metrics(data.test.features, x_synth, cfg.eval, seed, task["encoder"])
        if y_synth is None:
            row["utility"] = {
                "status": "not_applicable",
                "reason": "unconditioned_generator_has_no_class_labels",
            }
            row["lowdim_utility"] = {
                "status": "not_applicable",
                "reason": "unconditioned_generator_has_no_class_labels",
            }
        else:
            row["lowdim_utility"] = utility_metrics(
                z_synth,
                y_synth,
                rep.train,
                data.train.labels,
                rep.test,
                data.test.labels,
                cfg.eval,
                data.class_names,
            )
            row["utility"] = utility_metrics(
                x_synth,
                y_synth,
                data.train.features,
                data.train.labels,
                data.test.features,
                data.test.labels,
                cfg.eval,
                data.class_names,
            )
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }
        )
    row["runtime_seconds"] = perf_counter() - started
    return row


def _label_counts(labels: np.ndarray | None) -> dict[int, int]:
    if labels is None:
        return {}
    keys, values = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(keys, values)}


def _make_client(cfg: ExperimentConfig, n_workers: int | None):
    from dask.distributed import Client

    if cfg.runtime.require_cuml:
        try:
            from dask_cuda import LocalCUDACluster

            cluster = LocalCUDACluster(
                n_workers=n_workers,
                threads_per_worker=1,
                dashboard_address=cfg.runtime.dask_dashboard_address,
            )
            return Client(cluster)
        except ImportError:
            pass
    return Client(
        n_workers=n_workers,
        threads_per_worker=1,
        dashboard_address=cfg.runtime.dask_dashboard_address,
    )
