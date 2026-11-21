from __future__ import annotations

import importlib
from pathlib import Path

from .config import ExperimentConfig
from .data import resolve_embedding_root


def validate_runtime(cfg: ExperimentConfig) -> dict:
    report = {"paths": {}, "imports": {}, "embedding_roots": []}
    for name, path in {
        "threshold_source": cfg.paths.threshold_source,
        "intrinsic_source": cfg.paths.intrinsic_source,
        "non_icaird_embeddings_root": cfg.paths.non_icaird_embeddings_root,
        "icaird_embeddings_root": cfg.paths.icaird_embeddings_root,
    }.items():
        report["paths"][name] = {"path": str(path), "exists": Path(path).exists()}

    for module in ["numpy", "pandas", "sklearn", "torch", "dask.distributed"]:
        try:
            importlib.import_module(module)
            report["imports"][module] = True
        except Exception as exc:
            report["imports"][module] = f"{type(exc).__name__}: {exc}"
    for module in ["cupy", "cuml", "dask_cuda"]:
        try:
            importlib.import_module(module)
            report["imports"][module] = True
        except Exception as exc:
            report["imports"][module] = f"{type(exc).__name__}: {exc}"

    for dataset in cfg.datasets:
        for encoder in cfg.encoders:
            root = resolve_embedding_root(cfg, dataset, encoder)
            report["embedding_roots"].append(
                {"dataset": dataset, "encoder": encoder, "path": str(root), "exists": root.exists()}
            )
    return report

