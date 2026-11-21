from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lowgen.dgx.config import ExperimentConfig
from lowgen.dgx.correlation import (
    correlation_statistics,
    run_correlation_audit,
    top_correlated_pairs,
)
from lowgen.dgx.manifest import build_task_manifest
from lowgen.dgx.runner import run_task
from lowgen.dgx.targets import import_dimension_sources


def test_correlation_statistics_handles_known_pairs_and_zero_variance() -> None:
    base = np.arange(12, dtype=np.float32)
    x = np.column_stack([base, base, -base, np.ones_like(base)])

    stats = correlation_statistics(x)
    corr = stats["correlation"]

    assert stats["n_zero_variance_dims"] == 1
    assert corr[0, 1] == 1.0
    assert corr[0, 2] == -1.0
    assert np.isnan(corr[0, 3])
    assert stats["max_abs_corr"] == 1.0
    assert stats["fraction_abs_corr_ge_0_7"] == 1.0
    assert stats["effective_dimension"] < 2.0


def test_top_correlated_pairs_excludes_diagonal_and_sorts_by_abs_value() -> None:
    corr = np.asarray(
        [
            [1.0, -0.9, 0.2],
            [-0.9, 1.0, 0.5],
            [0.2, 0.5, 1.0],
        ]
    )

    pairs = top_correlated_pairs(corr, top_n=2)

    assert [(row["dim_i"], row["dim_j"]) for row in pairs] == [(0, 1), (1, 2)]
    assert [row["abs_correlation"] for row in pairs] == [0.9, 0.5]
    assert [row["rank"] for row in pairs] == [1, 2]


def test_correlation_audit_smoke_with_raw_and_cached_pca(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path)
    import_dimension_sources(cfg)
    tasks = build_task_manifest(cfg)
    pca_task = next(task for task in tasks if task["projector"] == "pca")
    run_task(cfg, pca_task)

    outputs = run_correlation_audit(
        cfg,
        top_n=3,
        include_projectors=("raw_highdim", "pca"),
    )

    summary = pd.read_csv(outputs["summary"])
    pairs = pd.read_csv(outputs["top_pairs"])
    audit = pd.read_csv(outputs["audit"])
    matrices = sorted(
        (cfg.paths.run_dir / "correlation_audit" / "matrices").glob("*.npz")
    )

    assert set(summary["projector"]) == {"raw_highdim", "pca"}
    assert not pairs.empty
    assert audit.empty
    assert len(matrices) == len(summary)
    with np.load(matrices[0]) as data:
        assert "correlation" in data.files


def test_correlation_audit_records_missing_projection_cache(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path)
    import_dimension_sources(cfg)
    build_task_manifest(cfg)

    outputs = run_correlation_audit(
        cfg,
        output_dir=cfg.paths.run_dir / "missing_cache_audit",
        top_n=1,
        include_projectors=("pca",),
    )

    summary = pd.read_csv(outputs["summary"])
    audit = pd.read_csv(outputs["audit"])

    assert summary.empty
    assert not audit.empty
    assert set(audit["error_type"]) == {"FileNotFoundError"}


def test_correlation_audit_deduplicates_manifest_rows(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path)
    import_dimension_sources(cfg)
    tasks = build_task_manifest(cfg)
    raw_task = next(task for task in tasks if task["projector"] == "raw_highdim")
    duplicate_manifest = cfg.paths.run_dir / "duplicate_manifest.jsonl"
    duplicate_manifest.write_text(
        "\n".join(
            [_json_line(raw_task), _json_line(raw_task | {"target_name": "alias"})]
        )
        + "\n"
    )

    outputs = run_correlation_audit(
        cfg,
        manifest=duplicate_manifest,
        output_dir=cfg.paths.run_dir / "dedupe_audit",
        include_projectors=("raw_highdim",),
        deduplicate=True,
    )

    summary = pd.read_csv(outputs["summary"])
    assert len(summary) == 1


def _tiny_cfg(tmp_path: Path) -> ExperimentConfig:
    root = tmp_path / "embeddings"
    _write_cache_dataset(root / "cache-tiny-uni", n_features=8)
    hypatia = tmp_path / "hypatia"
    (hypatia / "curve").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset": "tiny",
                "encoder": "uni",
                "projector": "pca",
                "classifier": "logistic-regression",
                "metric": "accuracy",
                "condition": 0.95,
                "baseline": 1.0,
                "threshold_value": 0.95,
                "achieved_dim": 3,
                "achieved_value": 0.97,
            },
        ]
    ).to_csv(hypatia / "curve" / "thresholds_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "tiny",
                "encoder": "uni",
                "num_dimensions": 8,
                "n_samples": 30,
                "intrinsic_dim_mle": 2,
            }
        ]
    ).to_csv(hypatia / "intrinsic_dims.csv", index=False)
    return ExperimentConfig.from_dict(
        {
            "paths": {
                "hypatia_run_root": str(hypatia),
                "non_icaird_embeddings_root": str(root),
                "output_root": str(tmp_path / "out"),
                "run_name": "run",
            },
            "datasets": ["tiny"],
            "encoders": ["uni"],
            "seeds": [0],
            "projection": {
                "projectors": ["pca"],
                "threshold_metrics": ["accuracy"],
                "threshold_conditions": [0.95],
                "include_intrinsic": False,
                "include_raw_highdim": True,
            },
            "generator": {
                "kinds": ["resample"],
                "conditioning_modes": ["unconditioned"],
                "gmm_components": 2,
                "timeout_seconds": 60,
            },
            "eval": {
                "knn_k": 3,
                "fid_subsample": 12,
                "kid_subsample": 12,
                "nn_subsample": 12,
                "pairwise_subsample": 12,
                "cluster_count_k": 3,
            },
        }
    )


def _write_cache_dataset(root: Path, n_features: int) -> None:
    rng = np.random.default_rng(0)
    for split, n in [("train", 30), ("test", 18)]:
        split_root = root / split
        split_root.mkdir(parents=True)
        y = np.tile(np.arange(3), n // 3)
        x = rng.normal(size=(len(y), n_features)).astype(np.float32)
        x += y[:, None].astype(np.float32) * 0.5
        torch.save(torch.from_numpy(x), split_root / "emb_0000.pt")
        torch.save(torch.from_numpy(y.astype(np.int64)), split_root / "labels_0000.pt")
        (split_root / "paths_0000.json").write_text(
            str([f"{split}_{i}" for i in range(len(y))]).replace("'", '"')
        )
        (split_root / "meta.json").write_text(
            f'{{"num_samples": {len(y)}, "embedding_dim": {n_features}}}'
        )
    (root / "label_map.json").write_text('{"0": "a", "1": "b", "2": "c"}')


def _json_line(row: dict) -> str:
    import json

    return json.dumps(row, sort_keys=True)
