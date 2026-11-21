from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lowgen.dgx.config import ExperimentConfig
from lowgen.dgx.manifest import build_task_manifest
from lowgen.dgx.runner import run_task
from lowgen.dgx.summary import aggregate
from lowgen.dgx.targets import import_dimension_sources


def test_manifest_and_raw_pca_smoke(tmp_path: Path) -> None:
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
        [{"dataset": "tiny", "encoder": "uni", "num_dimensions": 8, "n_samples": 30, "intrinsic_dim_mle": 2}]
    ).to_csv(hypatia / "intrinsic_dims.csv", index=False)
    cfg = ExperimentConfig.from_dict(
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
                "fixed_dimensions": [4],
                "threshold_metrics": ["accuracy"],
                "threshold_conditions": [0.95],
                "include_intrinsic": True,
                "include_raw_highdim": True,
            },
            "generator": {
                "kinds": ["resample", "diagonal_gaussian"],
                "conditioning_modes": ["unconditioned", "true_label"],
                "gmm_components": 2,
                "timeout_seconds": 60,
            },
            "eval": {
                "knn_k": 3,
                "fid_subsample": 20,
                "kid_subsample": 20,
                "nn_subsample": 20,
                "pairwise_subsample": 20,
                "cluster_count_k": 3,
            },
        }
    )
    import_dimension_sources(cfg)
    tasks = build_task_manifest(cfg)
    assert {task["projector"] for task in tasks} == {"raw_highdim", "pca"}
    assert any(task["target_name"] == "highdim_baseline" for task in tasks)
    assert any(task["target_name"] == "dim_4" for task in tasks)

    for task in tasks[:2]:
        result = run_task(cfg, task)
        assert result["status"] == "ok"
        assert "accuracy" in result["reduction"]["linear_probe"]
        assert len(result["generators"]) == 4

    outputs = aggregate(cfg)
    assert outputs["results_csv"].exists()
    assert outputs["classification_reports"].exists()
    assert outputs["latex_report"].exists()


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
        (split_root / "paths_0000.json").write_text(str([f"{split}_{i}" for i in range(len(y))]).replace("'", '"'))
        (split_root / "meta.json").write_text(
            f'{{"num_samples": {len(y)}, "embedding_dim": {n_features}}}'
        )
    (root / "label_map.json").write_text('{"0": "a", "1": "b", "2": "c"}')
