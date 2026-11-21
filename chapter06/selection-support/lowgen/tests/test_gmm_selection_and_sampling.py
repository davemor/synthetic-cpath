from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lowgen.dgx.config import ExperimentConfig
from lowgen.dgx.generators import (
    candidate_components_for_samples,
    fit_generator,
    select_gmm_components,
)
from lowgen.dgx.manifest import build_task_manifest
from lowgen.dgx.runner import run_task
from lowgen.dgx.summary import aggregate
from lowgen.dgx.targets import import_dimension_sources


def test_candidate_components_are_capped_by_subset_size() -> None:
    assert candidate_components_for_samples(50, [1, 2, 5, 10], 100) == [1]
    assert candidate_components_for_samples(250, [1, 2, 5, 10], 100) == [1, 2]
    assert candidate_components_for_samples(1000, [1, 2, 5, 10], 100) == [
        1,
        2,
        5,
        10,
    ]


def test_bic_selection_finds_two_component_mixture() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate(
        [
            rng.normal(-5.0, 0.25, size=(120, 1)),
            rng.normal(5.0, 0.25, size=(120, 1)),
        ]
    ).astype(np.float32)

    selected, class_components, metadata = select_gmm_components(
        x,
        None,
        [1, 2],
        "bic",
        10,
        0,
        100,
    )

    assert selected == 2
    assert class_components is None
    assert metadata["selected_k"] == 2
    assert [row["k"] for row in metadata["records"]] == [1, 2]


def test_true_label_selection_falls_back_to_one_for_small_classes() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 2)).astype(np.float32)
    y = np.repeat([0, 1, 2], 10)

    selected, class_components, metadata = select_gmm_components(
        x,
        y,
        [1, 2, 5],
        "bic",
        100,
        0,
        100,
    )

    assert selected == 1
    assert class_components == {0: 1, 1: 1, 2: 1}
    assert metadata["selected_k_by_label"] == {"0": 1, "1": 1, "2": 1}


def test_fixed_k_path_still_works_when_selection_disabled() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 3)).astype(np.float32)
    y = np.repeat([0, 1, 2], 20)

    fitted = fit_generator(
        "diagonal_gmm",
        "true_label",
        x,
        y,
        seed=0,
        n_components=2,
        timeout_seconds=60,
        fit_subsample=60,
        max_iter=20,
        select_k=False,
    )

    assert fitted.metadata["gmm_select_k"] is False
    assert fitted.metadata["n_components"] == 2
    assert fitted.metadata["gmm_selection"]["enabled"] is False


def test_run_task_samples_synthetic_data_at_train_size_and_aggregates(
    tmp_path: Path,
) -> None:
    cfg = _tiny_cfg(tmp_path)
    import_dimension_sources(cfg)
    task = next(
        task for task in build_task_manifest(cfg) if task["projector"] == "raw_highdim"
    )

    result = run_task(cfg, task)
    by_key = {
        (row["generator"], row["conditioning"]): row for row in result["generators"]
    }

    assert by_key[("resample", "unconditioned")]["synthetic_n_samples"] == 30
    assert by_key[("resample", "unconditioned")]["synthetic_label_counts"] == {}
    assert by_key[("resample", "true_label")]["synthetic_label_counts"] == {
        0: 10,
        1: 10,
        2: 10,
    }
    gmm = by_key[("diagonal_gmm", "true_label")]
    assert gmm["generator_metadata"]["gmm_selection"]["enabled"] is True
    assert gmm["generator_metadata"]["gmm_selection"]["selected_k_by_label"]
    assert "lowdim_fidelity" in gmm
    assert "lowdim_utility" in gmm
    assert "centroid_distance" in gmm["lowdim_fidelity"]
    assert "train_synthetic_test_real" in gmm["lowdim_utility"]

    outputs = aggregate(cfg)
    df = pd.read_csv(outputs["results_csv"])
    gen = df[df["row_type"] == "generator"]
    assert "synthetic_n_samples" in gen.columns
    assert "gmm_selected_k_min" in gen.columns
    assert "lowdim_fid" in gen.columns
    assert "lowdim_centroid_distance" in gen.columns
    assert "lowdim_tstr_accuracy" in gen.columns
    assert gen["synthetic_n_samples"].dropna().eq(30).all()

    reports = pd.read_json(outputs["classification_reports"], lines=True)
    assert "lowdim_train_synthetic_test_real" in set(reports["classifier_evaluation"])


def _tiny_cfg(tmp_path: Path) -> ExperimentConfig:
    root = tmp_path / "embeddings"
    _write_cache_dataset(root / "cache-tiny-uni", n_features=5)
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
                "achieved_dim": 2,
                "achieved_value": 0.97,
            },
        ]
    ).to_csv(hypatia / "curve" / "thresholds_results.csv", index=False)
    pd.DataFrame(
        [{"dataset": "tiny", "encoder": "uni", "n_samples": 30, "intrinsic_dim_mle": 2}]
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
                "kinds": ["resample", "diagonal_gaussian", "diagonal_gmm"],
                "conditioning_modes": ["unconditioned", "true_label"],
                "gmm_components": 2,
                "gmm_fit_subsample": 30,
                "gmm_max_iter": 20,
                "gmm_select_k": True,
                "gmm_candidate_components": [1, 2],
                "gmm_min_samples_per_component": 5,
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
