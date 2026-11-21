from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lowgen.dgx.cluster_conditioning import (
    cluster_label_guided_full_diag_gmm4_probe,
    cluster_guided_full_diag_gaussian_probe,
    cluster_guided_full_diag_gmm4_probe,
    label_guided_full_diag_gmm4_probe,
    load_cluster_split_labels,
    majority_label_map,
    run_cluster_conditioning_audit,
    select_metric_spread_cluster_rows,
)
from lowgen.dgx.config import ExperimentConfig
from lowgen.dgx.manifest import build_task_manifest


def test_cluster_conditioning_audit_reuses_saved_labels(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path)
    tasks = build_task_manifest(cfg)
    assert len(tasks) == 1

    labels_path = tmp_path / "cluster_labels.npz"
    train_ids = np.asarray([f"train_{i}" for i in range(36)], dtype=str)
    test_ids = np.asarray([f"test_{i}" for i in range(18)], dtype=str)
    train_labels = np.tile(np.arange(3), 12)
    test_labels = np.tile(np.arange(3), 6)
    np.savez_compressed(
        labels_path,
        labels=np.concatenate([train_labels, test_labels]),
        sample_ids=np.concatenate([train_ids, test_ids]),
        splits=np.asarray(["train"] * len(train_ids) + ["test"] * len(test_ids)),
    )
    clustering_results = tmp_path / "clustering_results.csv"
    pd.DataFrame(
        [
            {
                "task_id": "cluster-task",
                "dataset": "tiny",
                "encoder": "uni",
                "projector": "raw",
                "dimension": "",
                "target_name": "highdim_baseline",
                "projection_seed": 0,
                "cluster_seed": 0,
                "clusterer": "kmeans",
                "config_id": "k3",
                "cluster_params_json": '{"n_clusters": 3}',
                "labels_path": str(labels_path),
                "valid_cluster_count": True,
                "n_clusters": 3,
                "noise_fraction": 0.0,
                "silhouette": 0.4,
                "sd_index": 0.2,
                "mean_pairwise_ari": 0.9,
                "label_ari": 1.0,
            }
        ]
    ).to_csv(clustering_results, index=False)

    outputs = run_cluster_conditioning_audit(
        cfg,
        clustering_results_path=clustering_results,
    )

    results = pd.read_csv(outputs["results"])
    correlations = pd.read_csv(outputs["correlations"])
    skipped = pd.read_csv(outputs["skipped"])

    assert skipped.empty
    assert len(results) == 1
    assert results["generator"].iloc[0] == "cluster_guided_full_diag_gaussian_probe"
    assert results["conditioning"].iloc[0] == "cluster"
    assert results["status"].iloc[0] == "ok"
    assert results["synthetic_n_samples"].iloc[0] == 36
    assert results["synthetic_dimension"].iloc[0] == 8
    assert results["generation_space"].iloc[0] == "full_embedding"
    assert "silhouette" in correlations["cluster_metric"].to_numpy()
    assert "cluster_mean_full_embedding_diagonal_gaussian" in results["generator_metadata"].iloc[0]
    assert outputs["selected_cluster_configs"].exists()


def test_cluster_label_guided_audit_populates_utility(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path)
    cfg.cluster_conditioning.generator_kinds = [
        "cluster_label_guided_full_diag_gmm4_probe"
    ]
    cfg.cluster_conditioning.synthetic_total = 36
    tasks = build_task_manifest(cfg)
    assert len(tasks) == 1

    labels_path = tmp_path / "cluster_labels.npz"
    train_ids = np.asarray([f"train_{i}" for i in range(36)], dtype=str)
    test_ids = np.asarray([f"test_{i}" for i in range(18)], dtype=str)
    train_labels = np.tile(np.arange(3), 12)
    test_labels = np.tile(np.arange(3), 6)
    np.savez_compressed(
        labels_path,
        labels=np.concatenate([train_labels, test_labels]),
        sample_ids=np.concatenate([train_ids, test_ids]),
        splits=np.asarray(["train"] * len(train_ids) + ["test"] * len(test_ids)),
    )
    clustering_results = tmp_path / "clustering_results.csv"
    pd.DataFrame(
        [
            {
                "task_id": "cluster-task",
                "dataset": "tiny",
                "encoder": "uni",
                "projector": "raw",
                "dimension": "",
                "target_name": "highdim_baseline",
                "projection_seed": 0,
                "cluster_seed": 0,
                "clusterer": "kmeans",
                "config_id": "k3",
                "cluster_params_json": '{"n_clusters": 3}',
                "labels_path": str(labels_path),
                "valid_cluster_count": True,
                "n_clusters": 3,
                "noise_fraction": 0.0,
                "silhouette": 0.4,
                "sd_index": 0.2,
                "mean_pairwise_ari": 0.9,
                "label_ari": 1.0,
            }
        ]
    ).to_csv(clustering_results, index=False)

    outputs = run_cluster_conditioning_audit(
        cfg,
        clustering_results_path=clustering_results,
    )

    results = pd.read_csv(outputs["results"])
    assert len(results) == 1
    row = results.iloc[0]
    assert row["generator"] == "cluster_label_guided_full_diag_gmm4_probe"
    assert row["conditioning"] == "cluster_true_label"
    assert row["synthetic_count_source"] == "train_cluster_label_cells"
    assert pd.notna(row["tstr_macro_roc_auc_ovr"])
    assert pd.notna(row["aug_macro_roc_auc_ovr"])
    assert pd.notna(row["synthetic_label_counts"])


def test_label_guided_audit_populates_utility_without_cluster_conditioning(
    tmp_path: Path,
) -> None:
    cfg = _tiny_cfg(tmp_path)
    cfg.cluster_conditioning.generator_kinds = ["label_guided_full_diag_gmm4_probe"]
    cfg.cluster_conditioning.synthetic_total = 36
    tasks = build_task_manifest(cfg)
    assert len(tasks) == 1

    labels_path = tmp_path / "cluster_labels.npz"
    train_ids = np.asarray([f"train_{i}" for i in range(36)], dtype=str)
    test_ids = np.asarray([f"test_{i}" for i in range(18)], dtype=str)
    train_labels = np.tile(np.arange(3), 12)
    test_labels = np.tile(np.arange(3), 6)
    np.savez_compressed(
        labels_path,
        labels=np.concatenate([train_labels, test_labels]),
        sample_ids=np.concatenate([train_ids, test_ids]),
        splits=np.asarray(["train"] * len(train_ids) + ["test"] * len(test_ids)),
    )
    clustering_results = tmp_path / "clustering_results.csv"
    pd.DataFrame(
        [
            {
                "task_id": "cluster-task",
                "dataset": "tiny",
                "encoder": "uni",
                "projector": "raw",
                "dimension": "",
                "target_name": "highdim_baseline",
                "projection_seed": 0,
                "cluster_seed": 0,
                "clusterer": "kmeans",
                "config_id": "k3",
                "cluster_params_json": '{"n_clusters": 3}',
                "labels_path": str(labels_path),
                "valid_cluster_count": True,
                "n_clusters": 3,
                "noise_fraction": 0.0,
                "silhouette": 0.4,
                "sd_index": 0.2,
                "mean_pairwise_ari": 0.9,
                "label_ari": 1.0,
            }
        ]
    ).to_csv(clustering_results, index=False)

    outputs = run_cluster_conditioning_audit(
        cfg,
        clustering_results_path=clustering_results,
    )

    row = pd.read_csv(outputs["results"]).iloc[0]
    assert row["generator"] == "label_guided_full_diag_gmm4_probe"
    assert row["conditioning"] == "true_label"
    assert row["synthetic_count_source"] == "train_true_labels"
    assert pd.notna(row["tstr_macro_roc_auc_ovr"])
    assert pd.notna(row["aug_macro_roc_auc_ovr"])
    assert pd.notna(row["synthetic_label_counts"])


def test_cluster_label_helpers_align_and_map_majority(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.npz"
    np.savez_compressed(
        labels_path,
        labels=np.asarray([2, 1, 2, 0]),
        sample_ids=np.asarray(["train:b", "train:a", "test:d", "test:c"]),
    )

    train, test = load_cluster_split_labels(
        labels_path,
        np.asarray(["a", "b"]),
        np.asarray(["c", "d"]),
    )
    mapping = majority_label_map(
        np.asarray([0, 0, 1, 1, 1]),
        np.asarray([3, 3, 2, 2, 4]),
    )

    assert train.tolist() == [1, 2]
    assert test.tolist() == [0, 2]
    assert mapping == {0: 3, 1: 2}


def test_cluster_label_helpers_align_stringified_metadata(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.npz"
    np.savez_compressed(
        labels_path,
        labels=np.asarray([2, 1, 0]),
        sample_ids=np.asarray(
            [
                "nctcrche100k_train_000000.png",
                "nctcrche100k_train_000001.png",
                "nctcrche100k_test_000000.png",
            ]
        ),
    )

    train, test = load_cluster_split_labels(
        labels_path,
        np.asarray(
            [
                "{'patch_path': '/data/Kather100K/train/nctcrche100k_train_000000.png', "
                "'patch_filename': 'nctcrche100k_train_000000.png', "
                "'slide_id': 'train_000000', 'class': '2', 'split': 'train'}",
                "{'patch_path': '/data/Kather100K/train/nctcrche100k_train_000001.png', "
                "'patch_filename': 'nctcrche100k_train_000001.png', "
                "'slide_id': 'train_000001', 'class': '2', 'split': 'train'}",
            ]
        ),
        np.asarray(
            [
                "{'patch_path': '/data/Kather100K/test/nctcrche100k_test_000000.png', "
                "'patch_filename': 'nctcrche100k_test_000000.png', "
                "'slide_id': 'test_000000', 'class': '2', 'split': 'test'}",
            ]
        ),
    )

    assert train.tolist() == [2, 1]
    assert test.tolist() == [0]


def test_full_diag_probe_samples_full_embedding_dimensions() -> None:
    rng = np.random.default_rng(3)
    x_train = rng.normal(size=(12, 8)).astype(np.float32)
    labels = np.repeat(np.arange(3), 4)

    x_synth, metadata = cluster_guided_full_diag_gaussian_probe(
        x_train,
        labels,
        synthetic_total=5,
        min_cluster_size=2,
        min_variance=1e-6,
        rng=rng,
    )

    assert x_synth.shape == (5, 8)
    assert metadata["kind"] == "cluster_guided_full_diag_gaussian_probe"
    assert metadata["generation_space"] == "full_embedding"
    assert metadata["synthetic_dimension"] == 8


def test_full_diag_gmm4_probe_samples_full_embedding_dimensions() -> None:
    rng = np.random.default_rng(5)
    x_train = rng.normal(size=(24, 8)).astype(np.float32)
    labels = np.repeat(np.arange(3), 8)

    x_synth, metadata = cluster_guided_full_diag_gmm4_probe(
        x_train,
        labels,
        synthetic_total=9,
        min_cluster_size=2,
        min_variance=1e-6,
        rng=rng,
        n_components=4,
        fit_subsample=6,
        max_iter=20,
    )

    assert x_synth.shape == (9, 8)
    assert metadata["kind"] == "cluster_guided_full_diag_gmm4_probe"
    assert metadata["generation_space"] == "full_embedding"
    assert metadata["synthetic_dimension"] == 8
    assert metadata["gmm_components_requested"] == 4
    assert metadata["total_fit_samples"] == 18
    assert all(cluster["n_components_fitted"] == 4 for cluster in metadata["clusters"])


def test_cluster_label_guided_gmm4_probe_returns_true_labels() -> None:
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(36, 8)).astype(np.float32)
    cluster_labels = np.repeat(np.arange(3), 12)
    true_labels = np.tile(np.repeat(np.arange(2), 6), 3)

    x_synth, y_synth, metadata = cluster_label_guided_full_diag_gmm4_probe(
        x_train,
        cluster_labels,
        true_labels,
        synthetic_total=12,
        min_cluster_size=2,
        min_variance=1e-6,
        rng=rng,
        n_components=2,
        fit_subsample=6,
        max_iter=20,
    )

    assert x_synth.shape == (12, 8)
    assert y_synth.shape == (12,)
    assert set(y_synth.tolist()) == {0, 1}
    assert metadata["kind"] == "cluster_label_guided_full_diag_gmm4_probe"
    assert metadata["conditioning"] == "cluster_true_label"
    assert metadata["generation_space"] == "full_embedding"
    assert metadata["n_eligible_cluster_label_cells"] == 6
    assert all(cell["n_components_fitted"] == 2 for cell in metadata["cells"])


def test_label_guided_gmm4_probe_returns_true_labels() -> None:
    rng = np.random.default_rng(11)
    x_train = rng.normal(size=(36, 8)).astype(np.float32)
    true_labels = np.repeat(np.arange(3), 12)

    x_synth, y_synth, metadata = label_guided_full_diag_gmm4_probe(
        x_train,
        true_labels,
        synthetic_total=12,
        min_cluster_size=2,
        min_variance=1e-6,
        rng=rng,
        n_components=2,
        fit_subsample=6,
        max_iter=20,
    )

    assert x_synth.shape == (12, 8)
    assert y_synth.shape == (12,)
    assert set(y_synth.tolist()) == {0, 1, 2}
    assert metadata["kind"] == "label_guided_full_diag_gmm4_probe"
    assert metadata["conditioning"] == "true_label"
    assert metadata["generation_space"] == "full_embedding"
    assert metadata["n_eligible_labels"] == 3
    assert all(label["n_components_fitted"] == 2 for label in metadata["labels"])


def test_metric_spread_selection_takes_extremes_and_middle() -> None:
    rows = pd.DataFrame(
        {
            "config_id": [f"c{i}" for i in range(7)],
            "silhouette": [0.1, 0.9, 0.5, 0.2, 0.7, 0.4, 0.3],
            "sd_index": [7.0, 1.0, 4.0, 6.0, 2.0, 5.0, 3.0],
            "mean_pairwise_ari": [0.1, 0.3, 0.9, 0.2, 0.5, 0.7, 0.4],
        }
    )

    selected = select_metric_spread_cluster_rows(rows, cap=5)

    assert len(selected) == 5
    assert "c1" in set(selected["config_id"])
    assert "c0" in set(selected["config_id"])


def _tiny_cfg(tmp_path: Path) -> ExperimentConfig:
    root = tmp_path / "embeddings"
    _write_cache_dataset(root / "cache-tiny-uni", n_features=8)
    hypatia = tmp_path / "hypatia"
    (hypatia / "curve").mkdir(parents=True)
    pd.DataFrame(
        columns=[
            "dataset",
            "encoder",
            "projector",
            "classifier",
            "metric",
            "condition",
            "achieved_dim",
        ]
    ).to_csv(hypatia / "curve" / "thresholds_results.csv", index=False)
    pd.DataFrame(columns=["dataset", "encoder", "intrinsic_dim_mle"]).to_csv(
        hypatia / "intrinsic_dims.csv",
        index=False,
    )
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
                "projectors": [],
                "include_intrinsic": False,
                "include_raw_highdim": True,
            },
            "generator": {
                "kinds": ["diagonal_gaussian"],
                "conditioning_modes": ["unconditioned"],
                "timeout_seconds": 60,
            },
            "cluster_conditioning": {
                "min_clusters": 2,
                "max_noise_fraction": 0.5,
                "generator_kinds": ["cluster_guided_full_diag_gaussian_probe"],
                "synthetic_total": 50,
            },
            "eval": {
                "knn_k": 3,
                "fid_subsample": 12,
                "kid_subsample": 12,
                "nn_subsample": 12,
                "pairwise_subsample": 12,
                "cluster_count_k": 3,
                "classifier_max_iter": 200,
            },
        }
    )


def _write_cache_dataset(root: Path, n_features: int) -> None:
    rng = np.random.default_rng(0)
    for split, n in [("train", 36), ("test", 18)]:
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
