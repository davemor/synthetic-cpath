from pathlib import Path

import numpy as np

from cluster_exp.clusterers import compute_fit_subsample_size, run_clustering_task
from cluster_exp.config import ExperimentConfig
from cluster_exp.manifest import build_task_manifest, _cluster_grid_size
from cluster_exp.posthoc import add_sd_index_to_run
from cluster_exp.projections import (
    build_projection_manifest,
    deterministic_sample_indices,
    prepare_projection_cache,
    projection_cache_key,
)
from cluster_exp.summary import summarise_run
from cluster_exp.dimensions import ProjectionTarget
from cluster_exp.utils import read_jsonl


def _make_fixture(tmp_path: Path) -> ExperimentConfig:
    embeddings_root = tmp_path / "emb"
    root = embeddings_root / "cache-toy-uni"
    root.mkdir(parents=True)
    rng = np.random.default_rng(42)
    left = rng.normal(loc=-2, scale=0.2, size=(10, 4))
    right = rng.normal(loc=2, scale=0.2, size=(10, 4))
    features = np.vstack([left, right]).astype(np.float32)
    labels = np.asarray([0] * 10 + [1] * 10)
    np.savez(
        root / "features.npz",
        features=features,
        labels=labels,
        sample_ids=np.asarray([f"s{i}" for i in range(20)]),
        splits=np.asarray(["all"] * 20),
    )

    dim_root = tmp_path / "dim"
    (dim_root / "curve").mkdir(parents=True)
    (dim_root / "intrinsic_dims.csv").write_text(
        "dataset,encoder,intrinsic_dim_mle\n"
        "toy,uni,2\n"
    )
    (dim_root / "curve" / "thresholds_results.csv").write_text(
        "dataset,encoder,projector,classifier,metric,condition,achieved_dim,baseline\n"
        "toy,uni,pca,logistic-regression,accuracy,0.95,2,1.0\n"
        "toy,uni,pca,logistic-regression,accuracy,0.99,2,1.0\n"
        "toy,uni,pca,logistic-regression,roc_auc,0.95,3,1.0\n"
        "toy,uni,pca,logistic-regression,roc_auc,0.99,3,1.0\n"
    )

    return ExperimentConfig.from_dict(
        {
            "paths": {
                "embeddings_root": str(embeddings_root),
                "dimensionality_run_root": str(dim_root),
                "output_root": str(tmp_path / "out"),
                "run_name": "run",
            },
            "datasets": ["toy"],
            "encoders": ["uni"],
            "seeds": [0, 1],
            "projection": {
                "projectors": ["pca"],
                "threshold_metrics": ["roc_auc"],
                "include_one_dim": False,
            },
            "clustering": {
                "fit_subsample": 10,
                "metric_subsample": 10,
                "clusterers": ["kmeans", "hdbscan"],
                "kmeans": {"k": [2, 3], "n_init": 2, "max_iter": 50},
                "hdbscan": {"min_cluster_size": [2], "min_samples": [1]},
            },
            "runtime": {"require_cuml": False, "allow_cpu_fallback": True},
        }
    )


def test_deterministic_sample_indices() -> None:
    first = deterministic_sample_indices(100, 10, 7)
    second = deterministic_sample_indices(100, 10, 7)
    third = deterministic_sample_indices(100, 10, 8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)
    assert np.all(first[:-1] <= first[1:])


def test_projection_cache_key_includes_seed(tmp_path: Path) -> None:
    cfg = _make_fixture(tmp_path)
    target_a = ProjectionTarget("toy", "uni", "pca", 2, "intrinsic_mle", seed=0)
    target_b = ProjectionTarget("toy", "uni", "pca", 2, "intrinsic_mle", seed=1)
    assert projection_cache_key(cfg, target_a) != projection_cache_key(cfg, target_b)


def test_fit_subsample_fraction_with_floor_and_cap(tmp_path: Path) -> None:
    cfg = _make_fixture(tmp_path)
    cfg.clustering.fit_subsample = None
    cfg.clustering.fit_subsample_fraction = 0.10
    cfg.clustering.fit_subsample_min = 10000
    cfg.clustering.fit_subsample_max = 50000

    assert compute_fit_subsample_size(cfg, 20000) == 10000
    assert compute_fit_subsample_size(cfg, 200000) == 20000
    assert compute_fit_subsample_size(cfg, 1000000) == 50000
    assert compute_fit_subsample_size(cfg, 5000) == 5000


def test_leiden_grid_contributes_to_manifest_size(tmp_path: Path) -> None:
    cfg = _make_fixture(tmp_path)
    cfg.clustering.clusterers = ["kmeans", "hdbscan", "leiden"]
    cfg.clustering.leiden.n_neighbors = [15, 30]
    cfg.clustering.leiden.resolution = [0.5, 1.0, 2.0]

    assert _cluster_grid_size(cfg) == 9


def test_spherical_kmeans_grid_contributes_to_manifest_size(tmp_path: Path) -> None:
    cfg = _make_fixture(tmp_path)
    cfg.clustering.clusterers = ["spherical_kmeans"]
    cfg.clustering.spherical_kmeans.k = [4, 8, 16]

    assert _cluster_grid_size(cfg) == 3


def test_prepare_manifest_run_and_summarise(tmp_path: Path) -> None:
    cfg = _make_fixture(tmp_path)
    projection_task_manifest_path = tmp_path / "projection_tasks.jsonl"
    planned_manifest = build_projection_manifest(cfg, projection_task_manifest_path)
    assert len(read_jsonl(projection_task_manifest_path)) == len(
        planned_manifest[planned_manifest["projector"] != "raw"]
    )

    projection_manifest = prepare_projection_cache(cfg)
    assert {"logical_raw", "hit"}.issubset(set(projection_manifest["cache_status"]))

    manifest_path = tmp_path / "tasks.jsonl"
    tasks = build_task_manifest(cfg, manifest_path)
    loaded = read_jsonl(manifest_path)
    assert len(tasks) == len(loaded)
    assert all(task["cluster_grid_size"] == 3 for task in tasks)

    for idx in range(min(6, len(loaded))):
        result = run_clustering_task(cfg, loaded[idx])
        assert len(result) == 3
        assert all(Path(row["labels_path"]).exists() for row in result)
        assert all(row["fit_subsample"] == 10 for row in result)
        assert all(row["n_clusters"] >= 1 for row in result)
        assert all("sd_index" in row for row in result)
        assert all("sd_scat" in row for row in result)
        assert all("sd_dis" in row for row in result)

    outputs = summarise_run(cfg)
    assert outputs["results"].exists()
    assert outputs["stability"].exists()
    assert outputs["best"].exists()
    assert outputs["kmeans_stability"].exists()
    assert outputs["best_kmeans"].exists()
    assert outputs["spherical_kmeans_stability"].exists()
    assert outputs["best_spherical_kmeans"].exists()
    assert outputs["hdbscan_stability"].exists()
    assert outputs["best_hdbscan"].exists()
    stability = outputs["stability"].read_text()
    assert "mean_label_ari" in stability
    assert "mean_label_ami" in stability
    assert "mean_label_nmi" in stability
    assert "mean_label_purity" in stability
    assert "mean_sd_index" in stability


def test_posthoc_add_sd_index_updates_existing_results(tmp_path: Path) -> None:
    cfg = _make_fixture(tmp_path)
    prepare_projection_cache(cfg)
    manifest_path = tmp_path / "tasks.jsonl"
    build_task_manifest(cfg, manifest_path)
    loaded = read_jsonl(manifest_path)

    result = run_clustering_task(cfg, loaded[0])
    result_path = cfg.paths.run_dir / "tasks" / loaded[0]["task_id"] / "result.json"
    stripped = [
        {key: value for key, value in row.items() if key not in {"sd_index", "sd_scat", "sd_dis"}}
        for row in result
    ]
    result_path.write_text(__import__("json").dumps(stripped))

    stats = add_sd_index_to_run(cfg, manifest_paths=[manifest_path])
    assert stats["updated_task_files"] == 1
    assert stats["updated_result_rows"] == len(result)

    updated = __import__("json").loads(result_path.read_text())
    assert all("sd_index" in row for row in updated)
    assert all("sd_scat" in row for row in updated)
    assert all("sd_dis" in row for row in updated)
