import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synpath.metrics.quality import (
    cluster_adherence,
    load_or_extract_inception_features,
    precision_recall,
)
from synpath.paths import SynpathPaths
from synpath.reports.aggregate import collect_quality


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    monkeypatch.setenv("SYNPATH_PR_DEVICE", "cpu")


def test_precision_recall_is_complete_for_identical_features():
    features = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = precision_recall(
        features, features, nearest_neighbors=2, batch_size=2
    )

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["nearest_neighbors"] == 2
    assert result["method"] == "improved_precision_recall"


def test_precision_recall_rejects_separated_manifolds():
    real = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    synthetic = real + 100.0

    result = precision_recall(
        real, synthetic, nearest_neighbors=2, batch_size=2
    )

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


def test_precision_recall_validates_nearest_neighbors():
    features = np.zeros((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="smaller than both sample sets"):
        precision_recall(features, features, nearest_neighbors=3)


def test_quality_report_excludes_legacy_precision_recall(tmp_path):
    metrics = (
        tmp_path
        / "generators"
        / "data"
        / "model"
        / "mode"
        / "seed0"
        / "metrics"
    )
    metrics.mkdir(parents=True)
    current = metrics / "uni_precision_recall.json"
    legacy = metrics / "uni_precision_recall_legacy_global_radius.json"
    legacy_coverage = (
        metrics / "uni_cluster_coverage_legacy_requested_ids.json"
    )
    current.write_text(json.dumps({"precision": 0.5, "recall": 0.25}))
    legacy.write_text(json.dumps({"precision": 0.0, "recall": 0.0}))
    legacy_coverage.write_text(json.dumps({"coverage": 0.0}))

    report = collect_quality(tmp_path)

    assert set(report["source_file"]) == {"uni_precision_recall.json"}
    assert set(report["value"]) == {0.5, 0.25}


def test_cluster_adherence_predicts_morphology_from_features(tmp_path):
    paths = make_paths(tmp_path)
    cluster_root = paths.cache_root / "clusters" / "demo"
    cluster_root.mkdir(parents=True)
    real = embedding_frame(
        ["/real/a.png", "/real/b.png", "/real/c.png", "/real/d.png"],
        [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]],
        [0, 0, 1, 1],
    )
    real[["image_path", "cluster_id"]].to_csv(
        cluster_root / "pseudo_labels.csv", index=False
    )
    synthetic = embedding_frame(
        ["/fake/a.png", "/fake/b.png", "/fake/c.png", "/fake/d.png"],
        [[0.05, 0.0], [0.08, 0.0], [10.05, 10.0], [10.08, 10.0]],
        [0, 0, 1, 1],
    )

    result, contingency, counts = cluster_adherence(
        real,
        synthetic,
        "demo",
        "pseudo_label",
        paths,
    )

    assert result["coverage"] == 1.0
    assert result["exact_agreement"] == 1.0
    assert result["adjusted_rand_index"] == 1.0
    assert result["normalized_mutual_info"] == 1.0
    assert result["training_assignment_agreement"] == 1.0
    assert contingency["count"].sum() == 4
    assert counts["predicted_count"].sum() == 4
    assert (cluster_root / "global_kmeans_model.joblib").exists()


def test_cluster_adherence_detects_ignored_requested_condition(tmp_path):
    paths = make_paths(tmp_path)
    cluster_root = paths.cache_root / "clusters" / "demo"
    cluster_root.mkdir(parents=True)
    real = embedding_frame(
        ["/real/a.png", "/real/b.png", "/real/c.png", "/real/d.png"],
        [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]],
        [0, 0, 1, 1],
    )
    real[["image_path", "cluster_id"]].to_csv(
        cluster_root / "pseudo_labels.csv", index=False
    )
    synthetic = embedding_frame(
        ["/fake/a.png", "/fake/b.png", "/fake/c.png", "/fake/d.png"],
        [[0.05, 0.0], [0.08, 0.0], [0.06, 0.0], [0.09, 0.0]],
        [0, 0, 1, 1],
    )

    result, _contingency, _counts = cluster_adherence(
        real,
        synthetic,
        "demo",
        "label_global_cluster",
        paths,
    )

    assert result["coverage"] == 0.5
    assert result["exact_agreement"] == 0.5
    assert result["adjusted_rand_index"] == 0.0


def test_external_cluster_adherence_reports_counts_without_refitting(tmp_path):
    paths = make_paths(tmp_path)
    cluster_root = paths.cache_root / "clusters" / "demo"
    cluster_root.mkdir(parents=True)
    real = embedding_frame(
        ["/real/a.png", "/real/b.png", "/real/c.png", "/real/d.png"],
        [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]],
        [0, 0, 1, 1],
    )
    real[["image_path", "cluster_id"]].to_csv(
        cluster_root / "pseudo_labels.csv", index=False
    )
    (cluster_root / "balanced_global_cluster_metadata.json").write_text(
        json.dumps(
            {
                "conditioning_scope": "global",
                "encoder": "uni2",
                "projector": "pca",
                "pca_dim": 50,
                "clusterer": "kmeans",
                "n_clusters": 2,
                "cluster_config_path": "/calibration/selected.yaml",
            }
        )
    )
    synthetic = embedding_frame(
        ["/fake/a.png", "/fake/b.png", "/fake/c.png", "/fake/d.png"],
        [[100.0, 100.0]] * 4,
        [0, 0, 1, 1],
    )

    result, contingency, counts = cluster_adherence(
        real,
        synthetic,
        "demo",
        "label_global_cluster",
        paths,
    )

    assert result["status"] == "conditioning_counts_only"
    assert result["coverage"] == 1.0
    assert result["coverage_definition"] == (
        "requested conditioning-ID coverage only"
    )
    assert result["image_derived_adherence_status"] == "not_computed"
    assert "exact_agreement" not in result
    assert contingency is None
    assert counts["real_training_count"].sum() == 4
    assert counts["requested_count"].sum() == 4
    assert not (cluster_root / "global_kmeans_model.joblib").exists()


def test_load_or_extract_inception_features_caches_result(tmp_path):
    class FakeFid:
        calls = 0

        @classmethod
        def get_files_features(cls, image_paths, **_kwargs):
            cls.calls += 1
            return np.arange(len(image_paths) * 2).reshape(-1, 2)

    output = tmp_path / "features.npy"
    kwargs = {
        "image_paths": ["a.png", "b.png"],
        "output": output,
        "fid": FakeFid,
        "model": object(),
        "device": "cpu",
        "batch_size": 2,
        "num_workers": 0,
        "description": "test",
    }

    first = load_or_extract_inception_features(**kwargs)
    second = load_or_extract_inception_features(**kwargs)

    assert np.array_equal(first, second)
    assert FakeFid.calls == 1


def embedding_frame(paths, features, cluster_ids):
    frame = pd.DataFrame(
        {
            "image_path": paths,
            "label": ["class"] * len(paths),
            "cluster_id": cluster_ids,
        }
    )
    for index, values in enumerate(np.asarray(features).T):
        frame[f"f{index}"] = values
    return frame


def make_paths(root: Path) -> SynpathPaths:
    return SynpathPaths(
        data_root=root / "data",
        cache_root=root / "cache",
        outputs_root=root / "outputs",
        reports_root=root / "reports",
        external_root=root / "external",
    )
