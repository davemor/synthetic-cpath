from __future__ import annotations

import numpy as np

from scripts.run_fixed_kmeans24_cluster_followup import (
    KMEANS_PARAMS,
    fixed_cluster_specs,
    fit_predict_kmeans24,
)


def test_fixed_cluster_specs_are_the_requested_followup_settings() -> None:
    specs = fixed_cluster_specs()

    assert [(spec.dataset, spec.encoder, spec.projector, spec.dimension) for spec in specs] == [
        ("icaird100k", "uni2", "pca", 42),
        ("kather100k", "uni2", "pca", 50),
    ]
    assert {spec.clusterer for spec in specs} == {"kmeans"}
    assert {spec.scope for spec in specs} == {"global"}
    assert KMEANS_PARAMS == {
        "n_clusters": 24,
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 0,
    }


def test_fit_predict_kmeans24_returns_train_and_test_labels() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(96, 4)).astype(np.float32)
    test = rng.normal(size=(12, 4)).astype(np.float32)

    train_labels, test_labels, metadata = fit_predict_kmeans24(train, test)

    assert train_labels.shape == (96,)
    assert test_labels.shape == (12,)
    assert train_labels.min() >= 0
    assert train_labels.max() < 24
    assert test_labels.min() >= 0
    assert test_labels.max() < 24
    assert metadata["backend"] in {"sklearn", "cuml"}
    assert metadata["inertia"] >= 0
