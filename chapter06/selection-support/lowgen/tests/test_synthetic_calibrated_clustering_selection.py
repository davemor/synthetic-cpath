from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "select_clustering_configs_from_synthetic_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("synthetic_calibrated_selection", SCRIPT_PATH)
assert SPEC is not None
selector = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


def test_correlation_direction_conversion() -> None:
    synthetic = pd.DataFrame(
        {
            "good_metric": [1, 2, 3, 4, 5],
            "bad_metric": [1, 2, 3, 4, 5],
            "fid": [5, 4, 3, 2, 1],
            "kid": [5, 4, 3, 2, 1],
            "precision": [1, 2, 3, 4, 5],
            "recall": [1, 2, 3, 4, 5],
        }
    )
    synthetic["bad_metric"] = synthetic["bad_metric"] * -1
    specs = [
        selector.MetricSpec("good", "good_metric", "mean_good"),
        selector.MetricSpec("bad", "bad_metric", "mean_bad"),
    ]

    calibration = selector.build_metric_calibration(
        synthetic,
        metric_specs=specs,
        quality_metrics=["fid", "kid", "precision", "recall"],
        quality_weights={"fid": 0.25, "kid": 0.25, "precision": 0.25, "recall": 0.25},
        min_n=3,
    )

    no_label = calibration[calibration["scenario"].eq("no_label")]
    good = no_label[no_label["metric"].eq("good")].iloc[0]
    bad = no_label[no_label["metric"].eq("bad")].iloc[0]
    assert good["effect"] > 0
    assert bool(good["higher_is_better"])
    assert bad["effect"] < 0
    assert not bool(bad["higher_is_better"])


def test_effective_support_calculation() -> None:
    frame = pd.DataFrame(
        {
            "n_samples": [1000, 1000],
            "mean_noise_fraction": [0.0, 0.2],
            "mean_n_clusters": [10, 20],
        }
    )

    support = selector.effective_support(
        frame,
        n_samples_column="n_samples",
        noise_column="mean_noise_fraction",
        clusters_column="mean_n_clusters",
    )

    assert support.tolist() == [100.0, 40.0]


def test_full_selection_filters_noise_support_clusters_validity_and_clusterer() -> None:
    stability = pd.DataFrame(
        [
            _stability_row("ok", noise=0.0, clusters=16, valid_rate=1.0, clusterer="kmeans"),
            _stability_row("noise", noise=0.2, clusters=16, valid_rate=1.0, clusterer="kmeans"),
            _stability_row("support", noise=0.0, clusters=400, valid_rate=1.0, clusterer="kmeans"),
            _stability_row("clusters", noise=0.0, clusters=8, valid_rate=1.0, clusterer="kmeans"),
            _stability_row("validity", noise=0.0, clusters=16, valid_rate=0.5, clusterer="kmeans"),
            _stability_row("leiden", noise=0.0, clusters=16, valid_rate=1.0, clusterer="leiden"),
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scenario": "no_label",
                "metric": "mean_pairwise_ari",
                "clustering_column": "mean_pairwise_ari",
                "weight": 1.0,
                "higher_is_better": True,
            }
        ]
    )

    ranked, excluded = selector.rank_clustering_experiment_configs(
        stability,
        calibration,
        group_columns=["dataset", "encoder"],
        max_noise=0.10,
        min_effective_support=750,
        min_clusters=16,
        valid_cluster_count_rate=1.0,
        excluded_clusterers=["singleton", "leiden"],
    )

    eligible = ranked[ranked["eligible"]]
    assert eligible["target_name"].tolist() == ["ok"]
    reasons = dict(zip(excluded["target_name"], excluded["exclude_reason"], strict=True))
    assert "high_noise" in reasons["noise"]
    assert "low_effective_support" in reasons["support"]
    assert "too_few_clusters" in reasons["clusters"]
    assert "invalid_cluster_count_rate" in reasons["validity"]
    assert "excluded_clusterer" in reasons["leiden"]


def test_ranking_uses_learned_metric_direction() -> None:
    stability = pd.DataFrame(
        [
            _stability_row("low", ari=0.1),
            _stability_row("high", ari=0.9),
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scenario": "no_label",
                "metric": "mean_pairwise_ari",
                "clustering_column": "mean_pairwise_ari",
                "weight": 1.0,
                "higher_is_better": True,
            }
        ]
    )

    ranked, _ = selector.rank_clustering_experiment_configs(
        stability,
        calibration,
        group_columns=["dataset", "encoder"],
        max_noise=0.10,
        min_effective_support=750,
        min_clusters=16,
        valid_cluster_count_rate=1.0,
        excluded_clusterers=["singleton", "leiden"],
    )

    best = selector.best_by_group(ranked, ["scenario", "dataset", "encoder"])
    assert best["target_name"].tolist() == ["high"]


def _stability_row(
    target_name: str,
    *,
    noise: float = 0.0,
    clusters: int = 16,
    valid_rate: float = 1.0,
    clusterer: str = "kmeans",
    ari: float = 0.5,
) -> dict[str, object]:
    return {
        "dataset": "d",
        "encoder": "e",
        "projector": "pca",
        "dimension": "16",
        "target_name": target_name,
        "source_metric": "dim",
        "condition": "baseline",
        "clusterer": clusterer,
        "cluster_params_json": "{}",
        "n_samples": 16000,
        "mean_noise_fraction": noise,
        "mean_n_clusters": clusters,
        "valid_cluster_count_rate": valid_rate,
        "mean_pairwise_ari": ari,
    }
