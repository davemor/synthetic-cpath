from __future__ import annotations

import pandas as pd

from cluster_exp.dimension_plots import (
    select_best_dimension_configs,
    summarise_dimension_trends,
    summarise_seed_raw_baselines,
    summarise_seed_dimension_trends,
)


def test_dimension_trends_select_best_config_per_dimension() -> None:
    stability = pd.DataFrame(
        [
            _row("d1", "e1", "kmeans", 1, ami=0.20, stability=0.90),
            _row("d1", "e1", "kmeans", 1, ami=0.80, stability=0.50),
            _row("d2", "e1", "kmeans", 1, ami=0.60, stability=0.70),
            _row("d1", "e1", "hdbscan", 2, ami=0.50, stability=0.40, noise=0.30),
            _row("d2", "e1", "hdbscan", 2, ami=0.70, stability=0.50, noise=0.10),
            _row("d1", "e1", "kmeans", 51, ami=0.99, stability=0.99),
            {**_row("d1", "e1", "kmeans", 1, ami=0.99, stability=0.99), "projector": "umap"},
        ]
    )

    selected = select_best_dimension_configs(stability, projector="pca", min_dimension=1, max_dimension=50)
    hdbscan_selected = select_best_dimension_configs(
        stability,
        projector="pca",
        clusterer="hdbscan",
        min_dimension=1,
        max_dimension=50,
    )
    fixed_selected = select_best_dimension_configs(
        stability,
        projector="pca",
        dataset="d1",
        encoder="e1",
        min_dimension=1,
        max_dimension=50,
    )

    assert len(selected) == 4
    assert set(hdbscan_selected["clusterer"]) == {"hdbscan"}
    assert len(hdbscan_selected) == 2
    assert set(fixed_selected["dataset"]) == {"d1"}
    assert set(fixed_selected["encoder"]) == {"e1"}
    selected_kmeans = selected[
        (selected["dataset"] == "d1")
        & (selected["encoder"] == "e1")
        & (selected["clusterer"] == "kmeans")
        & (selected["dimension"] == 1)
    ].iloc[0]
    assert selected_kmeans["mean_label_ami"] == 0.80

    trends = summarise_dimension_trends(selected)
    kmeans_dim1 = trends[(trends["clusterer"] == "kmeans") & (trends["dimension"] == 1)].iloc[0]
    hdbscan_dim2 = trends[(trends["clusterer"] == "hdbscan") & (trends["dimension"] == 2)].iloc[0]
    assert round(kmeans_dim1["ami_mean"], 3) == 0.700
    assert round(hdbscan_dim2["noise_mean"], 3) == 0.200


def test_seed_dimension_trends_use_result_rows_for_error_bands() -> None:
    selected = pd.DataFrame(
        [
            _row("d1", "e1", "kmeans", 4, ami=0.80, stability=0.90),
        ]
    )
    results = pd.DataFrame(
        [
            _result_row("d1", "e1", "kmeans", 4, seed=0, ami=0.70, clusters=4),
            _result_row("d1", "e1", "kmeans", 4, seed=1, ami=0.80, clusters=8),
            _result_row("d1", "e1", "kmeans", 4, seed=2, ami=0.90, clusters=12),
            _result_row("d1", "e1", "kmeans", 5, seed=0, ami=0.99, clusters=16),
        ]
    )

    trends = summarise_seed_dimension_trends(selected, results)

    assert len(trends) == 1
    row = trends.iloc[0]
    assert round(row["ami_mean"], 3) == 0.800
    assert round(row["ami_sd"], 3) == 0.100
    assert round(row["clusters_mean"], 3) == 8.000
    assert pd.isna(row["stability_sd"])


def test_seed_raw_baselines_match_raw_result_rows() -> None:
    selected_raw = pd.DataFrame(
        [
            {**_row("d1", "e1", "kmeans", 0, ami=0.75, stability=0.85), "projector": "raw", "dimension": pd.NA},
        ]
    )
    results = pd.DataFrame(
        [
            {**_result_row("d1", "e1", "kmeans", 0, seed=0, ami=0.60, clusters=10), "projector": "raw", "dimension": pd.NA},
            {**_result_row("d1", "e1", "kmeans", 0, seed=1, ami=0.80, clusters=20), "projector": "raw", "dimension": pd.NA},
            _result_row("d1", "e1", "kmeans", 4, seed=0, ami=0.99, clusters=99),
        ]
    )

    baselines = summarise_seed_raw_baselines(selected_raw, results)

    assert len(baselines) == 1
    row = baselines.iloc[0]
    assert row["clusterer"] == "kmeans"
    assert round(row["ami_mean"], 3) == 0.700
    assert round(row["clusters_mean"], 3) == 15.000


def _row(
    dataset: str,
    encoder: str,
    clusterer: str,
    dimension: int,
    *,
    ami: float,
    stability: float,
    noise: float = 0.0,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "encoder": encoder,
        "clusterer": clusterer,
        "projector": "pca",
        "dimension": dimension,
        "target_name": "test",
        "source_metric": "roc_auc",
        "condition": "all",
        "cluster_params_json": "{}",
        "mean_label_ami": ami,
        "mean_label_ari": ami - 0.20,
        "mean_label_v_measure": ami - 0.05,
        "mean_pairwise_ari": stability,
        "mean_silhouette": 0.20,
        "mean_sd_index": 0.80,
        "mean_noise_fraction": noise,
        "mean_runtime_seconds": 1.0,
        "mean_n_clusters": 4.0,
    }


def _result_row(
    dataset: str,
    encoder: str,
    clusterer: str,
    dimension: int,
    *,
    seed: int,
    ami: float,
    clusters: float,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "encoder": encoder,
        "clusterer": clusterer,
        "projector": "pca",
        "dimension": dimension,
        "target_name": "test",
        "source_metric": "roc_auc",
        "condition": "all",
        "cluster_params_json": "{}",
        "cluster_seed": seed,
        "label_ami": ami,
        "label_ari": ami - 0.20,
        "label_v_measure": ami - 0.05,
        "silhouette": 0.20,
        "sd_index": 0.80,
        "noise_fraction": 0.0,
        "n_clusters": clusters,
    }
