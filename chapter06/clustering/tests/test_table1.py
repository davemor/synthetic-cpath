from __future__ import annotations

from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from cluster_exp.cli import main
from cluster_exp.table1 import generate_table_one, select_best_configs


def test_table_one_selects_best_per_dataset_encoder_method_projection(tmp_path: Path) -> None:
    rows = [
        _row("d1", "e1", "kmeans", "pca", ami=0.20, stability=0.90, clusters=4),
        _row("d1", "e1", "kmeans", "pca", ami=0.80, stability=0.70, clusters=8),
        _row("d2", "e1", "kmeans", "pca", ami=0.60, stability=0.60, clusters=6),
        _row("d1", "e1", "hdbscan", "umap", ami=0.50, stability=0.40, noise=0.20),
        _row("d2", "e1", "hdbscan", "umap", ami=0.70, stability=0.50, noise=0.10),
    ]
    stability = pd.DataFrame(rows)

    selected = select_best_configs(stability)
    assert len(selected) == 4
    best_kmeans = selected[
        (selected["dataset"] == "d1")
        & (selected["encoder"] == "e1")
        & (selected["clusterer"] == "kmeans")
        & (selected["projector"] == "pca")
    ].iloc[0]
    assert best_kmeans["mean_label_ami"] == 0.80

    stability_path = tmp_path / "clustering_stability.csv"
    stability.to_csv(stability_path, index=False)
    outputs = generate_table_one(tmp_path)
    table = pd.read_csv(outputs.csv_path)
    external = pd.read_csv(outputs.external_csv_path)
    structure = pd.read_csv(outputs.structure_csv_path)

    kmeans_pca = table[table["method_projection"] == "K-Means + PCA"].iloc[0]
    assert kmeans_pca["n"] == 2
    assert round(kmeans_pca["ami_mean"], 3) == 0.700
    assert "ari_mean" in external.columns
    assert "n" not in external.columns
    assert "stability_mean" in structure.columns
    assert "n" not in structure.columns
    latex = outputs.latex_path.read_text()
    external_latex = outputs.external_latex_path.read_text()
    structure_latex = outputs.structure_latex_path.read_text()
    assert latex.startswith("\\begin{table}")
    assert latex.count("\\begin{table}") == 2
    assert external_latex.count("\\begin{table}") == 1
    assert structure_latex.count("\\begin{table}") == 1
    assert "\\multirow" not in latex
    assert "\\multicolumn" in latex
    assert "Projection & AMI" in latex
    assert "Method & Projection" not in latex
    assert "ARI $\\uparrow$" in latex
    assert "Method & Projection & $n$" not in latex


def test_make_table1_cli_writes_csv_and_latex(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            _row("d1", "e1", "kmeans", "raw", ami=0.4, stability=0.6),
            _row("d1", "e1", "leiden", "pca", ami=0.5, stability=0.7),
        ]
    ).to_csv(tmp_path / "clustering_stability.csv", index=False)

    result = CliRunner().invoke(main, ["make-table1", "--run-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "clustering_results_method_projection_summary.csv").exists()
    assert (tmp_path / "clustering_results_method_projection_summary_external_information.csv").exists()
    assert (tmp_path / "clustering_results_method_projection_summary_cluster_structure_stability.csv").exists()
    assert (tmp_path / "clustering_results_method_projection_summary.tex").exists()
    assert (tmp_path / "clustering_results_method_projection_summary_external_information.tex").exists()
    assert (tmp_path / "clustering_results_method_projection_summary_cluster_structure_stability.tex").exists()


def _row(
    dataset: str,
    encoder: str,
    clusterer: str,
    projector: str,
    *,
    ami: float,
    stability: float,
    clusters: float = 4.0,
    noise: float = 0.0,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "encoder": encoder,
        "clusterer": clusterer,
        "projector": projector,
        "dimension": 2,
        "target_name": "test",
        "source_metric": "roc_auc",
        "condition": "all",
        "cluster_params_json": "{}",
        "mean_label_ami": ami,
        "mean_label_v_measure": ami - 0.05,
        "mean_pairwise_ari": stability,
        "mean_silhouette": 0.2,
        "mean_sd_index": 0.8,
        "mean_noise_fraction": noise,
        "mean_runtime_seconds": 1.0,
        "mean_n_clusters": clusters,
    }
