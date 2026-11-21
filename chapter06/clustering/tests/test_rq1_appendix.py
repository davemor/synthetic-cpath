from __future__ import annotations

from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from cluster_exp.cli import main
from cluster_exp.rq1_appendix import generate_rq1_appendix_dimension_tables, summarise_rq1_dimensions


def test_rq1_appendix_summarises_best_configs_over_dimensions(tmp_path: Path) -> None:
    rows = []
    for projector in ["pca", "umap"]:
        for dimension in [1, 2, 3]:
            rows.append(_row("d1", "e1", "kmeans", projector, dimension, ami=0.10 * dimension))
            rows.append(_row("d1", "e1", "kmeans", projector, dimension, ami=0.20 * dimension))
            rows.append(
                _row(
                    "d1",
                    "e1",
                    "hdbscan",
                    projector,
                    dimension,
                    ami=0.30 * dimension,
                    noise=0.05 * dimension,
                )
            )
    rows.extend(
        [
            _row("d1", "e1", "kmeans", "raw", None, ami=0.90),
            _row("d1", "e1", "hdbscan", "raw", None, ami=0.80, noise=0.25),
        ]
    )
    stability = pd.DataFrame(rows)

    summary = summarise_rq1_dimensions(stability, min_dimension=1, max_dimension=3)

    assert len(summary) == 6
    kmeans_pca = summary[
        (summary["clusterer"] == "kmeans")
        & (summary["projector"] == "pca")
    ].iloc[0]
    assert kmeans_pca["dims"] == "1-3"
    assert kmeans_pca["dimension_count"] == 3
    assert round(kmeans_pca["ami_mean"], 3) == 0.400
    assert round(kmeans_pca["ami_sd"], 3) == 0.200
    assert pd.isna(summary[(summary["projector"] == "raw")].iloc[0]["ami_sd"])

    stability.to_csv(tmp_path / "clustering_stability.csv", index=False)
    outputs = generate_rq1_appendix_dimension_tables(tmp_path, min_dimension=1, max_dimension=3)

    external = pd.read_csv(outputs.external_csv_path)
    internal = pd.read_csv(outputs.internal_csv_path)
    assert len(external) == 6
    assert len(internal) == 6
    assert "clusters_mean" not in internal.columns
    assert set(internal.loc[internal["clusterer"] != "hdbscan", "noise_mean"]) == {"-"}
    latex = outputs.external_latex_path.read_text()
    internal_latex = outputs.internal_latex_path.read_text()
    assert "\\begin{longtable}" not in latex
    assert "\\footnotesize" in latex
    assert "\\refstepcounter{table}" in latex
    assert "\\Needspace{16\\baselineskip}" in latex
    assert "\\textbf{d1 / e1}" in latex
    assert "\\begin{tabular}[t]" in latex
    assert "\\multirow" in latex
    assert "\\multirow{2}{*}{Raw}" in latex
    assert "\\cmidrule(lr){1-6}" in latex
    assert "\\textbf{" in latex
    assert "\\midrule" in latex
    assert "makecell" not in latex
    assert "1--3" in latex
    assert "Clusters" not in internal_latex
    assert "$\\log_{10}$(SD-index)" in internal_latex


def test_make_rq1_appendix_dimension_tables_cli(tmp_path: Path) -> None:
    rows = []
    for projector in ["pca", "umap"]:
        for dimension in [1, 2, 3]:
            rows.append(_row("d1", "e1", "kmeans", projector, dimension, ami=0.10 * dimension))
    rows.append(_row("d1", "e1", "kmeans", "raw", None, ami=0.90))
    pd.DataFrame(rows).to_csv(tmp_path / "clustering_stability.csv", index=False)

    result = CliRunner().invoke(
        main,
        [
            "make-rq1-appendix-dimension-tables",
            "--run-dir",
            str(tmp_path),
            "--min-dimension",
            "1",
            "--max-dimension",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "clustering-results-appendix-rq1-external-dimensions-1-3-table.csv").exists()
    assert (tmp_path / "clustering-results-appendix-rq1-external-dimensions-1-3-table.tex").exists()
    assert (tmp_path / "clustering-results-appendix-rq1-internal-dimensions-1-3-table.csv").exists()
    assert (tmp_path / "clustering-results-appendix-rq1-internal-dimensions-1-3-table.tex").exists()


def _row(
    dataset: str,
    encoder: str,
    clusterer: str,
    projector: str,
    dimension: int | None,
    *,
    ami: float,
    noise: float = 0.0,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "encoder": encoder,
        "clusterer": clusterer,
        "projector": projector,
        "dimension": dimension,
        "target_name": "test",
        "source_metric": "roc_auc",
        "condition": "all",
        "cluster_params_json": "{}",
        "mean_label_ami": ami,
        "mean_label_ari": ami - 0.10,
        "mean_label_v_measure": ami - 0.05,
        "mean_pairwise_ari": ami + 0.10,
        "mean_silhouette": ami + 0.20,
        "mean_sd_index": 1.0 / max(ami, 0.01),
        "mean_noise_fraction": noise,
        "mean_runtime_seconds": 1.0,
        "mean_n_clusters": 4.0,
    }
