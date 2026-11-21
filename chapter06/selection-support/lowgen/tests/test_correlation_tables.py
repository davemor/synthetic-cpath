from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.summarise_correlation_audit import (
    build_summary_tables,
    preprocess,
    reduction_effect,
    seed_averaged_targets,
    write_tables,
)


def test_seed_averaged_targets_and_n_targets() -> None:
    valid, _ = preprocess(_fixture())
    target_level = seed_averaged_targets(valid)
    tables = build_summary_tables(valid, target_level)

    by_projector = tables["correlation_by_projector"]
    raw = by_projector[by_projector["projector"] == "raw_highdim"].iloc[0]
    pca = by_projector[by_projector["projector"] == "pca"].iloc[0]

    assert raw["n_targets"] == 1
    assert raw["mean_abs_corr_mean"] == 0.2
    assert pca["n_targets"] == 2
    assert np.isclose(pca["mean_abs_corr_mean"], 0.015)


def test_missing_rows_are_excluded() -> None:
    valid, excluded = preprocess(_fixture())

    assert len(valid) == 8
    assert len(excluded) == 1
    assert excluded.iloc[0]["task_id"] == "missing"


def test_reduction_effect_matches_raw_by_dataset_encoder_seed() -> None:
    valid, _ = preprocess(_fixture())
    table = reduction_effect(valid)

    pca = table[table["projector"] == "pca"].iloc[0]
    umap = table[table["projector"] == "umap"].iloc[0]

    assert pca["n_matched"] == 2
    assert np.isclose(pca["raw_mean"], 0.2)
    assert np.isclose(pca["reduced_mean"], 0.015)
    assert np.isclose(pca["mean_change"], -0.185)
    assert umap["n_matched"] == 2
    assert np.isclose(umap["reduced_mean"], 0.3)


def test_write_tables_creates_csv_and_latex(tmp_path: Path) -> None:
    input_path = tmp_path / "correlation_summary.csv"
    _fixture().to_csv(input_path, index=False)

    outputs = write_tables(input_path, tmp_path / "tables")

    assert outputs["correlation_by_projector_csv"].exists()
    assert outputs["correlation_by_projector_tex"].exists()
    assert outputs["combined_latex"].exists()
    assert outputs["excluded_rows"].exists()
    assert "tab:correlation-by-projector" in outputs["combined_latex"].read_text()


def test_headers_are_stable(tmp_path: Path) -> None:
    input_path = tmp_path / "correlation_summary.csv"
    _fixture().to_csv(input_path, index=False)

    outputs = write_tables(input_path, tmp_path / "tables")
    table = pd.read_csv(outputs["correlation_reduction_effect_csv"])

    assert list(table.columns) == [
        "projector",
        "n_matched",
        "raw_mean",
        "reduced_mean",
        "mean_change",
        "std_change",
        "mean_ratio",
        "std_ratio",
    ]


def _fixture() -> pd.DataFrame:
    rows = []
    for seed in [0, 1]:
        rows.extend(
            [
                _row(
                    "raw",
                    "tiny",
                    "uni",
                    "raw_highdim",
                    "",
                    "highdim_baseline",
                    seed,
                    0.2,
                ),
                _row("pca1", "tiny", "uni", "pca", 2, "intrinsic_mle", seed, 0.01),
                _row("pca2", "tiny", "uni", "pca", 3, "roc_auc_95", seed, 0.02),
                _row("umap1", "tiny", "uni", "umap", 2, "intrinsic_mle", seed, 0.3),
            ]
        )
    rows.append(_row("missing", "tiny", "uni", "umap", 3, "missing", 0, ""))
    return pd.DataFrame(rows)


def _row(
    task_id: str,
    dataset: str,
    encoder: str,
    projector: str,
    dimension,
    target_name: str,
    seed: int,
    mean_abs_corr,
) -> dict:
    return {
        "task_id": task_id,
        "dataset": dataset,
        "encoder": encoder,
        "projector": projector,
        "dimension": dimension,
        "target_name": target_name,
        "seed": seed,
        "projection_cache_key": "cache",
        "n_samples": 10,
        "n_dimensions": 3,
        "n_zero_variance_dims": 0,
        "mean_abs_corr": mean_abs_corr,
    }
