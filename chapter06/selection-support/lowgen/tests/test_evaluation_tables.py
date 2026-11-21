from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.make_lowgen_evaluation_tables import write_evaluation_tables


def test_evaluation_tables_mark_missing_privacy_and_summarise_present_metrics(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "row_type": "generator",
                "dataset": "kather100k",
                "encoder": "uni",
                "projector": "pca",
                "dimension": 8,
                "target_name": "acc95",
                "generator": "resample",
                "conditioning": "true_label",
                "lowdim_fid": 1.0,
                "lowdim_precision": 0.8,
                "lowdim_recall": 0.7,
                "lowdim_tstr_accuracy": 0.6,
                "lowdim_tstr_macro_roc_auc_ovr": 0.65,
                "fid": 2.0,
                "precision": 0.75,
                "recall": 0.66,
                "tstr_accuracy": 0.55,
                "tstr_macro_roc_auc_ovr": 0.6,
            },
            {
                "row_type": "representation",
                "projector": "pca",
            },
        ]
    ).to_csv(results, index=False)

    outputs = write_evaluation_tables(results, tmp_path / "tables")
    inventory = pd.read_csv(outputs["metric_inventory_csv"])
    overall = pd.read_csv(outputs["evaluation_results_overall_csv"])
    low_selected = pd.read_csv(outputs["selected_low_dimensional_csv"])
    high_selected = pd.read_csv(outputs["selected_high_dimensional_csv"])
    selected_tex = outputs["selected_metric_tables_tex"].read_text()
    route_summary = pd.read_csv(outputs["thesis_route_summary_csv"])
    generator_comparison = pd.read_csv(outputs["thesis_generator_comparison_csv"])
    appendix = pd.read_csv(outputs["appendix_selected_metrics_csv"])

    privacy = inventory[
        (inventory["category"] == "privacy")
        & (inventory["space"] == "low_dimensional")
    ]
    assert set(privacy["status"]) == {"not_computed"}
    assert "lowdim_fid" in set(overall["column"])
    assert "fid" in set(overall["column"])
    assert outputs["evaluation_tables_tex"].exists()
    assert list(low_selected.columns) == ["Category", "Metric", "Mean", "SD"]
    assert list(high_selected.columns) == ["Category", "Metric", "Mean", "SD"]
    assert set(low_selected["Category"]) == {"Fidelity", "Diversity", "Utility"}
    assert set(high_selected["Category"]) == {"Fidelity", "Diversity", "Utility"}
    assert "Privacy" not in selected_tex
    assert "\\multirow" in selected_tex
    assert "status" not in selected_tex
    assert outputs["thesis_main_tables_tex"].exists()
    assert outputs["appendix_selected_metrics_tex"].exists()
    assert set(route_summary["Route"]) == {
        "Low-dimensional synthesis",
        "Direct full-dimensional synthesis",
        "Inverse-projected synthesis",
    }
    assert "Fréchet distance mean" in route_summary.columns
    assert "Diagonal Gaussian" not in set(generator_comparison["Generator"])
    assert "Low-D Fréchet distance" in appendix.columns
    assert "High-D LR macro ROC-AUC" in appendix.columns


def test_evaluation_tables_auto_prefers_more_complete_task_jsons(tmp_path: Path) -> None:
    results = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "row_type": "generator",
                "dataset": "toy",
                "encoder": "uni",
                "projector": "pca",
                "dimension": 8,
                "generator": "resample",
                "conditioning": "true_label",
                "lowdim_fid": 100.0,
            }
        ]
    ).to_csv(results, index=False)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for index, value in enumerate([1.0, 3.0]):
        (tasks / f"task-{index}.json").write_text(
            """{
  "task_id": "task-%d",
      "dataset": "toy",
      "encoder": "uni",
      "projector": "pca",
      "dimension": 8,
      "target_name": "acc95",
  "seed": %d,
  "generators": [
    {
      "status": "ok",
      "generator": "resample",
      "conditioning": "true_label",
      "lowdim_fidelity": {"fid": %.1f},
      "fidelity": {"fid": %.1f},
      "utility": {
        "train_synthetic_test_real": {
          "accuracy": 0.5,
          "macro_roc_auc_ovr": 0.6
        }
      },
      "lowdim_utility": {
        "train_synthetic_test_real": {
          "accuracy": 0.7,
          "macro_roc_auc_ovr": 0.8
        }
      }
    }
  ]
}"""
            % (index, index, value, value + 10.0)
        )

    outputs = write_evaluation_tables(results, tmp_path / "tables")
    overall = pd.read_csv(outputs["evaluation_results_overall_csv"])
    metadata = pd.read_json(outputs["metadata"], typ="series")
    low_by_rep = pd.read_csv(outputs["selected_low_dimensional_by_representation_csv"])
    route_summary = pd.read_csv(outputs["thesis_route_summary_csv"])
    appendix = pd.read_csv(outputs["appendix_selected_metrics_csv"])

    lowdim_fid = overall[overall["column"] == "lowdim_fid"].iloc[0]
    assert lowdim_fid["n"] == 2
    assert lowdim_fid["mean"] == 2.0
    assert metadata["selected_source"] == "task_json"
    assert low_by_rep["Fréchet distance"].iloc[0] == 2.0
    assert "Precision" in low_by_rep.columns
    assert outputs["selected_by_representation_tables_tex"].exists()
    low_route = route_summary[route_summary["Route"] == "Low-dimensional synthesis"].iloc[0]
    assert low_route["Fréchet distance mean"] == 2.0
    assert appendix["Low-D Fréchet distance"].iloc[0] == 2.0


def test_exemplar_table_selects_best_icaird_uni2_reduced_representation(tmp_path: Path) -> None:
    rows = []
    for projector, dimension, auc in [("pca", 8, 0.7), ("umap", 10, 0.9)]:
        for generator in ["resample", "diagonal_gaussian", "diagonal_gmm"]:
            for conditioning in ["true_label", "unconditioned"]:
                labelled = conditioning == "true_label"
                rows.append(
                    {
                        "row_type": "generator",
                        "dataset": "icaird100k",
                        "encoder": "uni2",
                        "projector": projector,
                        "dimension": dimension,
                        "target_name": "roc_auc_99",
                        "generator": generator,
                        "conditioning": conditioning,
                        "lowdim_fid": 3.0,
                        "lowdim_precision": 0.4,
                        "lowdim_recall": 0.8,
                        "lowdim_tstr_accuracy": 0.6 if labelled else None,
                        "lowdim_tstr_macro_roc_auc_ovr": auc if labelled else None,
                        "fid": 30.0,
                        "precision": 0.5,
                        "recall": 0.2,
                        "tstr_accuracy": 0.55 if labelled else None,
                        "tstr_macro_roc_auc_ovr": 0.75 if labelled else None,
                    }
                )
    for generator in ["resample", "diagonal_gaussian", "diagonal_gmm"]:
        for conditioning in ["true_label", "unconditioned"]:
            labelled = conditioning == "true_label"
            rows.append(
                {
                    "row_type": "generator",
                    "dataset": "icaird100k",
                    "encoder": "uni2",
                    "projector": "raw_highdim",
                    "dimension": "",
                    "target_name": "highdim_baseline",
                    "generator": generator,
                    "conditioning": conditioning,
                    "lowdim_fid": 10.0,
                    "lowdim_precision": 0.3,
                    "lowdim_recall": 0.6,
                    "lowdim_tstr_accuracy": 0.5 if labelled else None,
                    "lowdim_tstr_macro_roc_auc_ovr": 0.7 if labelled else None,
                    "fid": 100.0,
                    "precision": 0.25,
                    "recall": 0.65,
                    "tstr_accuracy": 0.58 if labelled else None,
                    "tstr_macro_roc_auc_ovr": 0.78 if labelled else None,
                }
            )
    results = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(results, index=False)

    outputs = write_evaluation_tables(results, tmp_path / "tables", source="results")
    selection = pd.read_csv(outputs["exemplar_selection_csv"])
    exemplar = pd.read_csv(outputs["exemplar_selected_metrics_csv"])
    latex = outputs["exemplar_selected_metrics_tex"].read_text()

    assert selection["Selected projector"].iloc[0] == "UMAP"
    assert selection["Selected dimension"].iloc[0] == 10
    assert selection["Selection value"].iloc[0] == 0.9
    assert exemplar["Route"].drop_duplicates().tolist() == [
        "Direct full-dimensional baseline",
        "Low-dimensional synthesis",
        "Inverse-projected synthesis",
    ]
    assert set(exemplar["Route"]) == {
        "Low-dimensional synthesis",
        "Inverse-projected synthesis",
        "Direct full-dimensional baseline",
    }
    assert set(exemplar["Generator"]) == {
        "Resample",
        "Diagonal Gaussian",
        "Diagonal GMM",
    }
    assert set(exemplar["Conditioning"]) == {"True label", "Unconditioned"}
    assert len(exemplar) == 18
    assert "highest mean true-label" in latex
    assert "\\multirow{6}{*}{Direct full-dimensional baseline}" in latex
    assert "\\multirow{2}{*}{Resample}" in latex
    assert "Dataset & Encoder" not in latex
    assert "\\resizebox{\\textwidth}{!}" in latex
    assert "\\small" in latex
