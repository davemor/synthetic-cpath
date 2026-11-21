from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig


def aggregate(cfg: ExperimentConfig) -> dict[str, Path]:
    task_dir = cfg.paths.run_dir / "tasks"
    rows: list[dict[str, Any]] = []
    classification_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in sorted(task_dir.glob("*.json")):
        with path.open() as handle:
            task = json.load(handle)
        base = {
            key: task.get(key)
            for key in ["task_id", "dataset", "encoder", "projector", "dimension", "target_name", "seed"]
        }
        rows.append(_task_row(base, task))
        _collect_classifier_artifacts(classification_reports, base, "linear_probe", task["reduction"]["linear_probe"])
        _collect_classifier_artifacts(classification_reports, base, "knn_probe", task["reduction"]["knn_probe"])
        for gen in task.get("generators", []):
            gen_base = base | {"generator": gen.get("generator"), "conditioning": gen.get("conditioning")}
            if gen.get("status") != "ok":
                failures.append(gen_base | {"error_type": gen.get("error_type"), "error": gen.get("error")})
            rows.append(_generator_row(gen_base, gen))
            utility = gen.get("utility", {})
            if isinstance(utility, dict):
                for name in ["train_synthetic_test_real", "real_plus_synthetic_test_real"]:
                    if name in utility:
                        _collect_classifier_artifacts(classification_reports, gen_base, name, utility[name])
            lowdim_utility = gen.get("lowdim_utility", {})
            if isinstance(lowdim_utility, dict):
                for name in ["train_synthetic_test_real", "real_plus_synthetic_test_real"]:
                    if name in lowdim_utility:
                        _collect_classifier_artifacts(
                            classification_reports,
                            gen_base,
                            f"lowdim_{name}",
                            lowdim_utility[name],
                        )

    df = pd.DataFrame(rows)
    outputs: dict[str, Path] = {}
    outputs["results_csv"] = cfg.paths.run_dir / "results.csv"
    df.to_csv(outputs["results_csv"], index=False)
    outputs["results_parquet"] = cfg.paths.run_dir / "results.parquet"
    try:
        df.to_parquet(outputs["results_parquet"], index=False)
    except Exception:
        outputs.pop("results_parquet")
    outputs["classification_reports"] = cfg.paths.run_dir / "classification_reports.jsonl"
    with outputs["classification_reports"].open("w") as handle:
        for row in classification_reports:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    outputs["generator_failures"] = cfg.paths.run_dir / "generator_failures.csv"
    pd.DataFrame(failures).to_csv(outputs["generator_failures"], index=False)
    _write_summaries(cfg, df, outputs)
    outputs["latex_report"] = write_latex_report(cfg, df, pd.DataFrame(failures))
    return outputs


def _task_row(base: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    red = task.get("reduction", {})
    recon = task.get("reconstruction", {})
    return {
        **base,
        "row_type": "representation",
        "linear_accuracy": red.get("linear_probe", {}).get("accuracy"),
        "linear_macro_roc_auc_ovr": red.get("linear_probe", {}).get("macro_roc_auc_ovr"),
        "knn_accuracy": red.get("knn_probe", {}).get("accuracy"),
        "knn_macro_roc_auc_ovr": red.get("knn_probe", {}).get("macro_roc_auc_ovr"),
        "trustworthiness": red.get("trustworthiness"),
        "pairwise_dist_correlation": red.get("pairwise_dist_correlation"),
        "reconstruction_mse": recon.get("mse"),
        "reconstruction_cosine_similarity": recon.get("cosine_similarity"),
    }


def _generator_row(base: dict[str, Any], gen: dict[str, Any]) -> dict[str, Any]:
    fid = gen.get("fidelity", {}) if gen.get("status") == "ok" else {}
    lowdim_fid = gen.get("lowdim_fidelity", {}) if gen.get("status") == "ok" else {}
    utility = gen.get("utility", {}) if gen.get("status") == "ok" else {}
    lowdim_utility = gen.get("lowdim_utility", {}) if gen.get("status") == "ok" else {}
    tstr = utility.get("train_synthetic_test_real", {}) if isinstance(utility, dict) else {}
    aug = utility.get("real_plus_synthetic_test_real", {}) if isinstance(utility, dict) else {}
    lowdim_tstr = (
        lowdim_utility.get("train_synthetic_test_real", {})
        if isinstance(lowdim_utility, dict)
        else {}
    )
    lowdim_aug = (
        lowdim_utility.get("real_plus_synthetic_test_real", {})
        if isinstance(lowdim_utility, dict)
        else {}
    )
    metadata = gen.get("generator_metadata", {}) if gen.get("status") == "ok" else {}
    gmm = _gmm_selection_columns(metadata)
    return {
        **base,
        "row_type": "generator",
        "status": gen.get("status"),
        "synthetic_n_samples": gen.get("synthetic_n_samples"),
        "synthetic_count_source": gen.get("synthetic_count_source"),
        **gmm,
        "lowdim_fid": lowdim_fid.get("fid"),
        "lowdim_kid": lowdim_fid.get("kid"),
        "lowdim_precision": lowdim_fid.get("precision"),
        "lowdim_recall": lowdim_fid.get("recall"),
        "lowdim_real_clusters": lowdim_fid.get("real_clusters"),
        "lowdim_synthetic_clusters": lowdim_fid.get("synthetic_clusters"),
        "lowdim_centroid_distance": lowdim_fid.get("centroid_distance"),
        "lowdim_covariance_frobenius": lowdim_fid.get("covariance_frobenius"),
        "lowdim_tstr_accuracy": lowdim_tstr.get("accuracy"),
        "lowdim_tstr_macro_roc_auc_ovr": lowdim_tstr.get("macro_roc_auc_ovr"),
        "lowdim_aug_accuracy": lowdim_aug.get("accuracy"),
        "lowdim_aug_macro_roc_auc_ovr": lowdim_aug.get("macro_roc_auc_ovr"),
        "fid": fid.get("fid"),
        "kid": fid.get("kid"),
        "uni_fid": fid.get("uni_fid"),
        "uni_kid": fid.get("uni_kid"),
        "precision": fid.get("precision"),
        "recall": fid.get("recall"),
        "real_clusters": fid.get("real_clusters"),
        "synthetic_clusters": fid.get("synthetic_clusters"),
        "tstr_accuracy": tstr.get("accuracy"),
        "tstr_macro_roc_auc_ovr": tstr.get("macro_roc_auc_ovr"),
        "aug_accuracy": aug.get("accuracy"),
        "aug_macro_roc_auc_ovr": aug.get("macro_roc_auc_ovr"),
    }


def _gmm_selection_columns(metadata: dict[str, Any]) -> dict[str, Any]:
    selection = metadata.get("gmm_selection", {}) if isinstance(metadata, dict) else {}
    if not selection:
        return {
            "gmm_selected_k": None,
            "gmm_selected_k_min": None,
            "gmm_selected_k_median": None,
            "gmm_selected_k_max": None,
            "gmm_selection_metric": None,
        }
    by_label = selection.get("selected_k_by_label") or {}
    if by_label:
        values = [int(value) for value in by_label.values()]
        return {
            "gmm_selected_k": None,
            "gmm_selected_k_min": min(values),
            "gmm_selected_k_median": float(np.median(values)),
            "gmm_selected_k_max": max(values),
            "gmm_selection_metric": selection.get("metric"),
        }
    selected = selection.get("selected_k", metadata.get("n_components"))
    return {
        "gmm_selected_k": selected,
        "gmm_selected_k_min": selected,
        "gmm_selected_k_median": selected,
        "gmm_selected_k_max": selected,
        "gmm_selection_metric": selection.get("metric"),
    }


def _collect_classifier_artifacts(
    output: list[dict[str, Any]], base: dict[str, Any], classifier_name: str, metrics: dict[str, Any]
) -> None:
    output.append(
        {
            **base,
            "classifier_evaluation": classifier_name,
            "accuracy": metrics.get("accuracy"),
            "macro_roc_auc_ovr": metrics.get("macro_roc_auc_ovr"),
            "labels": metrics.get("labels"),
            "label_names": metrics.get("label_names"),
            "confusion_matrix": metrics.get("confusion_matrix"),
            "classification_report": metrics.get("classification_report"),
        }
    )


def _write_summaries(cfg: ExperimentConfig, df: pd.DataFrame, outputs: dict[str, Path]) -> None:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    for name, groups in {
        "summary_by_seed": ["row_type", "seed"],
        "summary_by_dataset_encoder": ["row_type", "dataset", "encoder"],
        "summary_by_target": ["row_type", "projector", "target_name"],
        "summary_by_generator": ["row_type", "generator", "conditioning"],
    }.items():
        path = cfg.paths.run_dir / f"{name}.csv"
        present = [g for g in groups if g in df.columns]
        value_cols = [col for col in numeric if col not in present]
        if present and value_cols:
            df.groupby(present, dropna=False)[value_cols].mean().reset_index().to_csv(path, index=False)
        else:
            pd.DataFrame().to_csv(path, index=False)
        outputs[name] = path


def write_latex_report(cfg: ExperimentConfig, df: pd.DataFrame, failures: pd.DataFrame) -> Path:
    report_dir = cfg.paths.run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "lowgen_v2_dgx.tex"
    gen = df[df["row_type"] == "generator"].copy()
    rep = df[df["row_type"] == "representation"].copy()
    with path.open("w") as handle:
        handle.write("\\section*{Lowgen v2 DGX experiment}\n\n")
        handle.write(
            "Dimensions were copied from the dimensionality study and were not recomputed. "
            "The first run used unconditioned and true-label conditioning with resample, "
            "diagonal Gaussian, and diagonal GMM generators. Classification performance is "
            "reported as accuracy and macro one-vs-rest ROC-AUC. Standard trustworthiness is "
            "used for geometry preservation.\\n\\n"
        )
        handle.write("\\section*{Representation summary}\n\n")
        if not rep.empty:
            cols = [
                "dataset",
                "encoder",
                "projector",
                "target_name",
                "linear_accuracy",
                "linear_macro_roc_auc_ovr",
                "trustworthiness",
                "reconstruction_cosine_similarity",
            ]
            handle.write(rep[cols].groupby(cols[:4], dropna=False).mean().reset_index().to_latex(index=False))
        handle.write("\\section*{Generator summary}\n\n")
        if not gen.empty:
            cols = [
                "projector",
                "generator",
                "conditioning",
                "fid",
                "kid",
                "precision",
                "recall",
                "lowdim_fid",
                "lowdim_precision",
                "lowdim_recall",
                "tstr_accuracy",
                "lowdim_tstr_accuracy",
                "aug_accuracy",
                "lowdim_aug_accuracy",
            ]
            handle.write(gen[cols].groupby(cols[:3], dropna=False).mean().reset_index().to_latex(index=False))
        if not failures.empty:
            handle.write("\\section*{Generator failures}\n\n")
            handle.write(failures.head(100).to_latex(index=False))
    return path
