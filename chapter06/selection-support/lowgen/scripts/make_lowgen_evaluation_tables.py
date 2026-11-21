#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lowgen.dgx.summary import _generator_row


DEFAULT_RESULTS = Path(
    "/raid/experiments/lowgen_v2/output/lowgen-v2-trainsized-bic-k/results.csv"
)
DEFAULT_CLUSTER_RESULTS = Path(
    "/raid/experiments/lowgen_v2/output/lowgen-v2-cluster-kmeans24-followup/"
    "fixed_kmeans24_cluster_conditioning/cluster_guided_full_embedding_results.csv"
)


@dataclass(frozen=True)
class MetricSpec:
    space: str
    category: str
    metric: str
    column: str
    what_it_measures: str
    why_useful: str
    role: str = "primary"


@dataclass(frozen=True)
class PaperMetricSpec:
    space: str
    category: str
    metric: str
    column: str


PAPER_METRICS = [
    PaperMetricSpec("low_dimensional", "Fidelity", "Fréchet distance", "lowdim_fid"),
    PaperMetricSpec("low_dimensional", "Fidelity", "Precision", "lowdim_precision"),
    PaperMetricSpec("low_dimensional", "Diversity", "Recall", "lowdim_recall"),
    PaperMetricSpec(
        "low_dimensional",
        "Utility",
        "Logistic regression accuracy",
        "lowdim_tstr_accuracy",
    ),
    PaperMetricSpec(
        "low_dimensional",
        "Utility",
        "Logistic regression macro ROC-AUC",
        "lowdim_tstr_macro_roc_auc_ovr",
    ),
    PaperMetricSpec("high_dimensional", "Fidelity", "Fréchet distance", "fid"),
    PaperMetricSpec("high_dimensional", "Fidelity", "Precision", "precision"),
    PaperMetricSpec("high_dimensional", "Diversity", "Recall", "recall"),
    PaperMetricSpec(
        "high_dimensional",
        "Utility",
        "Logistic regression accuracy",
        "tstr_accuracy",
    ),
    PaperMetricSpec(
        "high_dimensional",
        "Utility",
        "Logistic regression macro ROC-AUC",
        "tstr_macro_roc_auc_ovr",
    ),
]


LOWDIM_METRIC_COLUMNS = {
    "Fréchet distance": "lowdim_fid",
    "Precision": "lowdim_precision",
    "Recall": "lowdim_recall",
    "LR accuracy": "lowdim_tstr_accuracy",
    "LR macro ROC-AUC": "lowdim_tstr_macro_roc_auc_ovr",
}

HIGHDIM_METRIC_COLUMNS = {
    "Fréchet distance": "fid",
    "Precision": "precision",
    "Recall": "recall",
    "LR accuracy": "tstr_accuracy",
    "LR macro ROC-AUC": "tstr_macro_roc_auc_ovr",
}


EXEMPLAR_DATASET = "icaird100k"
EXEMPLAR_ENCODER = "uni2"


METRICS = [
    MetricSpec(
        "low_dimensional",
        "fidelity",
        "Fréchet distance",
        "lowdim_fid",
        "Mean/covariance distance between real and synthetic low-dimensional vectors.",
        "Tests whether the generator matches the projected distribution before inverse projection.",
    ),
    MetricSpec(
        "low_dimensional",
        "fidelity",
        "Polynomial MMD",
        "lowdim_kid",
        "Kernel discrepancy between real and synthetic low-dimensional vectors.",
        "Provides a non-Gaussian distributional check alongside Fréchet distance.",
        "supplementary",
    ),
    MetricSpec(
        "low_dimensional",
        "fidelity",
        "Precision",
        "lowdim_precision",
        "Fraction of synthetic samples close to the real low-dimensional manifold.",
        "Checks whether generated low-dimensional points are locally plausible.",
    ),
    MetricSpec(
        "low_dimensional",
        "fidelity",
        "Centroid distance",
        "lowdim_centroid_distance",
        "Distance between real and synthetic low-dimensional means.",
        "Detects global location shifts in projected space.",
        "diagnostic",
    ),
    MetricSpec(
        "low_dimensional",
        "fidelity",
        "Covariance distance",
        "lowdim_covariance_frobenius",
        "Frobenius distance between real and synthetic covariance matrices.",
        "Checks whether projected-space spread and covariance structure are preserved.",
        "diagnostic",
    ),
    MetricSpec(
        "low_dimensional",
        "diversity",
        "Recall",
        "lowdim_recall",
        "Fraction of the real low-dimensional manifold covered by synthetic samples.",
        "Tests whether the generator covers projected-space modes rather than collapsing.",
    ),
    MetricSpec(
        "low_dimensional",
        "privacy",
        "NN minimum distance",
        "lowdim_nn_min_distance",
        "Minimum nearest-neighbour distance from synthetic samples to real training samples.",
        "Flags exact or near-exact memorisation in low-dimensional space.",
    ),
    MetricSpec(
        "low_dimensional",
        "privacy",
        "NN 1st percentile distance",
        "lowdim_nn_p01_distance",
        "First percentile nearest-neighbour distance to real training samples.",
        "Identifies whether a small tail of samples is near-memorised.",
    ),
    MetricSpec(
        "low_dimensional",
        "privacy",
        "NN 5th percentile distance",
        "lowdim_nn_p05_distance",
        "Fifth percentile nearest-neighbour distance to real training samples.",
        "Summarises near-neighbour behaviour beyond the single closest sample.",
    ),
    MetricSpec(
        "low_dimensional",
        "privacy",
        "NN median distance",
        "lowdim_nn_median_distance",
        "Median nearest-neighbour distance to real training samples.",
        "Provides a typical-case privacy distance baseline.",
    ),
    MetricSpec(
        "low_dimensional",
        "utility",
        "TSTR accuracy",
        "lowdim_tstr_accuracy",
        "Accuracy after training on synthetic low-dimensional data and testing on real data.",
        "Tests whether generated low-dimensional samples carry transferable class signal.",
    ),
    MetricSpec(
        "low_dimensional",
        "utility",
        "TSTR macro ROC-AUC OVR",
        "lowdim_tstr_macro_roc_auc_ovr",
        "Macro one-vs-rest ROC-AUC for low-dimensional TSTR classification.",
        "Gives a threshold-independent class-balanced utility measure.",
    ),
    MetricSpec(
        "low_dimensional",
        "utility",
        "Augmentation accuracy",
        "lowdim_aug_accuracy",
        "Accuracy after training on real plus synthetic low-dimensional data.",
        "Tests whether synthetic projected samples improve the real-data classifier.",
    ),
    MetricSpec(
        "low_dimensional",
        "utility",
        "Augmentation macro ROC-AUC OVR",
        "lowdim_aug_macro_roc_auc_ovr",
        "Macro one-vs-rest ROC-AUC for low-dimensional augmentation.",
        "Checks augmentation utility across classes.",
    ),
    MetricSpec(
        "high_dimensional",
        "fidelity",
        "Fréchet distance",
        "fid",
        "Mean/covariance distance between real and inverse-projected synthetic embeddings.",
        "Tests whether reconstructed synthetic embeddings match the high-dimensional distribution.",
    ),
    MetricSpec(
        "high_dimensional",
        "fidelity",
        "Polynomial MMD",
        "kid",
        "Kernel discrepancy between real and synthetic high-dimensional embeddings.",
        "Provides a non-Gaussian distributional check after inverse projection.",
        "supplementary",
    ),
    MetricSpec(
        "high_dimensional",
        "fidelity",
        "Precision",
        "precision",
        "Fraction of synthetic embeddings close to the real high-dimensional manifold.",
        "Checks local plausibility after inverse projection.",
    ),
    MetricSpec(
        "high_dimensional",
        "diversity",
        "Recall",
        "recall",
        "Fraction of the real high-dimensional manifold covered by synthetic embeddings.",
        "Tests whether inverse-projected samples preserve coverage of real modes.",
    ),
    MetricSpec(
        "high_dimensional",
        "privacy",
        "NN minimum distance",
        "nn_min_distance",
        "Minimum nearest-neighbour distance from synthetic embeddings to real training embeddings.",
        "Flags exact or near-exact memorisation after inverse projection.",
    ),
    MetricSpec(
        "high_dimensional",
        "privacy",
        "NN 1st percentile distance",
        "nn_p01_distance",
        "First percentile nearest-neighbour distance to real training embeddings.",
        "Identifies whether a small tail of high-dimensional samples is near-memorised.",
    ),
    MetricSpec(
        "high_dimensional",
        "privacy",
        "NN 5th percentile distance",
        "nn_p05_distance",
        "Fifth percentile nearest-neighbour distance to real training embeddings.",
        "Summarises near-neighbour behaviour beyond the single closest sample.",
    ),
    MetricSpec(
        "high_dimensional",
        "privacy",
        "NN median distance",
        "nn_median_distance",
        "Median nearest-neighbour distance to real training embeddings.",
        "Provides a typical-case privacy distance baseline after inverse projection.",
    ),
    MetricSpec(
        "high_dimensional",
        "utility",
        "TSTR accuracy",
        "tstr_accuracy",
        "Accuracy after training on reconstructed synthetic embeddings and testing on real data.",
        "Tests whether inverse-projected synthetic embeddings carry transferable class signal.",
    ),
    MetricSpec(
        "high_dimensional",
        "utility",
        "TSTR macro ROC-AUC OVR",
        "tstr_macro_roc_auc_ovr",
        "Macro one-vs-rest ROC-AUC for high-dimensional TSTR classification.",
        "Gives a threshold-independent class-balanced utility measure after reconstruction.",
    ),
    MetricSpec(
        "high_dimensional",
        "utility",
        "Augmentation accuracy",
        "aug_accuracy",
        "Accuracy after training on real plus reconstructed synthetic embeddings.",
        "Tests whether inverse-projected synthetic data improves a real-data classifier.",
    ),
    MetricSpec(
        "high_dimensional",
        "utility",
        "Augmentation macro ROC-AUC OVR",
        "aug_macro_roc_auc_ovr",
        "Macro one-vs-rest ROC-AUC for high-dimensional augmentation.",
        "Checks augmentation utility across classes after reconstruction.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--tasks-dir", type=Path, default=None)
    parser.add_argument("--source", choices=["auto", "results", "tasks"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--cluster-results",
        type=Path,
        default=DEFAULT_CLUSTER_RESULTS,
        help=(
            "Optional fixed cluster-conditioning follow-up result CSV. "
            "If present, a compact cluster-conditioning table is added."
        ),
    )
    parser.add_argument("--float-format", default="%.4g")
    args = parser.parse_args()
    outputs = write_evaluation_tables(
        args.results,
        args.output_dir,
        args.float_format,
        tasks_dir=args.tasks_dir,
        source=args.source,
        cluster_results_path=args.cluster_results,
    )
    print(f"Wrote evaluation tables to {outputs['output_dir']}")
    for name, path in outputs.items():
        if name != "output_dir":
            print(f"{name}: {path}")


def write_evaluation_tables(
    results_path: Path,
    output_dir: Path | None = None,
    float_format: str = "%.4g",
    tasks_dir: Path | None = None,
    source: str = "auto",
    cluster_results_path: Path | None = None,
) -> dict[str, Path]:
    output_dir = output_dir or results_path.parent / "evaluation_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    generator_df, input_metadata = load_generator_rows(results_path, tasks_dir, source)
    inventory = metric_inventory(generator_df)
    long = metric_long_table(generator_df, inventory)
    overall = summarise(long, ["space", "category", "metric", "column", "status"])
    by_model = summarise(
        long,
        ["space", "category", "metric", "projector", "generator", "conditioning", "status"],
    )

    outputs = {
        "output_dir": output_dir,
        "metric_inventory_csv": output_dir / "metric_inventory.csv",
        "metric_inventory_tex": output_dir / "metric_inventory.tex",
        "evaluation_results_overall_csv": output_dir / "evaluation_results_overall.csv",
        "evaluation_results_overall_tex": output_dir / "evaluation_results_overall.tex",
        "evaluation_results_by_model_csv": output_dir / "evaluation_results_by_model.csv",
        "evaluation_results_by_model_tex": output_dir / "evaluation_results_by_model.tex",
        "evaluation_tables_tex": output_dir / "evaluation_tables.tex",
        "selected_low_dimensional_csv": output_dir / "selected_low_dimensional_metrics.csv",
        "selected_low_dimensional_tex": output_dir / "selected_low_dimensional_metrics.tex",
        "selected_high_dimensional_csv": output_dir / "selected_high_dimensional_metrics.csv",
        "selected_high_dimensional_tex": output_dir / "selected_high_dimensional_metrics.tex",
        "selected_metric_tables_tex": output_dir / "selected_metric_tables.tex",
        "selected_low_dimensional_by_representation_csv": (
            output_dir / "selected_low_dimensional_by_representation.csv"
        ),
        "selected_low_dimensional_by_representation_tex": (
            output_dir / "selected_low_dimensional_by_representation.tex"
        ),
        "selected_high_dimensional_by_representation_csv": (
            output_dir / "selected_high_dimensional_by_representation.csv"
        ),
        "selected_high_dimensional_by_representation_tex": (
            output_dir / "selected_high_dimensional_by_representation.tex"
        ),
        "selected_by_representation_tables_tex": (
            output_dir / "selected_by_representation_tables.tex"
        ),
        "thesis_route_summary_csv": output_dir / "thesis_route_summary.csv",
        "thesis_route_summary_tex": output_dir / "thesis_route_summary.tex",
        "thesis_generator_comparison_csv": output_dir / "thesis_generator_comparison.csv",
        "thesis_generator_comparison_tex": output_dir / "thesis_generator_comparison.tex",
        "thesis_main_tables_tex": output_dir / "thesis_main_tables.tex",
        "appendix_selected_metrics_csv": output_dir / "appendix_selected_metrics.csv",
        "appendix_selected_metrics_tex": output_dir / "appendix_selected_metrics.tex",
        "exemplar_selection_csv": output_dir / "exemplar_selection.csv",
        "exemplar_selected_metrics_csv": output_dir / "exemplar_selected_metrics.csv",
        "exemplar_selected_metrics_tex": output_dir / "exemplar_selected_metrics.tex",
        "results_section_projection_quality_csv": (
            output_dir / "results_section_projection_quality.csv"
        ),
        "results_section_projection_quality_tex": (
            output_dir / "results_section_projection_quality.tex"
        ),
        "results_section_route_summary_csv": output_dir / "results_section_route_summary.csv",
        "results_section_route_summary_tex": output_dir / "results_section_route_summary.tex",
        "results_section_generator_conditioning_csv": (
            output_dir / "results_section_generator_conditioning.csv"
        ),
        "results_section_generator_conditioning_tex": (
            output_dir / "results_section_generator_conditioning.tex"
        ),
        "results_section_cluster_conditioning_csv": (
            output_dir / "results_section_cluster_conditioning.csv"
        ),
        "results_section_cluster_conditioning_tex": (
            output_dir / "results_section_cluster_conditioning.tex"
        ),
        "results_section_tables_tex": output_dir / "results_section_tables.tex",
        "metadata": output_dir / "metadata.json",
    }
    inventory.to_csv(outputs["metric_inventory_csv"], index=False)
    overall.to_csv(outputs["evaluation_results_overall_csv"], index=False, float_format=float_format)
    by_model.to_csv(outputs["evaluation_results_by_model_csv"], index=False, float_format=float_format)

    inventory_latex = inventory.to_latex(index=False, escape=True, longtable=True)
    overall_latex = overall.to_latex(
        index=False,
        escape=True,
        float_format=lambda value: float_format % value,
        na_rep="",
        caption="Summary of low- and high-dimensional synthetic embedding metrics.",
        label="tab:lowgen-evaluation-results-overall",
    )
    by_model_latex = by_model.to_latex(
        index=False,
        escape=True,
        float_format=lambda value: float_format % value,
        na_rep="",
        longtable=True,
        caption="Lowgen synthetic embedding metrics by representation and generator.",
        label="tab:lowgen-evaluation-results-by-model",
    )
    outputs["metric_inventory_tex"].write_text(inventory_latex)
    outputs["evaluation_results_overall_tex"].write_text(overall_latex)
    outputs["evaluation_results_by_model_tex"].write_text(by_model_latex)
    outputs["evaluation_tables_tex"].write_text(
        "\n\n".join([inventory_latex, overall_latex, by_model_latex]) + "\n"
    )

    selected_tables = selected_metric_tables(generator_df)
    low_selected = selected_tables["low_dimensional"]
    high_selected = selected_tables["high_dimensional"]
    low_selected.to_csv(outputs["selected_low_dimensional_csv"], index=False, float_format=float_format)
    high_selected.to_csv(outputs["selected_high_dimensional_csv"], index=False, float_format=float_format)
    low_selected_latex = selected_metrics_to_latex(
        low_selected,
        caption="Selected low-dimensional synthetic embedding evaluation metrics.",
        label="tab:lowgen-selected-low-dimensional",
        float_format=float_format,
    )
    high_selected_latex = selected_metrics_to_latex(
        high_selected,
        caption="Selected high-dimensional inverse-projected embedding evaluation metrics.",
        label="tab:lowgen-selected-high-dimensional",
        float_format=float_format,
    )
    outputs["selected_low_dimensional_tex"].write_text(low_selected_latex)
    outputs["selected_high_dimensional_tex"].write_text(high_selected_latex)
    outputs["selected_metric_tables_tex"].write_text(
        low_selected_latex + "\n\n" + high_selected_latex + "\n"
    )

    by_representation = selected_metric_tables_by_representation(generator_df)
    low_by_rep = by_representation["low_dimensional"]
    high_by_rep = by_representation["high_dimensional"]
    low_by_rep.to_csv(
        outputs["selected_low_dimensional_by_representation_csv"],
        index=False,
        float_format=float_format,
    )
    high_by_rep.to_csv(
        outputs["selected_high_dimensional_by_representation_csv"],
        index=False,
        float_format=float_format,
    )
    low_by_rep_latex = by_representation_to_latex(
        low_by_rep,
        caption=(
            "Selected low-dimensional synthetic embedding evaluation metrics by "
            "dataset, encoder, projector, and dimension."
        ),
        label="tab:lowgen-selected-low-dimensional-by-representation",
        float_format=float_format,
    )
    high_by_rep_latex = by_representation_to_latex(
        high_by_rep,
        caption=(
            "Selected high-dimensional inverse-projected embedding evaluation metrics "
            "by dataset, encoder, projector, and dimension."
        ),
        label="tab:lowgen-selected-high-dimensional-by-representation",
        float_format=float_format,
    )
    outputs["selected_low_dimensional_by_representation_tex"].write_text(low_by_rep_latex)
    outputs["selected_high_dimensional_by_representation_tex"].write_text(high_by_rep_latex)
    outputs["selected_by_representation_tables_tex"].write_text(
        low_by_rep_latex + "\n\n" + high_by_rep_latex + "\n"
    )

    thesis_tables = make_thesis_tables(generator_df, float_format)
    thesis_tables["route_summary_numeric"].to_csv(
        outputs["thesis_route_summary_csv"],
        index=False,
        float_format=float_format,
    )
    thesis_tables["generator_comparison_numeric"].to_csv(
        outputs["thesis_generator_comparison_csv"],
        index=False,
        float_format=float_format,
    )
    thesis_tables["appendix"].to_csv(
        outputs["appendix_selected_metrics_csv"],
        index=False,
        float_format=float_format,
    )
    outputs["thesis_route_summary_tex"].write_text(thesis_tables["route_summary_latex"])
    outputs["thesis_generator_comparison_tex"].write_text(
        thesis_tables["generator_comparison_latex"]
    )
    outputs["thesis_main_tables_tex"].write_text(
        thesis_tables["route_summary_latex"]
        + "\n\n"
        + thesis_tables["generator_comparison_latex"]
        + "\n"
    )
    outputs["appendix_selected_metrics_tex"].write_text(thesis_tables["appendix_latex"])
    exemplar = make_exemplar_table(generator_df, float_format)
    exemplar["selection"].to_csv(outputs["exemplar_selection_csv"], index=False, float_format=float_format)
    exemplar["table"].to_csv(
        outputs["exemplar_selected_metrics_csv"],
        index=False,
        float_format=float_format,
    )
    outputs["exemplar_selected_metrics_tex"].write_text(exemplar["latex"])

    results_section = make_results_section_tables(
        generator_df,
        results_df,
        float_format,
        cluster_results_path=cluster_results_path,
    )
    results_section["projection_quality"].to_csv(
        outputs["results_section_projection_quality_csv"],
        index=False,
        float_format=float_format,
    )
    results_section["route_summary_numeric"].to_csv(
        outputs["results_section_route_summary_csv"],
        index=False,
        float_format=float_format,
    )
    results_section["generator_conditioning_numeric"].to_csv(
        outputs["results_section_generator_conditioning_csv"],
        index=False,
        float_format=float_format,
    )
    results_section["cluster_conditioning"].to_csv(
        outputs["results_section_cluster_conditioning_csv"],
        index=False,
        float_format=float_format,
    )
    outputs["results_section_projection_quality_tex"].write_text(
        results_section["projection_quality_latex"]
    )
    outputs["results_section_route_summary_tex"].write_text(
        results_section["route_summary_latex"]
    )
    outputs["results_section_generator_conditioning_tex"].write_text(
        results_section["generator_conditioning_latex"]
    )
    outputs["results_section_cluster_conditioning_tex"].write_text(
        results_section["cluster_conditioning_latex"]
    )
    outputs["results_section_tables_tex"].write_text(
        "\n\n".join(
            [
                results_section["projection_quality_latex"],
                results_section["route_summary_latex"],
                results_section["generator_conditioning_latex"],
                results_section["cluster_conditioning_latex"],
                exemplar["latex"],
                thesis_tables["appendix_latex"],
            ]
        )
        + "\n"
    )
    outputs["metadata"].write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "results_path": str(results_path),
                "cluster_results_path": str(cluster_results_path)
                if cluster_results_path is not None
                else None,
                **input_metadata,
                "n_generator_rows": int(len(generator_df)),
                "n_metric_rows": int(len(long)),
                "note": "Privacy rows are marked not_computed unless NN distance columns exist.",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return outputs


def load_generator_rows(
    results_path: Path,
    tasks_dir: Path | None = None,
    source: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if source not in {"auto", "results", "tasks"}:
        raise ValueError(f"Unknown source: {source}")
    tasks_dir = tasks_dir or results_path.parent / "tasks"

    results_rows = pd.DataFrame()
    if results_path.exists():
        results_df = pd.read_csv(results_path)
        if "row_type" in results_df.columns:
            results_rows = results_df[results_df["row_type"] == "generator"].copy()

    task_rows = pd.DataFrame()
    task_files = sorted(tasks_dir.glob("*.json")) if tasks_dir.exists() else []
    if source in {"auto", "tasks"}:
        task_rows = load_generator_rows_from_tasks(task_files)
        if source == "tasks" and task_rows.empty:
            raise FileNotFoundError(f"No completed generator rows found in {tasks_dir}")

    if source == "results":
        selected = results_rows
        selected_source = "results_csv"
    elif source == "tasks":
        selected = task_rows
        selected_source = "task_json"
    elif len(task_rows) > len(results_rows):
        selected = task_rows
        selected_source = "task_json"
    else:
        selected = results_rows
        selected_source = "results_csv"

    return selected, {
        "selected_source": selected_source,
        "tasks_dir": str(tasks_dir),
        "n_task_files": int(len(task_files)),
        "n_generator_rows_from_results_csv": int(len(results_rows)),
        "n_generator_rows_from_task_json": int(len(task_rows)),
    }


def load_generator_rows_from_tasks(task_files: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in task_files:
        with path.open() as handle:
            task = json.load(handle)
        base = {
            key: task.get(key)
            for key in [
                "task_id",
                "dataset",
                "encoder",
                "projector",
                "dimension",
                "target_name",
                "seed",
            ]
        }
        for gen in task.get("generators", []):
            gen_base = base | {
                "generator": gen.get("generator"),
                "conditioning": gen.get("conditioning"),
            }
            rows.append(_generator_row(gen_base, gen))
    return pd.DataFrame(rows)


def metric_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in METRICS:
        present = spec.column in df.columns
        has_values = bool(present and pd.to_numeric(df[spec.column], errors="coerce").notna().any())
        rows.append(
            {
                "space": spec.space,
                "category": spec.category,
                "metric": spec.metric,
                "role": spec.role,
                "result_column": spec.column,
                "status": "computed" if has_values else "not_computed",
                "what_it_measures": spec.what_it_measures,
                "why_useful": spec.why_useful,
            }
        )
    return pd.DataFrame(rows)


def metric_long_table(df: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    id_columns = [c for c in ["projector", "generator", "conditioning"] if c in df.columns]
    rows = []
    for spec in METRICS:
        status = inventory.loc[inventory["result_column"] == spec.column, "status"].iloc[0]
        if status != "computed":
            rows.append(
                {
                    "space": spec.space,
                    "category": spec.category,
                    "metric": spec.metric,
                    "column": spec.column,
                    "status": status,
                    "value": np.nan,
                }
            )
            continue
        values = pd.to_numeric(df[spec.column], errors="coerce")
        present = df.loc[values.notna(), id_columns].copy()
        present["value"] = values[values.notna()].to_numpy()
        for _, row in present.iterrows():
            rows.append(
                {
                    "space": spec.space,
                    "category": spec.category,
                    "metric": spec.metric,
                    "column": spec.column,
                    "status": "computed",
                    **{column: row.get(column) for column in id_columns},
                    "value": float(row["value"]),
                }
            )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    present = [column for column in groups if column in df.columns]
    summary = (
        df.groupby(present, dropna=False, sort=True)
        .agg(
            n=("value", lambda value: int(pd.to_numeric(value, errors="coerce").notna().sum())),
            mean=("value", "mean"),
            std=("value", "std"),
        )
        .reset_index()
    )
    order = {category: index for index, category in enumerate(["fidelity", "diversity", "privacy", "utility"])}
    if "category" in summary.columns:
        summary["_category_order"] = summary["category"].map(order).fillna(99)
        summary = summary.sort_values(["space", "_category_order", "metric"], kind="mergesort")
        summary = summary.drop(columns=["_category_order"])
    return summary.reset_index(drop=True)


def selected_metric_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        space: selected_metric_table(df, space)
        for space in ["low_dimensional", "high_dimensional"]
    }


def selected_metric_table(df: pd.DataFrame, space: str) -> pd.DataFrame:
    rows = []
    for spec in PAPER_METRICS:
        if spec.space != space:
            continue
        values = (
            pd.to_numeric(df[spec.column], errors="coerce")
            if spec.column in df.columns
            else pd.Series(dtype=float)
        )
        values = values[np.isfinite(values)]
        rows.append(
            {
                "Category": spec.category,
                "Metric": spec.metric,
                "Mean": values.mean() if len(values) else np.nan,
                "SD": values.std(ddof=1) if len(values) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def selected_metric_tables_by_representation(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        space: selected_metric_table_by_representation(df, space)
        for space in ["low_dimensional", "high_dimensional"]
    }


def selected_metric_table_by_representation(df: pd.DataFrame, space: str) -> pd.DataFrame:
    specs = [spec for spec in PAPER_METRICS if spec.space == space]
    groups = ["dataset", "encoder", "projector", "dimension"]
    missing = [column for column in groups if column not in df.columns]
    if missing:
        raise ValueError(f"Missing grouping columns: {missing}")
    working = df.copy()
    working["dimension"] = working["dimension"].apply(format_dimension)
    rows = []
    for keys, group in working.groupby(groups, dropna=False, sort=True):
        row = {
            "Dataset": keys[0],
            "Encoder": keys[1],
            "Projector": format_projector(keys[2]),
            "Dimension": keys[3],
        }
        for spec in specs:
            values = (
                pd.to_numeric(group[spec.column], errors="coerce")
                if spec.column in group.columns
                else pd.Series(dtype=float)
            )
            values = values[np.isfinite(values)]
            row[spec.metric] = values.mean() if len(values) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    order = [
        "Dataset",
        "Encoder",
        "Projector",
        "Dimension",
        *[spec.metric for spec in specs],
    ]
    if table.empty:
        return pd.DataFrame(columns=order)
    return table[order].sort_values(
        ["Dataset", "Encoder", "Projector", "Dimension"],
        kind="mergesort",
    ).reset_index(drop=True)


def by_representation_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_format: str,
) -> str:
    return df.to_latex(
        index=False,
        escape=True,
        longtable=True,
        float_format=lambda value: float_format % value,
        na_rep="",
        caption=caption,
        label=label,
    )


def make_thesis_tables(df: pd.DataFrame, float_format: str) -> dict[str, Any]:
    route_specs = [
        (
            "Low-dimensional synthesis",
            "PCA/UMAP synthetic low-dimensional embeddings",
            df["projector"].isin(["pca", "umap"]),
            LOWDIM_METRIC_COLUMNS,
        ),
        (
            "Direct full-dimensional synthesis",
            "Synthetic embeddings generated in the original feature space",
            df["projector"].eq("raw_highdim"),
            HIGHDIM_METRIC_COLUMNS,
        ),
        (
            "Inverse-projected synthesis",
            "PCA/UMAP synthetic embeddings mapped back to the original feature space",
            df["projector"].isin(["pca", "umap"]),
            HIGHDIM_METRIC_COLUMNS,
        ),
    ]
    route_numeric_rows = []
    route_display_rows = []
    generator_numeric_rows = []
    generator_display_rows = []
    for route, description, mask, metrics in route_specs:
        route_df = df[mask].copy()
        route_numeric_rows.append(
            {
                "Route": route,
                "Evaluation": description,
                **metric_summary_columns(route_df, metrics),
            }
        )
        route_display_rows.append(
            {
                "Route": route,
                "Evaluation": description,
                **metric_display_columns(route_df, metrics, float_format),
            }
        )
        for generator, group in route_df.groupby("generator", dropna=False, sort=True):
            generator_numeric_rows.append(
                {
                    "Route": route,
                    "Generator": format_generator(generator),
                    **metric_summary_columns(group, metrics),
                }
            )
            generator_display_rows.append(
                {
                    "Route": route,
                    "Generator": format_generator(generator),
                    **metric_display_columns(group, metrics, float_format),
                }
            )
    route_numeric = pd.DataFrame(route_numeric_rows)
    route_display = pd.DataFrame(route_display_rows)
    generator_numeric = pd.DataFrame(generator_numeric_rows)
    generator_display = pd.DataFrame(generator_display_rows)
    appendix = appendix_selected_metrics(df)
    return {
        "route_summary_numeric": route_numeric,
        "route_summary_latex": thesis_table_to_latex(
            route_display,
            caption=(
                "Summary of the three synthesis routes used to answer whether "
                "low-dimensional embeddings can be generated, whether full-dimensional "
                "embeddings can be generated directly, and whether full-dimensional "
                "embeddings can be produced by inverse-projecting generated low-dimensional "
                "embeddings. Values are mean (SD) over completed datasets, encoders, "
                "dimensions, seeds, generators, and conditioning settings available at "
                "the time of table generation."
            ),
            label="tab:lowgen-thesis-route-summary",
            column_format="llrrrrr",
        ),
        "generator_comparison_numeric": generator_numeric,
        "generator_comparison_latex": thesis_table_to_latex(
            generator_display,
            caption=(
                "Comparison of the computationally inexpensive generators used in this "
                "study across the three synthesis routes. Resampling is an empirical "
                "baseline, the diagonal Gaussian is the simplest parametric model, and "
                "the diagonal Gaussian mixture model tests whether a modest multimodal "
                "extension improves fidelity, diversity, or downstream linear-probe "
                "utility."
            ),
            label="tab:lowgen-thesis-generator-comparison",
            column_format="llrrrrr",
        ),
        "appendix": appendix,
        "appendix_latex": appendix_to_latex(
            appendix,
            caption=(
                "Full selected-metric results by dataset, encoder, projector, dimension, "
                "dimension target, generator, and conditioning strategy. Metrics are "
                "averaged over completed seeds. This table provides the detailed support "
                "for the aggregated synthesis-route and generator-comparison results in "
                "the main text."
            ),
            label="tab:lowgen-appendix-selected-metrics",
            float_format=float_format,
        ),
    }


def metric_summary_columns(df: pd.DataFrame, metrics: dict[str, str]) -> dict[str, float]:
    row: dict[str, float] = {}
    for metric, column in metrics.items():
        values = numeric_values(df, column)
        row[f"{metric} mean"] = values.mean() if len(values) else np.nan
        row[f"{metric} SD"] = values.std(ddof=1) if len(values) > 1 else np.nan
    return row


def metric_display_columns(
    df: pd.DataFrame,
    metrics: dict[str, str],
    float_format: str,
    style: str = "paren",
) -> dict[str, str]:
    row: dict[str, str] = {}
    for metric, column in metrics.items():
        values = numeric_values(df, column)
        mean = values.mean() if len(values) else np.nan
        sd = values.std(ddof=1) if len(values) > 1 else np.nan
        if style == "pm":
            row[metric] = format_mean_pm(mean, sd, float_format)
        else:
            row[metric] = format_mean_sd(mean, sd, float_format)
    return row


def appendix_selected_metrics(df: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "dataset",
        "encoder",
        "projector",
        "dimension",
        "target_name",
        "generator",
        "conditioning",
    ]
    missing = [column for column in groups if column not in df.columns]
    if missing:
        raise ValueError(f"Missing appendix grouping columns: {missing}")
    working = df.copy()
    working["dimension"] = working["dimension"].apply(format_dimension)
    rows = []
    for keys, group in working.groupby(groups, dropna=False, sort=True):
        row = {
            "Dataset": keys[0],
            "Encoder": keys[1],
            "Projector": format_projector(keys[2]),
            "Dimension": keys[3],
            "Target": keys[4],
            "Generator": format_generator(keys[5]),
            "Conditioning": format_conditioning(keys[6]),
        }
        for prefix, metrics in [
            ("Low-D", LOWDIM_METRIC_COLUMNS),
            ("High-D", HIGHDIM_METRIC_COLUMNS),
        ]:
            for metric, column in metrics.items():
                values = numeric_values(group, column)
                row[f"{prefix} {metric}"] = values.mean() if len(values) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(
        ["Dataset", "Encoder", "Projector", "Dimension", "Target", "Generator", "Conditioning"],
        kind="mergesort",
    ).reset_index(drop=True)


def numeric_values(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(df[column], errors="coerce")
    return values[np.isfinite(values)]


def format_mean_sd(mean: float, sd: float, float_format: str) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(sd):
        return _format_float(mean, float_format)
    return f"{_format_float(mean, float_format)} ({_format_float(sd, float_format)})"


def format_mean_pm(mean: float, sd: float, float_format: str) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(sd):
        return _format_float(mean, float_format)
    return (
        r"$"
        + _format_float(mean, float_format)
        + r" \pm "
        + _format_float(sd, float_format)
        + r"$"
    )


def thesis_table_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_format: str,
) -> str:
    return df.to_latex(
        index=False,
        escape=True,
        column_format=column_format,
        caption=caption,
        label=label,
    )


def appendix_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_format: str,
) -> str:
    return df.to_latex(
        index=False,
        escape=True,
        longtable=True,
        float_format=lambda value: float_format % value,
        na_rep="",
        caption=caption,
        label=label,
    )


def make_exemplar_table(df: pd.DataFrame, float_format: str) -> dict[str, Any]:
    exemplar_df = df[
        df["dataset"].eq(EXEMPLAR_DATASET)
        & df["encoder"].eq(EXEMPLAR_ENCODER)
    ].copy()
    selection_df = exemplar_df[exemplar_df["conditioning"].eq("true_label")].copy()
    columns = [
        "Dataset",
        "Encoder",
        "Route",
        "Projector",
        "Dimension",
        "Target",
        "Generator",
        "Conditioning",
        "Fréchet distance",
        "Precision",
        "Recall",
        "LR accuracy",
        "LR macro ROC-AUC",
    ]
    selection_columns = [
        "Dataset",
        "Encoder",
        "Selected projector",
        "Selected dimension",
        "Selected target",
        "Selection metric",
        "Selection value",
        "Tie-breaker LR accuracy",
        "Tie-breaker Fréchet distance",
        "Tie-breaker precision",
    ]
    if exemplar_df.empty:
        empty_table = pd.DataFrame(columns=columns)
        empty_selection = pd.DataFrame(columns=selection_columns)
        return {
            "selection": empty_selection,
            "table": empty_table,
            "latex": exemplar_to_latex(empty_table, empty_selection, float_format),
        }

    reduced = selection_df[selection_df["projector"].isin(["pca", "umap"])].copy()
    selected = select_exemplar_representation(reduced)
    if selected.empty:
        empty_table = pd.DataFrame(columns=columns)
        empty_selection = pd.DataFrame(columns=selection_columns)
        return {
            "selection": empty_selection,
            "table": empty_table,
            "latex": exemplar_to_latex(empty_table, empty_selection, float_format),
        }

    selected_row = selected.iloc[0]
    selected_projector = selected_row["projector"]
    selected_dimension = selected_row["dimension"]
    selected_target = selected_row["target_name"]
    selected_reduced_source = exemplar_df[exemplar_df["projector"].isin(["pca", "umap"])].copy()
    selected_mask = (
        selected_reduced_source["projector"].eq(selected_projector)
        & selected_reduced_source["dimension"].apply(format_dimension).eq(selected_dimension)
        & selected_reduced_source["target_name"].eq(selected_target)
    )
    selected_reduced = selected_reduced_source[selected_mask].copy()
    raw_baseline = exemplar_df[exemplar_df["projector"].eq("raw_highdim")].copy()

    rows = []
    rows.extend(
        exemplar_route_rows(
            raw_baseline,
            "Direct full-dimensional baseline",
            HIGHDIM_METRIC_COLUMNS,
            "raw_highdim",
            "raw",
            "highdim_baseline",
        )
    )
    rows.extend(
        exemplar_route_rows(
            selected_reduced,
            "Low-dimensional synthesis",
            LOWDIM_METRIC_COLUMNS,
            selected_projector,
            selected_dimension,
            selected_target,
        )
    )
    rows.extend(
        exemplar_route_rows(
            selected_reduced,
            "Inverse-projected synthesis",
            HIGHDIM_METRIC_COLUMNS,
            selected_projector,
            selected_dimension,
            selected_target,
        )
    )
    table = pd.DataFrame(rows, columns=columns)
    selection = pd.DataFrame(
        [
            {
                "Dataset": format_dataset(EXEMPLAR_DATASET),
                "Encoder": format_encoder(EXEMPLAR_ENCODER),
                "Selected projector": format_projector(selected_projector),
                "Selected dimension": selected_dimension,
                "Selected target": selected_target,
                "Selection metric": "Mean true-label low-D LR macro ROC-AUC",
                "Selection value": selected_row["lowdim_tstr_macro_roc_auc_ovr"],
                "Tie-breaker LR accuracy": selected_row["lowdim_tstr_accuracy"],
                "Tie-breaker Fréchet distance": selected_row["lowdim_fid"],
                "Tie-breaker precision": selected_row["lowdim_precision"],
            }
        ],
        columns=selection_columns,
    )
    return {
        "selection": selection,
        "table": table,
        "latex": exemplar_to_latex(table, selection, float_format),
    }


def select_exemplar_representation(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    working = df.copy()
    working["dimension"] = working["dimension"].apply(format_dimension)
    summary = (
        working.groupby(["projector", "dimension", "target_name"], dropna=False)
        .agg(
            lowdim_tstr_macro_roc_auc_ovr=("lowdim_tstr_macro_roc_auc_ovr", "mean"),
            lowdim_tstr_accuracy=("lowdim_tstr_accuracy", "mean"),
            lowdim_fid=("lowdim_fid", "mean"),
            lowdim_precision=("lowdim_precision", "mean"),
        )
        .reset_index()
    )
    summary = summary.dropna(subset=["lowdim_tstr_macro_roc_auc_ovr"])
    if summary.empty:
        return summary
    return summary.sort_values(
        [
            "lowdim_tstr_macro_roc_auc_ovr",
            "lowdim_tstr_accuracy",
            "lowdim_fid",
            "lowdim_precision",
            "projector",
            "dimension",
            "target_name",
        ],
        ascending=[False, False, True, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def exemplar_route_rows(
    df: pd.DataFrame,
    route: str,
    metrics: dict[str, str],
    projector: Any,
    dimension: Any,
    target: Any,
) -> list[dict[str, Any]]:
    rows = []
    for (generator, conditioning), group in df.groupby(
        ["generator", "conditioning"],
        dropna=False,
        sort=True,
    ):
        row = {
            "Dataset": format_dataset(EXEMPLAR_DATASET),
            "Encoder": format_encoder(EXEMPLAR_ENCODER),
            "Route": route,
            "Projector": format_projector(projector),
            "Dimension": format_dimension(dimension),
            "Target": target,
            "Generator": format_generator(generator),
            "Conditioning": format_conditioning(conditioning),
        }
        for metric, column in metrics.items():
            values = numeric_values(group, column)
            row[metric] = values.mean() if len(values) else np.nan
        rows.append(row)
    generator_order = {"Resample": 0, "Diagonal Gaussian": 1, "Diagonal GMM": 2}
    conditioning_order = {"True label": 0, "Unconditioned": 1}
    return sorted(
        rows,
        key=lambda row: (
            generator_order.get(row["Generator"], 99),
            conditioning_order.get(row["Conditioning"], 99),
        ),
    )


def exemplar_to_latex(
    table: pd.DataFrame,
    selection: pd.DataFrame,
    float_format: str,
) -> str:
    if selection.empty:
        caption = (
            "Exemplar selected-metric results for iCAIRD Cervical 100K with UNI2. "
            "No completed true-label PCA/UMAP rows were available when the table was generated."
        )
    else:
        row = selection.iloc[0]
        caption = (
            "Exemplar selected-metric results for iCAIRD Cervical 100K with UNI2. "
            f"The reduced representation was selected as {row['Selected projector']} "
            f"with dimension {row['Selected dimension']} "
            f"({row['Selected target']}) because it had the highest mean true-label "
            "low-dimensional logistic-regression macro ROC-AUC across completed generators "
            "and seeds. The direct full-dimensional baseline uses the raw high-dimensional "
            "UNI2 embeddings for the same dataset and encoder. Rows compare the direct "
            "full-dimensional baseline, low-dimensional synthesis, and inverse-projected synthesis. "
            "Conditioning is shown explicitly; unconditioned rows have blank utility entries "
            "because no class labels are produced for train-synthetic-test-real evaluation."
        )
    return exemplar_table_to_latex(
        table,
        caption=caption,
        label="tab:lowgen-exemplar-icaird100k-uni2",
        float_format=float_format,
    )


def exemplar_table_to_latex(
    table: pd.DataFrame,
    caption: str,
    label: str,
    float_format: str,
) -> str:
    if table.empty:
        body = ""
    else:
        rows = []
        for route, group in table.groupby("Route", sort=False):
            route_span = len(group)
            route_row_index = 0
            for generator, generator_group in group.groupby("Generator", sort=False):
                generator_span = len(generator_group)
                for generator_index, (_, row) in enumerate(generator_group.iterrows()):
                    route_cell = (
                        f"\\multirow{{{route_span}}}{{*}}{{{_latex_escape(route)}}}"
                        if route_row_index == 0
                        else ""
                    )
                    generator_cell = (
                        f"\\multirow{{{generator_span}}}{{*}}{{{_latex_escape(generator)}}}"
                        if generator_index == 0
                        else ""
                    )
                    rows.append(
                        " & ".join(
                            [
                                route_cell,
                                generator_cell,
                                _latex_escape(row["Conditioning"]),
                                _format_float(row["Fréchet distance"], float_format),
                                _format_float(row["Precision"], float_format),
                                _format_float(row["Recall"], float_format),
                                _format_float(row["LR accuracy"], float_format),
                                _format_float(row["LR macro ROC-AUC"], float_format),
                            ]
                        )
                        + r" \\"
                    )
                    route_row_index += 1
        body = "\n".join(rows)
    return "\n".join(
        [
            r"\begin{table}",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\renewcommand{\arraystretch}{1.08}",
            rf"\caption{{{_latex_escape(caption)}}}",
            rf"\label{{{_latex_escape(label)}}}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lllrrrrr}",
            r"\toprule",
            (
                "Route & Generator & Conditioning & Fréchet distance & Precision & Recall & "
                r"LR accuracy & LR macro ROC-AUC \\"
            ),
            r"\midrule",
            body,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )


def make_results_section_tables(
    generator_df: pd.DataFrame,
    results_df: pd.DataFrame,
    float_format: str,
    cluster_results_path: Path | None = None,
) -> dict[str, Any]:
    projection_quality = make_projection_quality_table(results_df)
    projection_display = mean_sd_display_table(
        projection_quality,
        id_columns=["Projection", "n"],
        float_format=float_format,
    )

    route_numeric, route_display = make_results_route_summary(generator_df, float_format)
    generator_numeric, generator_display = make_generator_conditioning_summary(
        generator_df,
        float_format,
    )
    cluster_table = make_cluster_conditioning_table(cluster_results_path)

    return {
        "projection_quality": projection_quality,
        "projection_quality_latex": results_table_to_latex(
            projection_display,
            caption=(
                "Quality of the fitted dimensionality-reduction representations. "
                "Rows are averaged over completed datasets, encoders, dimensions, and seeds. "
                "The raw high-dimensional row is included as the baseline for the original "
                "embedding space."
            ),
            label="tab:lowgen-results-projection-quality",
            column_format="lrrrrrrrrr",
        ),
        "route_summary_numeric": route_numeric,
        "route_summary_latex": results_table_to_latex(
            route_display,
            caption=(
                "Summary of the three synthesis routes evaluated in the study. "
                "Rows are ordered by their position in the synthesis pipeline: direct "
                "full-dimensional synthesis, low-dimensional synthesis, then inverse-projected "
                "reconstruction. "
                "Low-dimensional synthesis measures synthetic samples in the PCA/UMAP space; "
                "direct full-dimensional synthesis fits generators to the original embeddings; "
                "inverse-projected synthesis evaluates low-dimensional synthetic samples after "
                "mapping them back to the original embedding space. Values are mean $\\pm$ SD."
            ),
            label="tab:lowgen-results-route-summary",
            column_format="lrrrrr",
            resize_to_textwidth=True,
        ),
        "generator_conditioning_numeric": generator_numeric,
        "generator_conditioning_latex": results_table_to_latex(
            generator_display,
            caption=(
                "Generator and conditioning comparison for each synthesis route. "
                "Unconditioned generators have blank utility entries because they do not "
                "produce class labels for train-synthetic-test-real classification. "
                "Values are mean $\\pm$ SD over completed datasets, encoders, dimensions, and seeds."
            ),
            label="tab:lowgen-results-generator-conditioning",
            column_format="llrrrrr",
            longtable=True,
        ),
        "cluster_conditioning": cluster_table,
        "cluster_conditioning_latex": cluster_conditioning_to_latex(
            cluster_table,
            float_format=float_format,
        ),
    }


def make_projection_quality_table(results_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Projection",
        "n",
        "Linear ACC mean",
        "Linear ACC SD",
        "Linear ROC-AUC mean",
        "Linear ROC-AUC SD",
        "KNN ACC mean",
        "KNN ACC SD",
        "KNN ROC-AUC mean",
        "KNN ROC-AUC SD",
        "Trustworthiness mean",
        "Trustworthiness SD",
        "Pairwise distance correlation mean",
        "Pairwise distance correlation SD",
        "Reconstruction MSE mean",
        "Reconstruction MSE SD",
        "Reconstruction cosine mean",
        "Reconstruction cosine SD",
    ]
    if results_df.empty or "row_type" not in results_df.columns:
        return pd.DataFrame(columns=columns)
    rep = results_df[results_df["row_type"].eq("representation")].copy()
    if rep.empty:
        return pd.DataFrame(columns=columns)
    metric_map = {
        "Linear ACC": "linear_accuracy",
        "Linear ROC-AUC": "linear_macro_roc_auc_ovr",
        "KNN ACC": "knn_accuracy",
        "KNN ROC-AUC": "knn_macro_roc_auc_ovr",
        "Trustworthiness": "trustworthiness",
        "Pairwise distance correlation": "pairwise_dist_correlation",
        "Reconstruction MSE": "reconstruction_mse",
        "Reconstruction cosine": "reconstruction_cosine_similarity",
    }
    rows = []
    for projector, group in rep.groupby("projector", dropna=False, sort=True):
        row: dict[str, Any] = {
            "Projection": format_projector(projector),
            "n": int(len(group)),
        }
        for metric, column in metric_map.items():
            values = numeric_values(group, column)
            row[f"{metric} mean"] = values.mean() if len(values) else np.nan
            row[f"{metric} SD"] = values.std(ddof=1) if len(values) > 1 else np.nan
        rows.append(row)
    order = {"Raw high-dimensional": 0, "PCA": 1, "UMAP": 2}
    table = pd.DataFrame(rows)
    return table.sort_values(
        by="Projection",
        key=lambda value: value.map(order).fillna(99),
        kind="mergesort",
    )[columns].reset_index(drop=True)


def make_results_route_summary(
    df: pd.DataFrame,
    float_format: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_specs = [
        (
            "Direct full-dimensional synthesis",
            df["projector"].eq("raw_highdim"),
            HIGHDIM_METRIC_COLUMNS,
        ),
        (
            "Low-dimensional synthesis",
            df["projector"].isin(["pca", "umap"]),
            LOWDIM_METRIC_COLUMNS,
        ),
        (
            "Inverse-projected synthesis",
            df["projector"].isin(["pca", "umap"]),
            HIGHDIM_METRIC_COLUMNS,
        ),
    ]
    numeric_rows = []
    display_rows = []
    for route, mask, metrics in route_specs:
        route_df = df[mask].copy()
        numeric_rows.append(
            {
                "Route": route,
                "n": int(len(route_df)),
                **metric_summary_columns(route_df, metrics),
            }
        )
        display_rows.append(
            {
                "Route": route,
                **metric_display_columns(
                    route_df,
                    metrics,
                    float_format,
                    style="pm",
                ),
            }
        )
    return pd.DataFrame(numeric_rows), pd.DataFrame(display_rows)


def make_generator_conditioning_summary(
    df: pd.DataFrame,
    float_format: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_specs = [
        (
            "Low-dimensional synthesis",
            df["projector"].isin(["pca", "umap"]),
            LOWDIM_METRIC_COLUMNS,
        ),
        (
            "Direct full-dimensional synthesis",
            df["projector"].eq("raw_highdim"),
            HIGHDIM_METRIC_COLUMNS,
        ),
        (
            "Inverse-projected synthesis",
            df["projector"].isin(["pca", "umap"]),
            HIGHDIM_METRIC_COLUMNS,
        ),
    ]
    numeric_rows = []
    display_rows = []
    generator_order = {"resample": 0, "diagonal_gaussian": 1, "diagonal_gmm": 2}
    conditioning_order = {"true_label": 0, "unconditioned": 1}
    for route, mask, metrics in route_specs:
        route_df = df[mask].copy()
        grouped = route_df.groupby(["generator", "conditioning"], dropna=False, sort=True)
        rows_for_route: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
        for (generator, conditioning), group in grouped:
            numeric = {
                "Route": route,
                "Generator": format_generator(generator),
                "Conditioning": format_conditioning(conditioning),
                "n": int(len(group)),
                **metric_summary_columns(group, metrics),
            }
            display = {
                "Route": route,
                "Generator": format_generator(generator),
                "Conditioning": format_conditioning(conditioning),
                **metric_display_columns(group, metrics, float_format, style="pm"),
            }
            rows_for_route.append(
                (
                    generator_order.get(str(generator), 99),
                    conditioning_order.get(str(conditioning), 99),
                    numeric,
                    display,
                )
            )
        for _, _, numeric, display in sorted(rows_for_route, key=lambda item: item[:2]):
            numeric_rows.append(numeric)
            display_rows.append(display)
    return pd.DataFrame(numeric_rows), pd.DataFrame(display_rows)


def make_cluster_conditioning_table(cluster_results_path: Path | None) -> pd.DataFrame:
    columns = [
        "Dataset",
        "Encoder",
        "Projection",
        "Dimension",
        "Conditioning",
        "Fréchet distance",
        "Precision",
        "Recall",
        "LR accuracy",
        "LR macro ROC-AUC",
    ]
    if cluster_results_path is None or not cluster_results_path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(cluster_results_path)
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Dataset": format_dataset(row.get("dataset")),
                "Encoder": format_encoder(row.get("encoder", "uni2")),
                "Projection": format_projector(row.get("projector", "pca")),
                "Dimension": format_dimension(row.get("dimension")),
                "Conditioning": format_conditioning(row.get("conditioning")),
                "Fréchet distance": row.get("fid"),
                "Precision": row.get("precision"),
                "Recall": row.get("recall"),
                "LR accuracy": row.get("tstr_accuracy"),
                "LR macro ROC-AUC": row.get("tstr_macro_roc_auc_ovr"),
            }
        )
    table = pd.DataFrame(rows, columns=columns)
    conditioning_order = {"Cluster": 0, "True label": 1, "Cluster + true label": 2}
    return table.sort_values(
        ["Dataset", "Conditioning"],
        key=lambda value: value.map(conditioning_order).fillna(value)
        if value.name == "Conditioning"
        else value,
        kind="mergesort",
    ).reset_index(drop=True)


def mean_sd_display_table(
    df: pd.DataFrame,
    id_columns: list[str],
    float_format: str,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    display = df[id_columns].copy()
    metric_prefixes = []
    for column in df.columns:
        if column.endswith(" mean"):
            metric_prefixes.append(column[: -len(" mean")])
    for metric in metric_prefixes:
        display[metric] = [
            format_mean_pm(mean, sd, float_format)
            for mean, sd in zip(
                df[f"{metric} mean"],
                df.get(f"{metric} SD", pd.Series([np.nan] * len(df))),
                strict=False,
            )
        ]
    return display


def results_table_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_format: str,
    longtable: bool = False,
    resize_to_textwidth: bool = False,
) -> str:
    latex = df.to_latex(
        index=False,
        escape=False,
        longtable=longtable,
        column_format=column_format,
        na_rep="",
        caption=caption,
        label=label,
    )
    if resize_to_textwidth and not longtable:
        return resize_tabular_latex(latex)
    return latex


def resize_tabular_latex(latex: str) -> str:
    lines = latex.splitlines()
    out = []
    inserted = False
    for line in lines:
        if line.startswith("\\begin{tabular}") and not inserted:
            out.append(r"\small")
            out.append(r"\setlength{\tabcolsep}{4pt}")
            out.append(r"\resizebox{\textwidth}{!}{%")
            out.append(line)
            inserted = True
            continue
        out.append(line)
        if inserted and line.startswith("\\end{tabular}"):
            out.append("}")
            inserted = False
    return "\n".join(out) + "\n"


def cluster_conditioning_to_latex(
    table: pd.DataFrame,
    float_format: str,
) -> str:
    caption = (
        "Fixed global KMeans cluster-conditioning follow-up. The cluster-only rows test "
        "whether global cluster assignments can structure synthetic embeddings without "
        "class labels; true-label rows are the class-conditioned baseline; cluster plus "
        "true label rows test whether adding global cluster information improves the "
        "class-conditioned generator. Utility entries are blank for cluster-only rows "
        "because no pathology class labels are generated."
    )
    if table.empty:
        return results_table_to_latex(
            table,
            caption=caption + " No completed cluster-conditioning rows were available.",
            label="tab:lowgen-results-cluster-conditioning",
            column_format="lllllrrrrr",
        )
    return table.to_latex(
        index=False,
        escape=True,
        float_format=lambda value: float_format % value,
        na_rep="",
        caption=caption,
        label="tab:lowgen-results-cluster-conditioning",
        column_format="lllllrrrrr",
    )


def format_dimension(value: Any) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "raw"
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if as_float.is_integer():
        return str(int(as_float))
    return str(value)


def format_projector(value: Any) -> str:
    mapping = {
        "raw_highdim": "Raw high-dimensional",
        "pca": "PCA",
        "umap": "UMAP",
    }
    return mapping.get(str(value), str(value))


def format_dataset(value: Any) -> str:
    mapping = {
        "icaird100k": "iCAIRD Cervical 100K",
        "kather100k": "Kather 100K",
        "spider-breast": "SPIDER breast",
        "spider-colorectal": "SPIDER colorectal",
    }
    return mapping.get(str(value), str(value))


def format_encoder(value: Any) -> str:
    mapping = {
        "uni2": "UNI2",
        "uni": "UNI",
        "dinov2": "DINOv2",
        "hibou": "Hibou",
    }
    return mapping.get(str(value), str(value))


def format_generator(value: Any) -> str:
    mapping = {
        "resample": "Resample",
        "diagonal_gaussian": "Diagonal Gaussian",
        "diagonal_gmm": "Diagonal GMM",
    }
    return mapping.get(str(value), str(value))


def format_conditioning(value: Any) -> str:
    mapping = {
        "unconditioned": "Unconditioned",
        "true_label": "True label",
        "cluster": "Cluster",
        "cluster_true_label": "Cluster + true label",
    }
    return mapping.get(str(value), str(value))


def selected_metrics_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_format: str,
) -> str:
    rows = []
    for _, group in df.groupby("Category", sort=False):
        span = len(group)
        for index, row in enumerate(group.itertuples(index=False)):
            category = f"\\multirow{{{span}}}{{*}}{{{_latex_escape(row.Category)}}}" if index == 0 else ""
            rows.append(
                " & ".join(
                    [
                        category,
                        _latex_escape(row.Metric),
                        _format_float(row.Mean, float_format),
                        _format_float(row.SD, float_format),
                    ]
                )
                + r" \\"
            )
    body = "\n".join(rows)
    return "\n".join(
        [
            r"\begin{table}",
            r"\centering",
            rf"\caption{{{_latex_escape(caption)}}}",
            rf"\label{{{_latex_escape(label)}}}",
            r"\begin{tabular}{llrr}",
            r"\toprule",
            r"Category & Metric & Mean & SD \\",
            r"\midrule",
            body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _format_float(value: float, float_format: str) -> str:
    if pd.isna(value):
        return ""
    return float_format % float(value)


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


if __name__ == "__main__":
    main()
