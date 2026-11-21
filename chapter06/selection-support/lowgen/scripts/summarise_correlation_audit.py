#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/raid/experiments/lowgen_v2/output/lowgen-v2-unconditioned-true-label/"
    "correlation_audit/correlation_summary.csv"
)
PROJECTOR_ORDER = ["raw_highdim", "pca", "umap"]
BASE_COLUMNS = ["dataset", "encoder", "projector", "dimension", "target_name"]


TABLE_SPECS = {
    "correlation_by_projector": {
        "caption": "Mean absolute dimension correlation by representation.",
        "label": "tab:correlation-by-projector",
    },
    "correlation_reduction_effect": {
        "caption": (
            "Effect of dimensionality reduction on mean absolute dimension "
            "correlation."
        ),
        "label": "tab:correlation-reduction-effect",
    },
    "correlation_by_dataset": {
        "caption": "Mean absolute dimension correlation by dataset.",
        "label": "tab:correlation-by-dataset",
    },
    "correlation_by_encoder": {
        "caption": "Mean absolute dimension correlation by foundation model.",
        "label": "tab:correlation-by-encoder",
    },
    "correlation_by_projector_dataset": {
        "caption": "Mean absolute dimension correlation by representation and dataset.",
        "label": "tab:correlation-by-projector-dataset",
    },
    "correlation_by_projector_encoder": {
        "caption": (
            "Mean absolute dimension correlation by representation and foundation "
            "model."
        ),
        "label": "tab:correlation-by-projector-encoder",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise lowgen DGX dimension-correlation audit outputs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--float-format", default="%.4g")
    args = parser.parse_args()

    outputs = write_tables(args.input, args.output_dir, args.float_format)
    print(f"Wrote tables to {outputs['output_dir']}")
    for name, path in outputs.items():
        if name != "output_dir":
            print(f"{name}: {path}")


def write_tables(
    input_path: Path,
    output_dir: Path | None = None,
    float_format: str = "%.4g",
) -> dict[str, Path]:
    output_dir = output_dir or input_path.parent / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    valid, excluded = preprocess(raw)
    excluded.to_csv(output_dir / "excluded_rows.csv", index=False)

    target_level = seed_averaged_targets(valid)
    target_level.to_csv(output_dir / "seed_averaged_targets.csv", index=False)

    tables = build_summary_tables(valid, target_level)
    outputs: dict[str, Path] = {"output_dir": output_dir}
    latex_parts = []

    for name, table in tables.items():
        csv_path = output_dir / f"{name}.csv"
        tex_path = output_dir / f"{name}.tex"
        table.to_csv(csv_path, index=False, float_format=float_format)
        latex = table_to_latex(name, table, float_format)
        tex_path.write_text(latex)
        outputs[f"{name}_csv"] = csv_path
        outputs[f"{name}_tex"] = tex_path
        latex_parts.append(latex)

    combined = output_dir / "correlation_tables.tex"
    combined.write_text("\n\n".join(latex_parts) + "\n")
    outputs["combined_latex"] = combined
    outputs["excluded_rows"] = output_dir / "excluded_rows.csv"
    outputs["seed_averaged_targets"] = output_dir / "seed_averaged_targets.csv"
    return outputs


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["mean_abs_corr"] = pd.to_numeric(df["mean_abs_corr"], errors="coerce")
    valid_mask = np.isfinite(df["mean_abs_corr"])
    valid = df.loc[valid_mask].copy()
    excluded = df.loc[~valid_mask].copy()

    valid["projector"] = pd.Categorical(
        valid["projector"], categories=PROJECTOR_ORDER, ordered=True
    )
    valid["dimension_sort"] = pd.to_numeric(valid["dimension"], errors="coerce")
    valid["dimension"] = valid["dimension"].fillna("").astype(str)
    valid = valid.sort_values(
        ["projector", "dataset", "encoder", "target_name", "dimension_sort", "seed"],
        kind="mergesort",
    ).reset_index(drop=True)
    excluded = excluded.sort_values(
        ["projector", "dataset", "encoder", "target_name", "seed"],
        kind="mergesort",
    ).reset_index(drop=True)
    return valid, excluded


def seed_averaged_targets(valid: pd.DataFrame) -> pd.DataFrame:
    target = (
        valid.groupby(BASE_COLUMNS, observed=True, dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_abs_corr=("mean_abs_corr", "mean"),
        )
        .reset_index()
    )
    return sort_table(
        target, ["projector", "dataset", "encoder", "target_name", "dimension"]
    )


def build_summary_tables(
    valid: pd.DataFrame,
    target_level: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "correlation_by_projector": group_summary(target_level, ["projector"]),
        "correlation_reduction_effect": reduction_effect(valid),
        "correlation_by_dataset": group_summary(target_level, ["dataset"]),
        "correlation_by_encoder": group_summary(target_level, ["encoder"]),
        "correlation_by_projector_dataset": group_summary(
            target_level, ["projector", "dataset"]
        ),
        "correlation_by_projector_encoder": group_summary(
            target_level, ["projector", "encoder"]
        ),
    }


def group_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(columns, observed=True, dropna=False)
        .agg(
            n_targets=("mean_abs_corr", "count"),
            mean_abs_corr_mean=("mean_abs_corr", "mean"),
            mean_abs_corr_std=("mean_abs_corr", "std"),
        )
        .reset_index()
    )
    return sort_table(out, columns)


def reduction_effect(valid: pd.DataFrame) -> pd.DataFrame:
    raw = valid[valid["projector"].astype(str) == "raw_highdim"][
        ["dataset", "encoder", "seed", "mean_abs_corr"]
    ].rename(columns={"mean_abs_corr": "raw_corr"})
    reduced = valid[valid["projector"].astype(str).isin(["pca", "umap"])].copy()
    reduced = (
        reduced.groupby(["dataset", "encoder", "seed", "projector"], observed=True)
        .agg(reduced_corr=("mean_abs_corr", "mean"))
        .reset_index()
    )
    matched = reduced.merge(raw, on=["dataset", "encoder", "seed"], how="inner")
    matched["change"] = matched["reduced_corr"] - matched["raw_corr"]
    matched["ratio"] = matched["reduced_corr"] / matched["raw_corr"]
    out = (
        matched.groupby("projector", observed=True)
        .agg(
            n_matched=("change", "count"),
            raw_mean=("raw_corr", "mean"),
            reduced_mean=("reduced_corr", "mean"),
            mean_change=("change", "mean"),
            std_change=("change", "std"),
            mean_ratio=("ratio", "mean"),
            std_ratio=("ratio", "std"),
        )
        .reset_index()
    )
    return sort_table(out, ["projector"])


def sort_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    if "projector" in out.columns:
        out["projector"] = pd.Categorical(
            out["projector"].astype(str), categories=PROJECTOR_ORDER, ordered=True
        )
    sort_columns = [
        c
        for c in ["projector", "dataset", "encoder", "target_name", "dimension"]
        if c in out.columns
    ]
    if sort_columns:
        out = out.sort_values(sort_columns, kind="mergesort")
    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(str)
    return out.reset_index(drop=True)


def table_to_latex(name: str, table: pd.DataFrame, float_format: str) -> str:
    spec = TABLE_SPECS[name]
    return table.to_latex(
        index=False,
        na_rep="",
        float_format=lambda value: float_format % value,
        caption=spec["caption"],
        label=spec["label"],
        escape=True,
    )


if __name__ == "__main__":
    main()
