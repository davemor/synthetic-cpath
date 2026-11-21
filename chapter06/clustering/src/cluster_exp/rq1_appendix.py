from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .dimension_plots import select_best_dimension_configs
from .table1 import (
    METHOD_LABELS,
    METHOD_ORDER,
    PROJECTOR_LABELS,
    PROJECTOR_ORDER,
    RANK_COLUMNS,
    TABLE_METRICS,
)


EXTERNAL_METRICS = [
    ("ami", "AMI", "higher"),
    ("ari", "ARI", "higher"),
    ("v_measure", "V-measure", "higher"),
]

INTERNAL_METRICS = [
    ("silhouette", "Silhouette", "higher"),
    ("sd_index", "$\\log_{10}$(SD-index)", "lower"),
    ("stability", "Stability", "higher"),
    ("noise", "Noise", "lower"),
]

DATASET_ORDER = [
    "icaird100k",
    "kather100k",
    "spider-colorectal",
    "spider-breast",
    "spider-skin",
    "spider-thorax",
]

ENCODER_ORDER = ["uni", "uni2", "hibou", "dinov2"]

METRIC_SOURCE_COLUMNS = {metric_key: source_col for metric_key, _, source_col, _ in TABLE_METRICS}


@dataclass(frozen=True)
class RQ1AppendixOutputs:
    external_csv_path: Path
    external_latex_path: Path
    internal_csv_path: Path
    internal_latex_path: Path


def generate_rq1_appendix_dimension_tables(
    run_dir: Path,
    stability_csv: Path | None = None,
    min_dimension: int = 1,
    max_dimension: int = 50,
) -> RQ1AppendixOutputs:
    run_dir = Path(run_dir)
    stability_csv = stability_csv or run_dir / "clustering_stability.csv"
    stability = pd.read_csv(stability_csv)

    summary = summarise_rq1_dimensions(stability, min_dimension=min_dimension, max_dimension=max_dimension)
    dimension_slug = f"{min_dimension}-{max_dimension}"
    dimension_text = f"{min_dimension}--{max_dimension}"

    external_csv = run_dir / f"clustering-results-appendix-rq1-external-dimensions-{dimension_slug}-table.csv"
    external_tex = run_dir / f"clustering-results-appendix-rq1-external-dimensions-{dimension_slug}-table.tex"
    internal_csv = run_dir / f"clustering-results-appendix-rq1-internal-dimensions-{dimension_slug}-table.csv"
    internal_tex = run_dir / f"clustering-results-appendix-rq1-internal-dimensions-{dimension_slug}-table.tex"

    external_csv.parent.mkdir(parents=True, exist_ok=True)
    _csv_view(summary, EXTERNAL_METRICS).to_csv(external_csv, index=False)
    _csv_view(summary, INTERNAL_METRICS).to_csv(internal_csv, index=False)
    external_tex.write_text(
        to_latex_rq1_appendix_table(
            summary,
            EXTERNAL_METRICS,
            caption=(
                f"External label-alignment metrics over projection dimensions {dimension_text} for RQ1. "
                f"Projected rows report mean $\\pm$ standard deviation across the "
                f"{max_dimension - min_dimension + 1} dimensions; "
                "at each dimension, the clustering hyperparameter setting was selected using the "
                "pre-specified ranking rule. Raw rows are full-embedding baselines and therefore "
                "do not have dimensional standard deviations. AMI, ARI, and V-measure use human "
                "pathology labels for evaluation only."
            ),
            label="tab:clustering-results-appendix-rq1-external-dimensions",
        )
    )
    internal_tex.write_text(
        to_latex_rq1_appendix_table(
            summary,
            INTERNAL_METRICS,
            caption=(
                "Internal validation, cluster-structure, and stability metrics over projection "
                f"dimensions {dimension_text} for RQ1. Projected rows report mean $\\pm$ standard deviation "
                f"across the {max_dimension - min_dimension + 1} dimensions; at each dimension, the clustering hyperparameter "
                "setting was selected using the pre-specified ranking rule. Raw rows are "
                "full-embedding baselines and therefore do not have dimensional standard "
                "deviations. SD-index is reported on a $\\log_{10}$ scale because raw values "
                "span several orders of magnitude; lower values remain better. Noise is reported only for HDBSCAN."
            ),
            label="tab:clustering-results-appendix-rq1-internal-dimensions",
        )
    )
    return RQ1AppendixOutputs(
        external_csv_path=external_csv,
        external_latex_path=external_tex,
        internal_csv_path=internal_csv,
        internal_latex_path=internal_tex,
    )


def summarise_rq1_dimensions(
    stability: pd.DataFrame,
    min_dimension: int = 1,
    max_dimension: int = 50,
) -> pd.DataFrame:
    projected = []
    for projector in ["pca", "umap"]:
        projected.append(
            select_best_dimension_configs(
                stability,
                projector=projector,
                min_dimension=min_dimension,
                max_dimension=max_dimension,
            )
        )
    selected_projected = pd.concat(projected, ignore_index=True)
    selected_raw = _select_best_raw_configs(stability)

    rows: list[dict[str, object]] = []
    dimension_set = set(range(min_dimension, max_dimension + 1))
    group_cols = ["dataset", "encoder", "projector", "clusterer"]
    for (dataset, encoder, projector, clusterer), group in selected_projected.groupby(
        group_cols, dropna=False
    ):
        dimensions = set(pd.to_numeric(group["dimension"], errors="coerce").dropna().astype(int))
        if dimensions != dimension_set:
            missing = sorted(dimension_set - dimensions)
            extra = sorted(dimensions - dimension_set)
            raise ValueError(
                "Projected RQ1 appendix row does not have the expected dimensions "
                f"{min_dimension}..{max_dimension}: dataset={dataset}, encoder={encoder}, "
                f"projector={projector}, clusterer={clusterer}, missing={missing}, extra={extra}."
            )
        row = _base_row(dataset, encoder, projector, clusterer, dims=f"{min_dimension}-{max_dimension}")
        row["dimension_count"] = len(dimensions)
        for metric_key, _, _ in EXTERNAL_METRICS + INTERNAL_METRICS:
            values = _metric_values(group, metric_key)
            row[f"{metric_key}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric_key}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)

    for _, row_raw in selected_raw.iterrows():
        row = _base_row(
            row_raw["dataset"],
            row_raw["encoder"],
            row_raw["projector"],
            row_raw["clusterer"],
            dims="Full",
        )
        row["dimension_count"] = np.nan
        for metric_key, _, _ in EXTERNAL_METRICS + INTERNAL_METRICS:
            values = _metric_values(pd.DataFrame([row_raw]), metric_key)
            row[f"{metric_key}_mean"] = float(values.iloc[0]) if len(values) else np.nan
            row[f"{metric_key}_sd"] = np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    return _sort_summary(summary)


def to_latex_rq1_appendix_table(
    summary: pd.DataFrame,
    metrics: list[tuple[str, str, str]],
    *,
    caption: str,
    label: str,
) -> str:
    nested_headers = ["Projector", "Clustering", "Dims"] + [
        f"{name} {_metric_arrow(direction)}" for _, name, direction in metrics
    ]
    nested_column_spec = "lll" + ("r" * len(metrics))
    nested_column_count = len(nested_headers)
    lines = [
        "\\begingroup",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\refstepcounter{table}",
        f"\\noindent\\textbf{{Table \\thetable.}} {caption}\\label{{{label}}}\\par",
        "\\vspace{0.75em}",
    ]

    blocks = list(summary.groupby(["dataset", "encoder"], sort=False))
    for block_index, ((dataset, encoder), block) in enumerate(blocks):
        block = block.sort_values(["__projector_order", "__method_order"])
        best_rows = _best_rows_for_block(block, metrics)
        nested = _nested_results_table(
            block,
            metrics,
            best_rows=best_rows,
            nested_headers=nested_headers,
            nested_column_spec=nested_column_spec,
            nested_column_count=nested_column_count,
        )
        lines.append("\\Needspace{16\\baselineskip}")
        lines.append(
            "\\textbf{"
            + _latex_escape(str(dataset))
            + " / "
            + _latex_escape(str(encoder))
            + "}\\par\\vspace{2pt}"
        )
        lines.append("\\noindent" + nested + "\\par")
        if block_index < len(blocks) - 1:
            lines.append("\\vspace{1.0em}")
            lines.append("")
    lines.extend(["\\endgroup", ""])
    return "\n".join(lines)


def _nested_results_table(
    block: pd.DataFrame,
    metrics: list[tuple[str, str, str]],
    *,
    best_rows: dict[str, int],
    nested_headers: list[str],
    nested_column_spec: str,
    nested_column_count: int,
) -> str:
    lines = [
        "\\begin{tabular}[t]{" + nested_column_spec + "}",
        "\\toprule",
        " & ".join(nested_headers) + " \\\\",
        "\\midrule",
    ]
    projector_blocks = list(block.groupby("projector", sort=False))
    for projector_block_index, (_, projector_block) in enumerate(projector_blocks):
        projector_span = len(projector_block)
        first_projector = True
        for row_index, row in projector_block.iterrows():
            cells = []
            if first_projector:
                cells.append(f"\\multirow{{{projector_span}}}{{*}}{{{_latex_escape(str(row['projection']))}}}")
                first_projector = False
            else:
                cells.append("")
            cells.extend(
                [
                    _latex_escape(str(row["method"])),
                    _latex_escape(str(row["dims"])),
                ]
            )
            for metric_key, _, _ in metrics:
                if metric_key == "noise" and str(row["clusterer"]) != "hdbscan":
                    cells.append("-")
                else:
                    value = _format_metric(row[f"{metric_key}_mean"], row[f"{metric_key}_sd"], metric_key)
                    if best_rows.get(metric_key) == row_index:
                        value = f"\\textbf{{{value}}}"
                    cells.append(value)
            lines.append(" & ".join(cells) + " \\\\")
        if projector_block_index < len(projector_blocks) - 1:
            lines.append(f"\\cmidrule(lr){{1-{nested_column_count}}}")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _select_best_raw_configs(stability: pd.DataFrame) -> pd.DataFrame:
    required = ["dataset", "encoder", "clusterer", "projector"]
    missing = [column for column in required if column not in stability.columns]
    if missing:
        raise ValueError(f"Missing required stability columns: {', '.join(missing)}")

    frame = stability.copy()
    frame = frame[frame["clusterer"].isin(METHOD_ORDER)]
    frame = frame[frame["projector"] == "raw"]
    if frame.empty:
        return frame

    for column, ascending, fill_value in RANK_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[f"__sort_{column}"] = frame[column].fillna(fill_value)

    group_cols = ["dataset", "encoder", "clusterer", "projector"]
    sort_cols = group_cols + [f"__sort_{column}" for column, _, _ in RANK_COLUMNS]
    ascending = [True] * len(group_cols) + [item[1] for item in RANK_COLUMNS]
    selected = frame.sort_values(sort_cols, ascending=ascending).groupby(
        group_cols, dropna=False
    ).head(1)
    return selected.drop(columns=[col for col in selected.columns if col.startswith("__sort_")])


def _base_row(
    dataset: object,
    encoder: object,
    projector: object,
    clusterer: object,
    *,
    dims: str,
) -> dict[str, object]:
    projector_text = str(projector)
    clusterer_text = str(clusterer)
    return {
        "dataset": str(dataset),
        "encoder": str(encoder),
        "projector": projector_text,
        "projection": PROJECTOR_LABELS.get(projector_text, projector_text),
        "clusterer": clusterer_text,
        "method": METHOD_LABELS.get(clusterer_text, clusterer_text),
        "dims": dims,
    }


def _metric_values(group: pd.DataFrame, metric_key: str) -> pd.Series:
    source_col = METRIC_SOURCE_COLUMNS[metric_key]
    if source_col not in group.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(group[source_col], errors="coerce").dropna()
    if metric_key == "sd_index":
        values = values[values > 0]
        return np.log10(values)
    return values


def _csv_view(summary: pd.DataFrame, metrics: list[tuple[str, str, str]]) -> pd.DataFrame:
    columns = [
        "dataset",
        "encoder",
        "projector",
        "projection",
        "clusterer",
        "method",
        "dims",
        "dimension_count",
    ]
    for metric_key, _, _ in metrics:
        columns.extend([f"{metric_key}_mean", f"{metric_key}_sd"])
    view = summary[columns].copy()
    if "noise_mean" in view.columns:
        view["noise_mean"] = view["noise_mean"].astype(object)
        view["noise_sd"] = view["noise_sd"].astype(object)
        non_hdbscan = view["clusterer"] != "hdbscan"
        view.loc[non_hdbscan, ["noise_mean", "noise_sd"]] = "-"
    return view


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    dataset_order = _order_map(DATASET_ORDER)
    encoder_order = _order_map(ENCODER_ORDER)
    projector_order = _order_map(PROJECTOR_ORDER)
    method_order = _order_map(METHOD_ORDER)
    summary["__dataset_order"] = summary["dataset"].map(dataset_order).fillna(len(dataset_order))
    summary["__encoder_order"] = summary["encoder"].map(encoder_order).fillna(len(encoder_order))
    summary["__projector_order"] = summary["projector"].map(projector_order).fillna(len(projector_order))
    summary["__method_order"] = summary["clusterer"].map(method_order).fillna(len(method_order))
    return summary.sort_values(
        [
            "__dataset_order",
            "dataset",
            "__encoder_order",
            "encoder",
            "__projector_order",
            "__method_order",
        ]
    ).reset_index(drop=True)


def _format_metric(mean: object, sd: object, metric_key: str) -> str:
    if pd.isna(mean):
        return "-"
    if metric_key == "clusters":
        mean_text = f"{float(mean):.1f}"
        sd_text = f"{float(sd):.1f}" if not pd.isna(sd) else None
    else:
        mean_text = f"{float(mean):.3f}"
        sd_text = f"{float(sd):.3f}" if not pd.isna(sd) else None
    return mean_text if sd_text is None else f"{mean_text} $\\pm$ {sd_text}"


def _best_rows_for_block(
    block: pd.DataFrame,
    metrics: list[tuple[str, str, str]],
) -> dict[str, int]:
    best: dict[str, int] = {}
    for metric_key, _, direction in metrics:
        if direction == "none":
            continue
        values = pd.to_numeric(block[f"{metric_key}_mean"], errors="coerce")
        if metric_key == "noise":
            values = values.where(block["clusterer"] == "hdbscan")
        values = values.dropna()
        if values.empty:
            continue
        best[metric_key] = int(values.idxmin() if direction == "lower" else values.idxmax())
    return best


def _metric_arrow(direction: str) -> str:
    if direction == "higher":
        return "$\\uparrow$"
    if direction == "lower":
        return "$\\downarrow$"
    return ""


def _order_map(values: list[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(values)}


def _latex_escape(value: str) -> str:
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
    return "".join(replacements.get(char, char) for char in value)
