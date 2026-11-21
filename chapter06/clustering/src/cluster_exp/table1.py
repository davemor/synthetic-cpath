from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "kmeans": "K-Means",
    "spherical_kmeans": "Spherical K-Means",
    "hdbscan": "HDBSCAN",
    "leiden": "Leiden",
}

PROJECTOR_LABELS = {
    "raw": "Raw",
    "pca": "PCA",
    "umap": "UMAP",
}

TABLE_METRICS = [
    ("ami", "AMI", "mean_label_ami", "higher"),
    ("ari", "ARI", "mean_label_ari", "higher"),
    ("v_measure", "V-measure", "mean_label_v_measure", "higher"),
    ("silhouette", "Silhouette", "mean_silhouette", "higher"),
    ("sd_index", "SD-index", "mean_sd_index", "lower"),
    ("stability", "Stability", "mean_pairwise_ari", "higher"),
    ("clusters", "Clusters", "mean_n_clusters", "none"),
    ("noise", "Noise", "mean_noise_fraction", "lower"),
]

EXTERNAL_TABLE_METRICS = [
    ("ami", "AMI", "higher"),
    ("ari", "ARI", "higher"),
    ("v_measure", "V-measure", "higher"),
]

STRUCTURE_TABLE_METRICS = [
    ("silhouette", "Silhouette", "higher"),
    ("sd_index", "SD-index", "lower"),
    ("stability", "Stability", "higher"),
    ("clusters", "Clusters", "none"),
    ("noise", "Noise", "lower"),
]

RANK_COLUMNS = [
    ("mean_label_ami", False, -np.inf),
    ("mean_label_v_measure", False, -np.inf),
    ("mean_pairwise_ari", False, -np.inf),
    ("mean_silhouette", False, -np.inf),
    ("mean_sd_index", True, np.inf),
    ("mean_noise_fraction", True, np.inf),
    ("mean_runtime_seconds", True, np.inf),
]

METHOD_ORDER = ["kmeans", "spherical_kmeans", "hdbscan", "leiden"]
PROJECTOR_ORDER = ["raw", "pca", "umap"]


@dataclass(frozen=True)
class TableOneOutputs:
    csv_path: Path
    external_csv_path: Path
    structure_csv_path: Path
    latex_path: Path
    external_latex_path: Path
    structure_latex_path: Path


def generate_table_one(
    run_dir: Path,
    output_csv: Path | None = None,
    output_tex: Path | None = None,
    stability_csv: Path | None = None,
    caption: str | None = None,
    label: str = "tab:clustering_method_projection",
    bold_best: bool = True,
) -> TableOneOutputs:
    """Generate the compact method/projection summary table.

    For each dataset/encoder/clusterer/projector, the best configuration is
    selected lexicographically using label alignment first, then stability,
    internal validation, noise, and runtime. The table then reports mean and
    sample standard deviation over the selected dataset-encoder rows.
    """

    run_dir = Path(run_dir)
    stability_csv = stability_csv or run_dir / "clustering_stability.csv"
    output_csv = output_csv or run_dir / "clustering_results_method_projection_summary.csv"
    output_tex = output_tex or run_dir / "clustering_results_method_projection_summary.tex"
    external_csv = _suffixed_path(output_csv, "external_information")
    structure_csv = _suffixed_path(output_csv, "cluster_structure_stability")
    external_tex = _suffixed_path(output_tex, "external_information")
    structure_tex = _suffixed_path(output_tex, "cluster_structure_stability")

    stability = pd.read_csv(stability_csv)
    selected = select_best_configs(stability)
    summary = summarise_method_projection(selected)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    external_csv.parent.mkdir(parents=True, exist_ok=True)
    structure_csv.parent.mkdir(parents=True, exist_ok=True)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    external_tex.parent.mkdir(parents=True, exist_ok=True)
    structure_tex.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    _csv_view(summary, EXTERNAL_TABLE_METRICS).to_csv(external_csv, index=False)
    _csv_view(summary, STRUCTURE_TABLE_METRICS).to_csv(structure_csv, index=False)
    external_latex = to_latex_external_table(
        summary,
        caption=caption,
        label=label,
        bold_best=bold_best,
    )
    structure_latex = to_latex_structure_table(
        summary,
        external_label=label,
        label=f"{label}_structure_stability",
        bold_best=bold_best,
    )
    external_tex.write_text(external_latex)
    structure_tex.write_text(structure_latex)
    output_tex.write_text("\n".join([external_latex, structure_latex]))
    return TableOneOutputs(
        csv_path=output_csv,
        external_csv_path=external_csv,
        structure_csv_path=structure_csv,
        latex_path=output_tex,
        external_latex_path=external_tex,
        structure_latex_path=structure_tex,
    )


def select_best_configs(stability: pd.DataFrame) -> pd.DataFrame:
    _require_columns(stability, ["dataset", "encoder", "clusterer", "projector"])
    frame = stability.copy()
    frame = frame[frame["clusterer"].isin(METHOD_ORDER)]
    frame = frame[frame["projector"].isin(PROJECTOR_ORDER)]
    if frame.empty:
        raise ValueError("No supported clusterer/projector rows found in stability results.")

    for column, ascending, fill_value in RANK_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[f"__sort_{column}"] = frame[column].fillna(fill_value)

    sort_cols = ["dataset", "encoder", "clusterer", "projector"] + [
        f"__sort_{column}" for column, _, _ in RANK_COLUMNS
    ]
    ascending = [True, True, True, True] + [item[1] for item in RANK_COLUMNS]
    frame = frame.sort_values(sort_cols, ascending=ascending)
    selected = frame.groupby(["dataset", "encoder", "clusterer", "projector"], dropna=False).head(1)
    return selected.drop(columns=[col for col in selected.columns if col.startswith("__sort_")])


def summarise_method_projection(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (clusterer, projector), group in selected.groupby(["clusterer", "projector"], dropna=False):
        row: dict[str, object] = {
            "clusterer": clusterer,
            "method": METHOD_LABELS.get(str(clusterer), str(clusterer)),
            "projector": projector,
            "projection": PROJECTOR_LABELS.get(str(projector), str(projector)),
            "method_projection": (
                f"{METHOD_LABELS.get(str(clusterer), str(clusterer))} + "
                f"{PROJECTOR_LABELS.get(str(projector), str(projector))}"
            ),
            "n": int(len(group)),
        }
        for metric_key, _, source_col, _ in TABLE_METRICS:
            if source_col not in group.columns:
                values = pd.Series(dtype=float)
            else:
                values = pd.to_numeric(group[source_col], errors="coerce").dropna()
            row[f"{metric_key}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric_key}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["__method_order"] = summary["clusterer"].map(_order_map(METHOD_ORDER))
    summary["__projector_order"] = summary["projector"].map(_order_map(PROJECTOR_ORDER))
    summary = summary.sort_values(["__method_order", "__projector_order"]).drop(
        columns=["__method_order", "__projector_order"]
    )
    return summary.reset_index(drop=True)


def to_latex_table(
    summary: pd.DataFrame,
    caption: str | None,
    label: str,
    bold_best: bool = True,
) -> str:
    return "\n".join(
        [
            to_latex_external_table(summary, caption=caption, label=label, bold_best=bold_best),
            to_latex_structure_table(
                summary,
                external_label=label,
                label=f"{label}_structure_stability",
                bold_best=bold_best,
            ),
        ]
    )


def to_latex_external_table(
    summary: pd.DataFrame,
    caption: str | None,
    label: str,
    bold_best: bool = True,
) -> str:
    caption = caption or (
        "External label-alignment metrics for the selected clustering configurations. "
        "For each dataset, encoder, clustering method, and projection, the best configuration "
        "was selected using the pre-specified ranking rule. Values are mean plus/minus "
        "standard deviation over dataset-encoder pairs. AMI, ARI, and V-measure use the "
        "human pathology annotations for evaluation only; the clustering itself was unsupervised."
    )
    return _latex_metric_table(
        summary,
        EXTERNAL_TABLE_METRICS,
        caption=caption,
        label=label,
        bold_best=bold_best,
    )


def to_latex_structure_table(
    summary: pd.DataFrame,
    external_label: str,
    label: str,
    bold_best: bool = True,
) -> str:
    caption = (
        "Cluster-structure and stability metrics for the same selected configurations as the "
        "external label-alignment table. Silhouette and SD-index are internal validation indices, stability is the "
        "mean pairwise ARI between cluster assignments across random seeds, clusters is the "
        "mean number of clusters found, and noise is reported only for HDBSCAN."
    )
    return _latex_metric_table(
        summary,
        STRUCTURE_TABLE_METRICS,
        caption=caption,
        label=label,
        bold_best=bold_best,
    )


def _latex_metric_table(
    summary: pd.DataFrame,
    metrics: list[tuple[str, str, str]],
    *,
    caption: str,
    label: str,
    bold_best: bool,
) -> str:
    best_rows = _best_rows(summary, metrics) if bold_best else {}
    column_count = 1 + len(metrics)
    column_spec = "l" + ("r" * len(metrics))
    headers = ["Projection"] + [
        f"{name} {_metric_arrow(direction)}" for _, name, direction in metrics
    ]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for _, method_group in summary.groupby("clusterer", sort=False):
        method_group = method_group.sort_values(
            "projector", key=lambda col: col.map(_order_map(PROJECTOR_ORDER))
        )
        method_name = _latex_escape(str(method_group.iloc[0]["method"]))
        lines.append(f"\\multicolumn{{{column_count}}}{{l}}{{\\textbf{{{method_name}}}}} \\\\")
        lines.append("\\midrule")
        for index, row in method_group.iterrows():
            cells = [_latex_escape(str(row["projection"]))]
            for metric_key, _, _ in metrics:
                value = _format_metric(row[f"{metric_key}_mean"], row[f"{metric_key}_sd"], metric_key)
                if metric_key == "noise" and str(row["clusterer"]) != "hdbscan":
                    value = "-"
                elif best_rows.get(metric_key) == index:
                    value = f"\\textbf{{{value}}}"
                cells.append(value)
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    if lines[-1] == "\\addlinespace":
        lines.pop()
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


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


def _csv_view(summary: pd.DataFrame, metrics: list[tuple[str, str, str]]) -> pd.DataFrame:
    columns = ["method", "projection"]
    for metric_key, _, _ in metrics:
        columns.extend([f"{metric_key}_mean", f"{metric_key}_sd"])
    return summary[columns].copy()


def _suffixed_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _best_rows(
    summary: pd.DataFrame,
    metrics: list[tuple[str, str, str]] | None = None,
) -> dict[str, int]:
    best: dict[str, int] = {}
    metrics = metrics or [(key, label, direction) for key, label, _, direction in TABLE_METRICS]
    for metric_key, _, direction in metrics:
        if direction == "none":
            continue
        values = pd.to_numeric(summary[f"{metric_key}_mean"], errors="coerce")
        if metric_key == "noise":
            values = values.where(summary["clusterer"] == "hdbscan")
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


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required stability columns: {', '.join(missing)}")


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
