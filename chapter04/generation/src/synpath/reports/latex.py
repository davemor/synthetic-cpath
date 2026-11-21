from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from synpath.artifacts import write_text
from synpath.reports.aggregate import collect_quality, collect_utility


DATASETS = (
    ("pathmnist224", "Kather 100K"),
    ("icaird_cervical100k", "iCAIRD"),
    ("pcam", "PCam"),
)
ARCHITECTURES = (
    ("stylegan2ada", "SG2"),
    ("sd_lora", "SD"),
    ("dcgan", "DC"),
)
QUALITY_CONDITIONING = (
    ("unconditioned", "Unconditioned"),
    ("true_label", "Class label"),
    ("pseudo_label", "Global cluster"),
    ("label_global_cluster", "Class + global"),
    ("label_local_cluster", "Class + local"),
)
UTILITY_CONDITIONING = (
    ("true_label", "Class label"),
    ("label_global_cluster", "Class + global"),
    ("label_local_cluster", "Class + local"),
)


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    source_file: str
    decimals: int
    higher_is_better: bool


QUALITY_METRICS = (
    MetricSpec("fid", "inception_fid.json", 1, False),
    MetricSpec("uni_fid", "uni_fid_kid.json", 1, False),
    MetricSpec("precision", "uni_precision_recall.json", 4, True),
    MetricSpec("recall", "uni_precision_recall.json", 4, True),
)
UTILITY_METRICS = (
    MetricSpec("accuracy", "", 3, True),
    MetricSpec("macro_roc_auc_ovr", "", 3, True),
)

Summary = dict[tuple[str, str, str, str], tuple[float, float]]


def build_appendix_tables(
    outputs_root: Path,
    output_path: Path,
    quality_seeds: Iterable[int] = (0, 1, 2),
    classifier_seeds: Iterable[int] = (0, 1, 2),
) -> Path:
    quality = summarise_quality(collect_quality(outputs_root), quality_seeds)
    utility = summarise_utility(collect_utility(outputs_root), classifier_seeds)
    write_text(output_path, render_appendix_tables(quality, utility))
    return output_path


def summarise_quality(frame: pd.DataFrame, seeds: Iterable[int]) -> Summary:
    selected = []
    for spec in QUALITY_METRICS:
        rows = frame[
            (frame["metric"] == spec.metric)
            & (frame["source_file"] == spec.source_file)
        ].copy()
        rows["table_metric"] = spec.metric
        selected.append(rows)
    metrics = pd.concat(selected, ignore_index=True)
    expected = {
        (dataset, model, conditioning, spec.metric)
        for dataset, _ in DATASETS
        for model, _ in ARCHITECTURES
        for conditioning, _ in QUALITY_CONDITIONING
        for spec in QUALITY_METRICS
    }
    return _summarise(metrics, seeds, expected, "model")


def summarise_utility(frame: pd.DataFrame, seeds: Iterable[int]) -> Summary:
    source_to_conditioning = {
        f"synthetic_only_{conditioning}": f"synthetic_only:{conditioning}"
        for conditioning, _ in UTILITY_CONDITIONING
    }
    source_to_conditioning.update(
        {
            f"real_plus_synthetic_{conditioning}": (
                f"real_plus_synthetic:{conditioning}"
            )
            for conditioning, _ in UTILITY_CONDITIONING
        }
    )
    metrics = frame[
        frame["conditioning"].isin(source_to_conditioning)
        & frame["metric"].isin(spec.metric for spec in UTILITY_METRICS)
        & frame["generator_model"].isin(model for model, _ in ARCHITECTURES)
    ].copy()
    metrics["table_metric"] = metrics.apply(
        lambda row: (
            f"{source_to_conditioning[row['conditioning']]}:{row['metric']}"
        ),
        axis=1,
    )
    expected = {
        (
            dataset,
            model,
            conditioning,
            f"{source}:{conditioning}:{spec.metric}",
        )
        for dataset, _ in DATASETS
        for model, _ in ARCHITECTURES
        for conditioning, _ in UTILITY_CONDITIONING
        for source in ("synthetic_only", "real_plus_synthetic")
        for spec in UTILITY_METRICS
    }
    return _summarise(
        metrics,
        seeds,
        expected,
        "generator_model",
        conditioning_in_metric=True,
    )


def _summarise(
    frame: pd.DataFrame,
    seeds: Iterable[int],
    expected: set[tuple[str, str, str, str]],
    model_column: str,
    conditioning_in_metric: bool = False,
) -> Summary:
    requested_seeds = tuple(sorted({int(seed) for seed in seeds}))
    if len(requested_seeds) < 2:
        raise ValueError("At least two independent seeds are required for SD")
    data = frame.copy()
    data["seed"] = pd.to_numeric(data["seed"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data[data["seed"].isin(requested_seeds) & data["value"].notna()]

    values: dict[tuple[str, str, str, str], dict[int, float]] = {}
    for row in data.itertuples(index=False):
        table_metric = str(row.table_metric)
        if conditioning_in_metric:
            _, conditioning, _ = table_metric.split(":", maxsplit=2)
        else:
            conditioning = str(row.conditioning)
        key = (
            str(row.dataset),
            str(getattr(row, model_column)),
            conditioning,
            table_metric,
        )
        seed = int(row.seed)
        if seed in values.setdefault(key, {}):
            raise ValueError(f"Duplicate metric row for {key}, seed {seed}")
        values[key][seed] = float(row.value)

    missing = []
    requested = set(requested_seeds)
    for key in sorted(expected):
        absent = requested - set(values.get(key, {}))
        if absent:
            missing.append(f"{key}: seeds {sorted(absent)}")
    if missing:
        preview = "\n  - ".join(missing[:20])
        remainder = len(missing) - min(len(missing), 20)
        suffix = f"\n  ... and {remainder} more" if remainder else ""
        raise ValueError(
            "Cannot report mean +/- SD; required metric replicates are missing:\n"
            f"  - {preview}{suffix}"
        )

    result: Summary = {}
    for key in expected:
        series = pd.Series(
            [values[key][seed] for seed in requested_seeds], dtype=float
        )
        result[key] = (float(series.mean()), float(series.std(ddof=1)))
    return result


def render_appendix_tables(quality: Summary, utility: Summary) -> str:
    return "\n".join(
        [
            "% Requires: \\usepackage{booktabs,longtable,multirow}",
            "% Designed for an A4 portrait appendix.",
            "",
            _render_quality_table(quality),
            "",
            "\\clearpage",
            "",
            _render_utility_table(utility),
            "",
        ]
    )


def _render_quality_table(summary: Summary) -> str:
    lines = [
        "\\begingroup",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{longtable}{@{}lllrrrr@{}}",
        "\\toprule",
        "Arch. & Conditioning & Dataset & FID $\\downarrow$ & "
        "UNI-FID $\\downarrow$ & Precision $\\uparrow$ & Recall $\\uparrow$ \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Arch. & Conditioning & Dataset & FID $\\downarrow$ & "
        "UNI-FID $\\downarrow$ & Precision $\\uparrow$ & Recall $\\uparrow$ \\\\",
        "\\midrule",
        "\\endhead",
    ]
    metrics = tuple(spec.metric for spec in QUALITY_METRICS)
    rows_per_architecture = len(QUALITY_CONDITIONING) * len(DATASETS)
    for architecture_index, (model, abbreviation) in enumerate(ARCHITECTURES):
        for conditioning_index, (conditioning, label) in enumerate(
            QUALITY_CONDITIONING
        ):
            for dataset_index, (dataset, dataset_label) in enumerate(DATASETS):
                architecture_cell = (
                    f"\\multirow{{{rows_per_architecture}}}{{*}}{{{abbreviation}}}"
                    if conditioning_index == 0 and dataset_index == 0
                    else ""
                )
                conditioning_cell = (
                    f"\\multirow{{{len(DATASETS)}}}{{*}}{{{label}}}"
                    if dataset_index == 0
                    else ""
                )
                cells = [architecture_cell, conditioning_cell, dataset_label]
                for metric in metrics:
                    cells.append(
                        _quality_cell(
                            summary,
                            dataset,
                            model,
                            conditioning,
                            metric,
                        )
                    )
                lines.append(" & ".join(cells) + " \\\\")
            if conditioning_index < len(QUALITY_CONDITIONING) - 1:
                lines.append("\\cmidrule(lr){2-7}")
        if architecture_index < len(ARCHITECTURES) - 1:
            lines.append("\\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\caption{Dataset-specific synthetic-patch quality in the main "
            "experiment. Values are mean $\\pm$ sample standard deviation over "
            "independent generator seeds 0, 1 and 2. SG2 denotes StyleGAN2-ADA, "
            "SD denotes Stable Diffusion v1.5 with LoRA, and DC denotes DCGAN. "
            "Precision and recall are the corrected per-sample improved "
            "$k$-nearest-neighbour estimates in UNI feature space. Bold denotes "
            "the best mean within each architecture--dataset block; all-zero "
            "ties are not emphasised.}",
            "\\label{tab:appendix-quality-by-dataset}\\\\",
            "\\end{longtable}",
            "\\endgroup",
        ]
    )
    return "\n".join(lines)


def _render_utility_table(summary: Summary) -> str:
    lines = [
        "\\begingroup",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{longtable}{@{}lllrrrr@{}}",
        "\\toprule",
        "& & & \\multicolumn{2}{c}{Synthetic only} & "
        "\\multicolumn{2}{c}{Real + synthetic} \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(l){6-7}",
        "Arch. & Conditioning & Dataset & Acc. $\\uparrow$ & ROC-AUC "
        "$\\uparrow$ & Acc. $\\uparrow$ & ROC-AUC $\\uparrow$ \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "& & & \\multicolumn{2}{c}{Synthetic only} & "
        "\\multicolumn{2}{c}{Real + synthetic} \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(l){6-7}",
        "Arch. & Conditioning & Dataset & Acc. $\\uparrow$ & ROC-AUC "
        "$\\uparrow$ & Acc. $\\uparrow$ & ROC-AUC $\\uparrow$ \\\\",
        "\\midrule",
        "\\endhead",
    ]
    rows_per_architecture = len(UTILITY_CONDITIONING) * len(DATASETS)
    for architecture_index, (model, abbreviation) in enumerate(ARCHITECTURES):
        for conditioning_index, (conditioning, label) in enumerate(
            UTILITY_CONDITIONING
        ):
            for dataset_index, (dataset, dataset_label) in enumerate(DATASETS):
                architecture_cell = (
                    f"\\multirow{{{rows_per_architecture}}}{{*}}{{{abbreviation}}}"
                    if conditioning_index == 0 and dataset_index == 0
                    else ""
                )
                conditioning_cell = (
                    f"\\multirow{{{len(DATASETS)}}}{{*}}{{{label}}}"
                    if dataset_index == 0
                    else ""
                )
                cells = [architecture_cell, conditioning_cell, dataset_label]
                for source in ("synthetic_only", "real_plus_synthetic"):
                    for spec in UTILITY_METRICS:
                        metric = f"{source}:{conditioning}:{spec.metric}"
                        cells.append(
                            _utility_cell(
                                summary,
                                dataset,
                                model,
                                conditioning,
                                metric,
                            )
                        )
                lines.append(" & ".join(cells) + " \\\\")
            if conditioning_index < len(UTILITY_CONDITIONING) - 1:
                lines.append("\\cmidrule(lr){2-7}")
        if architecture_index < len(ARCHITECTURES) - 1:
            lines.append("\\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\caption{Dataset-specific downstream ResNet-18 patch-classification "
            "utility. Values are mean $\\pm$ sample standard deviation over "
            "classifier seeds 0, 1 and 2; evaluation always used the real test "
            "split. SG2 denotes StyleGAN2-ADA, SD denotes Stable Diffusion v1.5 "
            "with LoRA, and DC denotes DCGAN. Bold denotes the highest mean "
            "within each architecture--dataset block.}",
            "\\label{tab:appendix-utility-by-dataset}\\\\",
            "\\end{longtable}",
            "\\endgroup",
        ]
    )
    return "\n".join(lines)


def _quality_cell(
    summary: Summary,
    dataset: str,
    model: str,
    conditioning: str,
    metric: str,
) -> str:
    spec = next(item for item in QUALITY_METRICS if item.metric == metric)
    return _metric_cell(
        summary,
        (dataset, model, conditioning, metric),
        spec,
        QUALITY_CONDITIONING,
    )


def _utility_cell(
    summary: Summary,
    dataset: str,
    model: str,
    conditioning: str,
    metric: str,
) -> str:
    source, _, base_metric = metric.split(":", maxsplit=2)
    spec = next(item for item in UTILITY_METRICS if item.metric == base_metric)
    key = (dataset, model, conditioning, metric)
    candidates = [
        summary[
            (
                dataset,
                model,
                candidate,
                f"{source}:{candidate}:{base_metric}",
            )
        ][0]
        for candidate, _ in UTILITY_CONDITIONING
    ]
    return _format_metric_cell(summary[key], spec, candidates)


def _metric_cell(
    summary: Summary,
    key: tuple[str, str, str, str],
    spec: MetricSpec,
    conditioning: tuple[tuple[str, str], ...],
) -> str:
    candidates = [
        summary[(key[0], key[1], condition, key[3])][0]
        for condition, _ in conditioning
    ]
    return _format_metric_cell(summary[key], spec, candidates)


def _format_metric_cell(
    value: tuple[float, float], spec: MetricSpec, candidates: list[float]
) -> str:
    mean, std = value
    format_string = f"{{:.{spec.decimals}f}}"
    content = f"{format_string.format(mean)} \\pm {format_string.format(std)}"
    target = max(candidates) if spec.higher_is_better else min(candidates)
    if any(value != 0 for value in candidates) and mean == target:
        content = f"\\mathbf{{{content}}}"
    return f"${content}$"
