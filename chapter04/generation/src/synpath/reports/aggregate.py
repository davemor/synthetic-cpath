from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from synpath.artifacts import write_text
from synpath.paths import SynpathPaths


def build_report(paths: SynpathPaths, cfg: dict) -> list[Path]:
    name = cfg.get("name", "experiment")
    root = paths.reports_root / name
    root.mkdir(parents=True, exist_ok=True)

    quality = collect_quality(paths.outputs_root)
    utility = collect_utility(paths.outputs_root)
    quality_path = root / "quality_metrics.csv"
    utility_path = root / "utility_metrics.csv"
    results_path = root / "results.csv"
    report_path = root / "report.md"
    quality.to_csv(quality_path, index=False)
    utility.to_csv(utility_path, index=False)
    pd.concat([quality, utility], ignore_index=True).to_csv(
        results_path, index=False
    )
    write_text(report_path, markdown_report(cfg, quality, utility))
    return [results_path, quality_path, utility_path, report_path]


def collect_quality(outputs_root: Path) -> pd.DataFrame:
    rows = []
    for path in outputs_root.glob("generators/*/*/*/seed*/metrics/*.json"):
        if path.name in {
            "uni_precision_recall_legacy_global_radius.json",
            "uni_cluster_coverage_legacy_requested_ids.json",
        }:
            continue
        parts = path.relative_to(outputs_root).parts
        _, dataset, model, conditioning, seed_name, _, metric_file = parts
        payload = load_json(path)
        for metric, value in payload.items():
            if isinstance(value, dict):
                for submetric, subvalue in value.items():
                    rows.append(
                        metric_row(
                            "quality",
                            dataset,
                            model,
                            conditioning,
                            seed_name,
                            f"{metric}.{submetric}",
                            subvalue,
                            metric_file,
                        )
                    )
            else:
                rows.append(
                    metric_row(
                        "quality",
                        dataset,
                        model,
                        conditioning,
                        seed_name,
                        metric,
                        value,
                        metric_file,
                    )
                )
    return pd.DataFrame(rows)


def collect_utility(outputs_root: Path) -> pd.DataFrame:
    rows = []
    for path in outputs_root.glob("utility/*/resnet18/*/seed*/metrics.json"):
        parts = path.relative_to(outputs_root).parts
        _, dataset, model, source, seed_name, _ = parts
        payload = load_json(path)
        for metric, value in payload.items():
            rows.append(
                {
                    "family": "utility",
                    "dataset": dataset,
                    "model": model,
                    "generator_model": "real"
                    if source.startswith("real_only")
                    else "unknown_legacy",
                    "conditioning": source,
                    "seed": seed_name.replace("seed", ""),
                    "metric": metric,
                    "value": value,
                    "source_file": str(path),
                }
            )
    for path in outputs_root.glob("utility/*/resnet18/*/*/seed*/metrics.json"):
        parts = path.relative_to(outputs_root).parts
        _, dataset, model, generator_model, source, seed_name, _ = parts
        payload = load_json(path)
        for metric, value in payload.items():
            rows.append(
                {
                    "family": "utility",
                    "dataset": dataset,
                    "model": model,
                    "generator_model": generator_model,
                    "conditioning": source,
                    "seed": seed_name.replace("seed", ""),
                    "metric": metric,
                    "value": value,
                    "source_file": str(path),
                }
            )
    return pd.DataFrame(rows)


def metric_row(
    family: str,
    dataset: str,
    model: str,
    conditioning: str,
    seed_name: str,
    metric: str,
    value,
    source_file: str,
) -> dict:
    return {
        "family": family,
        "dataset": dataset,
        "model": model,
        "conditioning": conditioning,
        "seed": seed_name.replace("seed", ""),
        "metric": metric,
        "value": value,
        "source_file": source_file,
    }


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def markdown_report(
    cfg: dict, quality: pd.DataFrame, utility: pd.DataFrame
) -> str:
    lines = [
        f"# synpath report: {cfg.get('name', 'experiment')}",
        "",
        "## Completed Rows",
        "",
        f"- Quality metric rows: {len(quality)}",
        f"- Utility metric rows: {len(utility)}",
        "",
    ]
    if not quality.empty:
        lines.extend(
            [
                "## Quality Metrics",
                "",
                quality.head(20).to_markdown(index=False),
                "",
            ]
        )
    if not utility.empty:
        lines.extend(
            [
                "## Utility Metrics",
                "",
                utility.head(20).to_markdown(index=False),
                "",
            ]
        )
    return "\n".join(lines)
