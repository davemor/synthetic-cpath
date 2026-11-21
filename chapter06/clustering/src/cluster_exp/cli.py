from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from .clusterers import run_clustering_task
from .config import ExperimentConfig
from .dimension_plots import plot_dimension_metrics
from .manifest import build_task_manifest
from .posthoc import add_sd_index_to_run
from .projections import build_projection_manifest, prepare_projection_cache, run_projection_task
from .runtime import validate_runtime as validate_runtime_impl
from .rq1_appendix import generate_rq1_appendix_dimension_tables
from .summary import summarise_run
from .table1 import generate_table_one
from .utils import read_jsonl


@click.group()
def main() -> None:
    """Standalone histology embedding clustering experiment."""


@main.command("validate-runtime")
@click.option("--no-require-cuml", is_flag=True, help="Skip mandatory cuML checks.")
def validate_runtime(no_require_cuml: bool) -> None:
    """Validate CUDA/RAPIDS support for production runs."""
    report = validate_runtime_impl(require_cuml=not no_require_cuml)
    click.echo(json.dumps(report, indent=2, sort_keys=True))


@main.command("prepare-projections")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--force", is_flag=True, help="Recompute projection artifacts even if cached.")
def prepare_projections(config_path: Path, force: bool) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    manifest = prepare_projection_cache(cfg, force=force)
    click.echo(f"Wrote projection cache manifest with {len(manifest)} rows.")


@main.command("make-projection-manifest")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
def make_projection_manifest(config_path: Path, output_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    manifest = build_projection_manifest(cfg, task_manifest_path=output_path)
    projected = manifest[manifest["projector"] != "raw"]
    click.echo(
        f"Wrote {len(projected)} projection tasks to {output_path} "
        f"and {len(manifest)} representation rows to the cache manifest."
    )


@main.command("run-projection-task")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--manifest", "manifest_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--task-index", type=int, required=True)
@click.option("--force", is_flag=True, help="Recompute projection output if it already exists.")
def run_projection_task_command(
    config_path: Path, manifest_path: Path, task_index: int, force: bool
) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    tasks = read_jsonl(manifest_path)
    if task_index < 0 or task_index >= len(tasks):
        raise click.ClickException(f"task-index {task_index} outside manifest range 0..{len(tasks)-1}.")
    result = run_projection_task(cfg, tasks[task_index], force=force)
    click.echo(f"Finished projection task {task_index}: {result['projection_cache_key']}")


@main.command("make-manifest")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
def make_manifest(config_path: Path, output_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    tasks = build_task_manifest(cfg, output_path)
    click.echo(f"Wrote {len(tasks)} clustering tasks to {output_path}.")


@main.command("run-task")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--manifest", "manifest_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--task-index", type=int, required=True)
@click.option("--force", is_flag=True, help="Recompute task output if it already exists.")
def run_task(config_path: Path, manifest_path: Path, task_index: int, force: bool) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    tasks = read_jsonl(manifest_path)
    if task_index < 0 or task_index >= len(tasks):
        raise click.ClickException(f"task-index {task_index} outside manifest range 0..{len(tasks)-1}.")
    result = run_clustering_task(cfg, tasks[task_index], force=force)
    click.echo(f"Finished task {task_index}: {tasks[task_index]['task_id']} ({len(result)} configs)")


@main.command("summarise")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--run-dir", type=click.Path(path_type=Path), default=None)
@click.option("--workers", type=int, default=1, show_default=True)
def summarise(config_path: Path, run_dir: Path | None, workers: int) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    paths = summarise_run(cfg, run_dir=run_dir, workers=workers)
    click.echo("Wrote summaries:")
    for name, path in paths.items():
        click.echo(f"  {name}: {path}")


@main.command("add-sd-index")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--run-dir", type=click.Path(path_type=Path), default=None)
@click.option(
    "--manifest",
    "manifest_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    help="Optional task manifest(s); when provided, only those task IDs are updated.",
)
@click.option("--workers", type=int, default=1, show_default=True)
@click.option("--force", is_flag=True, help="Recompute SD-index even when result rows already have it.")
def add_sd_index(
    config_path: Path,
    run_dir: Path | None,
    manifest_paths: tuple[Path, ...],
    workers: int,
    force: bool,
) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    stats = add_sd_index_to_run(
        cfg,
        run_dir=run_dir,
        manifest_paths=list(manifest_paths) or None,
        workers=workers,
        force=force,
    )
    click.echo(json.dumps(stats, indent=2, sort_keys=True))


@main.command("make-table1")
@click.option("--run-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--stability-csv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional path to clustering_stability.csv. Defaults to RUN_DIR/clustering_stability.csv.",
)
@click.option(
    "--output-csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Output CSV path. Defaults to RUN_DIR/table1_method_projection_summary.csv.",
)
@click.option(
    "--output-tex",
    type=click.Path(path_type=Path),
    default=None,
    help="Output LaTeX path. Defaults to RUN_DIR/table1_method_projection_summary.tex.",
)
@click.option(
    "--caption",
    type=str,
    default=None,
    help="Optional LaTeX caption.",
)
@click.option("--label", type=str, default="tab:clustering_method_projection", show_default=True)
@click.option("--no-bold-best", is_flag=True, help="Do not bold the best value in each metric column.")
def make_table1(
    run_dir: Path,
    stability_csv: Path | None,
    output_csv: Path | None,
    output_tex: Path | None,
    caption: str | None,
    label: str,
    no_bold_best: bool,
) -> None:
    outputs = generate_table_one(
        run_dir=run_dir,
        stability_csv=stability_csv,
        output_csv=output_csv,
        output_tex=output_tex,
        caption=caption,
        label=label,
        bold_best=not no_bold_best,
    )
    click.echo("Wrote Table 1 outputs:")
    click.echo(f"  wide csv: {outputs.csv_path}")
    click.echo(f"  external csv: {outputs.external_csv_path}")
    click.echo(f"  structure csv: {outputs.structure_csv_path}")
    click.echo(f"  combined latex: {outputs.latex_path}")
    click.echo(f"  external latex: {outputs.external_latex_path}")
    click.echo(f"  structure latex: {outputs.structure_latex_path}")


@main.command("make-rq1-appendix-dimension-tables")
@click.option("--run-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--stability-csv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional path to clustering_stability.csv. Defaults to RUN_DIR/clustering_stability.csv.",
)
@click.option("--min-dimension", type=int, default=1, show_default=True)
@click.option("--max-dimension", type=int, default=50, show_default=True)
def make_rq1_appendix_dimension_tables(
    run_dir: Path,
    stability_csv: Path | None,
    min_dimension: int,
    max_dimension: int,
) -> None:
    outputs = generate_rq1_appendix_dimension_tables(
        run_dir=run_dir,
        stability_csv=stability_csv,
        min_dimension=min_dimension,
        max_dimension=max_dimension,
    )
    click.echo("Wrote RQ1 appendix dimension table outputs:")
    click.echo(f"  external csv: {outputs.external_csv_path}")
    click.echo(f"  external latex: {outputs.external_latex_path}")
    click.echo(f"  internal csv: {outputs.internal_csv_path}")
    click.echo(f"  internal latex: {outputs.internal_latex_path}")


@main.command("plot-dimension-metrics")
@click.option("--run-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--projector", type=click.Choice(["pca", "umap", "all"]), default="pca", show_default=True)
@click.option(
    "--clusterer",
    type=click.Choice(["combined", "all", "kmeans", "spherical_kmeans", "hdbscan", "leiden"]),
    default="combined",
    show_default=True,
    help="Plot all methods together, all methods separately, or one clustering method.",
)
@click.option("--dataset", type=str, default=None, help="Optional dataset filter, e.g. icaird100k.")
@click.option("--encoder", type=str, default=None, help="Optional encoder filter, e.g. uni2.")
@click.option(
    "--error-source",
    type=click.Choice(["selected", "seeds"]),
    default="selected",
    show_default=True,
    help="Use selected summary-row variability or seed-level result-row variability for error bands.",
)
@click.option(
    "--stability-csv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional path to clustering_stability.csv. Defaults to RUN_DIR/clustering_stability.csv.",
)
@click.option(
    "--results-csv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional path to clustering_results.csv. Used when --error-source seeds.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output figure path. Defaults to RUN_DIR/clustering_results_PROJECTOR_dimension_metric_trends.png.",
)
@click.option(
    "--output-csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Output CSV path for plotted summary values.",
)
@click.option("--min-dimension", type=int, default=1, show_default=True)
@click.option("--max-dimension", type=int, default=50, show_default=True)
@click.option("--dpi", type=int, default=300, show_default=True)
@click.option("--no-error-band", is_flag=True, help="Plot means without standard-deviation bands.")
@click.option("--no-raw-baseline", is_flag=True, help="Do not draw dotted raw-embedding baselines.")
def plot_dimension_metrics_command(
    run_dir: Path,
    projector: str,
    clusterer: str,
    dataset: str | None,
    encoder: str | None,
    error_source: str,
    stability_csv: Path | None,
    results_csv: Path | None,
    output: Path | None,
    output_csv: Path | None,
    min_dimension: int,
    max_dimension: int,
    dpi: int,
    no_error_band: bool,
    no_raw_baseline: bool,
) -> None:
    if (projector == "all" or clusterer == "all") and (output is not None or output_csv is not None):
        raise click.ClickException(
            "--output and --output-csv can only be used with one projector/clusterer plot."
        )

    projectors = ["pca", "umap"] if projector == "all" else [projector]
    clusterers: list[str | None]
    if clusterer == "combined":
        clusterers = [None]
    elif clusterer == "all":
        clusterers = ["kmeans", "spherical_kmeans", "hdbscan", "leiden"]
    else:
        clusterers = [clusterer]
    click.echo("Wrote dimension metric plot outputs:")
    for projector_name in projectors:
        for clusterer_name in clusterers:
            outputs = plot_dimension_metrics(
                run_dir=run_dir,
                projector=projector_name,
                clusterer=clusterer_name,
                dataset=dataset,
                encoder=encoder,
                error_source=error_source,
                stability_csv=stability_csv,
                results_csv=results_csv,
                output=output,
                output_csv=output_csv,
                min_dimension=min_dimension,
                max_dimension=max_dimension,
                show_error_band=not no_error_band,
                include_raw_baseline=not no_raw_baseline,
                dpi=dpi,
            )
            label = clusterer_name or "combined"
            click.echo(f"  {projector_name}/{label} figure: {outputs.figure_path}")
            click.echo(f"  {projector_name}/{label} csv: {outputs.csv_path}")


@main.command("show-targets")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
def show_targets(config_path: Path) -> None:
    from .dimensions import load_projection_targets

    cfg = ExperimentConfig.from_yaml(config_path)
    targets = load_projection_targets(cfg)
    pd.DataFrame([target.__dict__ for target in targets]).to_csv(click.get_text_stream("stdout"), index=False)


if __name__ == "__main__":
    main()
