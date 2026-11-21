from __future__ import annotations

from pathlib import Path

import click

from .cluster_conditioning import run_cluster_conditioning_audit
from .config import ExperimentConfig
from .correlation import run_correlation_audit
from .manifest import build_task_manifest
from .runner import run_manifest_dask, run_task_by_index
from .runtime import validate_runtime
from .summary import aggregate
from .targets import import_dimension_sources
from .utils import atomic_write_json


@click.group("dgx")
def dgx_cli() -> None:
    """DGX lowgen v2 commands."""


@dgx_cli.command("validate")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def validate(config_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    report = validate_runtime(cfg)
    cfg.paths.run_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.run_dir / "runtime_validation.json"
    atomic_write_json(path, report)
    click.echo(f"Wrote {path}")


@dgx_cli.command("import-dimensions")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def import_dimensions(config_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    outputs = import_dimension_sources(cfg)
    click.echo(f"Wrote {outputs}")


@dgx_cli.command("build-manifest")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def build_manifest(config_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    rows = build_task_manifest(cfg)
    click.echo(
        f"Wrote {len(rows)} tasks to {cfg.paths.run_dir / 'task_manifest.jsonl'}"
    )


@dgx_cli.command("run-task")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option("--task-index", type=int, required=True)
@click.option("--force", is_flag=True)
def run_task(
    config_path: Path, manifest_path: Path | None, task_index: int, force: bool
) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    result = run_task_by_index(cfg, manifest_path, task_index, force=force)
    click.echo(f"Wrote task {result['task_id']}")


@dgx_cli.command("run-dask")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option("--n-workers", type=int, default=None)
@click.option("--force", is_flag=True)
def run_dask(
    config_path: Path, manifest_path: Path | None, n_workers: int | None, force: bool
) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    results = run_manifest_dask(cfg, manifest_path, force=force, n_workers=n_workers)
    click.echo(f"Finished {len(results)} Dask tasks.")


@dgx_cli.command("aggregate")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def aggregate_cmd(config_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    outputs = aggregate(cfg)
    click.echo(f"Wrote {outputs}")


@dgx_cli.command("correlation-audit")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--top-n", type=int, default=100, show_default=True)
@click.option("--force", is_flag=True)
@click.option(
    "--include-projector",
    "include_projectors",
    type=click.Choice(["raw_highdim", "pca", "umap"]),
    multiple=True,
)
@click.option("--deduplicate/--no-deduplicate", default=True, show_default=True)
@click.option("--max-rows", type=int, default=None)
@click.option("--seed", type=int, default=0, show_default=True)
def correlation_audit(
    config_path: Path,
    manifest_path: Path | None,
    output_dir: Path | None,
    top_n: int,
    force: bool,
    include_projectors: tuple[str, ...],
    deduplicate: bool,
    max_rows: int | None,
    seed: int,
) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    outputs = run_correlation_audit(
        cfg,
        manifest=manifest_path,
        output_dir=output_dir,
        top_n=top_n,
        force=force,
        include_projectors=include_projectors or None,
        deduplicate=deduplicate,
        max_rows=max_rows,
        seed=seed,
        config_path=config_path,
    )
    click.echo(f"Wrote {outputs}")


@dgx_cli.command("cluster-conditioning-audit")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--clustering-results",
    "clustering_results_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="clustering_experiment clustering_results.csv path.",
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--max-rows", type=int, default=None)
@click.option("--max-cluster-configs", type=int, default=None)
@click.option("--force", is_flag=True)
@click.option("--generator-kind", "generator_kinds", multiple=True)
@click.option("--include-noise-cluster", is_flag=True)
@click.option("--dataset", default=None)
@click.option("--encoder", default=None)
@click.option(
    "--projector",
    type=click.Choice(["raw", "raw_highdim", "pca", "umap"]),
    default=None,
)
@click.option("--target-name", default=None)
@click.option("--seed", type=int, default=None)
@click.option("--synthetic-total", type=int, default=None)
@click.option(
    "--n-workers",
    type=int,
    default=None,
    help="Run cluster rows through Dask with this many workers.",
)
@click.option(
    "--dask-chunk-size",
    type=int,
    default=None,
    help="Number of selected cluster rows per Dask task.",
)
def cluster_conditioning_audit(
    config_path: Path,
    clustering_results_path: Path | None,
    manifest_path: Path | None,
    output_dir: Path | None,
    max_rows: int | None,
    max_cluster_configs: int | None,
    force: bool,
    generator_kinds: tuple[str, ...],
    include_noise_cluster: bool,
    dataset: str | None,
    encoder: str | None,
    projector: str | None,
    target_name: str | None,
    seed: int | None,
    synthetic_total: int | None,
    n_workers: int | None,
    dask_chunk_size: int | None,
) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    outputs = run_cluster_conditioning_audit(
        cfg,
        clustering_results_path=clustering_results_path,
        output_dir=output_dir,
        manifest=manifest_path,
        max_rows=max_rows,
        max_cluster_configs=max_cluster_configs,
        force=force,
        generator_kinds=generator_kinds or None,
        include_noise_cluster=include_noise_cluster or None,
        dataset=dataset,
        encoder=encoder,
        projector=projector,
        target_name=target_name,
        seed=seed,
        synthetic_total=synthetic_total,
        n_workers=n_workers,
        dask_chunk_size=dask_chunk_size,
    )
    click.echo(f"Wrote {outputs}")
