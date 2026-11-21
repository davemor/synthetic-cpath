from __future__ import annotations

from datetime import datetime
from copy import deepcopy
import json
from pathlib import Path

import click
import pandas as pd
import yaml

from .citations import write_citations
from .cluster_pseudolabels import prepare_cluster_pseudolabels
from .config import load_config
from .images import run_image_task, run_images
from .metrics import bootstrap_metric_intervals_from_predictions, write_summary
from .mil import run_mil, run_mil_task
from .probes import run_probe_task, run_probes
from .synthetic_embeddings import (
    evaluate_synthetic_embeddings,
    prepare_synthetic_embedding_run,
    sample_synthetic_embeddings,
    train_synthetic_embedding_model,
    write_synthetic_baselines,
)
from .synthetic_bags import (
    finalize_synthetic_bag_shards,
    fit_bag_description_model,
    generate_synthetic_bag_shard,
    generate_synthetic_bag_datasets,
    sample_bag_descriptions,
)
from .timing import TimerRecorder


def _run_dir(output_dir: Path, name: str | None) -> Path:
    if name:
        return output_dir / name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / stamp


@click.group()
def main() -> None:
    """Benchmark iCAIRD Cervical 100K classifiers."""


@main.command("run")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--run-name", type=str, default=None)
def run_all(config_path: Path, run_name: str | None) -> None:
    """Run enabled image classifiers, embedding probes, and MIL tasks."""
    cfg = load_config(config_path)
    output_dir = _run_dir(cfg.run.output_dir, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    timer = TimerRecorder(output_dir)
    extra_citations = [
        embedding.citation
        for embedding in cfg.probes.embeddings.values()
        if embedding.citation
    ]
    write_citations(output_dir / "citations.md", extra_citations)
    with timer.time("run_all"):
        image_rows = run_images(cfg.data, cfg.images, output_dir, cfg.run.seeds, timer)
        probe_rows = run_probes(cfg.probes, output_dir, cfg.run.seeds, timer)
        mil_rows = run_mil(cfg.mil, output_dir, cfg.run.seeds, timer)
    _write_combined(output_dir, image_rows + probe_rows + mil_rows)
    write_summary(output_dir)
    click.echo(f"Wrote benchmark outputs to {output_dir}")


@main.command("train-images")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--run-name", type=str, default=None)
def train_images(config_path: Path, run_name: str | None) -> None:
    """Run image classifiers only."""
    cfg = load_config(config_path)
    output_dir = _run_dir(cfg.run.output_dir, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    timer = TimerRecorder(output_dir)
    write_citations(output_dir / "citations.md")
    rows = run_images(cfg.data, cfg.images, output_dir, cfg.run.seeds, timer)
    _write_combined(output_dir, rows)
    write_summary(output_dir)
    click.echo(f"Wrote image benchmark outputs to {output_dir}")


@main.command("train-probes")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--run-name", type=str, default=None)
def train_probes(config_path: Path, run_name: str | None) -> None:
    """Run embedding/manual-feature probes only."""
    cfg = load_config(config_path)
    output_dir = _run_dir(cfg.run.output_dir, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    timer = TimerRecorder(output_dir)
    extra_citations = [
        embedding.citation
        for embedding in cfg.probes.embeddings.values()
        if embedding.citation
    ]
    write_citations(output_dir / "citations.md", extra_citations)
    rows = run_probes(cfg.probes, output_dir, cfg.run.seeds, timer)
    _write_combined(output_dir, rows)
    write_summary(output_dir)
    click.echo(f"Wrote probe benchmark outputs to {output_dir}")


@main.command("train-mil")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--run-name", type=str, default=None)
def train_mil(config_path: Path, run_name: str | None) -> None:
    """Run transformer MIL tasks only."""
    cfg = load_config(config_path)
    output_dir = _run_dir(cfg.run.output_dir, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    timer = TimerRecorder(output_dir)
    write_citations(output_dir / "citations.md")
    rows = run_mil(cfg.mil, output_dir, cfg.run.seeds, timer)
    _write_combined(output_dir, rows)
    write_summary(output_dir)
    click.echo(f"Wrote MIL benchmark outputs to {output_dir}")


@main.command("synthetic-prepare")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
def synthetic_prepare(config_path: Path) -> None:
    """Prepare train-only statistics and metadata for synthetic embedding generation."""
    cfg = load_config(config_path)
    output_dir = prepare_synthetic_embedding_run(cfg.synthetic_embeddings)
    click.echo(f"Wrote synthetic embedding metadata to {output_dir / 'metadata'}")


@main.command("synthetic-train")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
def synthetic_train(config_path: Path) -> None:
    """Train the conditional latent DDIM embedding generator."""
    cfg = load_config(config_path)
    checkpoint_path = train_synthetic_embedding_model(cfg.synthetic_embeddings)
    click.echo(f"Wrote best latent DDIM checkpoint to {checkpoint_path}")


@main.command("synthetic-sample")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--checkpoint", "checkpoint_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
def synthetic_sample(config_path: Path, checkpoint_path: Path | None) -> None:
    """Sample synthetic embeddings from a trained conditional latent DDIM."""
    cfg = load_config(config_path)
    written = sample_synthetic_embeddings(cfg.synthetic_embeddings, checkpoint_path)
    for path in written:
        click.echo(f"Wrote synthetic DDIM dataset to {path}")


@main.command("synthetic-baselines")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
def synthetic_baselines(config_path: Path) -> None:
    """Write bootstrap and diagonal-Gaussian synthetic embedding baselines."""
    cfg = load_config(config_path)
    written = write_synthetic_baselines(cfg.synthetic_embeddings)
    for path in written:
        click.echo(f"Wrote synthetic baseline dataset to {path}")


@main.command("synthetic-evaluate")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--synthetic-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
def synthetic_evaluate(
    config_path: Path,
    synthetic_dir: Path,
    output_dir: Path | None,
) -> None:
    """Evaluate synthetic embeddings against real iCAIRD 100K embeddings."""
    cfg = load_config(config_path)
    result_dir = evaluate_synthetic_embeddings(cfg.synthetic_embeddings, synthetic_dir, output_dir)
    click.echo(f"Wrote synthetic embedding evaluation to {result_dir}")


@main.command("cluster-pseudolabels-prepare")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
def cluster_pseudolabels_prepare(config_path: Path) -> None:
    """Fit PCA/KMeans cluster pseudo-labels and write derived patch/bag roots."""
    cfg = load_config(config_path)
    outputs = prepare_cluster_pseudolabels(cfg.cluster_pseudolabels)
    for name, path in outputs.items():
        click.echo(f"Wrote {name} to {path}")


@main.command("synthetic-bags-fit")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
def synthetic_bags_fit(config_path: Path) -> None:
    """Fit train-only bag size and patch-label composition models."""
    cfg = load_config(config_path)
    metadata_dir = fit_bag_description_model(cfg.synthetic_bags)
    for multiplier in cfg.synthetic_bags.sample_multipliers:
        description_path = sample_bag_descriptions(cfg.synthetic_bags, multiplier)
        click.echo(f"Wrote synthetic bag descriptions to {description_path}")
    click.echo(f"Wrote synthetic bag model metadata to {metadata_dir}")


@main.command("synthetic-bags-generate")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--checkpoint", "checkpoint_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
def synthetic_bags_generate(config_path: Path, checkpoint_path: Path | None) -> None:
    """Generate Conditional-EDM synthetic MIL bags and MIL-ready bag roots."""
    cfg = load_config(config_path)
    roots = generate_synthetic_bag_datasets(cfg.synthetic_bags, cfg.synthetic_embeddings, checkpoint_path)
    for path in roots:
        click.echo(f"Wrote MIL-ready synthetic bag root to {path}")


@main.command("synthetic-bags-generate-shard")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--checkpoint", "checkpoint_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--multiplier", type=float, required=True)
@click.option("--num-shards", type=int, required=True)
@click.option("--shard-index", type=int, required=True)
def synthetic_bags_generate_shard(
    config_path: Path,
    checkpoint_path: Path | None,
    multiplier: float,
    num_shards: int,
    shard_index: int,
) -> None:
    """Generate one shard of Conditional-EDM synthetic MIL bags."""
    cfg = load_config(config_path)
    shard_dir = generate_synthetic_bag_shard(
        cfg.synthetic_bags,
        cfg.synthetic_embeddings,
        multiplier=multiplier,
        num_shards=num_shards,
        shard_index=shard_index,
        checkpoint_path=checkpoint_path,
    )
    click.echo(f"Wrote synthetic bag shard to {shard_dir}")


@main.command("synthetic-bags-finalize-shards")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--checkpoint", "checkpoint_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--num-shards", type=int, required=True)
@click.option("--multiplier", "multipliers", type=float, multiple=True)
def synthetic_bags_finalize_shards(
    config_path: Path,
    checkpoint_path: Path | None,
    num_shards: int,
    multipliers: tuple[float, ...],
) -> None:
    """Finalize sharded Conditional-EDM synthetic MIL bag generation."""
    cfg = load_config(config_path)
    selected_multipliers = list(multipliers) or cfg.synthetic_bags.sample_multipliers
    for multiplier in selected_multipliers:
        roots = finalize_synthetic_bag_shards(
            cfg.synthetic_bags,
            multiplier=multiplier,
            num_shards=num_shards,
            checkpoint_path=checkpoint_path,
        )
        for path in roots:
            click.echo(f"Wrote MIL-ready synthetic bag root to {path}")


@main.command("synthetic-bags-write-mil-configs")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
def synthetic_bags_write_mil_configs(config_path: Path, output_dir: Path | None) -> None:
    """Write MIL configs and manifests for generated synthetic bag roots."""
    cfg = load_config(config_path)
    output_dir = output_dir or cfg.synthetic_bags.output_dir / "mil_configs"
    manifest_dir = output_dir / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open() as handle:
        base_config = yaml.safe_load(handle) or {}

    root_specs = []
    for multiplier in cfg.synthetic_bags.sample_multipliers:
        ratio = f"{multiplier:g}".replace(".", "p")
        if cfg.synthetic_bags.materialize_synthetic_only:
            root_specs.append((f"synthetic_only_{ratio}x", cfg.synthetic_bags.output_dir / "mil_roots" / f"synthetic_only_{ratio}x"))
        if cfg.synthetic_bags.materialize_real_plus_synthetic:
            root_specs.append(
                (f"real_plus_synthetic_{ratio}x", cfg.synthetic_bags.output_dir / "mil_roots" / f"real_plus_synthetic_{ratio}x")
            )

    for root_name, bag_root in root_specs:
        mil_config = deepcopy(base_config)
        mil_config.setdefault("images", {})["enabled"] = False
        mil_config.setdefault("probes", {})["enabled"] = False
        mil_config.setdefault("mil", {})["enabled"] = True
        mil_config["mil"]["bag_root"] = str(bag_root)
        mil_config.setdefault("data", {})["dataset_root"] = str(bag_root)
        config_out = output_dir / f"icaird_cervical_mil_uni2h_{root_name}.yml"
        config_out.write_text(yaml.safe_dump(mil_config, sort_keys=False))

        manifest_out = manifest_dir / f"{root_name}.jsonl"
        with manifest_out.open("w") as handle:
            index = 0
            for task_name in cfg.mil.tasks:
                for seed in cfg.run.seeds:
                    handle.write(json.dumps({"index": index, "task_type": "mil", "task": task_name, "seed": seed}, sort_keys=True))
                    handle.write("\n")
                    index += 1
        click.echo(f"Wrote synthetic MIL config to {config_out}")
        click.echo(f"Wrote synthetic MIL manifest to {manifest_out}")


@main.command("make-task-manifest")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--include",
    type=click.Choice(["all", "images", "probes", "mil"]),
    default="all",
    show_default=True,
)
def make_task_manifest(config_path: Path, output_path: Path, include: str) -> None:
    """Write a JSONL task manifest for Slurm arrays."""
    cfg = load_config(config_path)
    tasks = []
    if include in {"all", "images"} and cfg.images.enabled:
        for model_name in cfg.images.models:
            for augmentation_suite in cfg.images.augmentation_suites:
                for seed in cfg.run.seeds:
                    tasks.append(
                        {
                            "task_type": "image",
                            "model": model_name,
                            "augmentation_suite": augmentation_suite,
                            "seed": seed,
                        }
                    )
    if include in {"all", "probes"} and cfg.probes.enabled:
        for embedding_name in cfg.probes.embeddings:
            for classifier_name in cfg.probes.classifiers:
                for seed in cfg.run.seeds:
                    tasks.append(
                        {
                            "task_type": "probe",
                            "embedding": embedding_name,
                            "classifier": classifier_name,
                            "seed": seed,
                        }
                    )
    if include in {"all", "mil"} and cfg.mil.enabled:
        for task_name in cfg.mil.tasks:
            for seed in cfg.run.seeds:
                tasks.append({"task_type": "mil", "task": task_name, "seed": seed})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for index, task in enumerate(tasks):
            handle.write(json.dumps({"index": index, **task}, sort_keys=True))
            handle.write("\n")
    click.echo(f"Wrote {len(tasks)} tasks to {output_path}")


@main.command("run-task")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--manifest", "manifest_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--task-index", type=int, required=True)
@click.option("--run-name", type=str, default=None)
def run_task(
    config_path: Path,
    manifest_path: Path,
    task_index: int,
    run_name: str | None,
) -> None:
    """Run one task from a JSONL task manifest."""
    cfg = load_config(config_path)
    task = _read_task(manifest_path, task_index)
    output_dir = _run_dir(cfg.run.output_dir, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_timer = TimerRecorder(output_dir / "array_tasks" / f"task_{task_index:05d}")
    extra_citations = [
        embedding.citation
        for embedding in cfg.probes.embeddings.values()
        if embedding.citation
    ]
    write_citations(output_dir / "citations.md", extra_citations)
    if task["task_type"] == "image":
        row = run_image_task(
            cfg.data,
            cfg.images,
            output_dir,
            task["model"],
            int(task["seed"]),
            task_timer,
            augmentation_suite=task.get("augmentation_suite", "none"),
        )
    elif task["task_type"] == "probe":
        row = run_probe_task(
            cfg.probes,
            output_dir,
            task["embedding"],
            task["classifier"],
            int(task["seed"]),
            task_timer,
        )
    elif task["task_type"] == "mil":
        row = run_mil_task(
            cfg.mil,
            output_dir,
            task["task"],
            int(task["seed"]),
            task_timer,
        )
    else:
        raise click.ClickException(f"Unknown task type {task['task_type']!r}.")
    pd.DataFrame([{**task, **row}]).to_csv(
        output_dir / "array_tasks" / f"task_{task_index:05d}" / "task_metrics.csv",
        index=False,
    )
    click.echo(f"Finished task {task_index}: {task}")


@main.command("report")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def report(run_dir: Path) -> None:
    """Rebuild all_metrics.csv and summary.csv for an existing run directory."""
    write_summary(run_dir)
    click.echo(f"Wrote summary files in {run_dir}")


@main.command("bootstrap-predictions")
@click.argument("predictions_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.option("--resamples", type=int, default=10_000, show_default=True)
@click.option("--confidence-level", type=float, default=0.95, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def bootstrap_predictions(
    predictions_csv: Path,
    output_path: Path | None,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> None:
    """Bootstrap one saved predictions.csv file and write metric confidence intervals."""
    intervals, class_names = bootstrap_metric_intervals_from_predictions(
        predictions_csv,
        n_resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    row = {**intervals, "predictions_csv": str(predictions_csv), "class_names": ";".join(class_names)}
    output_path = output_path or predictions_csv.with_name("bootstrap_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output_path, index=False)
    click.echo(f"Wrote bootstrap intervals to {output_path}")


def _write_combined(output_dir: Path, rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "all_metrics.csv", index=False)


def _read_task(manifest_path: Path, task_index: int) -> dict:
    with manifest_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            task = json.loads(line)
            if int(task["index"]) == task_index:
                return task
    raise click.ClickException(f"Task index {task_index} not found in {manifest_path}.")


if __name__ == "__main__":
    main()
