from pathlib import Path

import typer

from synpath.pipeline.jobs import Job
from synpath.pipeline.runner import PipelineRunner

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

app = typer.Typer(help="synpath synthetic histopathology benchmark CLI.")


@app.command("evaluate-authenticity")
def evaluate_authenticity_command(
    real_manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    real_embeddings: Path = typer.Option(..., exists=True, dir_okay=False),
    synthetic_manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    synthetic_embeddings: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
    dataset: str = typer.Option(...),
    model: str = typer.Option(...),
    conditioning: str = typer.Option(...),
    seed: int = typer.Option(0),
    family: str = typer.Option("manual"),
    batch_size: int = typer.Option(1024, min=1),
) -> None:
    """Calculate full-population UNI Authenticity for one generated arm."""
    from synpath.metrics.authenticity import (
        AuthenticityInputs,
        evaluate_authenticity,
    )
    from synpath.paths import SynpathPaths

    path = evaluate_authenticity(
        AuthenticityInputs(
            real_manifest=real_manifest,
            real_embeddings=real_embeddings,
            synthetic_manifest=synthetic_manifest,
            synthetic_embeddings=synthetic_embeddings,
            output=output,
            dataset=dataset,
            model=model,
            conditioning=conditioning,
            seed=seed,
            family=family,
        ),
        SynpathPaths.from_env().cache_root
        / "authenticity"
        / "real_neighbours",
        batch_size=batch_size,
    )
    typer.echo(f"artifact: {path}")


@app.command("backfill-authenticity")
def backfill_authenticity_command(
    experiment_root: list[Path] = typer.Option(
        ...,
        "--experiment-root",
        exists=True,
        file_okay=False,
        help="Repeat for each canonical or isolated experiment root.",
    ),
    main_root: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        help="Main SynPath data/output root containing the audit outputs.",
    ),
    include_audits: bool = typer.Option(True),
    gpus: str = typer.Option("0,1,2,3,4,5,6,7"),
    batch_size: int = typer.Option(1024, min=1),
    report: Path = typer.Option(..., dir_okay=False),
    resume: bool = typer.Option(True),
) -> None:
    """Backfill UNI Authenticity over the complete 59-output inventory."""
    from synpath.metrics.authenticity_backfill import run_backfill

    gpu_ids = [int(value.strip()) for value in gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise typer.BadParameter("At least one GPU ID is required")
    path = run_backfill(
        experiment_roots=experiment_root,
        main_root=main_root,
        include_audits=include_audits,
        gpu_ids=gpu_ids,
        batch_size=batch_size,
        report_path=report,
        resume=resume,
    )
    typer.echo(f"artifact: {path}")


@app.command("train-pcam-sd-full-class")
def train_pcam_sd_full_class(
    config: Path = typer.Option(
        Path("configs/experiments/pcam_sd_full_class.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to the PCam full-UNet class-label config.",
    ),
    stage: str = typer.Option(
        "all",
        "--stage",
        help=(
            "Stage: prepare, preflight, train, sample, evaluate, report, or all."
        ),
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from the latest complete checkpoint.",
    ),
) -> None:
    """Fully fine-tune SD 1.5's UNet on class-labelled PCam patches."""
    from synpath.audits.pcam_sd_full import run_pcam_sd_full

    outputs = run_pcam_sd_full(config, stage, resume)
    for output in outputs:
        typer.echo(f"artifact: {output}")


@app.command("audit-pcam-sd-lora")
def audit_pcam_sd_lora(
    config: Path = typer.Option(
        Path("configs/experiments/audit_pcam_sd_lora.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to the PCam SD-LoRA audit config.",
    ),
    stage: str = typer.Option(
        "all",
        "--stage",
        help=(
            "Audit stage: prepare, preflight, train, validate, sample, "
            "evaluate, report, or all."
        ),
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Reuse validated completed audit artifacts.",
    ),
) -> None:
    """Audit PCam SD-LoRA resolution and morphology-prompt balancing."""
    from synpath.audits.pcam_sd_lora import run_pcam_sd_lora_audit

    outputs = run_pcam_sd_lora_audit(config, stage, resume)
    for output in outputs:
        typer.echo(f"artifact: {output}")


@app.command("audit-sd-lora-convergence")
def audit_sd_lora_convergence(
    config: Path = typer.Option(
        Path("configs/experiments/convergence_sd_lora_kather_class_label.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to an SD-LoRA convergence-audit config.",
    ),
    stage: str = typer.Option(
        "all",
        "--stage",
        help="Audit stage: train, evaluate, validation, report, or all.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Reuse completed training and checkpoint evaluations.",
    ),
) -> None:
    """Audit rank-16 SD-LoRA convergence on Kather class labels."""
    from synpath.convergence.sd_lora import run_sd_lora_convergence_audit

    outputs = run_sd_lora_convergence_audit(config, stage, resume)
    for output in outputs:
        typer.echo(f"artifact: {output}")


@app.command("calibrate-sd-index")
def calibrate_sd_index(
    config: Path = typer.Option(
        Path("configs/experiments/sd_index_calibration.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to an SD-index calibration YAML file.",
    ),
    stage: str = typer.Option(
        "all",
        "--stage",
        help="Calibration stage: curate, embed, fit, report, or all.",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help="Run one configured dataset track.",
    ),
    encoder: str | None = typer.Option(
        None,
        "--encoder",
        help="Run one configured encoder track.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Reuse validated features and completed k fits.",
    ),
) -> None:
    """Calibrate global K-means cluster counts using the SD-index."""
    from synpath.calibration.pipeline import run_calibration

    outputs = run_calibration(config, stage, dataset, encoder, resume)
    for output in outputs:
        typer.echo(f"artifact: {output}")


@app.command("calibrate-cluster-count")
def calibrate_cluster_count(
    config: Path = typer.Option(
        Path("configs/experiments/cluster_count_calibration_2_50.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to a cluster-count calibration YAML file.",
    ),
    stage: str = typer.Option(
        "all",
        "--stage",
        help="Calibration stage: curate, embed, fit, report, or all.",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help="Run one configured training-population track.",
    ),
    encoder: str | None = typer.Option(
        None,
        "--encoder",
        help="Run one configured encoder track.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Reuse validated features and completed k fits.",
    ),
) -> None:
    """Calibrate global K-means training-set cluster counts from k=2 to 50."""
    from synpath.calibration.pipeline import run_calibration

    outputs = run_calibration(config, stage, dataset, encoder, resume)
    for output in outputs:
        typer.echo(f"artifact: {output}")


@app.command()
def plan(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to an experiment YAML file.",
    ),
) -> None:
    """Print a compact execution plan for an experiment config."""
    runner = PipelineRunner(config)
    typer.echo("\n".join(runner.plan_lines()))


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to an experiment YAML file.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Skip completed jobs when their expected artifacts already exist.",
    ),
) -> None:
    """Run an experiment end-to-end."""
    runner = PipelineRunner(config, resume=resume)
    for job in runner.jobs():
        typer.echo(f"running: {job.key}")
        result = runner.run_single(job)
        status = "skipped" if result.skipped else "completed"
        typer.echo(f"{status}: {result.job.key}")


@app.command("prepare-data")
def prepare_data(
    dataset: str,
    config: Path = typer.Option(
        Path("configs/experiments/main.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to an experiment YAML file.",
    ),
    resume: bool = False,
) -> None:
    """Prepare a dataset manifest in the common synpath schema."""
    runner = PipelineRunner(config, resume=resume)
    result = runner.run_single(Job(stage="prepare_data", dataset=dataset))
    typer.echo(f"completed: {result.job.key}")


@app.command()
def embed(
    dataset: str,
    encoder: str = "uni",
    config: Path = typer.Option(
        Path("configs/experiments/main.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to an experiment YAML file.",
    ),
    resume: bool = False,
) -> None:
    """Extract and cache embeddings for a prepared dataset."""
    del encoder
    runner = PipelineRunner(config, resume=resume)
    result = runner.run_single(Job(stage="embed", dataset=dataset))
    typer.echo(f"completed: {result.job.key}")


@app.command()
def cluster(
    dataset: str,
    config: Path = typer.Option(
        Path("configs/experiments/main.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to an experiment YAML file.",
    ),
    resume: bool = False,
) -> None:
    """Select KMeans pseudo-labels from cached UNI embeddings."""
    runner = PipelineRunner(config, resume=resume)
    result = runner.run_single(Job(stage="cluster", dataset=dataset))
    typer.echo(f"completed: {result.job.key}")


@app.command("select-subset")
def select_subset(
    dataset: str,
    config: Path = typer.Option(
        Path("configs/experiments/main.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to an experiment YAML file.",
    ),
    resume: bool = False,
) -> None:
    """Select the configured experiment subset after full-pool clustering."""
    runner = PipelineRunner(config, resume=resume)
    result = runner.run_single(Job(stage="select_subset", dataset=dataset))
    typer.echo(f"completed: {result.job.key}")


@app.command("build-conditions")
def build_conditions(
    dataset: str,
    conditioning: str,
    config: Path = typer.Option(
        Path("configs/experiments/main.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to an experiment YAML file.",
    ),
    resume: bool = False,
) -> None:
    """Build generator condition manifests for one conditioning mode."""
    runner = PipelineRunner(config, resume=resume)
    result = runner.run_single(
        Job(
            stage="build_conditions", dataset=dataset, conditioning=conditioning
        )
    )
    typer.echo(f"completed: {result.job.key}")


@app.command("train-generator")
def train_generator(
    dataset: str,
    model: str,
    conditioning: str,
    seed: int = 0,
    config: Path = typer.Option(
        Path("configs/experiments/main.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to an experiment YAML file.",
    ),
    resume: bool = False,
) -> None:
    """Train or fine-tune a generator for one experiment arm."""
    runner = PipelineRunner(config, resume=resume)
    result = runner.run_single(
        Job(
            stage="train_generator",
            dataset=dataset,
            model=model,
            conditioning=conditioning,
            seed=seed,
        )
    )
    typer.echo(f"completed: {result.job.key}")


@app.command()
def sample(
    dataset: str,
    model: str,
    conditioning: str,
    seed: int = 0,
    config: Path = typer.Option(
        Path("configs/experiments/main.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to an experiment YAML file.",
    ),
    resume: bool = False,
) -> None:
    """Generate PNG samples for one trained generator arm."""
    runner = PipelineRunner(config, resume=resume)
    result = runner.run_single(
        Job(
            stage="sample",
            dataset=dataset,
            model=model,
            conditioning=conditioning,
            seed=seed,
        )
    )
    typer.echo(f"completed: {result.job.key}")


@app.command("evaluate-quality")
def evaluate_quality(
    dataset: str,
    model: str,
    conditioning: str,
    seed: int = 0,
    config: Path = typer.Option(
        Path("configs/experiments/main.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to an experiment YAML file.",
    ),
    resume: bool = False,
) -> None:
    """Run fidelity, diversity, coverage, and privacy metrics."""
    runner = PipelineRunner(config, resume=resume)
    result = runner.run_single(
        Job(
            stage="evaluate_quality",
            dataset=dataset,
            model=model,
            conditioning=conditioning,
            seed=seed,
        )
    )
    typer.echo(f"completed: {result.job.key}")


@app.command("train-utility")
def train_utility(
    dataset: str,
    source: str,
    seed: int = 0,
    synthetic_model: str | None = typer.Option(
        None,
        "--synthetic-model",
        help="Generator model to use for synthetic utility data.",
    ),
    config: Path = typer.Option(
        Path("configs/experiments/main.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to an experiment YAML file.",
    ),
    resume: bool = False,
) -> None:
    """Train and evaluate ResNet-18 utility classifiers."""
    runner = PipelineRunner(config, resume=resume)
    result = runner.run_single(
        Job(
            stage="train_utility",
            dataset=dataset,
            source=source,
            seed=seed,
            synthetic_model=synthetic_model,
        )
    )
    typer.echo(f"completed: {result.job.key}")


@app.command()
def report(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="Path to an experiment YAML file.",
    ),
) -> None:
    """Aggregate experiment outputs into CSV files, figures, and report.md."""
    runner = PipelineRunner(config)
    result = runner.run_single(Job(stage="report"))
    typer.echo(f"completed: {result.job.key}")


@app.command("report-appendix")
def report_appendix(
    output: Path = typer.Option(
        Path("appendix_quality_utility_by_dataset.tex"),
        "--output",
        "-o",
        dir_okay=False,
        help="Destination for the appendix LaTeX tables.",
    ),
    quality_seeds: str = typer.Option(
        "0,1,2",
        "--quality-seeds",
        help="Comma-separated independent generator seeds.",
    ),
    classifier_seeds: str = typer.Option(
        "0,1,2",
        "--classifier-seeds",
        help="Comma-separated utility-classifier seeds.",
    ),
) -> None:
    """Create dataset-specific quality and utility appendix tables."""
    from synpath.paths import SynpathPaths
    from synpath.reports.latex import build_appendix_tables

    path = build_appendix_tables(
        SynpathPaths.from_env().outputs_root,
        output,
        quality_seeds=parse_seed_list(quality_seeds),
        classifier_seeds=parse_seed_list(classifier_seeds),
    )
    typer.echo(f"artifact: {path}")


def parse_seed_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    app()
