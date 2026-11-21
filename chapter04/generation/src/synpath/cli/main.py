from pathlib import Path

import typer

from synpath.calibration.pipeline import run_calibration
from synpath.pipeline.jobs import Job
from synpath.pipeline.runner import PipelineRunner

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

app = typer.Typer(help="synpath synthetic histopathology benchmark CLI.")


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
