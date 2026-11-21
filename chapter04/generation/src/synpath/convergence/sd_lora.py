from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

from synpath.artifacts import utc_now, write_json, write_text
from synpath.config import load_resolved_yaml
from synpath.generation.wrappers import (
    condition_manifest_path,
    generator_root,
    run_parallel_commands,
)
from synpath.manifests import (
    SYNTHETIC_COLUMNS,
    read_manifest,
    write_manifest,
)
from synpath.metrics.quality import evaluate_quality
from synpath.paths import SynpathPaths
from synpath.pipeline.jobs import Job
from synpath.pipeline.runner import PipelineRunner


def run_sd_lora_convergence_audit(
    config_path: Path,
    stage: str = "all",
    resume: bool = False,
) -> list[Path]:
    cfg = load_resolved_yaml(config_path)
    audit = cfg["convergence_audit"]
    paths = SynpathPaths.from_env()
    validate_config(cfg, audit)
    outputs = []

    if stage in {"train", "all"}:
        runner = PipelineRunner(config_path, resume=resume, paths=paths)
        result = runner.run_single(
            Job(
                stage="train_generator",
                dataset=audit["dataset"],
                model=audit["model"],
                conditioning=audit["conditioning"],
                seed=int(audit["training_seed"]),
            )
        )
        outputs.extend(result.outputs)
    if stage in {"evaluate", "all"}:
        outputs.extend(evaluate_checkpoints(cfg, audit, paths, resume))
    if stage in {"validation", "all"}:
        from synpath.convergence.validation_loss import run_validation_loss

        outputs.extend(
            run_validation_loss(config_path, cfg, audit, paths, resume)
        )
    if stage in {"report", "all"}:
        outputs.extend(write_report(cfg, audit, paths))
    if stage not in {"train", "evaluate", "validation", "report", "all"}:
        raise ValueError(f"Unknown convergence-audit stage: {stage}")
    return outputs


def validate_config(cfg: dict, audit: dict) -> None:
    expected = {
        "dataset": "pathmnist224",
        "model": "sd_lora",
        "conditioning": "true_label",
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise ValueError(f"Convergence audit requires {key}={value}")
    sd_cfg = cfg.get("generation", {}).get("sd_lora", {})
    if int(sd_cfg.get("rank", 0)) != 16:
        raise ValueError("Convergence audit requires SD-LoRA rank 16")
    checkpoints = [int(step) for step in audit.get("checkpoints", [])]
    if checkpoints != sorted(set(checkpoints)) or not checkpoints:
        raise ValueError("Audit checkpoints must be nonempty and increasing")
    if checkpoints[-1] != int(sd_cfg.get("max_train_steps", 0)):
        raise ValueError("Final audit checkpoint must equal max_train_steps")


def evaluate_checkpoints(
    cfg: dict,
    audit: dict,
    paths: SynpathPaths,
    resume: bool,
) -> list[Path]:
    training_root = audit_training_root(audit, paths)
    condition_manifest = condition_manifest_path(
        audit["dataset"], audit["conditioning"], paths
    )
    if not condition_manifest.exists():
        raise RuntimeError(f"Missing condition manifest: {condition_manifest}")
    outputs = []
    for step in [int(value) for value in audit["checkpoints"]]:
        step_root = audit_root(audit, paths) / f"step{step:05d}"
        metrics_root = step_root / "metrics"
        expected_metric = metrics_root / "inception_fid.json"
        if resume and expected_metric.exists():
            outputs.append(expected_metric)
            continue
        checkpoint = training_root / "sd_lora" / f"checkpoint-{step}"
        validate_lora_checkpoint(checkpoint)
        sample_dir = step_root / "samples"
        sample_checkpoint(
            checkpoint,
            condition_manifest,
            sample_dir,
            cfg,
            int(audit["sampling_seed"]),
        )
        manifest = write_checkpoint_manifest(
            condition_manifest,
            sample_dir,
            step_root / "synthetic_manifest.csv",
            audit,
            cfg,
        )
        metric_outputs = evaluate_quality(
            audit["dataset"],
            audit["model"],
            audit["conditioning"],
            int(audit["training_seed"]),
            paths,
            cfg,
            synthetic_manifest=manifest,
            metrics_root=metrics_root,
        )
        metadata_path = step_root / "metadata.json"
        write_json(
            metadata_path,
            {
                "checkpoint": str(checkpoint),
                "conditioning": audit["conditioning"],
                "dataset": audit["dataset"],
                "lora_rank": 16,
                "sampling_seed": int(audit["sampling_seed"]),
                "step": step,
                "training_seed": int(audit["training_seed"]),
            },
        )
        outputs.extend([manifest, metadata_path, *metric_outputs])
    return outputs


def sample_checkpoint(
    checkpoint: Path,
    condition_manifest: Path,
    output_dir: Path,
    cfg: dict,
    sampling_seed: int,
) -> None:
    sd_cfg = cfg["generation"]["sd_lora"]
    max_samples = int(cfg["metrics"]["max_synthetic"])
    sample_gpus = int(
        os.environ.get("SYNPATH_SD_SAMPLE_GPUS") or sd_cfg.get("sample_gpus", 1)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for shard_index in range(sample_gpus):
        command = [
            sys.executable,
            "-m",
            "synpath.generation.sd_sample",
            "--lora_dir",
            str(checkpoint),
            "--condition_manifest",
            str(condition_manifest),
            "--output_dir",
            str(output_dir),
            "--max_samples",
            str(max_samples),
            "--seed",
            str(sampling_seed),
            "--batch_size",
            str(sd_cfg.get("sample_batch_size", 8)),
            "--height",
            str(sd_cfg.get("sample_height", 256)),
            "--width",
            str(sd_cfg.get("sample_width", 256)),
            "--num_shards",
            str(sample_gpus),
            "--shard_index",
            str(shard_index),
        ]
        env = {"CUDA_VISIBLE_DEVICES": str(shard_index)}
        commands.append(
            (
                command,
                output_dir.parent / f"sample_shard_{shard_index:02d}.log",
                env,
            )
        )
    run_parallel_commands(commands)


def write_checkpoint_manifest(
    condition_manifest: Path,
    sample_dir: Path,
    output: Path,
    audit: dict,
    cfg: dict,
) -> Path:
    conditions = read_manifest(condition_manifest)
    max_samples = min(int(cfg["metrics"]["max_synthetic"]), len(conditions))
    rows = []
    for index, source in enumerate(conditions[:max_samples]):
        image_path = sample_dir / f"{index:08d}.png"
        if not image_path.exists():
            raise RuntimeError(f"Missing generated audit sample: {image_path}")
        rows.append(
            {
                "image_path": str(image_path),
                "dataset": audit["dataset"],
                "model": audit["model"],
                "conditioning": audit["conditioning"],
                "seed": int(audit["training_seed"]),
                "label": source["label"],
                "label_id": source["label_id"],
                "cluster_id": source["cluster_id"],
                "condition": source["condition"],
                "condition_id": source["condition_id"],
                "prompt": source["prompt"],
            }
        )
    write_manifest(output, rows, SYNTHETIC_COLUMNS)
    return output


def write_report(cfg: dict, audit: dict, paths: SynpathPaths) -> list[Path]:
    records = []
    for step in [int(value) for value in audit["checkpoints"]]:
        metrics_root = audit_root(audit, paths) / f"step{step:05d}" / "metrics"
        records.append(load_metric_record(step, metrics_root))
    frame = pd.DataFrame(records)
    root = audit_root(audit, paths)
    validation_path = root / "validation_loss" / "validation_denoising_loss.csv"
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        frame = frame.merge(
            validation, on="step", how="left", validate="one_to_one"
        )
    csv_path = root / "convergence_metrics.csv"
    figure_path = root / "convergence_curves.png"
    report_path = root / "report.md"
    frame.to_csv(csv_path, index=False)
    plot_convergence(frame, figure_path)
    plateau = assess_plateau(frame, audit["plateau"])
    write_text(report_path, markdown_report(cfg, audit, frame, plateau))
    write_json(root / "plateau_assessment.json", plateau)
    return [
        csv_path,
        figure_path,
        report_path,
        root / "plateau_assessment.json",
    ]


def load_metric_record(step: int, metrics_root: Path) -> dict:
    required = {
        "inception": metrics_root / "inception_fid.json",
        "uni": metrics_root / "uni_fid_kid.json",
        "pr": metrics_root / "uni_precision_recall.json",
        "privacy": metrics_root / "privacy_nearest_neighbor.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing checkpoint metrics: {', '.join(missing)}")
    payload = {name: load_json(path) for name, path in required.items()}
    return {
        "step": step,
        "fid": payload["inception"]["fid"],
        "uni_fid": payload["uni"]["uni_fid"],
        "uni_kid": payload["uni"]["uni_kid"],
        "precision": payload["pr"]["precision"],
        "recall": payload["pr"]["recall"],
        "privacy_median_distance": payload["privacy"]["median_distance"],
        "privacy_p01_distance": payload["privacy"]["p01_distance"],
        "privacy_min_distance": payload["privacy"]["min_distance"],
    }


def assess_plateau(frame: pd.DataFrame, thresholds: dict) -> dict:
    tail = frame.sort_values("step").tail(3)
    if len(tail) != 3:
        raise ValueError("Plateau assessment requires three final checkpoints")
    result = {}
    relative_limit = float(thresholds["distance_relative_change"])
    absolute_limit = float(thresholds["precision_recall_absolute_change"])
    for metric in ("fid", "uni_fid"):
        changes = tail[metric].pct_change().abs().dropna()
        result[metric] = {
            "maximum_relative_change": float(changes.max()),
            "plateau": bool((changes <= relative_limit).all()),
            "threshold": relative_limit,
        }
    for metric in ("precision", "recall"):
        changes = tail[metric].diff().abs().dropna()
        result[metric] = {
            "maximum_absolute_change": float(changes.max()),
            "plateau": bool((changes <= absolute_limit).all()),
            "threshold": absolute_limit,
        }
    result["overall_plateau"] = all(
        bool(result[metric]["plateau"])
        for metric in ("fid", "uni_fid", "precision", "recall")
    )
    return result


def plot_convergence(frame: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("fid", "FID", "lower is better"),
        ("uni_fid", "UNI-FID", "lower is better"),
        ("precision", "UNI precision", "higher is better"),
        ("recall", "UNI recall", "higher is better"),
        ("privacy_p01_distance", "Privacy p01 distance", "higher is safer"),
        (
            "privacy_median_distance",
            "Privacy median distance",
            "higher is safer",
        ),
    ]
    if "validation_loss" in frame:
        metrics.append(
            (
                "validation_loss",
                "Validation denoising MSE",
                "lower is better",
            )
        )
    rows = 3 if len(metrics) > 6 else 2
    figure, axes = plt.subplots(
        rows, 3, figsize=(12, 3.5 * rows), constrained_layout=True
    )
    for axis, (column, title, direction) in zip(
        axes.flat, metrics, strict=False
    ):
        axis.plot(frame["step"], frame[column], marker="o")
        axis.set_title(title)
        axis.set_xlabel("Training step")
        axis.set_ylabel(direction)
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(metrics) :]:
        axis.set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def markdown_report(
    cfg: dict, audit: dict, frame: pd.DataFrame, plateau: dict
) -> str:
    sd_cfg = cfg["generation"]["sd_lora"]
    lines = [
        "# SD-LoRA Convergence Audit",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Design",
        "",
        f"- Dataset: `{audit['dataset']}`",
        f"- Conditioning: `{audit['conditioning']}`",
        f"- LoRA rank: `{sd_cfg['rank']}`",
        f"- Training seed: `{audit['training_seed']}`",
        f"- Sampling seed: `{audit['sampling_seed']}`",
        f"- Maximum training steps: `{sd_cfg['max_train_steps']}`",
        f"- Checkpoints: `{audit['checkpoints']}`",
        "- Text encoder: frozen (Diffusers LoRA default)",
        "",
        "## Metrics",
        "",
        frame.to_markdown(index=False),
        "",
        "## Plateau Assessment",
        "",
        f"- Overall plateau: `{plateau['overall_plateau']}`",
    ]
    for metric in ("fid", "uni_fid", "precision", "recall"):
        lines.append(f"- {metric}: `{plateau[metric]['plateau']}`")
    lines.extend(
        [
            "",
            "The plateau rule is a prespecified computational diagnostic, not a "
            "formal proof of convergence.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_training_root(audit: dict, paths: SynpathPaths) -> Path:
    return generator_root(
        audit["dataset"],
        audit["model"],
        audit["conditioning"],
        int(audit["training_seed"]),
        paths,
    )


def audit_root(audit: dict, paths: SynpathPaths) -> Path:
    return audit_training_root(audit, paths) / "convergence_audit"


def validate_lora_checkpoint(checkpoint: Path) -> None:
    if not checkpoint.is_dir():
        raise RuntimeError(f"Missing LoRA checkpoint: {checkpoint}")
    weights = list(checkpoint.glob("*lora*"))
    if not weights:
        raise RuntimeError(
            f"Checkpoint has no LoRA weight artifact: {checkpoint}"
        )


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)
