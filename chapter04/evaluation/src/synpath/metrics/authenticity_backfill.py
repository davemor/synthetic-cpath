from __future__ import annotations

from dataclasses import asdict
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from synpath.metrics.authenticity import (
    AuthenticityInputs,
    evaluate_authenticity,
    prepare_neighbour_cache,
    select_gpu,
    validate_embedding_cache,
)


def discover_generator_jobs(experiment_root: Path) -> list[AuthenticityInputs]:
    outputs = experiment_root / "outputs"
    if not (outputs / "generators").exists():
        outputs = experiment_root
        experiment_root = outputs.parent
    jobs = []
    for manifest in sorted(
        (outputs / "generators").glob("*/*/*/seed*/synthetic_manifest.csv")
    ):
        relative = manifest.relative_to(outputs / "generators")
        dataset, model, conditioning, seed_name = relative.parts[:4]
        jobs.append(
            AuthenticityInputs(
                real_manifest=(
                    outputs
                    / "conditions"
                    / dataset
                    / conditioning
                    / "condition_manifest.csv"
                ),
                real_embeddings=(
                    experiment_root
                    / "cache"
                    / "embeddings"
                    / dataset
                    / "train_uni_224.parquet"
                ),
                synthetic_manifest=manifest,
                synthetic_embeddings=(
                    manifest.parent / "metrics" / "synthetic_uni_224.parquet"
                ),
                output=manifest.parent / "metrics" / "uni_authenticity.json",
                dataset=dataset,
                model=model,
                conditioning=conditioning,
                seed=int(seed_name.removeprefix("seed")),
                family=(
                    "balanced_global"
                    if "balanced_global" in experiment_root.name
                    else "main"
                ),
            )
        )
    return jobs


def discover_audit_jobs(main_root: Path) -> list[AuthenticityInputs]:
    outputs = main_root / "outputs"
    audits = outputs / "audits"
    pcam_embeddings = (
        main_root / "cache" / "embeddings" / "pcam" / "train_uni_224.parquet"
    )
    jobs = []
    lora = audits / "pcam_sd_lora_resolution_balance"
    for arm in ("control_256", "native_512", "balanced_256"):
        arm_root = lora / "arms" / arm
        distribution = "balanced" if arm == "balanced_256" else "unbalanced"
        real_manifest = lora / "datasets" / distribution / "train_conditions.csv"
        real_embeddings = lora / "datasets" / distribution / "train_uni_224.parquet"
        materialize_embedding_subset(
            real_manifest, pcam_embeddings, real_embeddings
        )
        jobs.append(
            _audit_job(
                arm_root,
                real_manifest,
                real_embeddings,
                model="sd_lora",
                conditioning=f"label_global_cluster_{arm}",
            )
        )

    canonical_manifest = (
        outputs / "conditions" / "pcam" / "true_label" / "condition_manifest.csv"
    )
    jobs.append(
        _audit_job(
            audits / "pcam_sd_full_class_label",
            canonical_manifest,
            pcam_embeddings,
            model="sd_full",
            conditioning="true_label_uncurated",
        )
    )

    curated_root = audits / "pcam_sd_full_class_curated"
    curated_manifest = curated_root / "metrics" / "authenticity_training_manifest.csv"
    curated_embeddings = curated_root / "metrics" / "real_train_uni_224.parquet"
    prepare_curated_training_embeddings(
        curated_root, curated_manifest, curated_embeddings
    )
    jobs.append(
        _audit_job(
            curated_root,
            curated_manifest,
            curated_embeddings,
            model="sd_full",
            conditioning="true_label_curated",
        )
    )
    return jobs


def _audit_job(
    root: Path,
    real_manifest: Path,
    real_embeddings: Path,
    model: str,
    conditioning: str,
) -> AuthenticityInputs:
    return AuthenticityInputs(
        real_manifest=real_manifest,
        real_embeddings=real_embeddings,
        synthetic_manifest=root / "synthetic_manifest.csv",
        synthetic_embeddings=root / "metrics" / "synthetic_uni_224.parquet",
        output=root / "metrics" / "uni_authenticity.json",
        dataset="pcam",
        model=model,
        conditioning=conditioning,
        seed=0,
        family="audit",
    )


def materialize_embedding_subset(
    manifest: Path, source_embeddings: Path, output: Path
) -> Path:
    if output.exists():
        validate_embedding_cache(manifest, output)
        return output
    rows = pd.read_csv(manifest)
    source = pd.read_parquet(source_embeddings)
    if source["image_path"].duplicated().any():
        raise RuntimeError(f"Duplicate paths in {source_embeddings}")
    indexed = source.set_index(source["image_path"].astype(str))
    paths = rows["image_path"].astype(str).tolist()
    missing = set(paths) - set(indexed.index)
    if missing:
        raise RuntimeError(
            f"Source UNI cache is missing {len(missing)} requested images"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    indexed.loc[paths].reset_index(drop=True).to_parquet(output, index=False)
    validate_embedding_cache(manifest, output)
    return output


def prepare_curated_training_embeddings(
    audit_root: Path, manifest: Path, embeddings: Path
) -> Path:
    if manifest.exists() and embeddings.exists():
        validate_embedding_cache(manifest, embeddings)
        return embeddings
    config = yaml.safe_load((audit_root / "resolved_config.yaml").read_text())
    train_root = Path(config["config"]["pcam_sd_full"]["train_data_dir"])
    image_paths = sorted(
        path
        for path in train_root.rglob("*")
        if path.is_file() and path.name != "metadata.jsonl"
    )
    expected = int(config["config"]["pcam_sd_full"]["expected_train_samples"])
    if len(image_paths) != expected:
        raise RuntimeError(
            f"Expected {expected} curated training images, found {len(image_paths)}"
        )
    rows = [
        {"image_path": str(path), "split": "train", "label": path.parent.name}
        for path in image_paths
    ]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)

    from synpath.embeddings.uni import embed_rows, load_uni_model, resolve_device
    from torchvision import transforms

    device = resolve_device()
    model = load_uni_model(device)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    frame = embed_rows(
        rows,
        model,
        transform,
        device,
        batch_size=128,
        num_workers=min(32, os.cpu_count() or 8),
    )
    frame.to_parquet(embeddings, index=False)
    validate_embedding_cache(manifest, embeddings)
    return embeddings


def run_backfill(
    experiment_roots: Iterable[Path],
    main_root: Path,
    include_audits: bool,
    gpu_ids: list[int],
    batch_size: int,
    report_path: Path,
    resume: bool,
) -> Path:
    jobs = []
    for root in experiment_roots:
        jobs.extend(discover_generator_jobs(root))
    if include_audits:
        jobs.extend(discover_audit_jobs(main_root))
    _validate_inventory(jobs)
    if resume:
        jobs = [job for job in jobs if not job.output.exists()]
    cache_root = main_root / "cache" / "authenticity" / "real_neighbours"
    if jobs:
        _precompute_real_populations(jobs, cache_root, batch_size, gpu_ids[0])
        partitions = [[] for _ in gpu_ids]
        for index, job in enumerate(jobs):
            partitions[index % len(gpu_ids)].append(job)
        context = mp.get_context("spawn")
        processes = [
            context.Process(
                target=_worker,
                args=(partition, gpu, cache_root, batch_size),
            )
            for partition, gpu in zip(partitions, gpu_ids, strict=True)
            if partition
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Authenticity worker failed with exit code {process.exitcode}"
                )
    return write_report(
        [
            *(
                job
                for root in experiment_roots
                for job in discover_generator_jobs(root)
            ),
            *(discover_audit_jobs(main_root) if include_audits else []),
        ],
        report_path,
    )


def _precompute_real_populations(jobs, cache_root, batch_size, gpu_id):
    select_gpu(gpu_id)
    unique = {}
    for job in jobs:
        unique[(job.real_manifest, job.real_embeddings)] = job
    for job in unique.values():
        frame = validate_embedding_cache(job.real_manifest, job.real_embeddings)
        prepare_neighbour_cache(frame, cache_root, batch_size)


def _worker(jobs, gpu_id, cache_root, batch_size):
    select_gpu(gpu_id)
    for job in jobs:
        evaluate_authenticity(job, cache_root, batch_size)


def _validate_inventory(jobs: list[AuthenticityInputs]) -> None:
    if len(jobs) != 59:
        raise RuntimeError(f"Expected 59 generated outputs, discovered {len(jobs)}")
    outputs = [job.output for job in jobs]
    if len(set(outputs)) != len(outputs):
        raise RuntimeError("Authenticity inventory contains duplicate outputs")
    for job in jobs:
        for path in (
            job.real_manifest,
            job.real_embeddings,
            job.synthetic_manifest,
            job.synthetic_embeddings,
        ):
            if not path.exists():
                raise RuntimeError(f"Missing Authenticity input: {path}")


def write_report(jobs: list[AuthenticityInputs], output: Path) -> Path:
    rows = []
    for job in jobs:
        if not job.output.exists():
            raise RuntimeError(f"Missing Authenticity result: {job.output}")
        payload = pd.read_json(job.output, typ="series")
        rows.append(
            {
                **asdict(job),
                "authenticity": payload["authenticity"],
                "memorisation_risk": payload["memorisation_risk"],
                "authentic_count": payload["authentic_count"],
                "suspicious_count": payload["suspicious_count"],
                "real_samples": payload["real_samples"],
                "synthetic_samples": payload["synthetic_samples"],
            }
        )
    frame = pd.DataFrame(rows)
    for column in (
        "real_manifest",
        "real_embeddings",
        "synthetic_manifest",
        "synthetic_embeddings",
        "output",
    ):
        frame[column] = frame[column].astype(str)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(
        ["family", "dataset", "model", "conditioning", "seed"]
    ).to_csv(output, index=False)
    return output
