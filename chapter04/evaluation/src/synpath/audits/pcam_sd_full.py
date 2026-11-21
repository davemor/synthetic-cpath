from __future__ import annotations

from collections import Counter
from copy import deepcopy
import errno
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from synpath.artifacts import utc_now, write_json, write_yaml
from synpath.config import load_resolved_yaml
from synpath.embeddings.uni import feature_matrix
from synpath.generation.wrappers import (
    ensure_diffusers_repo,
    run_parallel_commands,
    train_sd_full,
)
from synpath.manifests import (
    CONDITION_COLUMNS,
    SYNTHETIC_COLUMNS,
    read_manifest,
    write_manifest,
)
from synpath.metrics.quality import (
    frechet_distance,
    inception_fid,
    load_or_extract_synthetic_embeddings,
    polynomial_mmd,
    precision_recall,
)
from synpath.paths import SynpathPaths


VALID_STAGES = {
    "prepare",
    "preflight",
    "train",
    "sample",
    "evaluate",
    "report",
    "all",
}


def run_pcam_sd_full(
    config_path: Path,
    stage: str = "all",
    resume: bool = False,
) -> list[Path]:
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown full-UNet stage: {stage}")
    cfg = load_resolved_yaml(config_path)
    audit = cfg["pcam_sd_full"]
    paths = SynpathPaths.from_env()
    validate_config(cfg, audit)
    ensure_diffusers_repo(paths)
    outputs = []
    if stage in {"prepare", "all"}:
        outputs.extend(prepare_training_data(audit, paths, resume))
    validate_training_data(audit)
    if stage in {"preflight", "all"}:
        outputs.extend(run_preflight(cfg, audit, paths, resume))
    if stage in {"train", "all"}:
        require_preflight(audit, paths)
        outputs.extend(run_training(cfg, audit, paths, resume))
    if stage in {"sample", "all"}:
        require_training(audit, paths)
        outputs.extend(run_sampling(audit, paths, resume))
    if stage in {"evaluate", "all"}:
        require_samples(audit, paths)
        outputs.extend(run_evaluation(audit, paths, resume))
    if stage in {"report", "all"}:
        require_evaluation(audit, paths)
        outputs.extend(write_report(audit, paths))
    return outputs


def validate_config(cfg: dict, audit: dict) -> None:
    sd_cfg = cfg.get("generation", {}).get("sd_full", {})
    expected = {
        "num_processes": 8,
        "train_resolution": 512,
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "max_train_steps": 12500,
        "dataloader_num_workers": 8,
    }
    for key, value in expected.items():
        if int(sd_cfg.get(key, -1)) != value:
            raise ValueError(f"PCam full-UNet training requires {key}={value}")
    if sd_cfg.get("mixed_precision") != "fp16":
        raise ValueError("PCam full-UNet training requires fp16")
    if float(sd_cfg.get("learning_rate", 0)) != 1e-5:
        raise ValueError("PCam full-UNet training requires learning_rate=1e-5")
    if not sd_cfg.get("gradient_checkpointing"):
        raise ValueError("PCam full-UNet training requires checkpointing")
    if int(audit.get("expected_train_samples", -1)) != 50000:
        raise ValueError("PCam full-UNet training requires 50,000 patches")


def validate_training_data(audit: dict) -> None:
    root = Path(audit["train_data_dir"])
    metadata = root / "metadata.jsonl"
    if not metadata.exists():
        raise RuntimeError(f"Missing training metadata: {metadata}")
    records = [json.loads(line) for line in metadata.read_text().splitlines()]
    expected = int(audit["expected_train_samples"])
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} training rows, got {len(records)}")
    prompts = Counter(record["text"] for record in records)
    expected_prompts = {
        str(prompt): int(count)
        for prompt, count in audit["expected_prompts"].items()
    }
    if prompts != expected_prompts:
        raise RuntimeError(
            f"Unexpected class-prompt distribution: {dict(prompts)}"
        )
    missing = [
        record["file_name"]
        for record in records
        if not (root / record["file_name"]).exists()
    ]
    if missing:
        raise RuntimeError(f"Training data has {len(missing)} missing images")


def prepare_training_data(
    audit: dict,
    paths: SynpathPaths,
    resume: bool,
) -> list[Path]:
    curation = audit.get("curation")
    if not curation:
        validate_training_data(audit)
        return [Path(audit["train_data_dir"]) / "metadata.jsonl"]
    root = Path(audit["train_data_dir"])
    marker = root.parent / "curation_complete.json"
    if resume and marker.exists():
        validate_training_data(audit)
        return [marker]
    manifest = pd.read_csv(Path(curation["curated_manifest"]))
    labels = np.load(Path(curation["cluster_labels"]))
    if len(manifest) != len(labels):
        raise RuntimeError("Curated manifest and cluster labels are misaligned")
    if len(manifest) != int(curation["expected_curated_population"]):
        raise RuntimeError("Unexpected curated PCam population size")
    manifest = manifest.copy()
    manifest["cluster_id"] = labels.astype(int)
    selected, selection = select_morphology_balanced_rows(
        manifest,
        clusters_per_class=int(curation["clusters_per_class"]),
        target=int(audit["expected_train_samples"]),
        seed=int(audit["seed"]),
    )
    root.mkdir(parents=True, exist_ok=True)
    records = []
    prompt_map = {
        "non_metastatic": "Histology image of healthy tissue",
        "metastatic": "Histology image of cancer tissue",
    }
    for index, row in selected.reset_index(drop=True).iterrows():
        class_name = "healthy" if row["label"] == "non_metastatic" else "cancer"
        relative = Path(class_name) / f"{index:08d}{Path(row['image_path']).suffix}"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                os.link(row["image_path"], destination)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                destination.symlink_to(row["image_path"])
        records.append(
            {
                "file_name": str(relative),
                "text": prompt_map[row["label"]],
            }
        )
    metadata = root / "metadata.jsonl"
    metadata.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    write_json(
        marker,
        {
            "cluster_labels": curation["cluster_labels"],
            "clusters_per_class": int(curation["clusters_per_class"]),
            "completed_at": utc_now(),
            "curated_manifest": curation["curated_manifest"],
            "expected_curated_population": int(
                curation["expected_curated_population"]
            ),
            "fit_metadata": curation["fit_metadata"],
            "k": 33,
            "prompts": prompt_map,
            "selection": selection,
            "selection_role": "morphology_balance_only_not_conditioning",
            "status": "complete",
            "training_samples": len(records),
        },
    )
    validate_training_data(audit)
    return [metadata, marker]


def select_morphology_balanced_rows(
    frame: pd.DataFrame,
    clusters_per_class: int,
    target: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    if target % 2:
        raise ValueError("Morphology-balanced target must be even")
    rng = np.random.default_rng(seed)
    selected = []
    metadata = {}
    target_per_class = target // 2
    for label in ("non_metastatic", "metastatic"):
        group = frame[frame["label"] == label]
        support = group["cluster_id"].value_counts().sort_values(ascending=False)
        retained = [int(value) for value in support.index[:clusters_per_class]]
        base, remainder = divmod(target_per_class, clusters_per_class)
        counts = {
            cluster: base + int(index < remainder)
            for index, cluster in enumerate(retained)
        }
        for cluster, count in counts.items():
            candidates = group[group["cluster_id"] == cluster]
            if len(candidates) < count:
                raise RuntimeError(
                    f"Cluster {cluster} for {label} has {len(candidates)} "
                    f"patches but requires {count}"
                )
            indices = rng.choice(len(candidates), size=count, replace=False)
            selected.append(candidates.iloc[indices])
        metadata[label] = {
            "retained_clusters": retained,
            "selected_counts": {str(key): value for key, value in counts.items()},
            "source_support": {
                str(cluster): int(support.loc[cluster]) for cluster in retained
            },
        }
    result = pd.concat(selected, ignore_index=True)
    result = result.iloc[rng.permutation(len(result))].reset_index(drop=True)
    if result["image_path"].duplicated().any():
        raise RuntimeError("Morphology-balanced selection contains duplicates")
    return result, metadata


def run_preflight(
    cfg: dict,
    audit: dict,
    paths: SynpathPaths,
    resume: bool,
) -> list[Path]:
    root = output_root(audit, paths) / "preflight"
    marker = root / "preflight_complete.json"
    if resume and marker.exists():
        return [marker]
    reused = audit.get("reuse_preflight_from")
    if reused:
        source = Path(reused)
        if not source.exists():
            raise RuntimeError(f"Missing reusable preflight marker: {source}")
        write_json(
            marker,
            {
                "completed_at": utc_now(),
                "reused_from": str(source),
                "resolution": 512,
                "status": "complete",
            },
        )
        return [marker]
    resolved = deepcopy(cfg)
    full_cfg = resolved["generation"]["sd_full"]
    full_cfg["max_train_steps"] = int(audit["preflight_steps"])
    full_cfg.pop("checkpointing_steps", None)
    full_cfg.pop("checkpoints_total_limit", None)
    train_sd_full(
        root,
        Path(audit["train_data_dir"]),
        paths,
        resolved,
        seed=int(audit["seed"]),
    )
    validate_pipeline(root / "sd_full")
    write_json(
        marker,
        {
            "completed_at": utc_now(),
            "effective_batch_size": 64,
            "resolution": 512,
            "status": "complete",
            "steps": int(audit["preflight_steps"]),
        },
    )
    return [root / "sd_full_train.log", marker]


def run_training(
    cfg: dict,
    audit: dict,
    paths: SynpathPaths,
    resume: bool,
) -> list[Path]:
    root = output_root(audit, paths)
    marker = root / "training_complete.json"
    if resume and marker.exists():
        validate_pipeline(root / "sd_full")
        return [marker]
    root.mkdir(parents=True, exist_ok=True)
    write_yaml(
        root / "resolved_config.yaml",
        {
            "config": cfg,
            "created_at": utc_now(),
            "frozen_components": ["vae", "clip_text_encoder"],
            "trainable_component": "full_unet",
        },
    )
    train_sd_full(
        root,
        Path(audit["train_data_dir"]),
        paths,
        cfg,
        seed=int(audit["seed"]),
        resume=resume,
    )
    validate_pipeline(root / "sd_full")
    write_json(
        marker,
        {
            "completed_at": utc_now(),
            "effective_batch_size": 64,
            "frozen_components": ["vae", "clip_text_encoder"],
            "mixed_precision": "fp16",
            "resolution": 512,
            "status": "complete",
            "steps": 12500,
            "trainable_component": "full_unet",
        },
    )
    return [root / "sd_full_train.log", root / "sd_full", marker]


def validate_pipeline(path: Path) -> None:
    required = [
        path / "model_index.json",
        path / "unet" / "config.json",
        path / "unet" / "diffusion_pytorch_model.safetensors",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise RuntimeError(f"Incomplete full SD pipeline: {missing}")
    if list(path.glob("*lora*.safetensors")):
        raise RuntimeError("Full-UNet output unexpectedly contains LoRA weights")


def run_sampling(
    audit: dict,
    paths: SynpathPaths,
    resume: bool,
) -> list[Path]:
    root = output_root(audit, paths)
    conditions = root / "sample_conditions.csv"
    samples = root / "samples"
    manifest = root / "synthetic_manifest.csv"
    if resume and manifest.exists():
        validate_samples(audit, manifest)
        return [manifest]
    rows = balanced_sample_conditions(audit)
    write_manifest(conditions, rows, CONDITION_COLUMNS)
    samples.mkdir(parents=True, exist_ok=True)
    commands = []
    num_gpus = int(audit["sample_gpus"])
    for shard in range(num_gpus):
        commands.append(
            (
                [
                    sys.executable,
                    "-m",
                    "synpath.generation.sd_full_sample",
                    "--model_dir",
                    str(root / "sd_full"),
                    "--condition_manifest",
                    str(conditions),
                    "--output_dir",
                    str(samples),
                    "--seed",
                    str(audit["sampling_seed"]),
                    "--batch_size",
                    str(audit["sample_batch_size"]),
                    "--height",
                    "512",
                    "--width",
                    "512",
                    "--num_inference_steps",
                    str(audit["num_inference_steps"]),
                    "--guidance_scale",
                    str(audit["guidance_scale"]),
                    "--num_shards",
                    str(num_gpus),
                    "--shard_index",
                    str(shard),
                ],
                root / f"sample_shard_{shard:02d}.log",
                {"CUDA_VISIBLE_DEVICES": str(shard)},
            )
        )
    run_parallel_commands(commands)
    synthetic_rows = []
    for index, row in enumerate(rows):
        image_path = samples / f"{index:08d}.png"
        if not image_path.exists():
            raise RuntimeError(f"Missing generated sample: {image_path}")
        synthetic_rows.append(
            {
                "image_path": str(image_path),
                "dataset": "pcam",
                "model": "sd_full",
                "conditioning": "true_label",
                "seed": audit["seed"],
                "label": row["label"],
                "label_id": row["label_id"],
                "cluster_id": "",
                "condition": row["condition"],
                "condition_id": row["condition_id"],
                "prompt": row["prompt"],
            }
        )
    write_manifest(manifest, synthetic_rows, SYNTHETIC_COLUMNS)
    validate_samples(audit, manifest)
    return [conditions, samples, manifest]


def balanced_sample_conditions(audit: dict) -> list[dict[str, str]]:
    rows = read_manifest(Path(audit["source_condition_manifest"]))
    target_per_label = int(audit["synthetic_samples"]) // 2
    rng = np.random.default_rng(int(audit["sampling_seed"]))
    selected = []
    for label in ("non_metastatic", "metastatic"):
        group = [row for row in rows if row["label"] == label]
        indices = rng.choice(len(group), size=target_per_label, replace=False)
        selected.extend(group[int(index)] for index in indices)
    rng.shuffle(selected)
    return selected


def run_evaluation(
    audit: dict,
    paths: SynpathPaths,
    resume: bool,
) -> list[Path]:
    root = output_root(audit, paths)
    metrics = root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    marker = metrics / "evaluation_complete.json"
    if resume and marker.exists():
        return [marker]
    real_rows = read_manifest(Path(audit["real_test_manifest"]))
    real_full = pd.read_parquet(Path(audit["test_embeddings"]))
    real_lookup = real_full.set_index("image_path", drop=False)
    missing = [
        row["image_path"]
        for row in real_rows
        if row["image_path"] not in real_lookup.index
    ]
    if missing:
        raise RuntimeError(f"Test UNI cache is missing {len(missing)} rows")
    real_frame = real_lookup.loc[
        [row["image_path"] for row in real_rows]
    ].reset_index(drop=True)
    synthetic = load_or_extract_synthetic_embeddings(
        root / "synthetic_manifest.csv",
        metrics / "synthetic_uni_224.parquet",
    )
    real_x = feature_matrix(real_frame)
    synthetic_x = feature_matrix(synthetic)
    uni_path = metrics / "uni_fid_kid.json"
    write_json(
        uni_path,
        {
            "real_samples": len(real_x),
            "synthetic_samples": len(synthetic_x),
            "uni_fid": frechet_distance(real_x, synthetic_x),
            "uni_kid": polynomial_mmd(real_x, synthetic_x),
        },
    )
    pr = precision_recall(
        real_x,
        synthetic_x,
        nearest_neighbors=int(audit["precision_recall_k"]),
        batch_size=int(audit["precision_recall_batch_size"]),
    )
    pr.update(
        {
            "cluster_labels_discarded": True,
            "real_samples": len(real_x),
            "synthetic_samples": len(synthetic_x),
        }
    )
    pr_path = metrics / "uni_precision_recall.json"
    write_json(pr_path, pr)
    fid_path = metrics / "inception_fid.json"
    write_json(
        fid_path,
        {
            **inception_fid(
                real_frame,
                synthetic,
                metrics / "inception_real",
                metrics / "inception_synthetic",
            ),
            "real_samples": len(real_x),
            "synthetic_samples": len(synthetic_x),
        },
    )
    write_json(
        marker,
        {
            "completed_at": utc_now(),
            "precision_recall_method": "improved_precision_recall",
            "status": "complete",
        },
    )
    return [uni_path, pr_path, fid_path, marker]


def write_report(audit: dict, paths: SynpathPaths) -> list[Path]:
    root = output_root(audit, paths)
    metrics = root / "metrics"
    fid = load_json(metrics / "inception_fid.json")
    uni = load_json(metrics / "uni_fid_kid.json")
    pr = load_json(metrics / "uni_precision_recall.json")
    result = {
        "dataset": "pcam",
        "model": "sd_full",
        "conditioning": "true_label",
        "fid": fid["fid"],
        "uni_fid": uni["uni_fid"],
        "uni_kid": uni["uni_kid"],
        "precision": pr["precision"],
        "recall": pr["recall"],
        "real_samples": fid["real_samples"],
        "synthetic_samples": fid["synthetic_samples"],
    }
    output = root / "results.json"
    write_json(output, result)
    return [output]


def validate_samples(audit: dict, manifest: Path) -> None:
    rows = read_manifest(manifest)
    expected = int(audit["synthetic_samples"])
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} samples, got {len(rows)}")
    counts = Counter(row["label"] for row in rows)
    if set(counts.values()) != {expected // 2}:
        raise RuntimeError(f"Generated class counts are not balanced: {counts}")
    missing = [row["image_path"] for row in rows if not Path(row["image_path"]).exists()]
    if missing:
        raise RuntimeError(f"Generated manifest has {len(missing)} missing images")


def require_preflight(audit: dict, paths: SynpathPaths) -> None:
    marker = output_root(audit, paths) / "preflight" / "preflight_complete.json"
    if not marker.exists():
        raise RuntimeError("Full-UNet 512 px preflight has not completed")


def require_training(audit: dict, paths: SynpathPaths) -> None:
    marker = output_root(audit, paths) / "training_complete.json"
    if not marker.exists():
        raise RuntimeError("Full-UNet training has not completed")
    validate_pipeline(output_root(audit, paths) / "sd_full")


def require_samples(audit: dict, paths: SynpathPaths) -> None:
    manifest = output_root(audit, paths) / "synthetic_manifest.csv"
    if not manifest.exists():
        raise RuntimeError("Full-UNet sampling has not completed")
    validate_samples(audit, manifest)


def require_evaluation(audit: dict, paths: SynpathPaths) -> None:
    marker = output_root(audit, paths) / "metrics" / "evaluation_complete.json"
    if not marker.exists():
        raise RuntimeError("Full-UNet evaluation has not completed")


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def output_root(audit: dict, paths: SynpathPaths) -> Path:
    return paths.outputs_root / audit["output_subdir"]
