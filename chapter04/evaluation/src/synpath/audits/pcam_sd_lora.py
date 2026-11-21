from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from synpath.artifacts import (
    file_sha256,
    utc_now,
    write_json,
    write_text,
    write_yaml,
)
from synpath.config import load_resolved_yaml
from synpath.embeddings.uni import feature_matrix
from synpath.generation.wrappers import (
    ensure_diffusers_repo,
    prepare_training_images,
    run_parallel_commands,
    train_sd_lora,
    write_metadata_jsonl,
)
from synpath.manifests import (
    CONDITION_COLUMNS,
    DATASET_COLUMNS,
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
    "validate",
    "sample",
    "evaluate",
    "report",
    "all",
}


def run_pcam_sd_lora_audit(
    config_path: Path,
    stage: str = "all",
    resume: bool = False,
) -> list[Path]:
    if stage not in VALID_STAGES:
        choices = ", ".join(sorted(VALID_STAGES))
        raise ValueError(
            f"Unknown PCam SD-LoRA audit stage: {stage}; {choices}"
        )
    cfg = load_resolved_yaml(config_path)
    audit = cfg["pcam_sd_lora_audit"]
    paths = SynpathPaths.from_env()
    validate_audit_config(cfg, audit)

    outputs: list[Path] = []
    if stage in {"prepare", "all"}:
        outputs.extend(prepare_audit(cfg, audit, paths, resume))
    if stage in {"preflight", "all"}:
        require_prepared(audit, paths)
        outputs.extend(run_preflight(cfg, audit, paths, resume))
    if stage in {"train", "all"}:
        require_prepared(audit, paths)
        if bool(audit.get("require_preflight", True)):
            require_preflight(audit, paths)
        outputs.extend(train_arms(cfg, audit, paths, resume))
    if stage in {"validate", "all"}:
        require_trained(audit, paths)
        outputs.extend(run_validation(config_path, cfg, audit, paths, resume))
    if stage in {"sample", "all"}:
        require_trained(audit, paths)
        outputs.extend(sample_arms(cfg, audit, paths, resume))
    if stage in {"evaluate", "all"}:
        require_sampled(audit, paths)
        outputs.extend(
            evaluate_arms(config_path, cfg, audit, paths, resume)
        )
    if stage in {"report", "all"}:
        outputs.extend(write_report(cfg, audit, paths))
    return outputs


def validate_audit_config(cfg: dict, audit: dict) -> None:
    sd_cfg = cfg.get("generation", {}).get("sd_lora", {})
    expected = {
        "rank": 16,
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "num_processes": 8,
        "max_train_steps": 5000,
    }
    for key, value in expected.items():
        if int(sd_cfg.get(key, -1)) != value:
            raise ValueError(f"PCam audit requires sd_lora.{key}={value}")
    if sd_cfg.get("mixed_precision") != "fp16":
        raise ValueError("PCam audit requires SD-LoRA fp16 training")
    if float(sd_cfg.get("learning_rate", 0.0)) != 1e-5:
        raise ValueError("PCam audit requires SD-LoRA learning_rate=1e-5")
    if not sd_cfg.get("gradient_checkpointing"):
        raise ValueError("PCam audit requires gradient checkpointing")
    if sd_cfg.get("training_image_mode") != "hardlink":
        raise ValueError("PCam audit requires native-image hardlinks")

    arms = audit.get("arms", {})
    required = {
        "control_256": (256, "unbalanced"),
        "native_512": (512, "unbalanced"),
        "balanced_256": (256, "balanced"),
    }
    if set(arms) != set(required):
        raise ValueError("PCam audit requires the three prespecified arms")
    for name, (resolution, distribution) in required.items():
        arm = arms[name]
        if int(arm.get("resolution", 0)) != resolution:
            raise ValueError(f"Unexpected resolution for {name}")
        if arm.get("prompt_distribution") != distribution:
            raise ValueError(f"Unexpected prompt distribution for {name}")


def prepare_audit(
    cfg: dict,
    audit: dict,
    paths: SynpathPaths,
    resume: bool,
) -> list[Path]:
    root = audit_root(audit, paths)
    marker = root / "prepare_complete.json"
    if resume and marker.exists():
        validate_prepared_artifacts(audit, paths)
        return [marker]

    source_path = Path(audit["source_condition_manifest"])
    dataset_path = Path(audit["dataset_manifest"])
    source_rows = read_manifest(source_path)
    if len(source_rows) != int(audit["expected_source_count"]):
        raise RuntimeError(
            f"Expected {audit['expected_source_count']} source conditions, "
            f"found {len(source_rows)}"
        )
    ensure_unique_paths(source_rows, "source condition manifest")
    dataset_rows = read_manifest(dataset_path)
    split_by_path = {row["image_path"]: row["split"] for row in dataset_rows}
    invalid = [
        row["image_path"]
        for row in source_rows
        if split_by_path.get(row["image_path"]) != "train"
    ]
    if invalid:
        raise RuntimeError(
            f"Source conditions include {len(invalid)} non-training paths"
        )

    seed = int(audit["selection_seed"])
    target = int(audit["train_samples"])
    unbalanced = select_proportional_rows(source_rows, target, seed)
    balanced = select_balanced_rows(
        source_rows,
        retained_clusters=audit["retained_clusters"],
        per_prompt=int(audit["balanced_samples_per_prompt"]),
        seed=seed,
    )
    if len(unbalanced) != target or len(balanced) != target:
        raise RuntimeError("Prepared training arms have an unexpected size")

    distributions = {"unbalanced": unbalanced, "balanced": balanced}
    outputs: list[Path] = []
    for name, rows in distributions.items():
        distribution_root = root / "datasets" / name
        manifest = distribution_root / "train_conditions.csv"
        write_manifest(manifest, rows, CONDITION_COLUMNS)
        prepare_training_images(
            manifest,
            distribution_root / "training_images",
            cfg,
        )
        write_metadata_jsonl(
            manifest,
            distribution_root / "training_images" / "metadata.jsonl",
        )
        outputs.append(manifest)

    for arm_name, arm in audit["arms"].items():
        distribution = arm["prompt_distribution"]
        rows = distributions[distribution]
        schedule = sample_schedule(
            rows,
            int(audit["synthetic_samples"]),
            uniform=distribution == "balanced",
            seed=int(audit["sampling_seed"]),
        )
        arm_path = arm_root(audit, paths, arm_name)
        manifest = arm_path / "sample_conditions.csv"
        write_manifest(manifest, schedule, CONDITION_COLUMNS)
        outputs.append(manifest)

    validation_manifest = prepare_validation_manifest(audit, paths)
    real_test_manifest = prepare_real_test_manifest(audit)
    outputs.extend([validation_manifest, real_test_manifest])

    metadata_path = root / "selection_metadata.json"
    source_counts = condition_counts(source_rows)
    balanced_counts = condition_counts(balanced)
    retained = set(balanced_counts)
    excluded = {
        condition: count
        for condition, count in source_counts.items()
        if condition not in retained
    }
    write_json(
        metadata_path,
        {
            "created_at": utc_now(),
            "source_condition_manifest": str(source_path),
            "source_condition_manifest_sha256": file_sha256(source_path),
            "dataset_manifest": str(dataset_path),
            "dataset_manifest_sha256": file_sha256(dataset_path),
            "source_samples": len(source_rows),
            "train_samples_per_arm": target,
            "source_prompt_counts": source_counts,
            "unbalanced_prompt_counts": condition_counts(unbalanced),
            "balanced_prompt_counts": balanced_counts,
            "excluded_prompt_counts": excluded,
            "selection_seed": seed,
            "selection_population": "existing_50k_pcam_training_subset",
            "cluster_scope": "global",
            "cluster_count": 4,
            "training_split_only": True,
            "runner_versions": runner_versions(),
        },
    )
    write_json(
        marker,
        {
            "created_at": utc_now(),
            "metadata": str(metadata_path),
            "status": "complete",
        },
    )
    validate_prepared_artifacts(audit, paths)
    return [*outputs, metadata_path, marker]


def select_proportional_rows(
    rows: list[dict[str, str]], target: int, seed: int
) -> list[dict[str, str]]:
    groups = group_rows(rows)
    allocations = proportional_group_counts(
        {condition: len(values) for condition, values in groups.items()},
        target,
    )
    rng = np.random.default_rng(seed)
    selected: list[dict[str, str]] = []
    for condition in sorted(groups):
        values = groups[condition]
        take = allocations[condition]
        indices = rng.choice(len(values), size=take, replace=False)
        selected.extend(values[int(index)] for index in sorted(indices))
    return shuffled_rows(selected, seed)


def proportional_group_counts(
    counts: dict[str, int], target: int
) -> dict[str, int]:
    total = sum(counts.values())
    if target < 1 or target > total:
        raise ValueError("Proportional target must be between 1 and total rows")
    exact = {key: target * value / total for key, value in counts.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    remainder = target - sum(result.values())
    order = sorted(
        counts,
        key=lambda key: (-(exact[key] - result[key]), key),
    )
    for key in order[:remainder]:
        result[key] += 1
    return result


def select_balanced_rows(
    rows: list[dict[str, str]],
    retained_clusters: dict[str, list[int]],
    per_prompt: int,
    seed: int,
) -> list[dict[str, str]]:
    retained = {
        (str(label), str(cluster))
        for label, clusters in retained_clusters.items()
        for cluster in clusters
    }
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (str(row["label"]), str(row["cluster_id"]))
        if key in retained:
            groups[key].append(row)
    missing = retained - set(groups)
    if missing:
        raise RuntimeError(f"Retained prompts are absent: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    selected: list[dict[str, str]] = []
    for key in sorted(retained):
        values = groups[key]
        if len(values) < per_prompt:
            raise RuntimeError(
                f"Prompt {key} has {len(values)} rows; requires {per_prompt}"
            )
        indices = rng.choice(len(values), size=per_prompt, replace=False)
        selected.extend(values[int(index)] for index in sorted(indices))
    return shuffled_rows(selected, seed)


def sample_schedule(
    rows: list[dict[str, str]],
    target: int,
    uniform: bool,
    seed: int,
) -> list[dict[str, str]]:
    groups = group_rows(rows)
    if uniform:
        base, remainder = divmod(target, len(groups))
        allocations = {
            condition: base + (index < remainder)
            for index, condition in enumerate(sorted(groups))
        }
    else:
        allocations = proportional_group_counts(
            {condition: len(values) for condition, values in groups.items()},
            target,
        )
    rng = np.random.default_rng(seed)
    selected = []
    for condition in sorted(groups):
        values = groups[condition]
        take = allocations[condition]
        if take > len(values):
            raise RuntimeError(f"Sample schedule overdraws {condition}")
        indices = rng.choice(len(values), size=take, replace=False)
        selected.extend(values[int(index)] for index in sorted(indices))
    return shuffled_rows(selected, seed)


def group_rows(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[str(row["condition"])].append(row)
    return dict(groups)


def shuffled_rows(
    rows: list[dict[str, str]], seed: int
) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    return [rows[int(index)] for index in order]


def condition_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["condition"] for row in rows).items()))


def prepare_validation_manifest(audit: dict, paths: SynpathPaths) -> Path:
    embedding_path = Path(audit["validation_embeddings"])
    model_path = Path(audit["cluster_model"])
    dataset_rows = read_manifest(Path(audit["dataset_manifest"]))
    val_paths = {
        row["image_path"] for row in dataset_rows if row["split"] == "val"
    }
    frame = pd.read_parquet(embedding_path)
    if len(frame) != len(val_paths):
        raise RuntimeError("Validation embedding count does not match manifest")
    if set(frame["image_path"].astype(str)) != val_paths:
        raise RuntimeError("Validation embedding IDs do not match manifest")
    bundle = joblib.load(model_path)
    labels = bundle["models"].predict(feature_matrix(frame)).astype(int)
    frame = frame[["image_path", "label", "label_id"]].copy()
    frame["cluster_id"] = labels
    rows = []
    for row in frame.to_dict("records"):
        condition = f"{row['label']}__cluster_{row['cluster_id']}"
        rows.append(
            {
                "image_path": row["image_path"],
                "label": row["label"],
                "label_id": row["label_id"],
                "cluster_id": row["cluster_id"],
                "condition": condition,
                "condition_id": 0,
                "prompt": (
                    f"a histopathology patch of {row['label']}, "
                    f"morphology cluster {row['cluster_id']}"
                ),
            }
        )
    selected = select_balanced_rows(
        rows,
        retained_clusters=audit["retained_clusters"],
        per_prompt=int(audit["validation_samples_per_prompt"]),
        seed=int(audit["selection_seed"]),
    )
    condition_ids = {
        condition: index
        for index, condition in enumerate(
            sorted({row["condition"] for row in selected})
        )
    }
    for row in selected:
        row["condition_id"] = condition_ids[row["condition"]]
    output = audit_root(audit, paths) / "validation_conditions.csv"
    write_manifest(output, selected, CONDITION_COLUMNS)
    return output


def prepare_real_test_manifest(audit: dict) -> Path:
    rows = [
        row
        for row in read_manifest(Path(audit["dataset_manifest"]))
        if row["split"] == "test"
    ]
    target = int(audit["real_test_samples"])
    rng = np.random.default_rng(int(audit["evaluation_seed"]))
    indices = sorted(rng.choice(len(rows), size=target, replace=False))
    selected = [rows[int(index)] for index in indices]
    output = Path(audit["real_test_manifest"])
    write_manifest(output, selected, DATASET_COLUMNS)
    return output


def run_preflight(
    cfg: dict,
    audit: dict,
    paths: SynpathPaths,
    resume: bool,
) -> list[Path]:
    root = audit_root(audit, paths) / "preflight_512"
    marker = root / "preflight_complete.json"
    if resume and marker.exists():
        return [marker]
    preflight_cfg = arm_config(cfg, audit["arms"]["native_512"])
    preflight_cfg["generation"]["sd_lora"]["max_train_steps"] = int(
        audit["preflight_steps"]
    )
    preflight_cfg["generation"]["sd_lora"].pop("checkpointing_steps", None)
    ensure_diffusers_repo(paths)
    train_sd_lora(
        root,
        distribution_root(audit, paths, "unbalanced") / "training_images",
        distribution_root(audit, paths, "unbalanced") / "train_conditions.csv",
        paths,
        preflight_cfg,
        seed=int(audit["training_seed"]),
    )
    rank = inspect_lora_rank(root / "sd_lora")
    if rank != 16:
        raise RuntimeError(f"512 preflight produced LoRA rank {rank}")
    validate_training_log(
        root / "sd_lora_train.log",
        expected_steps=int(audit["preflight_steps"]),
    )
    write_json(
        marker,
        {
            "completed_at": utc_now(),
            "lora_rank": rank,
            "resolution": 512,
            "status": "complete",
        },
    )
    return [root / "sd_lora_train.log", marker]


def train_arms(
    cfg: dict,
    audit: dict,
    paths: SynpathPaths,
    resume: bool,
) -> list[Path]:
    ensure_diffusers_repo(paths)
    outputs: list[Path] = []
    for arm_name, arm in audit["arms"].items():
        root = arm_root(audit, paths, arm_name)
        marker = root / "training_complete.json"
        final_adapter = root / "sd_lora" / "pytorch_lora_weights.safetensors"
        if resume and (marker.exists() or final_adapter.exists()):
            validate_trained_arm(cfg, audit, paths, arm_name)
            if not marker.exists():
                write_training_marker(marker, arm_name, arm)
            outputs.append(marker)
            continue
        distribution = arm["prompt_distribution"]
        data_root = distribution_root(audit, paths, distribution)
        resolved = arm_config(cfg, arm)
        write_yaml(
            root / "resolved_config.yaml",
            {
                "audit": audit,
                "arm": {"name": arm_name, **arm},
                "created_at": utc_now(),
                "resolved_generation": resolved["generation"],
                "runner_versions": runner_versions(),
            },
        )
        train_sd_lora(
            root,
            data_root / "training_images",
            data_root / "train_conditions.csv",
            paths,
            resolved,
            seed=int(audit["training_seed"]),
        )
        validate_trained_arm(cfg, audit, paths, arm_name)
        write_training_marker(marker, arm_name, arm)
        outputs.extend(
            [root / "resolved_config.yaml", root / "sd_lora_train.log", marker]
        )
    return outputs


def write_training_marker(marker: Path, arm_name: str, arm: dict) -> None:
    write_json(
        marker,
        {
            "arm": arm_name,
            "completed_at": utc_now(),
            "effective_batch_size": 64,
            "lora_rank": 16,
            "mixed_precision": "fp16",
            "resolution": int(arm["resolution"]),
            "status": "complete",
        },
    )


def arm_config(cfg: dict, arm: dict) -> dict:
    resolved = deepcopy(cfg)
    sd_cfg = resolved["generation"]["sd_lora"]
    resolution = int(arm["resolution"])
    sd_cfg["train_resolution"] = resolution
    sd_cfg["sample_height"] = resolution
    sd_cfg["sample_width"] = resolution
    return resolved


def inspect_lora_rank(root: Path) -> int:
    from safetensors import safe_open

    candidates = sorted(root.glob("*lora*.safetensors"))
    if not candidates:
        raise RuntimeError(f"No LoRA safetensors found under {root}")
    ranks = set()
    with safe_open(candidates[0], framework="pt") as weights:
        for key in weights.keys():
            tensor = weights.get_tensor(key)
            if tensor.ndim == 2 and "lora" in key.lower():
                ranks.add(min(tensor.shape))
    if len(ranks) != 1:
        raise RuntimeError(f"Ambiguous LoRA ranks under {root}: {ranks}")
    return int(ranks.pop())


def validate_training_log(path: Path, expected_steps: int) -> None:
    text = path.read_text(errors="replace")
    expected = [
        "Num processes: 8",
        "Mixed precision type: fp16",
        "Instantaneous batch size per device = 8",
        "Total train batch size (w. parallel, distributed & accumulation) = 64",
        f"Total optimization steps = {expected_steps}",
    ]
    missing = [value for value in expected if value not in text]
    if missing:
        raise RuntimeError(f"Training log is missing invariants: {missing}")


def validate_trained_arm(
    cfg: dict,
    audit: dict,
    paths: SynpathPaths,
    arm_name: str,
) -> None:
    root = arm_root(audit, paths, arm_name)
    rank = inspect_lora_rank(root / "sd_lora")
    if rank != 16:
        raise RuntimeError(f"{arm_name} produced LoRA rank {rank}")
    steps = int(cfg["generation"]["sd_lora"]["max_train_steps"])
    validate_training_log(root / "sd_lora_train.log", steps)
    for step in audit["checkpoints"]:
        checkpoint = root / "sd_lora" / f"checkpoint-{int(step)}"
        if inspect_lora_rank(checkpoint) != 16:
            raise RuntimeError(f"Unexpected LoRA rank at {checkpoint}")


def run_validation(
    config_path: Path,
    cfg: dict,
    audit: dict,
    paths: SynpathPaths,
    resume: bool,
) -> list[Path]:
    num_shards = int(audit["validation_num_shards"])
    outputs: list[Path] = []
    for arm_name in audit["arms"]:
        root = arm_root(audit, paths, arm_name) / "validation_loss"
        commands = []
        for shard in range(num_shards):
            output = root / f"losses_shard_{shard:02d}.parquet"
            if resume and output.exists():
                continue
            commands.append(
                (
                    [
                        sys.executable,
                        "-m",
                        "synpath.audits.pcam_sd_lora",
                        "validation-worker",
                        "--config",
                        str(config_path),
                        "--arm",
                        arm_name,
                        "--num-shards",
                        str(num_shards),
                        "--shard-index",
                        str(shard),
                    ],
                    root / f"worker_{shard:02d}.log",
                    {"CUDA_VISIBLE_DEVICES": str(shard)},
                )
            )
        if commands:
            run_parallel_commands(commands)
        outputs.extend(aggregate_validation(audit, paths, arm_name, num_shards))
    return outputs


def validation_worker(
    config_path: Path,
    arm_name: str,
    num_shards: int,
    shard_index: int,
) -> Path:
    import torch
    import torch.nn.functional as functional
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    cfg = load_resolved_yaml(config_path)
    audit = cfg["pcam_sd_lora_audit"]
    paths = SynpathPaths.from_env()
    arm = audit["arms"][arm_name]
    rows = read_manifest(audit_root(audit, paths) / "validation_conditions.csv")
    indexed = [
        (index, row)
        for index, row in enumerate(rows)
        if index % num_shards == shard_index
    ]
    if not indexed:
        raise RuntimeError(f"Empty validation shard {shard_index}")
    resolution = int(arm["resolution"])

    class ValidationDataset(Dataset):
        def __init__(self, values):
            self.values = values
            self.transform = transforms.Compose(
                [
                    transforms.Resize((resolution, resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            global_index, row = self.values[index]
            image = Image.open(row["image_path"]).convert("RGB")
            return global_index, self.transform(image), row["prompt"]

    validation_cfg = audit["validation_loss"]
    loader = DataLoader(
        ValidationDataset(indexed),
        batch_size=int(validation_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(validation_cfg["num_workers"]),
        pin_memory=True,
    )
    device = torch.device("cuda")
    base_model = cfg["generation"]["sd_lora"]["base_model"]
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    for module in (pipe.vae, pipe.text_encoder, pipe.unet):
        module.eval().requires_grad_(False)

    cached = []
    seed = int(validation_cfg["seed"])
    with torch.inference_mode():
        for indices, images, prompts in loader:
            images = images.to(device, dtype=torch.float16, non_blocking=True)
            distribution = pipe.vae.encode(images).latent_dist
            posterior = deterministic_noise(
                distribution.mean.shape,
                indices,
                seed,
                dtype=torch.float16,
            ).to(device)
            latents = distribution.mean + distribution.std * posterior
            latents = latents * pipe.vae.config.scaling_factor
            input_ids = pipe.tokenizer(
                list(prompts),
                max_length=pipe.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            hidden = pipe.text_encoder(input_ids, return_dict=False)[0]
            noise = deterministic_noise(
                latents.shape,
                indices,
                seed + 10_000_000,
                dtype=torch.float16,
            ).to(device)
            timesteps = deterministic_timesteps(
                indices,
                seed + 20_000_000,
                int(scheduler.config.num_train_timesteps),
            ).to(device)
            noisy = scheduler.add_noise(latents, noise, timesteps)
            cached.append((indices, noisy, noise, timesteps, hidden))

    pipe.vae.to("cpu")
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    records = []
    steps = [0, *[int(value) for value in audit["checkpoints"]]]
    for step in steps:
        if step:
            checkpoint = (
                arm_root(audit, paths, arm_name)
                / "sd_lora"
                / f"checkpoint-{step}"
            )
            pipe.load_lora_weights(str(checkpoint))
        pipe.unet.eval()
        with torch.inference_mode():
            for indices, noisy, target, timesteps, hidden in cached:
                prediction = pipe.unet(
                    noisy,
                    timesteps,
                    hidden,
                    return_dict=False,
                )[0]
                losses = functional.mse_loss(
                    prediction.float(), target.float(), reduction="none"
                ).mean(dim=(1, 2, 3))
                records.extend(
                    {
                        "sample_index": int(index),
                        "step": step,
                        "timestep": int(timestep),
                        "loss": float(loss),
                    }
                    for index, timestep, loss in zip(
                        indices.tolist(),
                        timesteps.tolist(),
                        losses.tolist(),
                        strict=True,
                    )
                )
        if step:
            pipe.unload_lora_weights()
    output = (
        arm_root(audit, paths, arm_name)
        / "validation_loss"
        / f"losses_shard_{shard_index:02d}.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values(["step", "sample_index"]).to_parquet(
        output, index=False
    )
    return output


def deterministic_noise(shape, indices, seed: int, dtype):
    import torch

    rows = []
    for index in indices.tolist():
        generator = torch.Generator(device="cpu").manual_seed(seed + int(index))
        rows.append(torch.randn(shape[1:], generator=generator, dtype=dtype))
    return torch.stack(rows)


def deterministic_timesteps(indices, seed: int, count: int):
    import torch

    values = []
    for index in indices.tolist():
        generator = torch.Generator(device="cpu").manual_seed(seed + int(index))
        values.append(torch.randint(0, count, (1,), generator=generator).item())
    return torch.tensor(values, dtype=torch.long)


def aggregate_validation(
    audit: dict,
    paths: SynpathPaths,
    arm_name: str,
    num_shards: int,
) -> list[Path]:
    root = arm_root(audit, paths, arm_name) / "validation_loss"
    shard_paths = [
        root / f"losses_shard_{i:02d}.parquet" for i in range(num_shards)
    ]
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing validation shards: {missing}")
    frame = pd.concat([pd.read_parquet(path) for path in shard_paths])
    expected = int(audit["validation_samples_per_prompt"]) * 6
    counts = frame.groupby("step")["sample_index"].nunique()
    if not (counts == expected).all():
        raise RuntimeError(f"Unexpected validation counts: {counts.to_dict()}")
    per_sample = root / "validation_loss_per_sample.parquet"
    frame.to_parquet(per_sample, index=False)
    summary = (
        frame.groupby("step")["loss"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(
            columns={"mean": "validation_loss", "std": "validation_loss_sd"}
        )
    )
    summary["arm"] = arm_name
    summary_path = root / "validation_loss.csv"
    summary.to_csv(summary_path, index=False)
    return [per_sample, summary_path]


def sample_arms(
    cfg: dict,
    audit: dict,
    paths: SynpathPaths,
    resume: bool,
) -> list[Path]:
    outputs = []
    sample_gpus = int(audit["sample_gpus"])
    for arm_name, arm in audit["arms"].items():
        root = arm_root(audit, paths, arm_name)
        manifest = root / "sample_conditions.csv"
        sample_dir = root / "samples"
        synthetic_manifest = root / "synthetic_manifest.csv"
        if resume and synthetic_manifest.exists():
            validate_synthetic_manifest(audit, synthetic_manifest)
            outputs.append(synthetic_manifest)
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        commands = []
        for shard in range(sample_gpus):
            commands.append(
                (
                    [
                        sys.executable,
                        "-m",
                        "synpath.generation.sd_sample",
                        "--lora_dir",
                        str(root / "sd_lora"),
                        "--condition_manifest",
                        str(manifest),
                        "--output_dir",
                        str(sample_dir),
                        "--max_samples",
                        str(audit["synthetic_samples"]),
                        "--seed",
                        str(audit["sampling_seed"]),
                        "--batch_size",
                        str(audit["sample_batch_size"]),
                        "--height",
                        str(arm["resolution"]),
                        "--width",
                        str(arm["resolution"]),
                        "--num_inference_steps",
                        str(audit["num_inference_steps"]),
                        "--guidance_scale",
                        str(audit["guidance_scale"]),
                        "--num_shards",
                        str(sample_gpus),
                        "--shard_index",
                        str(shard),
                    ],
                    root / f"sample_shard_{shard:02d}.log",
                    {"CUDA_VISIBLE_DEVICES": str(shard)},
                )
            )
        run_parallel_commands(commands)
        write_audit_synthetic_manifest(
            audit, arm_name, manifest, sample_dir, synthetic_manifest
        )
        validate_synthetic_manifest(audit, synthetic_manifest)
        outputs.extend([sample_dir, synthetic_manifest])
    return outputs


def write_audit_synthetic_manifest(
    audit: dict,
    arm_name: str,
    conditions_path: Path,
    sample_dir: Path,
    output: Path,
) -> None:
    rows = []
    for index, source in enumerate(read_manifest(conditions_path)):
        image_path = sample_dir / f"{index:08d}.png"
        if not image_path.exists():
            raise RuntimeError(f"Missing generated sample: {image_path}")
        rows.append(
            {
                "image_path": str(image_path),
                "dataset": "pcam",
                "model": "sd_lora",
                "conditioning": f"label_global_cluster__{arm_name}",
                "seed": audit["training_seed"],
                "label": source["label"],
                "label_id": source["label_id"],
                "cluster_id": source["cluster_id"],
                "condition": source["condition"],
                "condition_id": source["condition_id"],
                "prompt": source["prompt"],
            }
        )
    write_manifest(output, rows, SYNTHETIC_COLUMNS)


def evaluate_arms(
    config_path: Path,
    cfg: dict,
    audit: dict,
    paths: SynpathPaths,
    resume: bool,
) -> list[Path]:
    commands = []
    outputs = []
    for gpu_index, arm_name in enumerate(audit["arms"]):
        root = arm_root(audit, paths, arm_name)
        marker = root / "metrics" / "evaluation_complete.json"
        if resume and marker.exists():
            outputs.append(marker)
            continue
        commands.append(
            (
                [
                    sys.executable,
                    "-m",
                    "synpath.audits.pcam_sd_lora",
                    "evaluation-worker",
                    "--config",
                    str(config_path),
                    "--arm",
                    arm_name,
                ],
                root / "evaluation.log",
                {"CUDA_VISIBLE_DEVICES": str(gpu_index)},
            )
        )
    if commands:
        run_parallel_commands(commands)
    for arm_name in audit["arms"]:
        marker = (
            arm_root(audit, paths, arm_name)
            / "metrics"
            / "evaluation_complete.json"
        )
        if not marker.exists():
            raise RuntimeError(f"Evaluation did not complete for {arm_name}")
        if marker not in outputs:
            outputs.append(marker)
    return outputs


def evaluate_arm(
    cfg: dict,
    audit: dict,
    paths: SynpathPaths,
    arm_name: str,
) -> list[Path]:
    real_manifest = Path(audit["real_test_manifest"])
    real_rows = read_manifest(real_manifest)
    real_full = pd.read_parquet(Path(audit["test_embeddings"]))
    real_lookup = real_full.set_index("image_path", drop=False)
    missing = [
        row["image_path"]
        for row in real_rows
        if row["image_path"] not in real_lookup.index
    ]
    if missing:
        raise RuntimeError(
            f"Test UNI cache is missing {len(missing)} audit rows"
        )
    real_frame = real_lookup.loc[
        [row["image_path"] for row in real_rows]
    ].reset_index(drop=True)
    real_x = feature_matrix(real_frame)
    root = arm_root(audit, paths, arm_name)
    metrics_root = root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    marker = metrics_root / "evaluation_complete.json"
    synthetic = load_or_extract_synthetic_embeddings(
        root / "synthetic_manifest.csv",
        metrics_root / "synthetic_uni_224.parquet",
    )
    synthetic_x = feature_matrix(synthetic)
    if len(synthetic_x) != int(audit["synthetic_samples"]):
        raise RuntimeError(f"Unexpected synthetic count for {arm_name}")
    uni_path = metrics_root / "uni_fid_kid.json"
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
    pr_path = metrics_root / "uni_precision_recall.json"
    write_json(pr_path, pr)
    fid_path = metrics_root / "inception_fid.json"
    write_json(
        fid_path,
        {
            **inception_fid(
                real_frame,
                synthetic,
                metrics_root / "inception_real",
                metrics_root / "inception_synthetic",
            ),
            "real_samples": len(real_x),
            "synthetic_samples": len(synthetic_x),
        },
    )
    write_json(
        marker,
        {
            "arm": arm_name,
            "cluster_labels_discarded": True,
            "completed_at": utc_now(),
            "precision_recall_method": "improved_precision_recall",
            "status": "complete",
        },
    )
    return [uni_path, pr_path, fid_path, marker]


def write_report(cfg: dict, audit: dict, paths: SynpathPaths) -> list[Path]:
    root = audit_root(audit, paths)
    records = []
    validation = []
    for arm_name, arm in audit["arms"].items():
        metrics = arm_root(audit, paths, arm_name) / "metrics"
        fid = load_json(metrics / "inception_fid.json")
        uni = load_json(metrics / "uni_fid_kid.json")
        pr = load_json(metrics / "uni_precision_recall.json")
        records.append(
            {
                "arm": arm_name,
                "resolution": int(arm["resolution"]),
                "prompt_distribution": arm["prompt_distribution"],
                "fid": fid["fid"],
                "uni_fid": uni["uni_fid"],
                "uni_kid": uni["uni_kid"],
                "precision": pr["precision"],
                "recall": pr["recall"],
            }
        )
        validation.append(
            pd.read_csv(
                arm_root(audit, paths, arm_name)
                / "validation_loss"
                / "validation_loss.csv"
            )
        )
    frame = pd.DataFrame(records)
    control = frame.set_index("arm").loc["control_256"]
    for metric in ("fid", "uni_fid", "uni_kid", "precision", "recall"):
        frame[f"delta_{metric}_vs_control"] = frame[metric] - control[metric]
    results_path = root / "results.csv"
    validation_path = root / "validation_loss.csv"
    frame.to_csv(results_path, index=False)
    validation_frame = pd.concat(validation, ignore_index=True)
    validation_frame.to_csv(validation_path, index=False)

    quality_figure = root / "quality_comparison.png"
    validation_figure = root / "validation_loss.png"
    prompt_figure = root / "prompt_counts.png"
    sample_figure = root / "sample_grid.png"
    plot_quality(frame, quality_figure)
    plot_validation(validation_frame, validation_figure)
    plot_prompt_counts(audit, paths, prompt_figure)
    plot_sample_grid(audit, paths, sample_figure)

    historical = load_historical_context(paths)
    historical_path = root / "historical_context.json"
    write_json(historical_path, historical)
    report_path = root / "report.md"
    write_text(
        report_path,
        markdown_report(cfg, audit, frame, historical),
    )
    return [
        results_path,
        validation_path,
        quality_figure,
        validation_figure,
        prompt_figure,
        sample_figure,
        historical_path,
        report_path,
    ]


def plot_quality(frame: pd.DataFrame, output: Path) -> None:
    metrics = ["fid", "uni_fid", "uni_kid", "precision", "recall"]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    labels = frame["arm"].str.replace("_", " ")
    for axis, metric in zip(axes.flat, metrics, strict=False):
        axis.bar(labels, frame[metric])
        axis.set_title(metric.replace("_", " ").upper())
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    axes.flat[-1].set_visible(False)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_validation(frame: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for arm, values in frame.groupby("arm"):
        axis.plot(
            values["step"], values["validation_loss"], marker="o", label=arm
        )
    axis.set_xlabel("Training step")
    axis.set_ylabel("Validation denoising MSE")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_prompt_counts(audit: dict, paths: SynpathPaths, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, distribution in zip(
        axes, ("unbalanced", "balanced"), strict=True
    ):
        path = (
            distribution_root(audit, paths, distribution)
            / "train_conditions.csv"
        )
        counts = condition_counts(read_manifest(path))
        axis.barh(list(counts), list(counts.values()))
        axis.set_title(distribution.capitalize())
        axis.set_xlabel("Training patches")
        axis.grid(axis="x", alpha=0.25)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_sample_grid(audit: dict, paths: SynpathPaths, output: Path) -> None:
    arms = list(audit["arms"])
    figure, axes = plt.subplots(
        len(arms), 8, figsize=(14, 6), constrained_layout=True
    )
    for row_index, arm_name in enumerate(arms):
        manifest = read_manifest(
            arm_root(audit, paths, arm_name) / "synthetic_manifest.csv"
        )
        indices = np.linspace(0, len(manifest) - 1, 8, dtype=int)
        for column, index in enumerate(indices):
            axes[row_index, column].imshow(
                Image.open(manifest[index]["image_path"])
            )
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_title(arm_name.replace("_", " "), loc="left")
    figure.savefig(output, dpi=200)
    plt.close(figure)


def markdown_report(
    cfg: dict, audit: dict, frame: pd.DataFrame, historical: dict
) -> str:
    lines = [
        "# PCam SD-LoRA Resolution and Prompt-Balancing Audit",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Protocol",
        "",
        "- Generator: Stable Diffusion 1.5 with rank-16 UNet LoRA.",
        "- Text encoder: frozen.",
        "- Training: 5,000 steps, fp16, effective batch size 64, seed 0.",
        "- Evaluation: 5,000 generated and 5,000 fixed real PCam test patches.",
        "- Morphology labels were discarded for aggregate quality metrics.",
        "- This is a single-seed diagnostic, not a main-experiment replacement.",
        "",
        "## Results",
        "",
        frame.to_markdown(index=False),
        "",
        "## Interpretation constraints",
        "",
        "The balanced arm changes both training-prompt and generation-prompt "
        "distributions. Validation losses at 256 and 512 use different latent "
        "spatial sizes and should be interpreted within each curve. The historical "
        "result is contextual only because it used rank 4 and fp32.",
        "",
        "## Historical context",
        "",
        f"```json\n{json.dumps(historical, indent=2, sort_keys=True)}\n```",
        "",
    ]
    return "\n".join(lines)


def load_historical_context(paths: SynpathPaths) -> dict[str, Any]:
    root = (
        paths.outputs_root
        / "generators"
        / "pcam"
        / "sd_lora"
        / "label_global_cluster"
        / "seed0"
    )
    result: dict[str, Any] = {
        "path": str(root),
        "role": "context_only_not_causal_control",
    }
    metric_paths = {
        "inception": root / "metrics" / "inception_fid.json",
        "uni": root / "metrics" / "uni_fid_kid.json",
        "precision_recall": root / "metrics" / "uni_precision_recall.json",
    }
    for key, path in metric_paths.items():
        result[key] = (
            load_json(path) if path.exists() else {"status": "missing"}
        )
    try:
        result["lora_rank"] = inspect_lora_rank(root / "sd_lora")
    except RuntimeError:
        result["lora_rank"] = None
    log = root / "sd_lora_train.log"
    if log.exists():
        text = log.read_text(errors="replace")
        result["mixed_precision"] = (
            "fp16" if "Mixed precision type: fp16" in text else "no"
        )
    return result


def validate_prepared_artifacts(audit: dict, paths: SynpathPaths) -> None:
    expected = int(audit["train_samples"])
    for distribution in ("unbalanced", "balanced"):
        manifest = (
            distribution_root(audit, paths, distribution)
            / "train_conditions.csv"
        )
        rows = read_manifest(manifest)
        if len(rows) != expected:
            raise RuntimeError(f"Unexpected {distribution} training count")
        ensure_unique_paths(rows, f"{distribution} training manifest")
        image_root = (
            distribution_root(audit, paths, distribution) / "training_images"
        )
        images = list(image_root.glob("*/*.png"))
        if len(images) != expected:
            raise RuntimeError(f"Unexpected {distribution} hardlink count")
    balanced = read_manifest(
        distribution_root(audit, paths, "balanced") / "train_conditions.csv"
    )
    expected_per_prompt = int(audit["balanced_samples_per_prompt"])
    if set(condition_counts(balanced).values()) != {expected_per_prompt}:
        raise RuntimeError("Balanced prompt supports are not exact")
    for arm_name in audit["arms"]:
        schedule = read_manifest(
            arm_root(audit, paths, arm_name) / "sample_conditions.csv"
        )
        if len(schedule) != int(audit["synthetic_samples"]):
            raise RuntimeError(f"Unexpected sample schedule for {arm_name}")


def ensure_unique_paths(rows: list[dict[str, str]], context: str) -> None:
    paths = [row["image_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"Duplicate image paths in {context}")


def validate_synthetic_manifest(audit: dict, path: Path) -> None:
    rows = read_manifest(path)
    if len(rows) != int(audit["synthetic_samples"]):
        raise RuntimeError(f"Unexpected generated sample count at {path}")
    missing = [
        row["image_path"]
        for row in rows
        if not Path(row["image_path"]).exists()
    ]
    if missing:
        raise RuntimeError(
            f"Synthetic manifest has {len(missing)} missing files"
        )


def require_prepared(audit: dict, paths: SynpathPaths) -> None:
    marker = audit_root(audit, paths) / "prepare_complete.json"
    if not marker.exists():
        raise RuntimeError("Audit preparation is incomplete")
    validate_prepared_artifacts(audit, paths)


def require_preflight(audit: dict, paths: SynpathPaths) -> None:
    marker = (
        audit_root(audit, paths) / "preflight_512" / "preflight_complete.json"
    )
    if not marker.exists():
        raise RuntimeError("The 512-resolution memory preflight is incomplete")


def require_trained(audit: dict, paths: SynpathPaths) -> None:
    missing = [
        arm_name
        for arm_name in audit["arms"]
        if not (
            arm_root(audit, paths, arm_name) / "training_complete.json"
        ).exists()
    ]
    if missing:
        raise RuntimeError(f"Audit training is incomplete for: {missing}")


def require_sampled(audit: dict, paths: SynpathPaths) -> None:
    for arm_name in audit["arms"]:
        path = arm_root(audit, paths, arm_name) / "synthetic_manifest.csv"
        if not path.exists():
            raise RuntimeError(f"Audit sampling is incomplete for {arm_name}")
        validate_synthetic_manifest(audit, path)


def audit_root(audit: dict, paths: SynpathPaths) -> Path:
    return paths.outputs_root / audit["output_subdir"]


def arm_root(audit: dict, paths: SynpathPaths, arm_name: str) -> Path:
    return audit_root(audit, paths) / "arms" / arm_name


def distribution_root(
    audit: dict, paths: SynpathPaths, distribution: str
) -> Path:
    return audit_root(audit, paths) / "datasets" / distribution


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def runner_versions() -> dict[str, str]:
    packages = ["accelerate", "diffusers", "peft", "torch", "transformers"]
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "not_installed"
    return result


def parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["validation-worker", "evaluation-worker"]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_worker_args()
    if args.command == "validation-worker":
        if args.num_shards is None or args.shard_index is None:
            raise ValueError("Validation workers require shard arguments")
        validation_worker(
            args.config,
            args.arm,
            args.num_shards,
            args.shard_index,
        )
        return
    cfg = load_resolved_yaml(args.config)
    audit = cfg["pcam_sd_lora_audit"]
    evaluate_arm(cfg, audit, SynpathPaths.from_env(), args.arm)


if __name__ == "__main__":
    main()
