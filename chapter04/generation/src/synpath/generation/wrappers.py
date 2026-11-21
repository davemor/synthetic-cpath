from __future__ import annotations

import json
from importlib import metadata
import os
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterator

from PIL import Image

from synpath.artifacts import utc_now, write_json, write_yaml
from synpath.generation.dcgan import sample_dcgan, train_dcgan
from synpath.manifests import (
    SYNTHETIC_COLUMNS,
    proportional_condition_schedule,
    read_manifest,
    write_manifest,
)
from synpath.paths import SynpathPaths


def train_generator(
    dataset: str,
    model: str,
    conditioning: str,
    seed: int,
    paths: SynpathPaths,
    cfg: dict,
) -> list[Path]:
    root = generator_root(dataset, model, conditioning, seed, paths)
    root.mkdir(parents=True, exist_ok=True)
    condition_manifest = condition_manifest_path(dataset, conditioning, paths)
    train_dir = root / "training_images"
    prepare_training_images(condition_manifest, train_dir, cfg)

    if model == "stylegan2ada":
        ensure_stylegan_repo(paths, cfg)
        outputs = train_stylegan2ada(
            root,
            condition_manifest,
            paths,
            cfg,
            conditional=conditioning != "unconditioned",
            seed=seed,
        )
    elif model == "sd_lora":
        ensure_diffusers_repo(paths)
        outputs = train_sd_lora(
            root,
            train_dir,
            condition_manifest,
            paths,
            cfg,
            seed=seed,
        )
    elif model == "dcgan":
        outputs = train_dcgan(
            root,
            condition_manifest,
            cfg,
            conditional=conditioning != "unconditioned",
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown generator model: {model}")

    write_yaml(
        root / "resolved_config.yaml",
        {"job": root.name, "created_at": utc_now()},
    )
    marker = root / "training_complete.json"
    write_json(
        marker, {"model": model, "conditioning": conditioning, "seed": seed}
    )
    return [root / "resolved_config.yaml", marker, *outputs]


def sample_generator(
    dataset: str,
    model: str,
    conditioning: str,
    seed: int,
    paths: SynpathPaths,
    cfg: dict,
) -> list[Path]:
    root = generator_root(dataset, model, conditioning, seed, paths)
    condition_manifest = condition_manifest_path(dataset, conditioning, paths)
    sample_dir = root / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    max_samples = synthetic_sample_limit(condition_manifest, cfg)

    existing_samples = sorted(sample_dir.glob("*.png"))
    if len(existing_samples) >= max_samples:
        pass
    elif model == "stylegan2ada":
        ensure_stylegan_repo(paths, cfg)
        sample_stylegan2ada(
            root, sample_dir, condition_manifest, paths, cfg, max_samples
        )
    elif model == "sd_lora":
        ensure_diffusers_repo(paths)
        sample_sd_lora(root, sample_dir, condition_manifest, cfg, max_samples)
    elif model == "dcgan":
        sample_dcgan(
            root, sample_dir, condition_manifest, cfg, max_samples, seed
        )
    else:
        raise ValueError(f"Unknown generator model: {model}")

    manifest = write_synthetic_manifest(
        dataset,
        model,
        conditioning,
        seed,
        condition_manifest,
        sample_dir,
        paths,
        cfg,
    )
    return [manifest, sample_dir]


def generator_root(
    dataset: str,
    model: str,
    conditioning: str,
    seed: int,
    paths: SynpathPaths,
) -> Path:
    return (
        paths.outputs_root
        / "generators"
        / dataset
        / model
        / conditioning
        / f"seed{seed}"
    )


def condition_manifest_path(
    dataset: str,
    conditioning: str,
    paths: SynpathPaths,
) -> Path:
    return (
        paths.outputs_root
        / "conditions"
        / dataset
        / conditioning
        / "condition_manifest.csv"
    )


def prepare_training_images(
    condition_manifest: Path,
    output_dir: Path,
    cfg: dict,
) -> None:
    rows = read_manifest(condition_manifest)
    size = int(cfg.get("images", {}).get("generator_size", 256))
    mode = (
        cfg.get("generation", {})
        .get("sd_lora", {})
        .get("training_image_mode", "resize")
    )
    if mode not in {"resize", "hardlink"}:
        raise ValueError(f"Unsupported training image mode: {mode}")
    for idx, row in enumerate(rows):
        label = row["condition"]
        dst = output_dir / label / f"{idx:08d}.png"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "hardlink":
            os.link(row["image_path"], dst)
        else:
            Image.open(row["image_path"]).convert("RGB").resize(
                (size, size)
            ).save(dst)


def train_stylegan2ada(
    root: Path,
    condition_manifest: Path,
    paths: SynpathPaths,
    cfg: dict,
    conditional: bool,
    seed: int,
) -> list[Path]:
    repo = stylegan_repo(paths, cfg)
    if not (repo / "train.py").exists():
        raise RuntimeError(f"Missing StyleGAN repo at {repo}")
    zip_path = root / "stylegan_dataset.zip"
    if not stylegan_zip_is_ready(zip_path, conditional):
        write_stylegan_dataset_zip(
            condition_manifest,
            zip_path,
            cfg,
            conditional=conditional,
        )
    python = stylegan_python(paths)
    stylegan_cfg = cfg.get("generation", {}).get("stylegan2ada", {})
    cmd = stylegan_train_command(
        repo,
        python,
        root,
        zip_path,
        cfg,
        stylegan_cfg,
        conditional,
        seed,
    )
    run_command(cmd, root / "stylegan_train.log")
    return [zip_path]


def stylegan_train_command(
    repo: Path,
    python: Path,
    root: Path,
    zip_path: Path,
    cfg: dict,
    stylegan_cfg: dict,
    conditional: bool,
    seed: int,
) -> list[str]:
    backend = stylegan_backend(cfg)
    common = [
        str(python),
        str(repo / "train.py"),
        "--outdir",
        str(root / "stylegan_runs"),
        "--data",
        str(zip_path),
        "--gpus",
        str(stylegan_gpu_count(cfg)),
        "--cond",
        "1" if conditional else "0",
        "--metrics",
        "none",
        "--aug",
        str(stylegan_cfg.get("aug", "ada")),
        "--kimg",
        str(stylegan_cfg.get("kimg", 25000)),
    ]
    if backend == "stylegan3":
        return [
            *common[:2],
            "--outdir",
            str(root / "stylegan_runs"),
            "--cfg",
            str(stylegan_cfg.get("cfg", "stylegan2")),
            "--data",
            str(zip_path),
            "--gpus",
            str(stylegan_gpu_count(cfg)),
            "--batch",
            str(stylegan_cfg.get("batch_size", 32)),
            "--gamma",
            str(stylegan_cfg.get("gamma", 0.8192)),
            "--cond",
            "1" if conditional else "0",
            "--metrics",
            "none",
            "--aug",
            str(stylegan_cfg.get("aug", "ada")),
            "--kimg",
            str(stylegan_cfg.get("kimg", 25000)),
            "--tick",
            str(stylegan_cfg.get("tick", 4)),
            "--snap",
            str(stylegan_cfg.get("snap", 50)),
            "--seed",
            str(seed),
        ]
    return [*common, "--seed", str(seed)]


def stylegan_zip_is_ready(zip_path: Path, conditional: bool) -> bool:
    if not zip_path.exists():
        return False
    try:
        with zipfile.ZipFile(zip_path) as zf:
            metadata = json.loads(zf.read("dataset.json"))
    except (KeyError, json.JSONDecodeError, zipfile.BadZipFile):
        return False
    labels = metadata.get("labels")
    return labels is not None if conditional else labels is None


def write_stylegan_dataset_zip(
    condition_manifest: Path,
    zip_path: Path,
    cfg: dict,
    conditional: bool,
) -> None:
    rows = read_manifest(condition_manifest)
    size = int(cfg.get("images", {}).get("generator_size", 256))
    labels = []
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for idx, row in enumerate(rows):
            archive_name = f"{idx // 1000:05d}/img{idx:08d}.png"
            image = (
                Image.open(row["image_path"])
                .convert("RGB")
                .resize((size, size))
            )
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            zf.writestr(archive_name, buffer.getvalue())
            if conditional:
                labels.append([archive_name, int(row["condition_id"])])
        zf.writestr(
            "dataset.json",
            json.dumps({"labels": labels if conditional else None}),
        )


def stylegan_gpu_count(cfg: dict) -> int:
    configured = cfg.get("stylegan_gpus") or os.environ.get(
        "SYNPATH_STYLEGAN_GPUS"
    )
    if configured:
        return max(1, int(configured))
    return detected_gpu_count()


def detected_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return max(1, len([item for item in visible.split(",") if item]))
    try:
        import torch

        count = torch.cuda.device_count()
        if count > 0:
            return count
    except ImportError:
        pass
    return int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))


def stylegan_python(paths: SynpathPaths) -> Path:
    configured = os.environ.get("SYNPATH_STYLEGAN_PYTHON")
    if configured:
        python = Path(configured)
        if not python.exists():
            raise RuntimeError(
                "SYNPATH_STYLEGAN_PYTHON points to a missing interpreter: "
                f"{python}. Run scripts/setup/install_stylegan2ada.sh first."
            )
        return python
    env_python = paths.external_root / "stylegan2ada-env" / "bin" / "python"
    if env_python.exists():
        return env_python
    return Path(sys.executable)


def stylegan_backend(cfg: dict | None = None) -> str:
    configured = (
        cfg.get("generation", {}).get("stylegan2ada", {}).get("backend")
        if cfg is not None
        else None
    )
    backend = configured or os.environ.get(
        "SYNPATH_STYLEGAN_BACKEND", "stylegan2ada"
    )
    if backend not in {"stylegan2ada", "stylegan3"}:
        raise ValueError(
            "SYNPATH_STYLEGAN_BACKEND must be 'stylegan2ada' or 'stylegan3', "
            f"got {backend!r}"
        )
    return backend


def stylegan_repo(paths: SynpathPaths, cfg: dict | None = None) -> Path:
    configured = os.environ.get("SYNPATH_STYLEGAN_REPO")
    if configured:
        return Path(configured)
    repo_name = (
        "stylegan3"
        if stylegan_backend(cfg) == "stylegan3"
        else "stylegan2-ada-pytorch"
    )
    return paths.external_root / repo_name


def ensure_stylegan_repo(paths: SynpathPaths, cfg: dict | None = None) -> None:
    repo = stylegan_repo(paths, cfg)
    backend = stylegan_backend(cfg)
    if (repo / "train.py").exists():
        patch_stylegan_repo(repo, backend)
        return
    repo.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://github.com/NVlabs/stylegan3.git"
        if stylegan_backend(cfg) == "stylegan3"
        else "https://github.com/NVlabs/stylegan2-ada-pytorch.git"
    )
    run_command(
        [
            "git",
            "clone",
            url,
            str(repo),
        ],
        repo.parent / "stylegan2ada_clone.log",
    )
    patch_stylegan_repo(repo, backend)


def patch_stylegan_repo(repo: Path, backend: str) -> None:
    sampler_path = repo / "torch_utils" / "misc.py"
    if not sampler_path.exists():
        return
    text = sampler_path.read_text()
    if backend == "stylegan3":
        if "super().__init__()" in text:
            sampler_path.write_text(
                text.replace("super().__init__()", "super().__init__(dataset)")
            )
        return
    if "super().__init__(dataset)" in text:
        sampler_path.write_text(
            text.replace("super().__init__(dataset)", "super().__init__()")
        )


def ensure_diffusers_repo(paths: SynpathPaths) -> None:
    repo = paths.external_root / "diffusers"
    script = repo / "examples" / "text_to_image" / "train_text_to_image_lora.py"
    if script.exists():
        align_diffusers_repo(repo)
        return
    repo.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "git",
            "clone",
            "https://github.com/huggingface/diffusers.git",
            str(repo),
        ],
        repo.parent / "diffusers_clone.log",
    )
    align_diffusers_repo(repo)


def align_diffusers_repo(repo: Path) -> None:
    if not (repo / ".git").exists():
        return
    try:
        version = metadata.version("diffusers")
    except metadata.PackageNotFoundError:
        return
    tag = f"v{version}"
    current = subprocess.run(
        ["git", "-C", str(repo), "describe", "--tags", "--exact-match"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if current.stdout.strip() == tag:
        return
    subprocess.run(
        ["git", "-C", str(repo), "fetch", "--tags", "--quiet"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "--quiet", tag],
        check=True,
    )


def sample_stylegan2ada(
    root: Path,
    sample_dir: Path,
    condition_manifest: Path,
    paths: SynpathPaths,
    cfg: dict,
    max_samples: int,
) -> None:
    repo = stylegan_repo(paths, cfg)
    network = latest_stylegan_network(root)
    if network is None:
        raise RuntimeError(f"No StyleGAN network snapshot found under {root}")
    rows = proportional_condition_schedule(
        read_manifest(condition_manifest), max_samples, seed=0
    )
    tmp_dir = sample_dir / "_stylegan_raw"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    by_class: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_class.setdefault(str(row["condition_id"]), []).append(idx)

    seed_chunk_size = int(
        cfg.get("generation", {})
        .get("stylegan2ada", {})
        .get("sample_seed_chunk_size", 1000)
    )
    for class_id, seeds in by_class.items():
        python = stylegan_python(paths)
        for chunk_index, seed_chunk in enumerate(
            chunked(seeds, seed_chunk_size)
        ):
            seed_text = ",".join(str(seed) for seed in seed_chunk)
            run_command(
                [
                    str(python),
                    str(stylegan_generate_script(repo)),
                    "--outdir",
                    str(tmp_dir),
                    "--trunc",
                    "1",
                    "--seeds",
                    seed_text,
                    "--class",
                    class_id,
                    "--network",
                    str(network),
                ],
                root
                / f"stylegan_sample_class_{class_id}_chunk_{chunk_index:04d}.log",
            )
    for idx in range(len(rows)):
        candidates = sorted(tmp_dir.glob(f"*{idx:04d}*.png"))
        if not candidates:
            candidates = sorted(tmp_dir.glob(f"*{idx}*.png"))
        if candidates:
            candidates[0].replace(sample_dir / f"{idx:08d}.png")


def chunked(values: list[int], size: int) -> Iterator[list[int]]:
    if size < 1:
        raise ValueError("Chunk size must be at least 1")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def latest_stylegan_network(root: Path) -> Path | None:
    snapshots = sorted(root.glob("stylegan_runs/**/network-snapshot-*.pkl"))
    return snapshots[-1] if snapshots else None


def stylegan_generate_script(repo: Path) -> Path:
    gen_images = repo / "gen_images.py"
    return gen_images if gen_images.exists() else repo / "generate.py"


def train_sd_lora(
    root: Path,
    train_dir: Path,
    condition_manifest: Path,
    paths: SynpathPaths,
    cfg: dict,
    seed: int,
) -> list[Path]:
    script = Path(
        cfg.get(
            "sd_lora_train_script",
            str(
                paths.external_root
                / "diffusers"
                / "examples"
                / "text_to_image"
                / "train_text_to_image_lora.py"
            ),
        )
    )
    if not script.exists():
        raise RuntimeError(
            "Missing Diffusers LoRA training script. Set sd_lora_train_script "
            f"or place it at {script}."
        )
    sd_cfg = cfg.get("generation", {}).get("sd_lora", {})
    base_model = str(sd_cfg.get("base_model", "runwayml/stable-diffusion-v1-5"))
    resolution = int(
        sd_cfg.get(
            "train_resolution",
            cfg.get("images", {}).get("generator_size", 256),
        )
    )
    launch_args = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
    ]
    if sd_cfg.get("num_processes") is not None:
        launch_args.extend(["--num_processes", str(sd_cfg["num_processes"])])
    if sd_cfg.get("mixed_precision"):
        launch_args.extend(
            ["--mixed_precision", str(sd_cfg["mixed_precision"])]
        )
    write_metadata_jsonl(condition_manifest, train_dir / "metadata.jsonl")
    run_command(
        [
            *launch_args,
            str(script),
            *sd_lora_training_precision_args(sd_cfg),
            "--pretrained_model_name_or_path",
            base_model,
            "--train_data_dir",
            str(train_dir),
            "--caption_column",
            "text",
            "--resolution",
            str(resolution),
            "--train_batch_size",
            str(sd_cfg.get("train_batch_size", 16)),
            "--gradient_accumulation_steps",
            str(sd_cfg.get("gradient_accumulation_steps", 1)),
            "--max_train_steps",
            str(sd_cfg.get("max_train_steps", 10000)),
            "--learning_rate",
            str(sd_cfg.get("learning_rate", 1e-4)),
            "--rank",
            str(sd_cfg.get("rank", 4)),
            "--seed",
            str(seed),
            *sd_lora_optional_training_args(sd_cfg),
            "--output_dir",
            str(root / "sd_lora"),
        ],
        root / "sd_lora_train.log",
    )
    return [root / "sd_lora"]


def train_sd_full(
    root: Path,
    train_dir: Path,
    paths: SynpathPaths,
    cfg: dict,
    seed: int,
    resume: bool = False,
) -> list[Path]:
    script = Path(
        cfg.get(
            "sd_full_train_script",
            str(
                paths.external_root
                / "diffusers"
                / "examples"
                / "text_to_image"
                / "train_text_to_image.py"
            ),
        )
    )
    if not script.exists():
        raise RuntimeError(
            "Missing Diffusers full text-to-image training script. Set "
            f"sd_full_train_script or place it at {script}."
        )
    sd_cfg = cfg.get("generation", {}).get("sd_full", {})
    output = root / "sd_full"
    launch_args = [sys.executable, "-m", "accelerate.commands.launch"]
    if sd_cfg.get("num_processes") is not None:
        launch_args.extend(["--num_processes", str(sd_cfg["num_processes"])])
    if sd_cfg.get("mixed_precision"):
        launch_args.extend(
            ["--mixed_precision", str(sd_cfg["mixed_precision"])]
        )
    command = [
        *launch_args,
        str(script),
        *sd_full_training_precision_args(sd_cfg),
        "--pretrained_model_name_or_path",
        str(sd_cfg.get("base_model", "runwayml/stable-diffusion-v1-5")),
        "--train_data_dir",
        str(train_dir),
        "--caption_column",
        "text",
        "--resolution",
        str(sd_cfg.get("train_resolution", 512)),
        "--train_batch_size",
        str(sd_cfg.get("train_batch_size", 8)),
        "--gradient_accumulation_steps",
        str(sd_cfg.get("gradient_accumulation_steps", 1)),
        "--max_train_steps",
        str(sd_cfg.get("max_train_steps", 12500)),
        "--learning_rate",
        str(sd_cfg.get("learning_rate", 1e-5)),
        "--dataloader_num_workers",
        str(sd_cfg.get("dataloader_num_workers", 8)),
        "--seed",
        str(seed),
        *sd_full_optional_training_args(sd_cfg),
        "--output_dir",
        str(output),
    ]
    if resume and any(output.glob("checkpoint-*")):
        command.extend(["--resume_from_checkpoint", "latest"])
    run_command(command, root / "sd_full_train.log")
    return [output]


def sd_full_optional_training_args(sd_cfg: dict) -> list[str]:
    args = []
    if sd_cfg.get("center_crop", True):
        args.append("--center_crop")
    if sd_cfg.get("gradient_checkpointing"):
        args.append("--gradient_checkpointing")
    if sd_cfg.get("enable_xformers_memory_efficient_attention"):
        args.append("--enable_xformers_memory_efficient_attention")
    if sd_cfg.get("checkpointing_steps") is not None:
        args.extend(
            ["--checkpointing_steps", str(sd_cfg["checkpointing_steps"])]
        )
    if sd_cfg.get("checkpoints_total_limit") is not None:
        args.extend(
            [
                "--checkpoints_total_limit",
                str(sd_cfg["checkpoints_total_limit"]),
            ]
        )
    return args


def sd_full_training_precision_args(sd_cfg: dict) -> list[str]:
    precision = sd_cfg.get("mixed_precision")
    return ["--mixed_precision", str(precision)] if precision else []


def sd_lora_optional_training_args(sd_cfg: dict) -> list[str]:
    args = []
    if sd_cfg.get("gradient_checkpointing"):
        args.append("--gradient_checkpointing")
    if sd_cfg.get("enable_xformers_memory_efficient_attention"):
        args.append("--enable_xformers_memory_efficient_attention")
    if sd_cfg.get("checkpointing_steps") is not None:
        args.extend(
            ["--checkpointing_steps", str(sd_cfg["checkpointing_steps"])]
        )
    if sd_cfg.get("checkpoints_total_limit") is not None:
        args.extend(
            [
                "--checkpoints_total_limit",
                str(sd_cfg["checkpoints_total_limit"]),
            ]
        )
    return args


def sd_lora_training_precision_args(sd_cfg: dict) -> list[str]:
    precision = sd_cfg.get("mixed_precision")
    return ["--mixed_precision", str(precision)] if precision else []


def sample_sd_lora(
    root: Path,
    sample_dir: Path,
    condition_manifest: Path,
    cfg: dict,
    max_samples: int,
) -> None:
    sd_cfg = cfg.get("generation", {}).get("sd_lora", {})
    sample_gpus = int(
        os.environ.get("SYNPATH_SD_SAMPLE_GPUS") or sd_cfg.get("sample_gpus", 1)
    )
    sample_gpus = max(sample_gpus, 1)
    commands: list[tuple[list[str], Path, dict[str, str] | None]] = []
    for shard_index in range(sample_gpus):
        env = None
        if sample_gpus > 1:
            env = {"CUDA_VISIBLE_DEVICES": str(shard_index)}
        commands.append(
            (
                [
                    sys.executable,
                    "-m",
                    "synpath.generation.sd_sample",
                    "--lora_dir",
                    str(root / "sd_lora"),
                    "--condition_manifest",
                    str(condition_manifest),
                    "--output_dir",
                    str(sample_dir),
                    "--max_samples",
                    str(max_samples),
                    "--seed",
                    "0",
                    "--batch_size",
                    str(sd_cfg.get("sample_batch_size", 8)),
                    "--height",
                    str(
                        sd_cfg.get(
                            "sample_height",
                            cfg.get("images", {}).get("generator_size", 256),
                        )
                    ),
                    "--width",
                    str(
                        sd_cfg.get(
                            "sample_width",
                            cfg.get("images", {}).get("generator_size", 256),
                        )
                    ),
                    "--num_shards",
                    str(sample_gpus),
                    "--shard_index",
                    str(shard_index),
                ],
                root / f"sd_lora_sample_shard_{shard_index:02d}.log",
                env,
            )
        )
    run_parallel_commands(commands)


def write_metadata_jsonl(condition_manifest: Path, output: Path) -> None:
    rows = read_manifest(condition_manifest)
    with output.open("w") as f:
        for idx, row in enumerate(rows):
            file_name = f"{row['condition']}/{idx:08d}.png"
            f.write(json.dumps({"file_name": file_name, "text": row["prompt"]}))
            f.write("\n")


def write_synthetic_manifest(
    dataset: str,
    model: str,
    conditioning: str,
    seed: int,
    condition_manifest: Path,
    sample_dir: Path,
    paths: SynpathPaths,
    cfg: dict,
) -> Path:
    condition_rows = proportional_condition_schedule(
        read_manifest(condition_manifest),
        synthetic_sample_limit(condition_manifest, cfg),
        seed=0,
    )
    samples = sorted(sample_dir.glob("*.png"))
    max_count = len(condition_rows)
    rows = []
    if not samples:
        raise RuntimeError(f"No generated PNG samples found in {sample_dir}")
    for idx, src_row in enumerate(condition_rows):
        if idx >= len(samples):
            break
        image_path = samples[idx]
        rows.append(
            {
                "image_path": str(image_path),
                "dataset": dataset,
                "model": model,
                "conditioning": conditioning,
                "seed": seed,
                "label": src_row["label"],
                "label_id": src_row["label_id"],
                "cluster_id": src_row["cluster_id"],
                "condition": src_row["condition"],
                "condition_id": src_row["condition_id"],
                "prompt": src_row["prompt"],
            }
        )
    output = (
        paths.outputs_root
        / "generators"
        / dataset
        / model
        / conditioning
        / f"seed{seed}"
        / "synthetic_manifest.csv"
    )
    write_manifest(output, rows, SYNTHETIC_COLUMNS)
    return output


def synthetic_sample_limit(condition_manifest: Path, cfg: dict) -> int:
    row_count = len(read_manifest(condition_manifest))
    configured = (
        cfg.get("limits", {}).get("max_synthetic_images")
        or cfg.get("metrics", {}).get("max_synthetic")
        or row_count
    )
    return min(int(configured), row_count)


def run_command(
    cmd: list[str], log_path: Path, env: dict[str, str] | None = None
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        subprocess.run(
            cmd,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **env} if env else None,
        )


def run_parallel_commands(
    commands: list[tuple[list[str], Path, dict[str, str] | None]],
) -> None:
    processes: list[tuple[subprocess.Popen, Path]] = []
    logs = []
    try:
        for cmd, log_path, env in commands:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w")
            logs.append(log)
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **env} if env else None,
            )
            processes.append((process, log_path))
        failures = []
        for process, log_path in processes:
            if process.wait() != 0:
                failures.append(log_path)
        if failures:
            failed_logs = ", ".join(str(path) for path in failures)
            raise RuntimeError(
                f"Parallel command failed; see logs: {failed_logs}"
            )
    finally:
        for log in logs:
            log.close()
