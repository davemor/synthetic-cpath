from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

from synpath.generation.wrappers import run_parallel_commands
from synpath.paths import SynpathPaths


def run_validation_loss(
    config_path: Path,
    cfg: dict,
    audit: dict,
    paths: SynpathPaths,
    resume: bool,
) -> list[Path]:
    validation_cfg = audit["validation_loss"]
    root = validation_root(audit, paths)
    root.mkdir(parents=True, exist_ok=True)
    num_shards = int(validation_cfg["num_shards"])
    commands = []
    for shard_index in range(num_shards):
        output = root / f"losses_shard_{shard_index:02d}.parquet"
        if resume and output.exists():
            continue
        commands.append(
            (
                [
                    sys.executable,
                    "-m",
                    "synpath.convergence.validation_loss",
                    "--config",
                    str(config_path),
                    "--num-shards",
                    str(num_shards),
                    "--shard-index",
                    str(shard_index),
                ],
                root / f"worker_{shard_index:02d}.log",
                {"CUDA_VISIBLE_DEVICES": str(shard_index)},
            )
        )
    if commands:
        run_parallel_commands(commands)
    return aggregate_validation_loss(audit, paths, num_shards)


def evaluate_shard(
    config_path: Path,
    num_shards: int,
    shard_index: int,
) -> Path:
    import torch
    import torch.nn.functional as functional
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    from synpath.config import load_resolved_yaml
    from synpath.manifests import read_manifest

    cfg = load_resolved_yaml(config_path)
    audit = cfg["convergence_audit"]
    validation_cfg = audit["validation_loss"]
    paths = SynpathPaths.from_env()
    output = validation_root(audit, paths) / (
        f"losses_shard_{shard_index:02d}.parquet"
    )
    manifest = paths.data_root / "manifests" / f"{audit['dataset']}.csv"
    all_rows = [row for row in read_manifest(manifest) if row["split"] == "val"]
    indexed_rows = [
        (index, row)
        for index, row in enumerate(all_rows)
        if index % num_shards == shard_index
    ]
    if not indexed_rows:
        raise RuntimeError(f"Validation shard {shard_index} is empty")

    class ValidationDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            global_index, row = self.rows[index]
            image = Image.open(row["image_path"]).convert("RGB")
            prompt = f"a histopathology patch of {row['label']}"
            return global_index, self.transform(image), prompt

    loader = DataLoader(
        ValidationDataset(indexed_rows),
        batch_size=int(validation_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(validation_cfg["num_workers"]),
        pin_memory=True,
        persistent_workers=int(validation_cfg["num_workers"]) > 0,
    )
    device = torch.device("cuda")
    base_model = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.unet.eval()
    for module in (pipe.vae, pipe.text_encoder, pipe.unet):
        module.requires_grad_(False)

    cached_batches = []
    base_seed = int(validation_cfg["seed"])
    with torch.inference_mode():
        for global_indices, images, prompts in loader:
            images = images.to(device, non_blocking=True)
            latent_distribution = pipe.vae.encode(images).latent_dist
            posterior_noise = []
            for global_index in global_indices.tolist():
                generator = torch.Generator(device="cpu").manual_seed(
                    base_seed + int(global_index)
                )
                posterior_noise.append(
                    torch.randn(
                        latent_distribution.mean.shape[1:], generator=generator
                    )
                )
            posterior_noise_tensor = torch.stack(posterior_noise).to(device)
            latents = (
                latent_distribution.mean
                + latent_distribution.std * posterior_noise_tensor
            )
            latents = latents * pipe.vae.config.scaling_factor
            input_ids = pipe.tokenizer(
                list(prompts),
                max_length=pipe.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            hidden_states = pipe.text_encoder(input_ids, return_dict=False)[0]
            noise_rows = []
            timesteps = []
            for global_index in global_indices.tolist():
                generator = torch.Generator(device="cpu").manual_seed(
                    base_seed + 10_000_000 + int(global_index)
                )
                noise_rows.append(
                    torch.randn(latents.shape[1:], generator=generator)
                )
                timesteps.append(
                    torch.randint(
                        0,
                        scheduler.config.num_train_timesteps,
                        (1,),
                        generator=generator,
                    ).item()
                )
            noise = torch.stack(noise_rows).to(device)
            timestep_tensor = torch.tensor(
                timesteps, dtype=torch.long, device=device
            )
            noisy_latents = scheduler.add_noise(latents, noise, timestep_tensor)
            cached_batches.append(
                (
                    global_indices.clone(),
                    noisy_latents,
                    noise,
                    timestep_tensor,
                    hidden_states,
                )
            )

    pipe.vae.to("cpu")
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    records = []
    training_root = (
        paths.outputs_root
        / "generators"
        / audit["dataset"]
        / audit["model"]
        / audit["conditioning"]
        / f"seed{audit['training_seed']}"
        / "sd_lora"
    )
    for step in [int(value) for value in audit["checkpoints"]]:
        checkpoint = training_root / f"checkpoint-{step}"
        pipe.load_lora_weights(str(checkpoint))
        pipe.unet.eval()
        with torch.inference_mode():
            for (
                indices,
                noisy,
                target,
                timesteps,
                hidden_states,
            ) in cached_batches:
                prediction = pipe.unet(
                    noisy,
                    timesteps,
                    hidden_states,
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
        pipe.unload_lora_weights()
    frame = pd.DataFrame(records).sort_values(["step", "sample_index"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return output


def aggregate_validation_loss(
    audit: dict, paths: SynpathPaths, num_shards: int
) -> list[Path]:
    root = validation_root(audit, paths)
    shard_paths = [
        root / f"losses_shard_{index:02d}.parquet"
        for index in range(num_shards)
    ]
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise RuntimeError(
            f"Missing validation-loss shards: {', '.join(missing)}"
        )
    frame = pd.concat(
        [pd.read_parquet(path) for path in shard_paths], ignore_index=True
    )
    expected_samples = 10004
    counts = frame.groupby("step")["sample_index"].nunique()
    if not (counts == expected_samples).all():
        raise RuntimeError(f"Unexpected validation counts: {counts.to_dict()}")
    per_sample_path = root / "validation_denoising_loss_per_sample.parquet"
    frame.to_parquet(per_sample_path, index=False)
    summary = (
        frame.groupby("step")["loss"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(
            columns={"mean": "validation_loss", "std": "validation_loss_sd"}
        )
    )
    summary["validation_loss_se"] = summary["validation_loss_sd"] / summary[
        "count"
    ].map(math.sqrt)
    summary["validation_loss_ci95_low"] = (
        summary["validation_loss"] - 1.96 * summary["validation_loss_se"]
    )
    summary["validation_loss_ci95_high"] = (
        summary["validation_loss"] + 1.96 * summary["validation_loss_se"]
    )
    summary_path = root / "validation_denoising_loss.csv"
    summary.to_csv(summary_path, index=False)
    frame["timestep_decile"] = pd.cut(
        frame["timestep"],
        bins=list(range(0, 1001, 100)),
        right=False,
        include_lowest=True,
    )
    by_timestep = (
        frame.groupby(["step", "timestep_decile"], observed=True)["loss"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    by_timestep["timestep_decile"] = by_timestep["timestep_decile"].astype(str)
    timestep_path = root / "validation_denoising_loss_by_timestep.csv"
    by_timestep.to_csv(timestep_path, index=False)
    figure_path = root / "validation_denoising_loss.png"
    plot_validation_loss(summary, figure_path)
    return [per_sample_path, summary_path, timestep_path, figure_path]


def plot_validation_loss(frame: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(frame["step"], frame["validation_loss"], marker="o")
    axis.fill_between(
        frame["step"],
        frame["validation_loss_ci95_low"],
        frame["validation_loss_ci95_high"],
        alpha=0.2,
    )
    axis.set_xlabel("Training step")
    axis.set_ylabel("Validation denoising MSE")
    axis.set_title("Kather validation denoising loss")
    axis.grid(alpha=0.25)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def validation_root(audit: dict, paths: SynpathPaths) -> Path:
    return (
        paths.outputs_root
        / "generators"
        / audit["dataset"]
        / audit["model"]
        / audit["conditioning"]
        / f"seed{audit['training_seed']}"
        / "convergence_audit"
        / "validation_loss"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_shard(args.config, args.num_shards, args.shard_index)


if __name__ == "__main__":
    main()
