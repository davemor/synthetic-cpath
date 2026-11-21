from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import SyntheticEmbeddingConfig
from .metrics import write_evaluation_artifacts


SPLITS = ("train", "valid", "test")
BASELINE_GENERATORS = ("gaussian", "bootstrap")
EDM_CONDITIONAL_MODES = (
    "edm_conditional",
    "classifier_free_edm_conditional",
    "edm_adaln_mlp_conditional",
)


class EmbeddingSplitDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.features = features.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = (self.features[index] - self.mean) / self.std
        return torch.from_numpy(z), torch.tensor(self.labels[index], dtype=torch.long)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class DiffAESkipMLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cond_dim: int,
        *,
        use_condition: bool,
        use_norm: bool,
        use_activation: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_dim)) if use_condition else None
        self.norm = nn.LayerNorm(out_dim) if use_norm else None
        self.activation = nn.SiLU() if use_activation else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        if self.cond is not None:
            h = h * (1.0 + self.cond(cond))
        if self.norm is not None:
            h = self.norm(h)
        return self.dropout(self.activation(h))


class MLPSkipDenoiser(nn.Module):
    """DiffAE-style skip MLP for latent DDIM noise prediction."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 2048,
        depth: int = 10,
        dropout: float = 0.0,
        use_class_condition: bool | None = None,
        extra_condition_tokens: int = 0,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.n_classes = n_classes
        self.use_class_condition = n_classes > 1 if use_class_condition is None else use_class_condition
        self.time_embedding = SinusoidalTimeEmbedding(64)
        self.time_mlp = nn.Sequential(
            nn.Linear(64, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim),
        )
        self.class_embedding = (
            nn.Embedding(n_classes + extra_condition_tokens, input_dim)
            if self.use_class_condition
            else None
        )
        blocks = []
        for layer_idx in range(depth):
            is_last = layer_idx == depth - 1
            in_dim = input_dim if layer_idx == 0 else hidden_dim + input_dim
            out_dim = input_dim if is_last else hidden_dim
            blocks.append(
                DiffAESkipMLPBlock(
                    in_dim,
                    out_dim,
                    input_dim,
                    use_condition=not is_last,
                    use_norm=not is_last,
                    use_activation=not is_last,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, z_t: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.time_mlp(self.time_embedding(timesteps))
        if self.class_embedding is not None:
            cond = cond + self.class_embedding(labels)
        h = z_t
        for layer_idx, block in enumerate(self.blocks):
            block_input = torch.cat([h, z_t], dim=1) if layer_idx > 0 else h
            h = block(block_input, cond)
        return h


class AdaLNZeroMLPBlock(nn.Module):
    """Vector-native AdaLN-Zero residual MLP block inspired by DiT conditioning."""

    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(cond).chunk(3, dim=1)
        modulated = self.norm(h) * (1.0 + scale) + shift
        return h + gate * self.mlp(modulated)


class AdaLNZeroMLPDenoiser(nn.Module):
    """EDM denoiser with DiT-style AdaLN-Zero conditioning for whole embeddings."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 2048,
        depth: int = 10,
        dropout: float = 0.0,
        extra_condition_tokens: int = 0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.n_classes = n_classes
        self.time_embedding = SinusoidalTimeEmbedding(64)
        self.time_mlp = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.class_embedding = nn.Embedding(n_classes + extra_condition_tokens, hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [AdaLNZeroMLPBlock(hidden_dim, hidden_dim, dropout) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z_t: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.time_mlp(self.time_embedding(timesteps)) + self.class_embedding(labels)
        h = self.input_proj(z_t)
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(self.output_norm(h))


class LatentDiffusion(nn.Module):
    def __init__(self, model: MLPSkipDenoiser, timesteps: int, beta: float) -> None:
        super().__init__()
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be in (0, 1).")
        self.model = model
        self.timesteps = int(timesteps)
        betas = torch.full((timesteps,), float(beta), dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def training_loss(self, z0: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        timesteps = torch.randint(0, self.timesteps, (z0.shape[0],), device=z0.device)
        eps = torch.randn_like(z0)
        z_t = (
            self.sqrt_alpha_bars[timesteps].unsqueeze(1) * z0
            + self.sqrt_one_minus_alpha_bars[timesteps].unsqueeze(1) * eps
        )
        eps_hat = self.model(z_t, timesteps, labels)
        return torch.nn.functional.l1_loss(eps_hat, eps)

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        *,
        steps: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        z = torch.randn(
            labels.shape[0],
            self.model.input_dim,
            device=device,
            generator=generator,
        )
        labels = labels.to(device)
        step_indices = self._space_timesteps(steps, device=device)
        spaced_alpha_bars = self.alpha_bars[step_indices]
        for position in range(len(step_indices) - 1, -1, -1):
            original_t = int(step_indices[position].item())
            t = torch.full((labels.shape[0],), original_t, device=device, dtype=torch.long)
            alpha_bar_t = spaced_alpha_bars[position].view(1, 1)
            eps_hat = self.model(z, t, labels)
            z0_hat = (z - torch.sqrt(1.0 - alpha_bar_t) * eps_hat) / torch.sqrt(alpha_bar_t)
            if position == 0:
                z = z0_hat
                continue
            alpha_bar_prev = spaced_alpha_bars[position - 1].view(1, 1)
            eps = (z - torch.sqrt(alpha_bar_t) * z0_hat) / torch.sqrt(1.0 - alpha_bar_t)
            z = torch.sqrt(alpha_bar_prev) * z0_hat + torch.sqrt(1.0 - alpha_bar_prev) * eps
        return z

    def _space_timesteps(self, steps: int, *, device: torch.device) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be at least 1.")
        if steps > self.timesteps:
            raise ValueError(f"steps ({steps}) cannot exceed training timesteps ({self.timesteps}).")
        if steps == 1:
            return torch.tensor([self.timesteps - 1], device=device, dtype=torch.long)
        frac_stride = (self.timesteps - 1) / (steps - 1)
        indices = [round(i * frac_stride) for i in range(steps)]
        return torch.tensor(indices, device=device, dtype=torch.long)


class EDMLatentDiffusion(nn.Module):
    """EDM-style conditional diffusion for continuous embedding vectors."""

    def __init__(
        self,
        model: nn.Module,
        *,
        sigma_min: float,
        sigma_max: float,
        sigma_data: float,
        rho: float,
        class_dropout: float,
    ) -> None:
        super().__init__()
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError("EDM requires 0 < sigma_min < sigma_max.")
        if sigma_data <= 0.0:
            raise ValueError("sigma_data must be positive.")
        if not 0.0 <= class_dropout < 1.0:
            raise ValueError("class_dropout must be in [0, 1).")
        self.model = model
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_data = float(sigma_data)
        self.rho = float(rho)
        self.class_dropout = float(class_dropout)
        self.null_label = model.n_classes

    def training_loss(self, z0: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        rnd_normal = torch.randn((z0.shape[0],), device=z0.device)
        sigma = (rnd_normal * 1.2 - 1.2).exp().clamp(self.sigma_min, self.sigma_max)
        noise = torch.randn_like(z0) * sigma.unsqueeze(1)
        noisy = z0 + noise
        train_labels = labels
        if self.class_dropout > 0.0:
            drop = torch.rand_like(labels.float()) < self.class_dropout
            train_labels = labels.clone()
            train_labels[drop] = self.null_label
        denoised = self.denoise(noisy, sigma, train_labels)
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        loss = weight.unsqueeze(1) * (denoised - z0).pow(2)
        return loss.mean()

    def denoise(self, noisy: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sigma = sigma.to(noisy.device, dtype=noisy.dtype)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_in = 1.0 / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_noise = sigma.log() / 4.0
        model_out = self.model(
            c_in.unsqueeze(1) * noisy,
            c_noise,
            labels,
        )
        return c_skip.unsqueeze(1) * noisy + c_out.unsqueeze(1) * model_out

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        *,
        steps: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        labels = labels.to(device)
        sigmas = self._sigma_schedule(steps, device=device)
        z = torch.randn(
            labels.shape[0],
            self.model.input_dim,
            device=device,
            generator=generator,
        ) * sigmas[0]
        for step_idx in range(len(sigmas) - 1):
            sigma = sigmas[step_idx].expand(labels.shape[0])
            next_sigma = sigmas[step_idx + 1].expand(labels.shape[0])
            denoised = self._guided_denoise(z, sigma, labels, guidance_scale)
            d = (z - denoised) / sigma.unsqueeze(1)
            z_next = z + (next_sigma - sigma).unsqueeze(1) * d
            if step_idx < len(sigmas) - 2:
                denoised_next = self._guided_denoise(z_next, next_sigma, labels, guidance_scale)
                d_next = (z_next - denoised_next) / next_sigma.unsqueeze(1)
                z = z + (next_sigma - sigma).unsqueeze(1) * (0.5 * d + 0.5 * d_next)
            else:
                z = z_next
        return z

    def _guided_denoise(
        self,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        labels: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        denoised_cond = self.denoise(noisy, sigma, labels)
        if guidance_scale == 1.0:
            return denoised_cond
        null_labels = torch.full_like(labels, self.null_label)
        denoised_uncond = self.denoise(noisy, sigma, null_labels)
        return denoised_uncond + guidance_scale * (denoised_cond - denoised_uncond)

    def _sigma_schedule(self, steps: int, *, device: torch.device) -> torch.Tensor:
        if steps < 2:
            raise ValueError("EDM sampling requires at least two steps.")
        ramp = torch.linspace(0, 1, steps, device=device)
        min_inv_rho = self.sigma_min ** (1.0 / self.rho)
        max_inv_rho = self.sigma_max ** (1.0 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return torch.cat([sigmas, torch.zeros(1, device=device)])


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if torch.is_floating_point(param)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if name not in self.shadow or not torch.is_floating_point(value):
                continue
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.cpu() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {name: value.detach().clone() for name, value in state.items()}

    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, value in self.shadow.items():
            if name in state:
                state[name].copy_(value.to(state[name].device))
        model.load_state_dict(state)


def prepare_synthetic_embedding_run(config: SyntheticEmbeddingConfig) -> Path:
    _require_embedding_dir(config)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    features, index = _load_real_dataset(config)
    labels, class_names = _encode_labels(index, config)
    train_mask = _split_mask(index, config, "train")
    train_features = features[train_mask]
    mean = train_features.mean(axis=0).astype(np.float32)
    std = train_features.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1.0e-6).astype(np.float32)
    class_means = []
    class_stds = []
    for class_id in range(len(class_names)):
        class_features = features[train_mask & (labels == class_id)]
        class_mean = class_features.mean(axis=0).astype(np.float32)
        class_std = np.maximum(class_features.std(axis=0), 1.0e-6).astype(np.float32)
        class_means.append(class_mean)
        class_stds.append(class_std)

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    np.savez(metadata_dir / "train_stats.npz", mean=mean, std=std)
    np.savez(
        metadata_dir / "class_train_stats.npz",
        mean=np.stack(class_means).astype(np.float32),
        std=np.stack(class_stds).astype(np.float32),
        class_names=np.asarray(class_names),
    )
    (metadata_dir / "class_mapping.json").write_text(
        json.dumps(
            {
                "class_names": class_names,
                "class_to_id": {name: idx for idx, name in enumerate(class_names)},
                "label_column": config.label_column,
            },
            indent=2,
        )
    )
    (metadata_dir / "config_snapshot.json").write_text(
        json.dumps(_jsonable_config(config), indent=2, sort_keys=True)
    )
    _write_dataset_summary(index, labels, config, metadata_dir / "dataset_summary.csv")
    return output_dir


def train_synthetic_embedding_model(config: SyntheticEmbeddingConfig) -> Path:
    if config.model_mode == "independent_per_class":
        return train_independent_class_embedding_models(config)
    if config.model_mode in EDM_CONDITIONAL_MODES:
        return train_edm_conditional_embedding_model(config)
    if config.model_mode != "conditional":
        raise ValueError(f"Unsupported synthetic embedding model_mode {config.model_mode!r}.")

    output_dir = prepare_synthetic_embedding_run(config)
    _set_seeds(config.seed)
    features, index = _load_real_dataset(config)
    labels, class_names = _encode_labels(index, config)
    stats = np.load(output_dir / "metadata" / "train_stats.npz")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    train_dataset = EmbeddingSplitDataset(
        features[_split_mask(index, config, "train")],
        labels[_split_mask(index, config, "train")],
        mean,
        std,
    )
    valid_dataset = EmbeddingSplitDataset(
        features[_split_mask(index, config, "valid")],
        labels[_split_mask(index, config, "valid")],
        mean,
        std,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_dataset) >= config.batch_size,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = _make_diffusion(config, len(class_names)).to(device)
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler(enabled=config.use_amp and device.type == "cuda")
    ema = EMA(diffusion, config.ema_decay)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    dynamics = []
    best_val = float("inf")
    best_path = checkpoints_dir / "best.pt"
    validations_without_improvement = 0
    seen_samples = 0
    step = 0
    started = perf_counter()
    train_iter = iter(train_loader)
    progress = tqdm(total=config.max_seen_samples, desc="latent-ddim seen samples", unit="sample")

    while seen_samples < config.max_seen_samples:
        try:
            batch, batch_labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch, batch_labels = next(train_iter)
        batch = batch.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=config.use_amp and device.type == "cuda"):
            loss = diffusion.training_loss(batch, batch_labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(diffusion)
        step += 1
        seen_samples += batch.shape[0]
        progress.update(batch.shape[0])

        should_validate = step == 1 or step % config.validation_interval_steps == 0
        if should_validate:
            val_loss = _validation_loss(diffusion, valid_loader, device, config)
            row = {
                "step": step,
                "seen_samples": seen_samples,
                "train_loss": float(loss.detach().cpu()),
                "valid_loss": val_loss,
                "elapsed_seconds": perf_counter() - started,
            }
            dynamics.append(row)
            pd.DataFrame(dynamics).to_csv(output_dir / "training_curves.csv", index=False)
            if val_loss < best_val:
                best_val = val_loss
                validations_without_improvement = 0
                _save_checkpoint(
                    best_path,
                    diffusion,
                    optimizer,
                    ema,
                    config,
                    class_names,
                    step,
                    seen_samples,
                    best_val,
                )
            else:
                validations_without_improvement += 1
                if validations_without_improvement >= config.patience_validations:
                    break

    progress.close()
    _save_checkpoint(
        checkpoints_dir / "last.pt",
        diffusion,
        optimizer,
        ema,
        config,
        class_names,
        step,
        seen_samples,
        best_val,
    )
    return best_path


def train_independent_class_embedding_models(config: SyntheticEmbeddingConfig) -> Path:
    output_dir = prepare_synthetic_embedding_run(config)
    _set_seeds(config.seed)
    features, index = _load_real_dataset(config)
    labels, class_names = _encode_labels(index, config)
    stats = np.load(output_dir / "metadata" / "class_train_stats.npz")
    means = np.asarray(stats["mean"], dtype=np.float32)
    stds = np.asarray(stats["std"], dtype=np.float32)
    train_mask = _split_mask(index, config, "train")
    valid_mask = _split_mask(index, config, "valid")
    checkpoint_root = output_dir / "checkpoints"
    best_paths = []
    for class_id, class_name in enumerate(class_names):
        class_train_mask = train_mask & (labels == class_id)
        class_valid_mask = valid_mask & (labels == class_id)
        zero_train_labels = np.zeros(int(class_train_mask.sum()), dtype=np.int64)
        zero_valid_labels = np.zeros(int(class_valid_mask.sum()), dtype=np.int64)
        class_dir = checkpoint_root / _safe_name(class_name)
        best_path = _train_diffusion_from_arrays(
            config=config,
            train_features=features[class_train_mask],
            train_labels=zero_train_labels,
            valid_features=features[class_valid_mask],
            valid_labels=zero_valid_labels,
            mean=means[class_id],
            std=stds[class_id],
            n_classes=1,
            checkpoint_dir=class_dir,
            dynamics_path=output_dir / f"training_curves_{_safe_name(class_name)}.csv",
            checkpoint_class_names=[class_name],
            progress_desc=f"latent-ddim {class_name}",
            seed=config.seed + class_id,
        )
        best_paths.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "best_checkpoint": str(best_path),
            }
        )
    manifest_path = checkpoint_root / "independent_class_checkpoints.json"
    manifest_path.write_text(json.dumps(best_paths, indent=2))
    return manifest_path


def train_edm_conditional_embedding_model(config: SyntheticEmbeddingConfig) -> Path:
    output_dir = prepare_synthetic_embedding_run(config)
    _set_seeds(config.seed)
    features, index = _load_real_dataset(config)
    labels, class_names = _encode_labels(index, config)
    stats = np.load(output_dir / "metadata" / "train_stats.npz")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return _train_diffusion_from_arrays(
        config=config,
        train_features=features[_split_mask(index, config, "train")],
        train_labels=labels[_split_mask(index, config, "train")],
        valid_features=features[_split_mask(index, config, "valid")],
        valid_labels=labels[_split_mask(index, config, "valid")],
        mean=mean,
        std=std,
        n_classes=len(class_names),
        checkpoint_dir=output_dir / "checkpoints",
        dynamics_path=output_dir / "training_curves.csv",
        checkpoint_class_names=class_names,
        progress_desc=_edm_progress_desc(config),
        seed=config.seed,
    )


def sample_synthetic_embeddings(
    config: SyntheticEmbeddingConfig,
    checkpoint_path: Path | None = None,
) -> list[Path]:
    if config.model_mode == "independent_per_class":
        return sample_independent_class_embeddings(config)
    if config.model_mode in EDM_CONDITIONAL_MODES:
        return sample_edm_conditional_embeddings(config, checkpoint_path)
    if config.model_mode != "conditional":
        raise ValueError(f"Unsupported synthetic embedding model_mode {config.model_mode!r}.")

    output_dir = prepare_synthetic_embedding_run(config)
    checkpoint_path = checkpoint_path or output_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = list(checkpoint["class_names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = _make_diffusion(config, len(class_names)).to(device)
    diffusion.load_state_dict(checkpoint["model"])
    if "ema" in checkpoint:
        ema = EMA(diffusion, config.ema_decay)
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(diffusion)
    diffusion.eval()

    features, index = _load_real_dataset(config)
    labels, _ = _encode_labels(index, config)
    train_labels = labels[_split_mask(index, config, "train")]
    stats = np.load(output_dir / "metadata" / "train_stats.npz")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    written = []
    for sample_size in config.sample_sizes:
        counts = _counts_from_distribution(train_labels, len(class_names), sample_size)
        generated = _sample_diffusion_by_counts(
            diffusion,
            counts,
            config,
            device,
            seed=config.seed + sample_size,
        )
        raw = (generated * std + mean).astype(np.float32)
        labels_out = np.concatenate(
            [np.full(count, class_id, dtype=np.int64) for class_id, count in enumerate(counts)]
        )
        path = output_dir / "samples" / f"ddim_n{sample_size}"
        _write_synthetic_dataset(
            path,
            raw,
            labels_out,
            class_names,
            generator_name="conditional_latent_ddim",
            checkpoint_path=checkpoint_path,
            seed=config.seed,
            label_column=config.label_column,
        )
        written.append(path)
    return written


def sample_edm_conditional_embeddings(
    config: SyntheticEmbeddingConfig,
    checkpoint_path: Path | None = None,
) -> list[Path]:
    output_dir = prepare_synthetic_embedding_run(config)
    checkpoint_path = checkpoint_path or output_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = list(checkpoint["class_names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = _make_diffusion(config, len(class_names)).to(device)
    diffusion.load_state_dict(checkpoint["model"])
    if "ema" in checkpoint:
        ema = EMA(diffusion, config.ema_decay)
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(diffusion)
    diffusion.eval()

    features, index = _load_real_dataset(config)
    labels, _ = _encode_labels(index, config)
    train_labels = labels[_split_mask(index, config, "train")]
    stats = np.load(output_dir / "metadata" / "train_stats.npz")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    written = []
    for sample_size in config.sample_sizes:
        counts = _counts_from_distribution(train_labels, len(class_names), sample_size)
        generated = _sample_diffusion_by_counts(
            diffusion,
            counts,
            config,
            device,
            seed=config.seed + sample_size,
        )
        raw = (generated * std + mean).astype(np.float32)
        labels_out = np.concatenate(
            [np.full(count, class_id, dtype=np.int64) for class_id, count in enumerate(counts)]
        )
        path = output_dir / "samples" / _edm_sample_dir_name(config, sample_size)
        _write_synthetic_dataset(
            path,
            raw,
            labels_out,
            class_names,
            generator_name=_edm_generator_name(config),
            checkpoint_path=checkpoint_path,
            seed=config.seed,
            label_column=config.label_column,
        )
        written.append(path)
    return written


def _edm_generator_name(config: SyntheticEmbeddingConfig) -> str:
    if config.model_mode == "edm_adaln_mlp_conditional":
        return "edm_adaln_mlp_conditional"
    if config.model_mode == "classifier_free_edm_conditional":
        return "classifier_free_edm_conditional"
    return "edm_conditional"


def _edm_sample_dir_name(config: SyntheticEmbeddingConfig, sample_size: int) -> str:
    if config.model_mode == "edm_adaln_mlp_conditional":
        return f"edm_adaln_mlp_cfg{config.guidance_scale:g}_n{sample_size}"
    if config.model_mode == "classifier_free_edm_conditional":
        return f"classifier_free_edm_cfg{config.guidance_scale:g}_n{sample_size}"
    return f"edm_conditional_cfg{config.guidance_scale:g}_n{sample_size}"


def _edm_progress_desc(config: SyntheticEmbeddingConfig) -> str:
    if config.model_mode == "edm_adaln_mlp_conditional":
        return "edm-adaln-mlp seen samples"
    if config.model_mode == "classifier_free_edm_conditional":
        return "classifier-free-edm seen samples"
    return "edm-conditional seen samples"


def sample_independent_class_embeddings(config: SyntheticEmbeddingConfig) -> list[Path]:
    output_dir = prepare_synthetic_embedding_run(config)
    features, index = _load_real_dataset(config)
    labels, class_names = _encode_labels(index, config)
    train_labels = labels[_split_mask(index, config, "train")]
    stats = np.load(output_dir / "metadata" / "class_train_stats.npz")
    means = np.asarray(stats["mean"], dtype=np.float32)
    stds = np.asarray(stats["std"], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    written = []
    for sample_size in config.sample_sizes:
        counts = _counts_from_distribution(train_labels, len(class_names), sample_size)
        generated_by_class = []
        labels_out = []
        checkpoint_paths = []
        for class_id, count in enumerate(counts):
            class_name = class_names[class_id]
            checkpoint_path = output_dir / "checkpoints" / _safe_name(class_name) / "best.pt"
            checkpoint_paths.append(str(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            diffusion = _make_diffusion(config, n_classes=1).to(device)
            diffusion.load_state_dict(checkpoint["model"])
            if "ema" in checkpoint:
                ema = EMA(diffusion, config.ema_decay)
                ema.load_state_dict(checkpoint["ema"])
                ema.copy_to(diffusion)
            diffusion.eval()
            generated = _sample_diffusion_by_counts(
                diffusion,
                np.asarray([int(count)]),
                config,
                device,
                seed=config.seed + sample_size + class_id,
            )
            raw = (generated * stds[class_id] + means[class_id]).astype(np.float32)
            generated_by_class.append(raw)
            labels_out.append(np.full(int(count), class_id, dtype=np.int64))
            del diffusion
            if device.type == "cuda":
                torch.cuda.empty_cache()
        path = output_dir / "samples" / f"ddim_independent_n{sample_size}"
        _write_synthetic_dataset(
            path,
            np.vstack(generated_by_class).astype(np.float32),
            np.concatenate(labels_out),
            class_names,
            generator_name="independent_class_latent_ddim",
            checkpoint_path=None,
            seed=config.seed,
            label_column=config.label_column,
        )
        (path / "checkpoint_paths.json").write_text(json.dumps(checkpoint_paths, indent=2))
        written.append(path)
    return written


def write_synthetic_baselines(config: SyntheticEmbeddingConfig) -> list[Path]:
    output_dir = prepare_synthetic_embedding_run(config)
    features, index = _load_real_dataset(config)
    labels, class_names = _encode_labels(index, config)
    train_mask = _split_mask(index, config, "train")
    train_features = features[train_mask]
    train_labels = labels[train_mask]
    stats = np.load(output_dir / "metadata" / "train_stats.npz")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    train_z = ((train_features - mean) / std).astype(np.float32)
    rng = np.random.default_rng(config.seed)
    written = []
    for sample_size in config.sample_sizes:
        counts = _counts_from_distribution(train_labels, len(class_names), sample_size)
        for baseline in BASELINE_GENERATORS:
            synthetic, synthetic_labels = _sample_baseline(
                baseline,
                train_features,
                train_z,
                train_labels,
                counts,
                mean,
                std,
                rng,
            )
            path = output_dir / "samples" / f"{baseline}_n{sample_size}"
            _write_synthetic_dataset(
                path,
                synthetic,
                synthetic_labels,
                class_names,
                generator_name=baseline,
                checkpoint_path=None,
                seed=config.seed,
                label_column=config.label_column,
            )
            written.append(path)
    return written


def evaluate_synthetic_embeddings(
    config: SyntheticEmbeddingConfig,
    synthetic_dir: Path,
    output_dir: Path | None = None,
) -> Path:
    prepare_synthetic_embedding_run(config)
    output_dir = output_dir or config.output_dir / "evaluation" / synthetic_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    real_features, real_index = _load_real_dataset(config)
    real_labels, class_names = _encode_labels(real_index, config)
    synthetic_features, synthetic_index = _load_npz_parquet(synthetic_dir)
    synthetic_labels = synthetic_index["label_id"].to_numpy(dtype=np.int64)
    if synthetic_features.shape[1] != config.embedding_dim:
        raise ValueError(
            f"Synthetic features have dim {synthetic_features.shape[1]}, expected {config.embedding_dim}."
        )
    stats = np.load(config.output_dir / "metadata" / "train_stats.npz")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    fidelity = _fidelity_metrics(
        real_features,
        real_labels,
        real_index,
        synthetic_features,
        synthetic_labels,
        class_names,
        mean,
        std,
        config,
        output_dir,
    )
    fidelity.to_csv(output_dir / "fidelity_metrics.csv", index=False)
    utility = _utility_metrics(
        real_features,
        real_labels,
        real_index,
        synthetic_features,
        synthetic_labels,
        class_names,
        config,
        output_dir / "utility",
    )
    utility.to_csv(output_dir / "utility_metrics.csv", index=False)
    return output_dir


def _make_diffusion(config: SyntheticEmbeddingConfig, n_classes: int) -> LatentDiffusion | EDMLatentDiffusion:
    if config.model_mode in EDM_CONDITIONAL_MODES:
        if config.model_mode == "edm_adaln_mlp_conditional":
            denoiser = AdaLNZeroMLPDenoiser(
                input_dim=config.embedding_dim,
                n_classes=n_classes,
                hidden_dim=config.hidden_dim,
                depth=config.depth,
                dropout=config.dropout,
                extra_condition_tokens=1,
            )
        else:
            denoiser = MLPSkipDenoiser(
                input_dim=config.embedding_dim,
                n_classes=n_classes,
                hidden_dim=config.hidden_dim,
                depth=config.depth,
                dropout=config.dropout,
                use_class_condition=True,
                extra_condition_tokens=1,
            )
        return EDMLatentDiffusion(
            denoiser,
            sigma_min=config.edm_sigma_min,
            sigma_max=config.edm_sigma_max,
            sigma_data=config.edm_sigma_data,
            rho=config.edm_rho,
            class_dropout=config.edm_class_dropout,
        )
    denoiser = MLPSkipDenoiser(
        input_dim=config.embedding_dim,
        n_classes=n_classes,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        dropout=config.dropout,
        use_class_condition=config.model_mode == "conditional",
    )
    return LatentDiffusion(denoiser, timesteps=config.timesteps, beta=config.beta)


def _train_diffusion_from_arrays(
    *,
    config: SyntheticEmbeddingConfig,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    valid_features: np.ndarray,
    valid_labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    n_classes: int,
    checkpoint_dir: Path,
    dynamics_path: Path,
    checkpoint_class_names: list[str],
    progress_desc: str,
    seed: int,
) -> Path:
    _set_seeds(seed)
    train_dataset = EmbeddingSplitDataset(train_features, train_labels, mean, std)
    valid_dataset = EmbeddingSplitDataset(valid_features, valid_labels, mean, std)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_dataset) >= config.batch_size,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = _make_diffusion(config, n_classes).to(device)
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler(enabled=config.use_amp and device.type == "cuda")
    ema = EMA(diffusion, config.ema_decay)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dynamics_path.parent.mkdir(parents=True, exist_ok=True)
    dynamics = []
    best_val = float("inf")
    best_path = checkpoint_dir / "best.pt"
    validations_without_improvement = 0
    seen_samples = 0
    step = 0
    started = perf_counter()
    train_iter = iter(train_loader)
    progress = tqdm(total=config.max_seen_samples, desc=progress_desc, unit="sample")

    while seen_samples < config.max_seen_samples:
        try:
            batch, batch_labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch, batch_labels = next(train_iter)
        batch = batch.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=config.use_amp and device.type == "cuda"):
            loss = diffusion.training_loss(batch, batch_labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(diffusion)
        step += 1
        seen_samples += batch.shape[0]
        progress.update(batch.shape[0])

        should_validate = step == 1 or step % config.validation_interval_steps == 0
        if should_validate:
            val_loss = _validation_loss(diffusion, valid_loader, device, config)
            row = {
                "step": step,
                "seen_samples": seen_samples,
                "train_loss": float(loss.detach().cpu()),
                "valid_loss": val_loss,
                "elapsed_seconds": perf_counter() - started,
            }
            dynamics.append(row)
            pd.DataFrame(dynamics).to_csv(dynamics_path, index=False)
            if val_loss < best_val:
                best_val = val_loss
                validations_without_improvement = 0
                _save_checkpoint(
                    best_path,
                    diffusion,
                    optimizer,
                    ema,
                    config,
                    checkpoint_class_names,
                    step,
                    seen_samples,
                    best_val,
                )
            else:
                validations_without_improvement += 1
                if validations_without_improvement >= config.patience_validations:
                    break

    progress.close()
    _save_checkpoint(
        checkpoint_dir / "last.pt",
        diffusion,
        optimizer,
        ema,
        config,
        checkpoint_class_names,
        step,
        seen_samples,
        best_val,
    )
    return best_path


def _validation_loss(
    diffusion: LatentDiffusion | EDMLatentDiffusion,
    valid_loader: DataLoader,
    device: torch.device,
    config: SyntheticEmbeddingConfig,
) -> float:
    diffusion.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, (batch, labels) in enumerate(valid_loader):
            if batch_idx >= config.validation_batches:
                break
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(enabled=config.use_amp and device.type == "cuda"):
                losses.append(float(diffusion.training_loss(batch, labels).detach().cpu()))
    diffusion.train()
    return float(np.mean(losses)) if losses else float("nan")


def _save_checkpoint(
    path: Path,
    diffusion: LatentDiffusion | EDMLatentDiffusion,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    config: SyntheticEmbeddingConfig,
    class_names: list[str],
    step: int,
    seen_samples: int,
    best_val: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": diffusion.state_dict(),
            "ema": ema.state_dict(),
            "config": _jsonable_config(config),
            "class_names": class_names,
            "step": step,
            "seen_samples": seen_samples,
            "best_valid_loss": best_val,
            "optimizer_state": "omitted_to_keep_checkpoint_below_filesystem_limits",
        },
        path,
    )


def _load_real_dataset(config: SyntheticEmbeddingConfig) -> tuple[np.ndarray, pd.DataFrame]:
    _require_embedding_dir(config)
    if config.embedding_dir is None:
        raise ValueError("Synthetic embedding config requires embedding_dir.")
    if (config.embedding_dir / "features.npz").exists() and (config.embedding_dir / "index.parquet").exists():
        return _load_npz_parquet(config.embedding_dir)
    if _is_chunked_embedding_cache(config.embedding_dir):
        return _load_chunked_embedding_cache(config.embedding_dir, config)
    raise FileNotFoundError(
        f"Unsupported embedding layout under {config.embedding_dir}. Expected either "
        "features.npz + index.parquet or a chunked cache with split directories containing emb_*.pt."
    )


def _load_npz_parquet(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    features_path = path / "features.npz"
    index_path = path / "index.parquet"
    if not features_path.exists() or not index_path.exists():
        raise FileNotFoundError(f"Expected features.npz and index.parquet under {path}.")
    with np.load(features_path, allow_pickle=True) as data:
        if "features" not in data.files:
            raise ValueError(f"{features_path} must contain a features array.")
        features = np.asarray(data["features"], dtype=np.float32)
    index = pd.read_parquet(index_path)
    if len(index) != len(features):
        raise ValueError(f"{features_path} rows ({len(features)}) do not match {index_path} ({len(index)}).")
    return features, index


def _is_chunked_embedding_cache(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "label_map.json").exists()
        and (path / "train").is_dir()
        and any((path / "train").glob("emb_[0-9][0-9][0-9][0-9].pt"))
    )


def _load_chunked_embedding_cache(path: Path, config: SyntheticEmbeddingConfig) -> tuple[np.ndarray, pd.DataFrame]:
    label_map = {
        str(key): str(value)
        for key, value in json.loads((path / "label_map.json").read_text()).items()
    }
    split_arrays = {}
    split_indexes = {}
    for split in SPLITS:
        split_dir = path / split
        if not split_dir.is_dir():
            continue
        split_arrays[split], split_indexes[split] = _load_chunked_split(split_dir, split, label_map)

    if "train" not in split_arrays or "test" not in split_arrays:
        raise ValueError(f"{path} must contain at least train and test chunked embedding splits.")
    if "valid" not in split_arrays:
        train_features = split_arrays["train"]
        train_index = split_indexes["train"].copy()
        valid_mask = _stratified_validation_mask(
            train_index["label_id"].to_numpy(dtype=np.int64),
            validation_fraction=config.validation_fraction,
            seed=config.seed,
        )
        train_index.loc[valid_mask, "split"] = "valid"
        split_indexes["train"] = train_index

    features = np.vstack([split_arrays[split] for split in split_arrays]).astype(np.float32, copy=False)
    index = pd.concat([split_indexes[split] for split in split_indexes], ignore_index=True)
    if len(index) != len(features):
        raise ValueError(f"Loaded {len(features)} feature rows from {path}, but built {len(index)} index rows.")
    return features, index


def _load_chunked_split(
    split_dir: Path,
    split: str,
    label_map: dict[str, str],
) -> tuple[np.ndarray, pd.DataFrame]:
    features = []
    rows = []
    offset = 0
    for emb_path in sorted(split_dir.glob("emb_[0-9][0-9][0-9][0-9].pt")):
        suffix = emb_path.stem.split("_", 1)[1]
        labels_path = split_dir / f"labels_{suffix}.pt"
        paths_path = split_dir / f"paths_{suffix}.json"
        if not labels_path.exists() or not paths_path.exists():
            raise FileNotFoundError(f"Expected {labels_path.name} and {paths_path.name} beside {emb_path}.")
        emb = torch.load(emb_path, map_location="cpu").detach().cpu().numpy().astype(np.float32, copy=False)
        labels = torch.load(labels_path, map_location="cpu").detach().cpu().numpy().astype(np.int64, copy=False)
        paths = json.loads(paths_path.read_text())
        if len(emb) != len(labels) or len(emb) != len(paths):
            raise ValueError(
                f"{emb_path} has {len(emb)} embeddings, {labels_path} has {len(labels)} labels, "
                f"and {paths_path} has {len(paths)} path records."
            )
        features.append(emb)
        for row_idx, label_id in enumerate(labels.astype(int)):
            metadata = paths[row_idx] if isinstance(paths[row_idx], dict) else {}
            class_name = label_map.get(str(label_id), str(label_id))
            sample_id = str(
                metadata.get("slide_id")
                or metadata.get("sample_id")
                or metadata.get("patch_filename")
                or f"{split}_{offset + row_idx:06d}"
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "source_split": split,
                    "label_id": int(label_id),
                    "class_name": class_name,
                    "annotation_label_name": class_name,
                    "category": class_name,
                    "label": class_name,
                    "patch_path": str(metadata.get("patch_path", "")),
                    "patch_filename": str(metadata.get("patch_filename", "")),
                    "slide_id": str(metadata.get("slide_id", sample_id)),
                }
            )
        offset += len(labels)
    if not features:
        raise ValueError(f"No numbered emb_*.pt shard files found in {split_dir}.")
    return np.vstack(features).astype(np.float32, copy=False), pd.DataFrame(rows)


def _stratified_validation_mask(labels: np.ndarray, validation_fraction: float, seed: int) -> np.ndarray:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1) when creating a validation split.")
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for label_id in np.unique(labels):
        indices = np.flatnonzero(labels == label_id)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_valid = max(1, int(round(len(shuffled) * validation_fraction)))
        mask[shuffled[:n_valid]] = True
    return mask


def _encode_labels(index: pd.DataFrame, config: SyntheticEmbeddingConfig) -> tuple[np.ndarray, list[str]]:
    if config.label_column not in index.columns:
        raise ValueError(f"{config.label_column!r} is not in index columns: {list(index.columns)}")
    train_values = index.loc[_split_mask(index, config, "train"), config.label_column].astype(str)
    present = set(train_values.unique().tolist())
    class_names = [name for name in config.class_order if name in present]
    class_names.extend(sorted(present - set(class_names)))
    mapping = {name: idx for idx, name in enumerate(class_names)}
    labels = index[config.label_column].astype(str).map(mapping)
    if labels.isna().any():
        missing = sorted(index.loc[labels.isna(), config.label_column].astype(str).unique())
        raise ValueError(f"Labels found outside train mapping: {missing}")
    return labels.to_numpy(dtype=np.int64), class_names


def _split_mask(index: pd.DataFrame, config: SyntheticEmbeddingConfig, split: str) -> np.ndarray:
    if config.split_column not in index.columns:
        raise ValueError(f"{config.split_column!r} is not in index columns: {list(index.columns)}")
    return index[config.split_column].astype(str).to_numpy() == split


def _write_dataset_summary(
    index: pd.DataFrame,
    labels: np.ndarray,
    config: SyntheticEmbeddingConfig,
    output_path: Path,
) -> None:
    rows = []
    for split in SPLITS:
        mask = _split_mask(index, config, split)
        for label_id in sorted(np.unique(labels)):
            count = int(np.sum(mask & (labels == label_id)))
            rows.append({"split": split, "label_id": int(label_id), "count": count})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _counts_from_distribution(labels: np.ndarray, n_classes: int, total: int) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=n_classes).astype(np.float64)
    proportions = counts / counts.sum()
    expected = proportions * int(total)
    base = np.floor(expected).astype(int)
    remainder = int(total) - int(base.sum())
    if remainder:
        order = np.argsort(expected - base)[::-1]
        base[order[:remainder]] += 1
    return base


def _sample_diffusion_by_counts(
    diffusion: LatentDiffusion | EDMLatentDiffusion,
    counts: np.ndarray,
    config: SyntheticEmbeddingConfig,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    chunks = []
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for class_id, count in enumerate(counts):
            remaining = int(count)
            while remaining > 0:
                batch_size = min(config.batch_size, remaining)
                labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
                batch = diffusion.sample(
                    labels,
                    steps=config.ddim_steps,
                    device=device,
                    generator=generator,
                    **({"guidance_scale": config.guidance_scale} if isinstance(diffusion, EDMLatentDiffusion) else {}),
                )
                chunks.append(batch.cpu().numpy())
                remaining -= batch_size
    return np.vstack(chunks).astype(np.float32)


def _sample_baseline(
    baseline: str,
    train_features: np.ndarray,
    train_z: np.ndarray,
    train_labels: np.ndarray,
    counts: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    synthetic = []
    labels = []
    for class_id, count in enumerate(counts):
        class_mask = train_labels == class_id
        if baseline == "bootstrap":
            source = train_features[class_mask]
            indices = rng.integers(0, len(source), size=int(count))
            samples = source[indices].astype(np.float32)
        elif baseline == "gaussian":
            source = train_z[class_mask]
            class_mean = source.mean(axis=0)
            class_std = np.maximum(source.std(axis=0), 1.0e-6)
            samples_z = rng.normal(class_mean, class_std, size=(int(count), source.shape[1])).astype(np.float32)
            samples = (samples_z * std + mean).astype(np.float32)
        else:
            raise ValueError(f"Unknown baseline {baseline!r}.")
        synthetic.append(samples)
        labels.append(np.full(int(count), class_id, dtype=np.int64))
    return np.vstack(synthetic).astype(np.float32), np.concatenate(labels)


def _write_synthetic_dataset(
    path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    *,
    generator_name: str,
    checkpoint_path: Path | None,
    seed: int,
    label_column: str,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "features.npz", features=features.astype(np.float32, copy=False))
    rows = []
    for idx, label_id in enumerate(labels.astype(int)):
        class_name = class_names[label_id]
        row = {
            "sample_id": f"synthetic_{generator_name}_{idx:06d}",
            "split": "synthetic",
            "annotation_label_name": class_name,
            "class_name": class_name,
            "label": class_name,
            "category": class_name,
            "label_id": label_id,
            "generator": generator_name,
            "checkpoint": str(checkpoint_path) if checkpoint_path else "",
            "seed": seed,
        }
        row[label_column] = class_name
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path / "index.parquet", index=False)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "generator": generator_name,
                "num_embeddings": int(len(features)),
                "embedding_dim": int(features.shape[1]),
                "class_names": class_names,
                "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                "seed": seed,
            },
            indent=2,
        )
    )


def _fidelity_metrics(
    real_features: np.ndarray,
    real_labels: np.ndarray,
    real_index: pd.DataFrame,
    synthetic_features: np.ndarray,
    synthetic_labels: np.ndarray,
    class_names: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    config: SyntheticEmbeddingConfig,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(config.seed)
    real_z = ((real_features - mean) / std).astype(np.float32)
    synth_z = ((synthetic_features - mean) / std).astype(np.float32)
    for class_id, class_name in enumerate(class_names):
        synth_idx = np.flatnonzero(synthetic_labels == class_id)
        if len(synth_idx) == 0:
            continue
        synth_sample_idx = _subsample_indices(synth_idx, config.eval_max_per_class, rng)
        synth_sample = synthetic_features[synth_sample_idx]
        synth_sample_z = synth_z[synth_sample_idx]
        for split in SPLITS:
            real_idx = np.flatnonzero(_split_mask(real_index, config, split) & (real_labels == class_id))
            if len(real_idx) == 0:
                continue
            real_sample_idx = _subsample_indices(real_idx, config.eval_max_per_class, rng)
            real_sample = real_features[real_sample_idx]
            real_sample_z = real_z[real_sample_idx]
            row = {
                "class_name": class_name,
                "class_id": class_id,
                "reference_split": split,
                "real_count": int(len(real_idx)),
                "synthetic_count": int(len(synth_idx)),
                "real_norm_mean": float(np.linalg.norm(real_sample, axis=1).mean()),
                "synthetic_norm_mean": float(np.linalg.norm(synth_sample, axis=1).mean()),
                "mean_l2": float(np.linalg.norm(real_sample.mean(axis=0) - synth_sample.mean(axis=0))),
                "std_l2": float(np.linalg.norm(real_sample.std(axis=0) - synth_sample.std(axis=0))),
                "mmd_rbf": _mmd_rbf(real_sample_z, synth_sample_z),
                "frechet_z": _frechet_distance(real_sample_z, synth_sample_z),
            }
            if split == "train":
                row.update(_manifold_metrics(real_sample_z, synth_sample_z, config.eval_knn_k))
            rows.append(row)
    _write_pca_plot(real_z, real_labels, synth_z, synthetic_labels, class_names, output_dir / "pca_real_synthetic.png", config)
    return pd.DataFrame(rows)


def _utility_metrics(
    real_features: np.ndarray,
    real_labels: np.ndarray,
    real_index: pd.DataFrame,
    synthetic_features: np.ndarray,
    synthetic_labels: np.ndarray,
    class_names: list[str],
    config: SyntheticEmbeddingConfig,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    train_mask = _split_mask(real_index, config, "train")
    test_mask = _split_mask(real_index, config, "test")
    x_train = real_features[train_mask]
    y_train = real_labels[train_mask]
    x_test = real_features[test_mask]
    y_test = real_labels[test_mask]

    scenarios = [
        ("real_train_to_real_test", x_train, y_train),
        ("synthetic_to_real_test", synthetic_features, synthetic_labels),
    ]
    rng = np.random.default_rng(config.seed)
    for ratio in (0.5, 1.0, 2.0):
        extra_n = int(round(len(x_train) * ratio))
        indices = rng.integers(0, len(synthetic_features), size=extra_n)
        scenarios.append(
            (
                f"real_plus_synthetic_{ratio:g}x_to_real_test",
                np.vstack([x_train, synthetic_features[indices]]),
                np.concatenate([y_train, synthetic_labels[indices]]),
            )
        )

    for name, train_x, train_y in scenarios:
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=10,
                max_iter=2000,
                solver="lbfgs",
                tol=1.0e-5,
                random_state=config.seed,
            ),
        )
        estimator.fit(train_x, train_y)
        proba = estimator.predict_proba(x_test)
        pred = np.argmax(proba, axis=1)
        run_dir = output_dir / name
        metrics = write_evaluation_artifacts(
            run_dir,
            y_test,
            pred,
            proba,
            [f"test_{idx}" for idx in range(len(y_test))],
            class_names,
        )
        rows.append(
            {
                **metrics,
                "lr_accuracy": metrics["accuracy"],
                "lr_macro_roc_auc": metrics["auc"],
                "lr_C": 10,
                "lr_max_iter": 2000,
                "lr_linesearch_max_iter": 300,
                "lr_tol": 1.0e-5,
                "scenario": name,
                "train_rows": int(len(train_x)),
                "test_rows": int(len(x_test)),
            }
        )
    return pd.DataFrame(rows)


def _subsample_indices(indices: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= max_count:
        return indices
    return rng.choice(indices, size=max_count, replace=False)


def _mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    combined = np.vstack([x, y])
    sample = combined
    if len(sample) > 1000:
        sample = sample[np.linspace(0, len(sample) - 1, 1000).astype(int)]
    distances = pairwise_distances(sample, metric="euclidean")
    median_sq = np.median(distances[distances > 0]) ** 2
    gamma = 1.0 / max(2.0 * median_sq, 1.0e-6)
    xx = np.exp(-gamma * pairwise_distances(x, x, metric="sqeuclidean"))
    yy = np.exp(-gamma * pairwise_distances(y, y, metric="sqeuclidean"))
    xy = np.exp(-gamma * pairwise_distances(x, y, metric="sqeuclidean"))
    return float(xx.mean() + yy.mean() - 2.0 * xy.mean())


def _frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    mean_diff = x64.mean(axis=0) - y64.mean(axis=0)
    cov_x = np.cov(x64, rowvar=False)
    cov_y = np.cov(y64, rowvar=False)
    cov_prod_sqrt = linalg.sqrtm(cov_x @ cov_y)
    if np.iscomplexobj(cov_prod_sqrt):
        cov_prod_sqrt = cov_prod_sqrt.real
    value = mean_diff @ mean_diff + np.trace(cov_x) + np.trace(cov_y) - 2.0 * np.trace(cov_prod_sqrt)
    return float(max(value, 0.0))


def _manifold_metrics(real: np.ndarray, synthetic: np.ndarray, k: int) -> dict[str, float]:
    if len(real) <= k or len(synthetic) <= k:
        return {"precision_knn": float("nan"), "recall_knn": float("nan")}
    real_nn = NearestNeighbors(n_neighbors=k + 1).fit(real)
    real_radii = real_nn.kneighbors(real, return_distance=True)[0][:, -1]
    synth_to_real_dist, synth_to_real_idx = real_nn.kneighbors(synthetic, n_neighbors=1)
    precision = np.mean(synth_to_real_dist[:, 0] <= real_radii[synth_to_real_idx[:, 0]])

    synth_nn = NearestNeighbors(n_neighbors=min(k + 1, len(synthetic))).fit(synthetic)
    synth_radii = synth_nn.kneighbors(synthetic, return_distance=True)[0][:, -1]
    real_to_synth_dist, real_to_synth_idx = synth_nn.kneighbors(real, n_neighbors=1)
    recall = np.mean(real_to_synth_dist[:, 0] <= synth_radii[real_to_synth_idx[:, 0]])
    nearest_mean = float(synth_to_real_dist[:, 0].mean())
    return {
        "precision_knn": float(precision),
        "recall_knn": float(recall),
        "synthetic_to_real_nn_distance_mean": nearest_mean,
    }


def _write_pca_plot(
    real_z: np.ndarray,
    real_labels: np.ndarray,
    synth_z: np.ndarray,
    synth_labels: np.ndarray,
    class_names: list[str],
    path: Path,
    config: SyntheticEmbeddingConfig,
) -> None:
    try:
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(config.seed)
        real_indices = []
        synth_indices = []
        per_class = max(config.eval_max_per_class // 4, 250)
        for class_id in range(len(class_names)):
            real_indices.extend(_subsample_indices(np.flatnonzero(real_labels == class_id), per_class, rng))
            synth_indices.extend(_subsample_indices(np.flatnonzero(synth_labels == class_id), per_class, rng))
        real_sample = real_z[np.asarray(real_indices)]
        synth_sample = synth_z[np.asarray(synth_indices)]
        pca = PCA(n_components=2, random_state=config.seed)
        coords = pca.fit_transform(np.vstack([real_sample, synth_sample]))
        source = np.array(["real"] * len(real_sample) + ["synthetic"] * len(synth_sample))
        labels = np.concatenate([real_labels[np.asarray(real_indices)], synth_labels[np.asarray(synth_indices)]])
        fig, ax = plt.subplots(figsize=(8, 6))
        for class_id, class_name in enumerate(class_names):
            for source_name, marker in [("real", "o"), ("synthetic", "x")]:
                mask = (labels == class_id) & (source == source_name)
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=8,
                    alpha=0.45,
                    marker=marker,
                    label=f"{class_name} {source_name}",
                )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".error.txt").write_text(f"{type(exc).__name__}: {exc}\n")


def _require_embedding_dir(config: SyntheticEmbeddingConfig) -> None:
    if config.embedding_dir is None:
        raise ValueError("synthetic_embeddings.embedding_dir is required.")


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable_config(config: SyntheticEmbeddingConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
