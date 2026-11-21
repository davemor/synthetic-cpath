from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from synpath.artifacts import write_json
from synpath.manifests import proportional_condition_schedule, read_manifest


@dataclass(frozen=True)
class DCGANConfig:
    image_size: int
    latent_dim: int
    base_channels: int
    batch_size: int
    epochs: int
    max_steps: int | None
    lr: float
    beta1: float
    beta2: float
    num_workers: int
    conditional: bool
    use_data_parallel: bool


class ManifestImageDataset(Dataset):
    def __init__(
        self, manifest: Path, image_size: int, conditional: bool
    ) -> None:
        self.rows = read_manifest(manifest)
        self.conditional = conditional
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        condition = int(row["condition_id"]) if self.conditional else 0
        return self.transform(image), torch.tensor(condition, dtype=torch.long)


class DCGANGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        base_channels: int,
        image_size: int,
        n_conditions: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_conditions = n_conditions
        input_dim = latent_dim
        if n_conditions > 0:
            self.condition_embedding = nn.Embedding(n_conditions, latent_dim)
            input_dim += latent_dim
        else:
            self.condition_embedding = None

        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                input_dim, base_channels * 16, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(True),
        ]
        channels = base_channels * 16
        current_size = 4
        while current_size < image_size:
            out_channels = max(base_channels, channels // 2)
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels,
                        out_channels,
                        4,
                        2,
                        1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                ]
            )
            channels = out_channels
            current_size *= 2
        layers.extend([nn.Conv2d(channels, 3, 3, 1, 1), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(
        self, noise: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if self.condition_embedding is not None:
            embedded = self.condition_embedding(labels)
            noise = torch.cat([noise, embedded], dim=1)
        return self.net(noise[:, :, None, None])


class DCGANDiscriminator(nn.Module):
    def __init__(
        self,
        base_channels: int,
        image_size: int,
        n_conditions: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.n_conditions = n_conditions
        input_channels = 3
        if n_conditions > 0:
            self.condition_embedding = nn.Embedding(
                n_conditions, image_size * image_size
            )
            input_channels += 1
        else:
            self.condition_embedding = None

        layers: list[nn.Module] = []
        channels = base_channels
        layers.extend(
            [
                nn.Conv2d(input_channels, channels, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        current_size = image_size // 2
        while current_size > 4:
            out_channels = min(base_channels * 16, channels * 2)
            layers.extend(
                [
                    nn.Conv2d(channels, out_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = out_channels
            current_size //= 2
        layers.append(nn.Conv2d(channels, 1, 4, 1, 0, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if self.condition_embedding is not None:
            condition = self.condition_embedding(labels)
            condition = condition.view(-1, 1, self.image_size, self.image_size)
            images = torch.cat([images, condition], dim=1)
        return self.net(images).flatten()


def train_dcgan(
    root: Path,
    condition_manifest: Path,
    cfg: dict,
    conditional: bool,
    seed: int,
) -> list[Path]:
    dcgan_cfg = resolve_config(cfg, conditional)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ManifestImageDataset(
        condition_manifest,
        dcgan_cfg.image_size,
        dcgan_cfg.conditional,
    )
    n_conditions = (
        condition_count(condition_manifest) if dcgan_cfg.conditional else 0
    )
    generator = DCGANGenerator(
        dcgan_cfg.latent_dim,
        dcgan_cfg.base_channels,
        dcgan_cfg.image_size,
        n_conditions,
    )
    discriminator = DCGANDiscriminator(
        dcgan_cfg.base_channels,
        dcgan_cfg.image_size,
        n_conditions,
    )
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    generator = maybe_parallel(generator.to(device), dcgan_cfg)
    discriminator = maybe_parallel(discriminator.to(device), dcgan_cfg)

    loader = DataLoader(
        dataset,
        batch_size=dcgan_cfg.batch_size,
        shuffle=True,
        num_workers=dcgan_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    criterion = nn.BCEWithLogitsLoss()
    optim_g = torch.optim.Adam(
        generator.parameters(),
        lr=dcgan_cfg.lr,
        betas=(dcgan_cfg.beta1, dcgan_cfg.beta2),
    )
    optim_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=dcgan_cfg.lr,
        betas=(dcgan_cfg.beta1, dcgan_cfg.beta2),
    )

    root.mkdir(parents=True, exist_ok=True)
    step = 0
    generator.train()
    discriminator.train()
    for _epoch in range(dcgan_cfg.epochs):
        progress = tqdm(loader, desc="dcgan", leave=False)
        for real_images, labels in progress:
            real_images = real_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = real_images.shape[0]

            real_targets = torch.ones(batch_size, device=device)
            fake_targets = torch.zeros(batch_size, device=device)
            noise = torch.randn(batch_size, dcgan_cfg.latent_dim, device=device)

            optim_d.zero_grad(set_to_none=True)
            fake_images = generator(noise, labels)
            real_loss = criterion(
                discriminator(real_images, labels), real_targets
            )
            fake_loss = criterion(
                discriminator(fake_images.detach(), labels),
                fake_targets,
            )
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optim_d.step()

            optim_g.zero_grad(set_to_none=True)
            g_loss = criterion(discriminator(fake_images, labels), real_targets)
            g_loss.backward()
            optim_g.step()

            step += 1
            progress.set_postfix(
                d_loss=f"{d_loss.item():.3f}",
                g_loss=f"{g_loss.item():.3f}",
            )
            if dcgan_cfg.max_steps is not None and step >= dcgan_cfg.max_steps:
                break
        if dcgan_cfg.max_steps is not None and step >= dcgan_cfg.max_steps:
            break

    checkpoint = root / "dcgan.pt"
    torch.save(
        {
            "generator": unwrap(generator).state_dict(),
            "discriminator": unwrap(discriminator).state_dict(),
            "config": dcgan_cfg.__dict__,
            "n_conditions": n_conditions,
            "seed": seed,
            "steps": step,
        },
        checkpoint,
    )
    write_json(
        root / "dcgan_training.json",
        {
            "steps": step,
            "n_conditions": n_conditions,
            "seed": seed,
            **dcgan_cfg.__dict__,
        },
    )
    return [checkpoint, root / "dcgan_training.json"]


def sample_dcgan(
    root: Path,
    sample_dir: Path,
    condition_manifest: Path,
    cfg: dict,
    max_samples: int,
    seed: int,
) -> None:
    checkpoint = torch.load(
        root / "dcgan.pt",
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )
    model_cfg = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = DCGANGenerator(
        int(model_cfg["latent_dim"]),
        int(model_cfg["base_channels"]),
        int(model_cfg["image_size"]),
        int(checkpoint["n_conditions"]),
    ).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    rows = proportional_condition_schedule(
        read_manifest(condition_manifest), max_samples, seed=seed
    )
    sample_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(
        cfg.get("generation", {}).get("dcgan", {}).get("sample_batch_size", 128)
    )
    latent_dim = int(model_cfg["latent_dim"])
    with torch.no_grad():
        for start in tqdm(range(0, len(rows), batch_size), desc="dcgan sample"):
            batch_rows = rows[start : start + batch_size]
            labels = [
                int(row["condition_id"])
                if int(checkpoint["n_conditions"]) > 0
                else 0
                for row in batch_rows
            ]
            label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
            generator_seed = seed + start
            torch.manual_seed(generator_seed)
            noise = torch.randn(len(batch_rows), latent_dim, device=device)
            images = generator(noise, label_tensor).cpu()
            for offset, image in enumerate(images):
                save_image(
                    (image + 1.0) / 2.0,
                    sample_dir / f"{start + offset:08d}.png",
                )


def resolve_config(cfg: dict, conditional: bool) -> DCGANConfig:
    dcgan_cfg = cfg.get("generation", {}).get("dcgan", {})
    image_size = int(
        dcgan_cfg.get(
            "image_size", cfg.get("images", {}).get("generator_size", 256)
        )
    )
    if not math.log2(image_size).is_integer() or image_size < 16:
        raise ValueError(
            "DCGAN image_size must be a power of two and at least 16"
        )
    return DCGANConfig(
        image_size=image_size,
        latent_dim=int(dcgan_cfg.get("latent_dim", 128)),
        base_channels=int(dcgan_cfg.get("base_channels", 64)),
        batch_size=int(dcgan_cfg.get("batch_size", 128)),
        epochs=int(dcgan_cfg.get("epochs", 50)),
        max_steps=optional_int(dcgan_cfg.get("max_steps")),
        lr=float(dcgan_cfg.get("lr", 2e-4)),
        beta1=float(dcgan_cfg.get("beta1", 0.5)),
        beta2=float(dcgan_cfg.get("beta2", 0.999)),
        num_workers=int(dcgan_cfg.get("num_workers", 8)),
        conditional=conditional,
        use_data_parallel=bool(dcgan_cfg.get("use_data_parallel", True)),
    )


def optional_int(value: object) -> int | None:
    if value in {None, "", 0, "0"}:
        return None
    return int(value)


def condition_count(condition_manifest: Path) -> int:
    rows = read_manifest(condition_manifest)
    if not rows:
        return 0
    return max(int(row["condition_id"]) for row in rows) + 1


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d | nn.ConvTranspose2d):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def maybe_parallel(module: nn.Module, cfg: DCGANConfig) -> nn.Module:
    if cfg.use_data_parallel and torch.cuda.device_count() > 1:
        return nn.DataParallel(module)
    return module


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
