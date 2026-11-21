from __future__ import annotations

import json
import random
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_f
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as vision_f
from tqdm.auto import tqdm

from .config import DataConfig, ImageConfig
from .data import PatchDataset, discover_image_splits, write_split_manifest
from .metrics import compute_metrics, write_evaluation_artifacts
from .timing import TimerRecorder


def _image_param_candidates(config: ImageConfig, seed: int) -> list[dict[str, Any]]:
    base = [
        {"learning_rate": lr, "weight_decay": wd}
        for lr, wd in product(config.search.learning_rate, config.search.weight_decay)
    ]
    if config.search.method == "fixed":
        if len(base) != 1:
            raise ValueError(
                "Image search method 'fixed' requires exactly one learning_rate and one weight_decay."
            )
        return base
    if config.search.method == "random":
        rng = np.random.default_rng(seed)
        count = min(config.search.n_iter, len(base))
        indexes = rng.choice(len(base), size=count, replace=False)
        return [base[int(idx)] for idx in indexes]
    return base


_NORMALIZE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
_AUGMENTATION_ALIASES = {
    "none": "none",
    "baseline": "none",
    "geometric": "geometric",
    "geometry": "geometric",
    "colour": "colour",
    "color": "colour",
    "hed": "hed",
    "histology": "hed",
    "histology_specific": "hed",
    "combined": "combined",
    "all": "combined",
}


class _RandomFiveCrop:
    def __call__(self, img: Image.Image) -> Image.Image:
        side = max(1, min(img.size) // 2)
        return random.choice(vision_f.five_crop(img, side))


class _RandomRightAngleRotation:
    def __call__(self, img: Image.Image) -> Image.Image:
        return vision_f.rotate(img, random.choice((90, 180, 270)))


class _RandomAxisFlip:
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.choice((True, False)):
            return vision_f.hflip(img)
        return vision_f.vflip(img)


class _Morphology:
    def __init__(self, operation: str, kernel_size: int = 4) -> None:
        if operation not in {"dilation", "erosion"}:
            raise ValueError(f"Unknown morphology operation {operation!r}.")
        self.operation = operation
        self.kernel_size = kernel_size

    def __call__(self, img: Image.Image) -> Image.Image:
        tensor = vision_f.to_tensor(img).unsqueeze(0)
        height, width = tensor.shape[-2:]
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size // 2
        padding = (pad_left, pad_right, pad_left, pad_right)
        if self.operation == "dilation":
            padded = torch_f.pad(tensor, padding, mode="constant", value=0.0)
            morphed = torch_f.max_pool2d(padded, kernel_size=self.kernel_size, stride=1)
        else:
            padded = torch_f.pad(tensor, padding, mode="constant", value=1.0)
            morphed = -torch_f.max_pool2d(-padded, kernel_size=self.kernel_size, stride=1)
        morphed = morphed[..., :height, :width].squeeze(0).clamp(0.0, 1.0)
        return vision_f.to_pil_image(morphed)


class _RandomGamma:
    def __init__(self, min_gamma: float = 0.5, max_gamma: float = 1.5) -> None:
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, img: Image.Image) -> Image.Image:
        return vision_f.adjust_gamma(img, gamma=random.uniform(self.min_gamma, self.max_gamma))


class _RandomBrightness:
    def __call__(self, img: Image.Image) -> Image.Image:
        return vision_f.adjust_brightness(img, brightness_factor=random.uniform(0.5, 1.5))


class _RandomContrast:
    def __call__(self, img: Image.Image) -> Image.Image:
        return vision_f.adjust_contrast(img, contrast_factor=random.uniform(0.5, 1.5))


class _RandomSaturation:
    def __call__(self, img: Image.Image) -> Image.Image:
        return vision_f.adjust_saturation(img, saturation_factor=random.uniform(0.5, 1.5))


class _RandomHue:
    def __call__(self, img: Image.Image) -> Image.Image:
        return vision_f.adjust_hue(img, hue_factor=random.uniform(-0.5, 0.5))


class _RandomHED:
    def __init__(self, sigma: float = 0.03, min_param: float = -0.5, max_param: float = 0.5) -> None:
        hed_to_rgb = np.asarray(
            [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
            dtype=np.float32,
        )
        self.hed_to_rgb = torch.from_numpy(hed_to_rgb)
        self.rgb_to_hed = torch.linalg.inv(self.hed_to_rgb)
        self.sigma = sigma
        self.min_param = min_param
        self.max_param = max_param

    def __call__(self, img: Image.Image) -> Image.Image:
        tensor = vision_f.to_tensor(img).clamp(0.0, 1.0)
        channels, height, width = tensor.shape
        flat_rgb = tensor.reshape(channels, -1).permute(1, 0).clamp_min(1.0e-6)
        hed = -torch.log(flat_rgb) @ self.rgb_to_hed
        param = random.uniform(self.min_param, self.max_param)
        channel_factors = torch.tensor([-2.0, 2.0, -3.0], dtype=hed.dtype)
        alpha = 1.0 + param * channel_factors * self.sigma
        beta = param * channel_factors * self.sigma
        shifted = hed * alpha + beta
        rgb = torch.exp(-(shifted @ self.hed_to_rgb))
        rgb = rgb.clamp(0.0, 1.0).permute(1, 0).reshape(channels, height, width)
        return vision_f.to_pil_image(rgb)


class _ControlledAugmentationSampler:
    def __init__(self, transforms_: list[Any], probability: float, min_transforms: int = 1, max_transforms: int = 3) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("augmentation probability must be between 0 and 1.")
        if min_transforms < 1:
            raise ValueError("min_transforms must be at least 1.")
        if max_transforms < min_transforms:
            raise ValueError("max_transforms must be greater than or equal to min_transforms.")
        self.transforms = list(transforms_)
        self.probability = probability
        self.min_transforms = min_transforms
        self.max_transforms = max_transforms

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.transforms or random.random() >= self.probability:
            return img
        max_count = min(self.max_transforms, len(self.transforms))
        min_count = min(self.min_transforms, max_count)
        count = random.randint(min_count, max_count)
        for transform in random.sample(self.transforms, count):
            img = transform(img)
        return img


def _normalise_augmentation_suite(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    try:
        return _AUGMENTATION_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(set(_AUGMENTATION_ALIASES.values())))
        raise ValueError(f"Unknown image augmentation suite {name!r}. Choose one of: {choices}.") from exc


def _augmentation_operations_for_suite(suite: str, hed_sigma: float) -> list[Any]:
    geometric = [
        _RandomFiveCrop(),
        _Morphology("dilation"),
        _Morphology("erosion"),
        transforms.GaussianBlur(kernel_size=15),
        _RandomAxisFlip(),
        _RandomRightAngleRotation(),
    ]
    colour = [
        _RandomBrightness(),
        _RandomContrast(),
        _RandomSaturation(),
        _RandomHue(),
        _RandomGamma(),
    ]
    hed = [_RandomHED(sigma=hed_sigma)]
    if suite == "geometric":
        return geometric
    if suite == "colour":
        return colour
    if suite == "hed":
        return hed
    if suite == "combined":
        return [*geometric, *colour, *hed]
    return []


def _build_augmentation_transforms(suite: str, probability: float, hed_sigma: float) -> list[Any]:
    operations = _augmentation_operations_for_suite(suite, hed_sigma)
    if not operations:
        return []
    return [_ControlledAugmentationSampler(operations, probability)]


def _build_transform(
    image_size: int,
    augmentation_suite: str = "none",
    *,
    train: bool = False,
    probability: float = 0.75,
    hed_sigma: float = 0.03,
):
    suite = _normalise_augmentation_suite(augmentation_suite)
    steps: list[Any] = []
    if train:
        steps.extend(_build_augmentation_transforms(suite, probability, hed_sigma))
    steps.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            _NORMALIZE,
        ]
    )
    return transforms.Compose(steps)


def _image_run_dir(output_dir: Path, model_name: str, seed: int, augmentation_suite: str) -> Path:
    if augmentation_suite == "none":
        return output_dir / "images" / model_name / f"seed_{seed}"
    return output_dir / "images" / model_name / augmentation_suite / f"seed_{seed}"


def _build_model(name: str, n_classes: int, pretrained: bool):
    import timm

    model = timm.create_model(name, pretrained=pretrained, num_classes=n_classes)
    return model


def run_images(
    data_config: DataConfig,
    image_config: ImageConfig,
    output_dir: Path,
    seeds: list[int],
    timer: TimerRecorder,
) -> list[dict]:
    results = []
    if not image_config.enabled:
        return results
    with timer.time("discover_image_splits"):
        splits = discover_image_splits(
            data_config.dataset_root,
            label_index_path=data_config.label_index_path,
            label_column=data_config.label_column,
        )
        write_split_manifest(splits, output_dir / "data")
    class_names = splits.class_names
    augmentation_suites = [
        _normalise_augmentation_suite(suite) for suite in image_config.augmentation_suites
    ]
    for model_name in image_config.models:
        for augmentation_suite in augmentation_suites:
            train_transform = _build_transform(
                data_config.image_size,
                augmentation_suite,
                train=True,
                probability=image_config.augmentation_probability,
                hed_sigma=image_config.hed_sigma,
            )
            eval_transform = _build_transform(data_config.image_size)
            for seed in seeds:
                run_dir = _image_run_dir(output_dir, model_name, seed, augmentation_suite)
                started = perf_counter()
                best_params, best_score, search_dynamics = _select_image_params(
                    model_name,
                    image_config,
                    data_config,
                    splits,
                    train_transform,
                    eval_transform,
                    seed,
                    class_names,
                    augmentation_suite,
                )
                final_dynamics, y_true, y_pred, y_proba, sample_ids = _fit_final_image_model(
                    model_name,
                    image_config,
                    data_config,
                    splits,
                    train_transform,
                    eval_transform,
                    seed,
                    class_names,
                    best_params,
                    augmentation_suite,
                )
                metrics = write_evaluation_artifacts(
                    run_dir,
                    y_true,
                    y_pred,
                    y_proba,
                    sample_ids,
                    class_names,
                )
                elapsed = perf_counter() - started
                timer.add(
                    "image_model",
                    elapsed,
                    model=model_name,
                    seed=seed,
                    augmentation_suite=augmentation_suite,
                )
                dynamics = search_dynamics + final_dynamics
                pd.DataFrame(dynamics).to_csv(run_dir / "training_dynamics.csv", index=False)
                enriched = {
                    **metrics,
                    "task": "image",
                    "model_id": model_name,
                    "augmentation_suite": augmentation_suite,
                    "seed": seed,
                    "best_params": json.dumps(best_params, sort_keys=True),
                    "valid_auc": best_score,
                    "runtime_seconds": elapsed,
                }
                pd.DataFrame([enriched]).to_csv(run_dir / "metrics.csv", index=False)
                results.append(enriched)
    return results


def run_image_task(
    data_config: DataConfig,
    image_config: ImageConfig,
    output_dir: Path,
    model_name: str,
    seed: int,
    timer: TimerRecorder,
    augmentation_suite: str = "none",
) -> dict:
    if not image_config.enabled:
        raise ValueError("Image benchmarks are disabled in config.")
    with timer.time("discover_image_splits"):
        splits = discover_image_splits(
            data_config.dataset_root,
            label_index_path=data_config.label_index_path,
            label_column=data_config.label_column,
        )
        write_split_manifest(splits, output_dir / "data")
    class_names = splits.class_names
    augmentation_suite = _normalise_augmentation_suite(augmentation_suite)
    train_transform = _build_transform(
        data_config.image_size,
        augmentation_suite,
        train=True,
        probability=image_config.augmentation_probability,
        hed_sigma=image_config.hed_sigma,
    )
    eval_transform = _build_transform(data_config.image_size)
    run_dir = _image_run_dir(output_dir, model_name, seed, augmentation_suite)
    started = perf_counter()
    best_params, best_score, search_dynamics = _select_image_params(
        model_name,
        image_config,
        data_config,
        splits,
        train_transform,
        eval_transform,
        seed,
        class_names,
        augmentation_suite,
    )
    final_dynamics, y_true, y_pred, y_proba, sample_ids = _fit_final_image_model(
        model_name,
        image_config,
        data_config,
        splits,
        train_transform,
        eval_transform,
        seed,
        class_names,
        best_params,
        augmentation_suite,
    )
    metrics = write_evaluation_artifacts(run_dir, y_true, y_pred, y_proba, sample_ids, class_names)
    elapsed = perf_counter() - started
    timer.add("image_model", elapsed, model=model_name, seed=seed, augmentation_suite=augmentation_suite)
    pd.DataFrame(search_dynamics + final_dynamics).to_csv(
        run_dir / "training_dynamics.csv", index=False
    )
    enriched = {
        **metrics,
        "task": "image",
        "model_id": model_name,
        "augmentation_suite": augmentation_suite,
        "seed": seed,
        "best_params": json.dumps(best_params, sort_keys=True),
        "valid_auc": best_score,
        "runtime_seconds": elapsed,
    }
    pd.DataFrame([enriched]).to_csv(run_dir / "metrics.csv", index=False)
    return enriched


def _select_image_params(
    model_name: str,
    image_config: ImageConfig,
    data_config: DataConfig,
    splits,
    train_transform,
    eval_transform,
    seed: int,
    class_names: list[str],
    augmentation_suite: str,
) -> tuple[dict[str, Any], float, list[dict]]:
    best_score = -np.inf
    best_params = {}
    dynamics = []
    for candidate_idx, params in enumerate(_image_param_candidates(image_config, seed)):
        candidate_started = perf_counter()
        epoch_rows, valid_auc, best_epoch, _model = _train_image_once(
            model_name,
            image_config,
            data_config,
            splits.train,
            splits.valid,
            train_transform,
            eval_transform,
            seed,
            class_names,
            params,
            stage="search",
            candidate=candidate_idx,
            augmentation_suite=augmentation_suite,
        )
        dynamics.extend(epoch_rows)
        dynamics.append(
            {
                "stage": "search",
                "candidate": candidate_idx,
                "epoch": "summary",
                "model": model_name,
                "augmentation_suite": augmentation_suite,
                "seed": seed,
                "params": json.dumps(params, sort_keys=True),
                "valid_auc": valid_auc,
                "runtime_seconds": perf_counter() - candidate_started,
            }
        )
        if valid_auc > best_score:
            best_score = valid_auc
            best_params = {**params, "epochs": best_epoch}
    return best_params, best_score, dynamics


def _fit_final_image_model(
    model_name: str,
    image_config: ImageConfig,
    data_config: DataConfig,
    splits,
    train_transform,
    eval_transform,
    seed: int,
    class_names: list[str],
    params: dict[str, Any],
    augmentation_suite: str,
):
    train_valid = [*splits.train, *splits.valid]
    dynamics, _valid_auc, _best_epoch, model = _train_image_once(
        model_name,
        image_config,
        data_config,
        train_valid,
        None,
        train_transform,
        eval_transform,
        seed,
        class_names,
        params,
        stage="final_train",
        candidate="best",
        augmentation_suite=augmentation_suite,
    )
    device = _device(image_config)
    loader = _loader(splits.test, eval_transform, image_config.batch_size, False, data_config.num_workers)
    y_true, y_pred, y_proba, sample_ids = _evaluate_model(model, loader, device, len(class_names))
    return dynamics, y_true, y_pred, y_proba, sample_ids


def _device(image_config: ImageConfig) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if image_config.require_cuda:
        raise RuntimeError(
            "Image benchmark requires CUDA, but torch.cuda.is_available() is false. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r}."
        )
    return torch.device("cpu")


def _train_image_once(
    model_name: str,
    image_config: ImageConfig,
    data_config: DataConfig,
    train_records,
    valid_records,
    train_transform,
    eval_transform,
    seed: int,
    class_names: list[str],
    params: dict[str, Any],
    stage: str,
    candidate,
    augmentation_suite: str,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(image_config)
    model = _build_model(model_name, len(class_names), image_config.pretrained).to(device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    max_epochs = int(params.get("epochs", image_config.epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()
    train_loader = _loader(train_records, train_transform, image_config.batch_size, True, data_config.num_workers)
    valid_loader = (
        _loader(valid_records, eval_transform, image_config.batch_size, False, data_config.num_workers)
        if valid_records is not None
        else None
    )
    best_auc = -np.inf
    best_epoch = 1
    best_state = None
    stale_epochs = 0
    dynamics = []
    for epoch in range(1, max_epochs + 1):
        epoch_started = perf_counter()
        model.train()
        losses = []
        for images, labels, _sample_ids in tqdm(train_loader, desc=f"{model_name} {stage} epoch {epoch}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        row = {
            "stage": stage,
            "candidate": candidate,
            "epoch": epoch,
            "model": model_name,
            "augmentation_suite": augmentation_suite,
            "seed": seed,
            "params": json.dumps(params, sort_keys=True),
            "train_loss": float(np.mean(losses)) if losses else np.nan,
            "learning_rate": scheduler.get_last_lr()[0],
            "runtime_seconds": perf_counter() - epoch_started,
        }
        if valid_loader is not None:
            y_true, y_pred, y_proba, _ = _evaluate_model(model, valid_loader, device, len(class_names))
            metrics = compute_metrics(y_true, y_pred, y_proba)
            valid_auc = metrics["auc"]
            row["valid_accuracy"] = metrics["accuracy"]
            row["valid_auc"] = valid_auc
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
        dynamics.append(row)
        if valid_loader is not None and stale_epochs >= image_config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return dynamics, best_auc, best_epoch, model


def _loader(records, transform, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        PatchDataset(records, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _evaluate_model(model, loader, device, n_classes: int):
    model.eval()
    y_true = []
    y_pred = []
    y_proba = []
    sample_ids = []
    with torch.no_grad():
        for images, labels, batch_sample_ids in loader:
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images.to(device))
            proba = torch.softmax(logits, dim=1).detach().cpu().numpy()
            y_proba.append(proba)
            y_pred.append(np.argmax(proba, axis=1))
            y_true.append(labels.numpy())
            sample_ids.extend([str(value) for value in batch_sample_ids])
    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.vstack(y_proba).reshape(-1, n_classes),
        sample_ids,
    )
