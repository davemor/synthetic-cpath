from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from synpath.artifacts import write_json
from synpath.manifests import read_manifest
from synpath.paths import SynpathPaths


class ClassificationDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), int(row["label_id"])


def train_utility_classifier(
    dataset: str,
    source: str,
    seed: int,
    paths: SynpathPaths,
    cfg: dict,
    synthetic_model: str | None = None,
) -> list[Path]:
    torch.manual_seed(seed)
    output_root = paths.outputs_root / "utility" / dataset / "resnet18"
    if synthetic_model is not None:
        output_root = output_root / synthetic_model
    output_root = output_root / source / f"seed{seed}"
    output_root.mkdir(parents=True, exist_ok=True)

    train_rows, test_rows = load_utility_rows(
        dataset,
        source,
        paths,
        cfg,
        synthetic_model=synthetic_model,
    )
    num_classes = len({int(row["label_id"]) for row in train_rows + test_rows})
    model_cfg = cfg.get("utility", {})
    epochs = int(model_cfg.get("epochs", 20))
    batch_size = int(model_cfg.get("batch_size", 128))
    num_workers = int(
        os.environ.get("SYNPATH_UTILITY_NUM_WORKERS")
        or model_cfg.get("num_workers", 8)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    train_tf, test_tf = transforms_for_resnet()
    train_loader = DataLoader(
        ClassificationDataset(train_rows, train_tf),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        ClassificationDataset(test_rows, test_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.get("lr", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
    )
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"utility {source}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

    y_true, y_pred, y_score = evaluate(model, test_loader, device)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
    }
    try:
        metrics["macro_roc_auc_ovr"] = utility_roc_auc(y_true, y_score)
    except ValueError:
        metrics["macro_roc_auc_ovr"] = None

    metrics_path = output_root / "metrics.json"
    report_path = output_root / "classification_report.json"
    cm_path = output_root / "confusion_matrix.csv"
    write_json(metrics_path, metrics)
    write_json(
        report_path,
        classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    )
    np.savetxt(cm_path, confusion_matrix(y_true, y_pred), delimiter=",", fmt="%d")
    return [metrics_path, report_path, cm_path]


def utility_roc_auc(y_true: list[int], y_score: np.ndarray) -> float:
    if y_score.ndim != 2:
        raise ValueError("Expected a two-dimensional class-probability array")
    if y_score.shape[1] == 2:
        return float(roc_auc_score(y_true, y_score[:, 1]))
    return float(
        roc_auc_score(
            y_true,
            y_score,
            multi_class="ovr",
            average="macro",
        )
    )


def load_utility_rows(
    dataset: str,
    source: str,
    paths: SynpathPaths,
    cfg: dict,
    synthetic_model: str | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    manifest = read_manifest(paths.data_root / "manifests" / f"{dataset}.csv")
    real_train = [row for row in manifest if row["split"] == "train"]
    real_test = [row for row in manifest if row["split"] == "test"]
    if source == "real_only":
        return real_train, real_test
    if source == "real_only_10k":
        utility_cfg = cfg.get("utility", {})
        subset = stratified_limit_rows(
            real_train,
            limit=int(utility_cfg.get("real_train_limit", 10_000)),
            seed=int(utility_cfg.get("real_subset_seed", 0)),
        )
        return subset, real_test

    conditioning = source_conditioning(source)
    if synthetic_model is None:
        synthetic_model = default_synthetic_model(cfg)
    synthetic = read_manifest(
        find_synthetic_manifest(paths.outputs_root, dataset, synthetic_model, conditioning)
    )
    if source.startswith("synthetic_only"):
        return synthetic, real_test
    if source.startswith("real_plus"):
        return real_train + synthetic, real_test
    raise ValueError(f"Unknown utility source: {source}")


def stratified_limit_rows(
    rows: list[dict[str, str]],
    limit: int,
    seed: int,
) -> list[dict[str, str]]:
    if limit <= 0:
        raise ValueError("real_train_limit must be positive")
    if len(rows) < limit:
        raise ValueError(
            f"Cannot select {limit} real training rows from a population of {len(rows)}"
        )
    if len(rows) == limit:
        return rows

    labels = [int(row["label_id"]) for row in rows]
    selected, _ = train_test_split(
        rows,
        train_size=limit,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return list(selected)


def default_synthetic_model(cfg: dict) -> str:
    models = cfg.get("utility", {}).get("synthetic_generators")
    if models is None:
        models = cfg.get("generators", [])
    if not models:
        raise ValueError(
            "Synthetic utility source requested, but no generator model was provided. "
            "Pass --synthetic-model or set utility.synthetic_generators."
        )
    return str(models[0])


def source_conditioning(source: str) -> str:
    if "label_global_cluster" in source:
        return "label_global_cluster"
    if "label_local_cluster" in source:
        return "label_local_cluster"
    if "true_label" in source:
        return "true_label"
    raise ValueError(f"Cannot infer conditioning from {source}")


def find_synthetic_manifest(
    outputs_root: Path,
    dataset: str,
    synthetic_model: str,
    conditioning: str,
) -> Path:
    matches = list(
        (outputs_root / "generators" / dataset / synthetic_model / conditioning).glob(
            "seed*/synthetic_manifest.csv"
        )
    )
    if not matches:
        raise RuntimeError(
            "No synthetic manifest found for "
            f"{dataset} model={synthetic_model} conditioning={conditioning}"
        )
    return sorted(matches)[0]


def transforms_for_resnet():
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return train_tf, test_tf


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_score.append(probs)
            y_pred.extend(probs.argmax(axis=1).tolist())
            y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred), np.concatenate(y_score, axis=0)
