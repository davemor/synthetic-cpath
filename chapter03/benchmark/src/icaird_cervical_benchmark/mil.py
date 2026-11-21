from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import MilConfig
from .metrics import compute_metrics, write_evaluation_artifacts
from .timing import TimerRecorder


SPLITS = ("train", "valid", "test")
TASK_CLASS_NAMES = {
    "malignant": ["benign", "malignant"],
    "category": ["normal_inflammation", "low_grade", "high_grade", "malignant"],
    "subcategory": [
        "normal_inflammation",
        "hpv",
        "cin1",
        "cin2",
        "cin3",
        "cgin",
        "adenocarcinoma",
        "squamous_carcinoma",
        "other",
    ],
}


@dataclass(frozen=True)
class BagRecord:
    slide_id: str
    split: str
    label: int
    class_name: str
    feature_path: Path
    patch_count: int


class BagDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[BagRecord], expected_dim: int):
        self.records = records
        self.expected_dim = expected_dim

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with np.load(record.feature_path, allow_pickle=False) as data:
            features = np.asarray(data["features"], dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features for {record.slide_id}, got {features.shape}.")
        if features.shape[1] != self.expected_dim:
            raise ValueError(
                f"Expected {self.expected_dim} features for {record.slide_id}, got {features.shape[1]}."
            )
        return torch.from_numpy(features), torch.tensor(record.label), record.slide_id


class TransformerMilClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 512,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.15,
        activation: str = "gelu",
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("MIL classifier requires at least two classes.")
        self.num_classes = num_classes
        self.projection = nn.Sequential(nn.Linear(input_dim, d_model), nn.ReLU())
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_encoder_layers)
        output_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(d_model, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        encoded = self.encoder(self.projection(features))
        pooled = encoded.mean(dim=1)
        logits = self.classifier(pooled)
        if self.num_classes == 2:
            return logits.squeeze(-1)
        return logits


def run_mil(config: MilConfig, output_dir: Path, seeds: list[int], timer: TimerRecorder) -> list[dict]:
    results = []
    if not config.enabled:
        return results
    _validate_mil_config(config)
    for task in config.tasks:
        for seed in seeds:
            results.append(run_mil_task(config, output_dir, task, seed, timer))
    return results


def run_mil_task(
    config: MilConfig,
    output_dir: Path,
    task: str,
    seed: int,
    timer: TimerRecorder,
) -> dict:
    _validate_mil_config(config)
    if task not in TASK_CLASS_NAMES:
        raise ValueError(f"Unknown MIL task {task!r}. Valid tasks: {sorted(TASK_CLASS_NAMES)}")
    run_dir = output_dir / "mil" / config.embedding_name / task / f"seed_{seed}"
    started = perf_counter()
    with timer.time("load_mil_bags", task=task):
        records = load_bag_records(config, task)
    write_bag_manifest(records, run_dir / "bag_manifest.csv")
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = TASK_CLASS_NAMES[task]
    num_classes = len(class_names)
    model = TransformerMilClassifier(
        input_dim=config.embedding_dim,
        num_classes=num_classes,
        d_model=config.d_model,
        nhead=config.heads,
        num_encoder_layers=config.encoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        activation=config.activation,
    ).to(device)
    criterion = make_loss(_records_for_split(records, "train"), num_classes, config.balance_classes, device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = GradScaler(enabled=config.use_amp and device.type == "cuda")
    train_dataset = BagDataset(_records_for_split(records, "train"), config.embedding_dim)
    valid_dataset = BagDataset(_records_for_split(records, "valid"), config.embedding_dim)
    best_state = None
    best_auc = -np.inf
    dynamics = []

    for epoch in range(config.epochs):
        train_metrics = _run_epoch(
            model,
            train_dataset,
            criterion,
            device,
            phase="train",
            optimiser=optimiser,
            scaler=scaler,
            use_amp=config.use_amp,
        )
        valid_metrics, _, _, _ = _predict_dataset(
            model,
            valid_dataset,
            criterion,
            device,
            use_amp=config.use_amp,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_auc": train_metrics["auc"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_auc": valid_metrics["auc"],
        }
        dynamics.append(row)
        if valid_metrics["auc"] > best_auc:
            best_auc = valid_metrics["auc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"No checkpoint was selected for MIL task {task}.")
    model.load_state_dict(best_state)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "task": task,
            "class_names": class_names,
            "config": _mil_config_dict(config),
            "best_val_auc": best_auc,
        },
        run_dir / "best_model_val_auroc.pt",
    )
    pd.DataFrame(dynamics).to_csv(run_dir / "training_dynamics.csv", index=False)

    valid_summary = _evaluate_split(model, records, "valid", criterion, config, device, run_dir, class_names)
    test_summary = _evaluate_split(model, records, "test", criterion, config, device, run_dir, class_names)
    elapsed = perf_counter() - started
    timer.add("mil", elapsed, task=task, seed=seed, embedding=config.embedding_name)
    enriched = {
        **test_summary,
        "task": "mil",
        "mil_task": task,
        "embedding": config.embedding_name,
        "model_id": f"{config.embedding_name}:transformer-mil:{task}",
        "seed": seed,
        "feature_dim": config.embedding_dim,
        "best_val_auc": best_auc,
        "valid_accuracy": valid_summary["accuracy"],
        "valid_auc": valid_summary["auc"],
        "runtime_seconds": elapsed,
    }
    pd.DataFrame([enriched]).to_csv(run_dir / "metrics.csv", index=False)
    return enriched


def load_bag_records(config: MilConfig, task: str) -> list[BagRecord]:
    _validate_mil_config(config)
    assert config.bag_root is not None
    index_path = config.bag_root / "dataset_index_ometiff.csv"
    if not index_path.exists():
        raise ValueError(f"Expected iCAIRD bag index at {index_path}.")
    df = pd.read_csv(index_path)
    required = {"id", "split", "category", "subcategory"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{index_path} is missing columns {missing}.")
    class_names = TASK_CLASS_NAMES[task]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    records = []
    for row in df.itertuples(index=False):
        row_data = row._asdict()
        split = str(row_data["split"])
        if split not in SPLITS:
            continue
        slide_id = str(row_data["id"])
        class_name = _class_name_for_task(row_data, task)
        if class_name not in class_to_idx:
            raise ValueError(f"Unexpected class {class_name!r} for MIL task {task}.")
        feature_path = (
            config.bag_root
            / "embeddings"
            / config.embedding_name
            / "slides"
            / f"{slide_id}.{config.embedding_name}.embedding"
            / "features.npz"
        )
        if not feature_path.exists():
            raise ValueError(f"Missing UNI2-h features for slide {slide_id}: {feature_path}")
        patch_count = _patch_count(feature_path, config.embedding_dim)
        records.append(
            BagRecord(
                slide_id=slide_id,
                split=split,
                label=class_to_idx[class_name],
                class_name=class_name,
                feature_path=feature_path,
                patch_count=patch_count,
            )
        )
    _validate_records(records, class_names)
    return records


def make_loss(
    train_records: list[BagRecord],
    num_classes: int,
    balance_classes: bool,
    device: torch.device,
) -> nn.Module:
    labels = np.asarray([record.label for record in train_records], dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"Cannot build loss with empty classes: counts={counts.tolist()}")
    if num_classes == 2:
        pos_weight = None
        if balance_classes:
            pos_weight = torch.tensor([counts[0] / counts[1]], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    weight = None
    if balance_classes:
        total = counts.sum()
        weight = torch.tensor(total / (counts * num_classes), dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight)


def write_bag_manifest(records: list[BagRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "slide_id": record.slide_id,
                "split": record.split,
                "class_name": record.class_name,
                "label": record.label,
                "feature_path": str(record.feature_path),
                "patch_count": record.patch_count,
            }
            for record in records
        ]
    ).to_csv(path, index=False)


def _run_epoch(
    model: nn.Module,
    dataset: BagDataset,
    criterion: nn.Module,
    device: torch.device,
    phase: Literal["train"],
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    losses = []
    y_true = []
    y_proba = []
    for features, label, _ in tqdm(dataset, desc=phase, unit="WSI", leave=False):
        features = features.to(device)
        label = label.to(device)
        optimiser.zero_grad()
        with autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(features)
            loss = _loss_from_logits(criterion, logits, label, model.num_classes)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        y_true.append(int(label.cpu()))
        y_proba.append(_probabilities(logits.detach(), model.num_classes).cpu().numpy()[0])
    proba = np.asarray(y_proba)
    pred = np.argmax(proba, axis=1)
    metrics = compute_metrics(np.asarray(y_true), pred, proba)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def _predict_dataset(
    model: nn.Module,
    dataset: BagDataset,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, list[str]]:
    model.eval()
    losses = []
    y_true = []
    y_proba = []
    sample_ids = []
    with torch.no_grad():
        for features, label, slide_id in tqdm(dataset, desc="evaluate", unit="WSI", leave=False):
            features = features.to(device)
            label = label.to(device)
            with autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(features)
                loss = _loss_from_logits(criterion, logits, label, model.num_classes)
            losses.append(float(loss.detach().cpu()))
            y_true.append(int(label.cpu()))
            y_proba.append(_probabilities(logits.detach(), model.num_classes).cpu().numpy()[0])
            sample_ids.append(str(slide_id))
    proba = np.asarray(y_proba)
    truth = np.asarray(y_true)
    pred = np.argmax(proba, axis=1)
    metrics = compute_metrics(truth, pred, proba)
    metrics["loss"] = float(np.mean(losses))
    return metrics, truth, proba, sample_ids


def _evaluate_split(
    model: nn.Module,
    records: list[BagRecord],
    split: str,
    criterion: nn.Module,
    config: MilConfig,
    device: torch.device,
    run_dir: Path,
    class_names: list[str],
) -> dict[str, float]:
    dataset = BagDataset(_records_for_split(records, split), config.embedding_dim)
    metrics, y_true, proba, sample_ids = _predict_dataset(
        model,
        dataset,
        criterion,
        device,
        use_amp=config.use_amp,
    )
    pred = np.argmax(proba, axis=1)
    artifact_metrics = write_evaluation_artifacts(
        run_dir / split,
        y_true,
        pred,
        proba,
        sample_ids,
        class_names,
    )
    split_metrics_path = run_dir / split / "metrics.csv"
    if split_metrics_path.exists():
        split_metrics_path.replace(run_dir / split / f"{split}_metrics.csv")
    artifact_metrics["loss"] = metrics["loss"]
    return artifact_metrics


def _loss_from_logits(
    criterion: nn.Module,
    logits: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if num_classes == 2:
        return criterion(logits.reshape(1), label.float().reshape(1))
    return criterion(logits.reshape(1, num_classes), label.long().reshape(1))


def _probabilities(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes == 2:
        positive = torch.sigmoid(logits.reshape(-1))
        return torch.stack([1.0 - positive, positive], dim=1)
    return torch.softmax(logits.reshape(1, num_classes), dim=1)


def _class_name_for_task(row_data: dict, task: str) -> str:
    if task == "malignant":
        return "malignant" if str(row_data["category"]) == "malignant" else "benign"
    if task == "category":
        return str(row_data["category"])
    if task == "subcategory":
        return str(row_data["subcategory"])
    raise ValueError(f"Unknown MIL task {task!r}.")


def _patch_count(feature_path: Path, expected_dim: int) -> int:
    manifest_path = feature_path.with_name("manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        dim = int(manifest.get("embedding_dim", expected_dim))
        if dim != expected_dim:
            raise ValueError(
                f"{manifest_path} reports embedding_dim={dim}, expected {expected_dim}."
            )
        return int(manifest["num_embeddings"])
    with np.load(feature_path, allow_pickle=False) as data:
        if "features" not in data.files:
            raise ValueError(f"{feature_path} does not contain a features array.")
        features = data["features"]
        if features.ndim != 2 or features.shape[1] != expected_dim:
            raise ValueError(
                f"{feature_path} features have shape {features.shape}, expected (*, {expected_dim})."
            )
        return int(features.shape[0])


def _records_for_split(records: list[BagRecord], split: str) -> list[BagRecord]:
    return [record for record in records if record.split == split]


def _validate_records(records: list[BagRecord], class_names: list[str]) -> None:
    if not records:
        raise ValueError("No MIL bag records were loaded.")
    for split in SPLITS:
        split_records = _records_for_split(records, split)
        if not split_records:
            raise ValueError(f"No MIL bag records found for split {split!r}.")
        present = {record.label for record in split_records}
        expected = set(range(len(class_names)))
        if present != expected:
            missing = [class_names[idx] for idx in sorted(expected - present)]
            raise ValueError(f"MIL split {split!r} is missing classes {missing}.")


def _validate_mil_config(config: MilConfig) -> None:
    if config.bag_root is None:
        raise ValueError("config.mil.bag_root is required when MIL is enabled.")
    for task in config.tasks:
        if task not in TASK_CLASS_NAMES:
            raise ValueError(f"Unknown MIL task {task!r}. Valid tasks: {sorted(TASK_CLASS_NAMES)}")
    if config.batch_size != 1:
        raise ValueError("MIL currently supports batch_size=1 because each slide bag has variable length.")


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mil_config_dict(config: MilConfig) -> dict:
    data = dict(config.__dict__)
    data["bag_root"] = str(data["bag_root"]) if data["bag_root"] is not None else None
    return json.loads(json.dumps(data))
