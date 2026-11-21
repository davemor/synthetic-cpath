#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_SPLIT_DIR = Path(
    "/sharedscratch/dm236/data/icaird_cervical/resnet_patch_subset/"
    "stratified_120_consensus_100k/split_dataset"
)
CLASS_TO_INDEX = {"background": 0, "foreground": 1}
INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class TrainConfig:
    train_manifest: str
    test_manifest: str
    output_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    image_size: int
    num_workers: int
    seed: int
    device: str


def require_training_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError as exc:
        raise RuntimeError(
            "Training requires torch and scikit-learn. Install them in the "
            "Hypatia environment, for example with the histokit train/embed "
            "extras or explicit torch/scikit-learn packages."
        ) from exc
    return torch, nn, functional, (classification_report, confusion_matrix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a ResNet-19 binary classifier for foreground "
            "vs background patch labels."
        )
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Directory written by create_resnet_slide_split_dataset.py.",
    )
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <split-dir>/resnet19_run.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=236)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Use cuda if available by default.",
    )
    return parser.parse_args()


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str, torch: Any) -> Any:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class PatchManifestDataset:
    def __init__(
        self,
        manifest_path: Path,
        *,
        image_size: int,
        train: bool,
    ) -> None:
        self.manifest = pd.read_parquet(manifest_path)
        self.image_size = image_size
        self.train = train
        path_column = (
            "split_patch_path"
            if "split_patch_path" in self.manifest.columns
            else "sample_patch_path"
        )
        required = {path_column, "majority_label"}
        missing = required - set(self.manifest.columns)
        if missing:
            raise ValueError(f"{manifest_path} missing columns: {sorted(missing)}")
        self.path_column = path_column

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        import torch

        row = self.manifest.iloc[idx]
        image = Image.open(row[self.path_column]).convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        array = np.asarray(image, dtype=np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:
                array = np.flip(array, axis=1)
            if random.random() < 0.5:
                array = np.flip(array, axis=0)
            rotations = random.randint(0, 3)
            if rotations:
                array = np.rot90(array, rotations, axes=(0, 1))

        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        array = np.ascontiguousarray(array.transpose(2, 0, 1))
        label = CLASS_TO_INDEX[str(row["majority_label"])]
        return torch.from_numpy(array), torch.tensor(label, dtype=torch.long)


def conv3x3(nn: Any, in_channels: int, out_channels: int, stride: int = 1) -> Any:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(nn: Any, in_channels: int, out_channels: int, stride: int = 1) -> Any:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def make_basic_block(nn: Any) -> type:
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: Any | None = None,
        ) -> None:
            super().__init__()
            self.conv1 = conv3x3(nn, in_channels, out_channels, stride)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(nn, out_channels, out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = downsample

        def forward(self, x: Any) -> Any:
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return self.relu(out)

    return BasicBlock


def make_resnet19(nn: Any) -> Any:
    BasicBlock = make_basic_block(nn)

    class ResNet19(nn.Module):
        """ResNet-18 style backbone plus one extra post-stage conv layer.

        Counting weighted layers: initial conv (1) + 8 basic blocks x 2 convs
        (16) + extra conv (1) + classifier (1) = 19.
        """

        def __init__(self) -> None:
            super().__init__()
            self.in_channels = 64
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(BasicBlock, 64, 2)
            self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
            self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
            self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
            self.extra_conv = nn.Sequential(
                conv3x3(nn, 512, 512),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 2)

        def _make_layer(
            self,
            block: type,
            out_channels: int,
            blocks: int,
            stride: int = 1,
        ) -> Any:
            downsample = None
            if stride != 1 or self.in_channels != out_channels:
                downsample = nn.Sequential(
                    conv1x1(nn, self.in_channels, out_channels, stride),
                    nn.BatchNorm2d(out_channels),
                )
            layers = [block(self.in_channels, out_channels, stride, downsample)]
            self.in_channels = out_channels
            for _ in range(1, blocks):
                layers.append(block(self.in_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.extra_conv(x)
            x = self.avgpool(x)
            x = x.flatten(1)
            return self.fc(x)

    return ResNet19()


def train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
    torch: Any,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(labels)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(len(labels))

    return {"loss": total_loss / total, "accuracy": correct / total}


def predict(model: Any, loader: Any, device: Any, torch: Any) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            y_true.extend(int(label) for label in labels.tolist())
            y_pred.extend(int(pred) for pred in logits.argmax(dim=1).cpu().tolist())
    return y_true, y_pred


def main() -> None:
    args = parse_args()
    torch, nn, _, metrics = require_training_dependencies()
    classification_report, confusion_matrix = metrics
    seed_everything(args.seed, torch)
    device = resolve_device(args.device, torch)

    train_manifest = args.train_manifest or args.split_dir / "sample_manifest_train.parquet"
    test_manifest = args.test_manifest or args.split_dir / "sample_manifest_test.parquet"
    output_dir = args.output_dir or args.split_dir / "resnet19_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        train_manifest=str(train_manifest),
        test_manifest=str(test_manifest),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=str(device),
    )
    (output_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    train_dataset = PatchManifestDataset(
        train_manifest, image_size=args.image_size, train=True
    )
    test_dataset = PatchManifestDataset(
        test_manifest, image_size=args.image_size, train=False
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = make_resnet19(nn).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        metrics_row = train_one_epoch(
            model, train_loader, optimizer, criterion, device, torch
        )
        metrics_row["epoch"] = epoch
        history.append(metrics_row)
        print(
            f"epoch={epoch} loss={metrics_row['loss']:.4f} "
            f"accuracy={metrics_row['accuracy']:.4f}"
        )
        torch.save(model.state_dict(), output_dir / "checkpoint_latest.pt")

    y_true, y_pred = predict(model, test_loader, device, torch)
    target_names = [INDEX_TO_CLASS[index] for index in sorted(INDEX_TO_CLASS)]
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    report_text = classification_report(y_true, y_pred, target_names=target_names)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "classification_report.txt").write_text(report_text)
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    pd.DataFrame(matrix, index=target_names, columns=target_names).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    torch.save(model.state_dict(), output_dir / "resnet19_final.pt")
    print(report_text)
    print("Confusion matrix:")
    print(matrix)


if __name__ == "__main__":
    main()
