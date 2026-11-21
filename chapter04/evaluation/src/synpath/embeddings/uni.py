from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from synpath.manifests import read_manifest
from synpath.paths import SynpathPaths


class ManifestImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), row


def collate_batch(batch):
    images, rows = zip(*batch, strict=True)
    return torch.stack(list(images)), list(rows)


def load_uni_model(device: torch.device):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("Install timm to extract UNI embeddings.") from exc

    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    )
    model.eval().to(device)
    return model


def extract_uni_embeddings(
    dataset: str,
    paths: SynpathPaths,
    batch_size: int = 128,
    num_workers: int = 8,
) -> list[Path]:
    manifest_path = paths.data_root / "manifests" / f"{dataset}.csv"
    rows = read_manifest(manifest_path)
    device = resolve_device()
    model = load_uni_model(device)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    outputs = []
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        output = (
            paths.cache_root
            / "embeddings"
            / dataset
            / f"{split}_uni_224.parquet"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        frame = embed_rows(
            split_rows,
            model,
            transform,
            device,
            batch_size,
            num_workers,
        )
        frame.to_parquet(output, index=False)
        outputs.append(output)
    return outputs


def resolve_device() -> torch.device:
    requested = os.environ.get("SYNPATH_EMBED_DEVICE", "auto")
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embed_rows(
    rows: list[dict[str, str]],
    model,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    loader = DataLoader(
        ManifestImageDataset(rows, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    all_features = []
    all_rows = []
    with torch.inference_mode():
        for images, batch_rows in tqdm(loader, desc="UNI embeddings"):
            features = model(images.to(device))
            features = features.detach().float().cpu().numpy()
            all_features.append(features)
            all_rows.extend(batch_rows)

    features_np = np.concatenate(all_features, axis=0)
    meta = pd.DataFrame(all_rows)
    feature_cols = {
        f"f{i}": features_np[:, i] for i in range(features_np.shape[1])
    }
    return pd.concat([meta.reset_index(drop=True), pd.DataFrame(feature_cols)], axis=1)


def load_embedding_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    cols = [col for col in frame.columns if col.startswith("f")]
    return frame[cols].to_numpy(dtype=np.float32)
