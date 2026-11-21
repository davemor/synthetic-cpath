from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), index


def load_uni2_h() -> torch.nn.Module:
    kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    return timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kwargs)


def embed_images(
    image_paths: list[str],
    features_path: Path,
    batch_size: int,
    num_workers: int,
    device: str,
) -> np.ndarray:
    if features_path.exists():
        print(f"loading cached features: {features_path}", flush=True)
        return np.load(features_path)["features"]

    model = load_uni2_h().eval().to(device)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    loader = DataLoader(
        ImagePathDataset(image_paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    features = np.empty((len(image_paths), 1536), dtype=np.float32)

    seen = 0
    with torch.inference_mode():
        for batch, indices in loader:
            batch = batch.to(device, non_blocking=True)
            output = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
            features[indices.numpy()] = output
            seen += len(indices)
            if seen == len(image_paths) or seen % (batch_size * 50) == 0:
                print(f"embedded {seen}/{len(image_paths)} images", flush=True)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(features_path, features=features)
    print(f"wrote cached features: {features_path}", flush=True)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="pathmnist224")
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--n-clusters", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    train = manifest.loc[manifest["split"] == "train"].copy().reset_index(drop=True)
    if train.empty:
        raise RuntimeError(f"No train rows found in {manifest_path}")

    missing = [p for p in train["image_path"].tolist() if not Path(p).exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} image files. First missing: {missing[0]}")

    image_paths = train["image_path"].tolist()
    features_path = out_dir / "uni2_train_features.npz"
    features = embed_images(
        image_paths=image_paths,
        features_path=features_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    print(
        f"fitting PCA(n_components={args.pca_dim}, svd_solver='full', random_state=0)",
        flush=True,
    )
    pca = PCA(n_components=args.pca_dim, svd_solver="full", random_state=0)
    reduced = pca.fit_transform(features).astype(np.float32, copy=False)
    reduced_path = out_dir / f"uni2_pca{args.pca_dim}_train_features.npz"
    np.savez_compressed(reduced_path, features=reduced)

    print(
        "fitting KMeans("
        f"n_clusters={args.n_clusters}, init='k-means++', n_init=10, max_iter=300, random_state=0"
        ")",
        flush=True,
    )
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=0,
    )
    cluster_ids = kmeans.fit_predict(reduced).astype(int)

    labels = pd.DataFrame(
        {"image_path": train["image_path"].to_numpy(), "cluster_id": cluster_ids}
    )
    labels.to_csv(out_dir / "pseudo_labels.csv", index=False)
    labels.to_csv(out_dir / "label_local_cluster_labels.csv", index=False)

    counts = (
        pd.Series(cluster_ids)
        .value_counts()
        .sort_index()
        .rename_axis("cluster_id")
        .reset_index(name="count")
    )
    counts.to_csv(out_dir / "cluster_counts.csv", index=False)

    metadata = {
        "assignment_generated_by": "Codex standalone UNI2 PCA+KMeans assignment script",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "manifest_path": str(manifest_path),
        "label_artifact_path": str(out_dir / "pseudo_labels.csv"),
        "cluster_config_path": str(out_dir / "selected_global_k.yaml"),
        "split_used_for_fit": "train",
        "n_train_images": int(len(train)),
        "encoder": "uni2",
        "encoder_source": "hf-hub:MahmoodLab/UNI2-h",
        "projector": "pca",
        "pca_dim": args.pca_dim,
        "pca_svd_solver": "full",
        "pca_random_state": 0,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "clusterer": "kmeans",
        "n_clusters": args.n_clusters,
        "kmeans_init": "k-means++",
        "kmeans_n_init": 10,
        "kmeans_max_iter": 300,
        "kmeans_random_state": 0,
        "conditioning_scope": "global",
        "intraclass_clustering_used": False,
        "true_class_labels_used_as_clustering_criterion": False,
        "selection_rule": (
            "support/noise/validity filters; stability hard filter; Pareto front over "
            "calibrated proxies for ROC-AUC, FID, KID, precision, recall, stability, "
            "and support; closest-to-utopia balanced compromise selection"
        ),
        "cluster_counts": {str(row.cluster_id): int(row.count) for row in counts.itertuples()},
    }
    (out_dir / "balanced_global_cluster_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    (out_dir / "selected_global_k.yaml").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
