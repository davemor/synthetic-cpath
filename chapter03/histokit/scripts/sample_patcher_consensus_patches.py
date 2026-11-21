#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.vq import kmeans2
from tqdm import tqdm


DEFAULT_CONSENSUS = Path(
    "/sharedscratch/dm236/data/icaird_cervical/patcher_annotations/"
    "stratified_120/consensus/patch_consensus.parquet"
)
DEFAULT_METADATA = Path(
    "/sharedscratch/dm236/data/icaird_cervical/patcher_annotations/"
    "stratified_120/consensus/annotation_metadata.parquet"
)
DEFAULT_PATCHES_ROOT = Path(
    "/sharedscratch/dm236/data/icaird_cervical/patcher_cache/stratified_120/"
    "pipeline_research/patches"
)
DEFAULT_OUTPUT_DIR = Path(
    "/sharedscratch/dm236/data/icaird_cervical/resnet_patch_subset/"
    "stratified_120_consensus_100k"
)
PATCH_ID_COLUMNS = ["slide_id", "row", "column"]
LABEL_COLUMN = "majority_label"
CLASS_LABELS = ["background", "foreground"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a slide-stratified, cluster-diverse subset of patcher "
            "consensus patches for ResNet training."
        )
    )
    parser.add_argument("--consensus", type=Path, default=DEFAULT_CONSENSUS)
    parser.add_argument("--patches-root", type=Path, default=DEFAULT_PATCHES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Optional annotation metadata parquet to preserve alongside outputs.",
    )
    parser.add_argument("--target-per-class", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=236)
    parser.add_argument(
        "--copy-images",
        dest="copy_images",
        action="store_true",
        default=True,
        help="Copy selected PNGs into output images/{class}/ directories.",
    )
    parser.add_argument(
        "--no-copy-images",
        dest="copy_images",
        action="store_false",
        help="Only write manifests; do not copy PNGs.",
    )
    parser.add_argument(
        "--max-candidates-per-slide-class",
        type=int,
        default=None,
        help=(
            "Maximum patches to feature/cluster per slide/class. Defaults to "
            "max(quota * 10, 5000). Use 0 to cluster all available patches."
        ),
    )
    parser.add_argument(
        "--clusters-per-slide-class",
        type=int,
        default=None,
        help="Fixed number of clusters per slide/class. Defaults from quota.",
    )
    parser.add_argument(
        "--require-unanimous",
        action="store_true",
        help="Only sample patches where all annotators agreed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute quotas and selections but do not copy image files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def stable_seed(*parts: object, seed: int) -> int:
    value = seed & 0xFFFFFFFF
    for part in parts:
        for byte in str(part).encode("utf-8"):
            value = ((value * 131) + byte) & 0xFFFFFFFF
    return value


def ensure_inputs(args: argparse.Namespace) -> None:
    if not args.consensus.exists():
        raise FileNotFoundError(f"Consensus parquet not found: {args.consensus}")
    if not args.patches_root.exists():
        raise FileNotFoundError(f"Patches root not found: {args.patches_root}")
    if args.target_per_class <= 0:
        raise ValueError("--target-per-class must be positive")


def load_consensus(path: Path, require_unanimous: bool) -> pd.DataFrame:
    consensus = pd.read_parquet(path)
    required = {*PATCH_ID_COLUMNS, LABEL_COLUMN, "any_missing_annotation"}
    missing = required - set(consensus.columns)
    if missing:
        raise ValueError(f"Consensus is missing required columns: {sorted(missing)}")

    consensus = consensus.loc[~consensus["any_missing_annotation"]].copy()
    if require_unanimous:
        if "unanimous" not in consensus.columns:
            raise ValueError("--require-unanimous needs a consensus 'unanimous' column")
        consensus = consensus.loc[consensus["unanimous"]].copy()

    consensus = consensus.loc[consensus[LABEL_COLUMN].isin(CLASS_LABELS)].copy()
    consensus["row"] = consensus["row"].astype(int)
    consensus["column"] = consensus["column"].astype(int)
    consensus["patch_id"] = (
        consensus["slide_id"].astype(str)
        + "_"
        + consensus["row"].astype(str)
        + "_"
        + consensus["column"].astype(str)
    )
    return consensus


def allocate_class_quotas(
    frame: pd.DataFrame, target: int
) -> dict[tuple[str, str], int]:
    quotas: dict[tuple[str, str], int] = {}
    for label in CLASS_LABELS:
        counts = frame.loc[frame[LABEL_COLUMN] == label].groupby("slide_id").size()
        if counts.empty:
            continue

        total_available = int(counts.sum())
        class_target = min(target, total_available)
        base = class_target // len(counts)
        remainder = class_target % len(counts)

        ordered_slides = sorted(
            counts.index,
            key=lambda slide_id: (-int(counts.loc[slide_id]), str(slide_id)),
        )
        for rank, slide_id in enumerate(ordered_slides):
            desired = base + int(rank < remainder)
            quotas[(str(slide_id), label)] = min(desired, int(counts.loc[slide_id]))

        while sum(quotas[(str(slide_id), label)] for slide_id in counts.index) < class_target:
            changed = False
            for slide_id in ordered_slides:
                key = (str(slide_id), label)
                if quotas[key] < int(counts.loc[slide_id]):
                    quotas[key] += 1
                    changed = True
                    if (
                        sum(quotas[(str(s), label)] for s in counts.index)
                        >= class_target
                    ):
                        break
            if not changed:
                break
    return quotas


def patch_filename(slide_id: str, row: int, column: int) -> str:
    return f"{slide_id}_{row}_{column}.png"


def build_patch_path_index(patches_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in patches_root.glob("*/*.png"):
        paths[path.name] = path
    return paths


def attach_patch_paths(frame: pd.DataFrame, patches_root: Path) -> pd.DataFrame:
    path_index = build_patch_path_index(patches_root)
    frame = frame.copy()
    filenames = [
        patch_filename(str(slide_id), int(row), int(column))
        for slide_id, row, column in frame[PATCH_ID_COLUMNS].itertuples(
            index=False, name=None
        )
    ]
    frame["patch_filename"] = filenames
    frame["source_patch_path"] = [str(path_index.get(name, "")) for name in filenames]
    missing = frame["source_patch_path"].eq("")
    if missing.any():
        examples = ", ".join(frame.loc[missing, "patch_filename"].head(10).tolist())
        raise FileNotFoundError(
            f"Missing {int(missing.sum())} patch PNGs under {patches_root}. "
            f"Examples: {examples}"
        )
    return frame


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    diff = maxc - minc

    hue = np.zeros_like(maxc)
    mask = diff > 0
    red = mask & (maxc == rgb[..., 0])
    green = mask & (maxc == rgb[..., 1])
    blue = mask & (maxc == rgb[..., 2])
    hue[red] = ((rgb[..., 1][red] - rgb[..., 2][red]) / diff[red]) % 6
    hue[green] = ((rgb[..., 2][green] - rgb[..., 0][green]) / diff[green]) + 2
    hue[blue] = ((rgb[..., 0][blue] - rgb[..., 1][blue]) / diff[blue]) + 4
    hue /= 6.0

    saturation = np.zeros_like(maxc)
    nonzero = maxc > 0
    saturation[nonzero] = diff[nonzero] / maxc[nonzero]
    value = maxc
    return np.stack([hue, saturation, value], axis=-1)


def entropy_from_gray(gray: np.ndarray) -> float:
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum() / 8.0)


def image_features(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        small = image.resize((16, 16), Image.Resampling.BILINEAR)
        rgb = np.asarray(small, dtype=np.uint8)

    rgb_float = rgb.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(rgb)
    gray = np.dot(rgb_float, np.array([0.299, 0.587, 0.114], dtype=np.float32))
    gray_u8 = np.clip(gray * 255, 0, 255).astype(np.uint8)

    rgb_mean = rgb_float.reshape(-1, 3).mean(axis=0)
    rgb_std = rgb_float.reshape(-1, 3).std(axis=0)
    hsv_mean = hsv.reshape(-1, 3).mean(axis=0)
    hsv_std = hsv.reshape(-1, 3).std(axis=0)

    gray_stats = np.array(
        [
            gray.mean(),
            gray.std(),
            np.quantile(gray, 0.1),
            np.quantile(gray, 0.5),
            np.quantile(gray, 0.9),
            entropy_from_gray(gray_u8),
        ],
        dtype=np.float32,
    )

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    edge_density = np.array(
        [np.mean(np.abs(gx)), np.mean(np.abs(gy))], dtype=np.float32
    )

    hist_features = []
    for channel in range(3):
        hist, _ = np.histogram(rgb_float[..., channel], bins=8, range=(0.0, 1.0))
        hist_features.extend((hist / hist.sum()).astype(np.float32))
    for channel in range(3):
        hist, _ = np.histogram(hsv[..., channel], bins=8, range=(0.0, 1.0))
        hist_features.extend((hist / hist.sum()).astype(np.float32))

    thumbnail = image_thumbnail_features(rgb_float)
    return np.concatenate(
        [
            rgb_mean,
            rgb_std,
            hsv_mean,
            hsv_std,
            gray_stats,
            edge_density,
            np.asarray(hist_features, dtype=np.float32),
            thumbnail,
        ]
    ).astype(np.float32)


def image_thumbnail_features(rgb_float: np.ndarray) -> np.ndarray:
    pooled = rgb_float.reshape(8, 2, 8, 2, 3).mean(axis=(1, 3))
    return pooled.reshape(-1).astype(np.float32)


def candidate_pool(
    group: pd.DataFrame,
    quota: int,
    label: str,
    seed: int,
    max_candidates_per_slide_class: int | None,
) -> pd.DataFrame:
    if max_candidates_per_slide_class == 0:
        max_candidates = len(group)
    elif max_candidates_per_slide_class is None:
        max_candidates = max(quota * 10, 5_000)
    else:
        max_candidates = max_candidates_per_slide_class

    if len(group) <= max_candidates:
        return group

    rng_seed = stable_seed(group["slide_id"].iloc[0], label, "candidate", seed=seed)
    return group.sample(n=max_candidates, random_state=rng_seed)


def standardize(features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return features
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    return (features - mean) / std


def cluster_count(quota: int, available: int, fixed: int | None) -> int:
    if quota <= 0 or available <= 0:
        return 0
    if fixed is not None:
        return max(1, min(fixed, available, quota))
    return max(1, min(available, quota, int(math.ceil(math.sqrt(quota) * 2))))


def balanced_cluster_sample(
    candidates: pd.DataFrame,
    labels: np.ndarray,
    quota: int,
    seed: int,
) -> pd.DataFrame:
    selected_indices: list[int] = []
    rng = np.random.default_rng(seed)
    clusters = sorted(set(int(label) for label in labels))
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for position, label in enumerate(labels):
        by_cluster[int(label)].append(position)

    base = quota // len(clusters)
    remainder = quota % len(clusters)
    spare_clusters: list[int] = []
    for rank, cluster in enumerate(clusters):
        positions = by_cluster[cluster]
        rng.shuffle(positions)
        take = min(len(positions), base + int(rank < remainder))
        selected_indices.extend(positions[:take])
        if take < len(positions):
            spare_clusters.append(cluster)

    while len(selected_indices) < quota and spare_clusters:
        changed = False
        for cluster in list(spare_clusters):
            already = sum(1 for idx in selected_indices if labels[idx] == cluster)
            positions = by_cluster[cluster]
            if already < len(positions):
                selected_indices.append(positions[already])
                changed = True
                if len(selected_indices) >= quota:
                    break
            else:
                spare_clusters.remove(cluster)
        if not changed:
            break

    selected_positions = sorted(selected_indices[:quota])
    return candidates.iloc[selected_positions].copy()


def sample_slide_class(
    group: pd.DataFrame,
    quota: int,
    label: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    slide_id = str(group["slide_id"].iloc[0])
    if quota <= 0:
        return group.head(0).copy(), {}
    if len(group) <= quota:
        sampled = group.copy()
        sampled["cluster_id"] = 0
        return sampled, {
            "slide_id": slide_id,
            "majority_label": label,
            "available_count": int(len(group)),
            "candidate_count": int(len(group)),
            "selected_count": int(len(sampled)),
            "quota": int(quota),
            "cluster_count": 1,
        }

    candidates = candidate_pool(
        group,
        quota=quota,
        label=label,
        seed=args.seed,
        max_candidates_per_slide_class=args.max_candidates_per_slide_class,
    ).reset_index(drop=True)

    features = np.vstack(
        [
            image_features(Path(path))
            for path in tqdm(
                candidates["source_patch_path"],
                desc=f"{slide_id} {label}",
                leave=False,
            )
        ]
    )
    features = standardize(features)
    k = cluster_count(quota, len(candidates), args.clusters_per_slide_class)
    rng_seed = stable_seed(slide_id, label, "kmeans", seed=args.seed)

    try:
        _, labels = kmeans2(
            features,
            k,
            minit="++",
            iter=30,
            seed=np.random.default_rng(rng_seed),
        )
    except np.linalg.LinAlgError:
        labels = np.zeros(len(candidates), dtype=int)
        k = 1

    candidates = candidates.copy()
    candidates["cluster_id"] = labels.astype(int)
    sampled = balanced_cluster_sample(
        candidates,
        labels=labels,
        quota=quota,
        seed=stable_seed(slide_id, label, "sample", seed=args.seed),
    )

    summary = {
        "slide_id": slide_id,
        "majority_label": label,
        "available_count": int(len(group)),
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(sampled)),
        "quota": int(quota),
        "cluster_count": int(k),
    }
    return sampled, summary


def copy_selected_images(sampled: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    sampled = sampled.copy()
    dest_paths = []
    for row in tqdm(
        sampled.itertuples(index=False),
        total=len(sampled),
        desc="Copying selected PNGs",
    ):
        label = getattr(row, LABEL_COLUMN)
        dest_dir = output_dir / "images" / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        source = Path(getattr(row, "source_patch_path"))
        dest_name = f"{getattr(row, 'slide_id')}_{getattr(row, 'row')}_{getattr(row, 'column')}.png"
        dest = dest_dir / dest_name
        shutil.copy2(source, dest)
        dest_paths.append(str(dest))
    sampled["sample_patch_path"] = dest_paths
    return sampled


def copy_metadata(metadata_path: Path, output_dir: Path) -> None:
    if not metadata_path.exists():
        return
    metadata = pd.read_parquet(metadata_path)
    metadata.to_parquet(output_dir / "annotation_metadata.parquet", index=False)
    metadata.to_csv(output_dir / "annotation_metadata.csv", index=False)


def write_outputs(
    sampled: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(output_dir / "sample_manifest.parquet", index=False)
    sampled.to_csv(output_dir / "sample_manifest.csv", index=False)
    cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False)
    copy_metadata(args.metadata, output_dir)

    class_counts = sampled[LABEL_COLUMN].value_counts().to_dict()
    slide_counts = sampled.groupby(["slide_id", LABEL_COLUMN]).size().reset_index(
        name="selected_count"
    )
    slide_counts.to_csv(output_dir / "slide_class_counts.csv", index=False)

    summary = {
        "consensus": str(args.consensus),
        "patches_root": str(args.patches_root),
        "metadata": str(args.metadata),
        "output_dir": str(output_dir),
        "target_per_class": int(args.target_per_class),
        "seed": int(args.seed),
        "copy_images": bool(args.copy_images and not args.dry_run),
        "require_unanimous": bool(args.require_unanimous),
        "selected_total": int(len(sampled)),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "slide_count": int(sampled["slide_id"].nunique()),
    }
    (output_dir / "sampling_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Pass --overwrite to write into it."
        )
    if output_dir.exists() and overwrite:
        for path in [
            output_dir / "images" / "foreground",
            output_dir / "images" / "background",
        ]:
            if path.exists():
                shutil.rmtree(path)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_inputs(args)
    prepare_output_dir(args.output_dir, args.overwrite)

    consensus = load_consensus(args.consensus, args.require_unanimous)
    consensus = attach_patch_paths(consensus, args.patches_root)
    quotas = allocate_class_quotas(consensus, args.target_per_class)

    sampled_frames = []
    summaries = []
    grouped = consensus.groupby(["slide_id", LABEL_COLUMN], sort=True)
    for (slide_id, label), group in grouped:
        quota = quotas.get((str(slide_id), str(label)), 0)
        sampled, summary = sample_slide_class(group, quota, str(label), args)
        if not sampled.empty:
            sampled_frames.append(sampled)
        if summary:
            summaries.append(summary)

    if not sampled_frames:
        raise RuntimeError("Sampling produced no patches")

    sampled = pd.concat(sampled_frames, ignore_index=True)
    sampled = sampled.sort_values(
        [LABEL_COLUMN, "slide_id", "cluster_id", "row", "column"]
    ).reset_index(drop=True)

    if args.copy_images and not args.dry_run:
        sampled = copy_selected_images(sampled, args.output_dir)
    else:
        sampled["sample_patch_path"] = ""

    cluster_summary = pd.DataFrame(summaries)
    write_outputs(sampled, cluster_summary, args.output_dir, args)

    print(f"Selected {len(sampled)} patches")
    print(sampled[LABEL_COLUMN].value_counts().to_string())
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
