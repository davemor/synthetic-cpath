#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm


DEFAULT_TEST_MANIFEST = Path(
    "/sharedscratch/dm236/data/icaird_cervical/resnet_patch_subset/"
    "stratified_120_consensus_100k/split_dataset/sample_manifest_test.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "/sharedscratch/dm236/data/icaird_cervical/resnet_patch_subset/"
    "stratified_120_consensus_100k/umap_test_foreground_background"
)
DEFAULT_SPLIT_IMAGES_ROOT = Path(
    "/sharedscratch/dm236/data/icaird_cervical/resnet_patch_subset/"
    "stratified_120_consensus_100k/split_dataset/split_images/test"
)
PATCH_ID_COLUMNS = ["slide_id", "row", "column"]
PATH_COLUMNS = [
    "split_patch_path",
    "sample_patch_path",
    "source_patch_path",
    "patch_png_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project the iCAIRD test patch subset with UMAP and colour points by "
            "a manifest label such as majority_label or cluster_id."
        )
    )
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--image-path-column",
        default="auto",
        help=(
            "Column containing patch PNG paths. Use 'auto' to try split_patch_path, "
            "sample_patch_path, source_patch_path, then patch_png_path."
        ),
    )
    parser.add_argument(
        "--split-images-root",
        type=Path,
        default=DEFAULT_SPLIT_IMAGES_ROOT,
        help=(
            "Fallback image root used when no usable path column exists. Expected "
            "layout is <root>/{majority_label}/{slide_id}_{row}_{column}.png."
        ),
    )
    parser.add_argument(
        "--cluster-column",
        default="cluster_id",
        help="Cluster column retained for backwards-compatible cluster labels.",
    )
    parser.add_argument(
        "--colour-column",
        default="majority_label",
        help="Manifest column used to colour the UMAP dots.",
    )
    parser.add_argument(
        "--global-cluster-label",
        action="store_true",
        help=(
            "Colour by slide_id/majority_label/cluster_id instead of raw cluster_id. "
            "This makes labels unique across slide/class groups, but can create many "
            "colours."
        ),
    )
    parser.add_argument("--seed", type=int, default=236)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="PCA dimensions before UMAP. Use 0 to disable PCA.",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=0,
        help="Optional deterministic subsample size for quick exploratory plots.",
    )
    parser.add_argument(
        "--label-column",
        default="majority_label",
        help="Class label column retained in the output coordinates.",
    )
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads for PNG feature extraction.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
    return np.stack([hue, saturation, maxc], axis=-1)


def entropy_from_gray(gray: np.ndarray) -> float:
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum() / 8.0)


def thumbnail_features(rgb_float: np.ndarray) -> np.ndarray:
    pooled = rgb_float.reshape(8, 2, 8, 2, 3).mean(axis=(1, 3))
    return pooled.reshape(-1).astype(np.float32)


def image_features(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        small = image.resize((16, 16), Image.Resampling.BILINEAR)
        rgb = np.asarray(small, dtype=np.uint8)

    rgb_float = rgb.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(rgb)
    gray = np.dot(rgb_float, np.array([0.299, 0.587, 0.114], dtype=np.float32))
    gray_u8 = np.clip(gray * 255, 0, 255).astype(np.uint8)

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

    return np.concatenate(
        [
            rgb_float.reshape(-1, 3).mean(axis=0),
            rgb_float.reshape(-1, 3).std(axis=0),
            hsv.reshape(-1, 3).mean(axis=0),
            hsv.reshape(-1, 3).std(axis=0),
            gray_stats,
            edge_density,
            np.asarray(hist_features, dtype=np.float32),
            thumbnail_features(rgb_float),
        ]
    ).astype(np.float32)


def load_manifest(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Test manifest not found: {path}")
    manifest = pd.read_parquet(path)
    required = {*PATCH_ID_COLUMNS, args.colour_column, args.label_column}
    if args.global_cluster_label:
        required.add(args.cluster_column)
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Test manifest missing columns: {sorted(missing)}")
    if args.max_patches > 0 and len(manifest) > args.max_patches:
        manifest = manifest.sample(n=args.max_patches, random_state=args.seed)
    return manifest.reset_index(drop=True)


def resolve_image_paths(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.image_path_column == "auto":
        for column in PATH_COLUMNS:
            if column in manifest.columns and manifest[column].astype(str).ne("").any():
                paths = manifest[column].astype(str)
                if paths.ne("").all():
                    print(f"Using image paths from manifest column: {column}", flush=True)
                    return paths
    elif args.image_path_column in manifest.columns:
        paths = manifest[args.image_path_column].astype(str)
        missing = paths.eq("") | ~paths.map(lambda value: Path(value).exists())
        if missing.any():
            examples = ", ".join(paths[missing].head(10).tolist())
            raise FileNotFoundError(
                f"{int(missing.sum())} paths in {args.image_path_column} do not "
                f"exist. Examples: {examples}"
            )
        return paths
    else:
        raise ValueError(f"Image path column not found: {args.image_path_column}")

    paths = []
    for row in manifest.itertuples(index=False):
        label = getattr(row, args.label_column)
        filename = f"{getattr(row, 'slide_id')}_{int(getattr(row, 'row'))}_{int(getattr(row, 'column'))}.png"
        paths.append(args.split_images_root / str(label) / filename)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        examples = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} patch PNGs. Examples: {examples}"
        )
    return pd.Series([str(path) for path in paths], index=manifest.index)


def standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    return (features - mean) / std


def project_umap(features: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    try:
        import umap
    except ImportError as error:
        raise ImportError(
            "umap-learn is required. Install/update the environment with "
            "`uv sync` or `uv pip install umap-learn`."
        ) from error

    features = standardize(features)
    if args.pca_components > 0:
        n_components = min(args.pca_components, features.shape[0], features.shape[1])
        features = PCA(n_components=n_components, random_state=args.seed).fit_transform(
            features
        )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
        transform_seed=args.seed,
    )
    return reducer.fit_transform(features)


def extract_features(image_paths: pd.Series, workers: int) -> np.ndarray:
    paths = [Path(path) for path in image_paths]
    if workers <= 1:
        features = [
            image_features(path) for path in tqdm(paths, desc="Extracting image features")
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            features = list(
                tqdm(
                    executor.map(image_features, paths),
                    total=len(paths),
                    desc=f"Extracting image features ({workers} workers)",
                )
            )
    return np.vstack(features)


def add_colour_label(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if not args.global_cluster_label:
        return manifest[args.colour_column].astype(str)
    return (
        manifest["slide_id"].astype(str)
        + "/"
        + manifest[args.label_column].astype(str)
        + "/c"
        + manifest[args.cluster_column].astype(str)
    )


def plot_projection(coords: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    labels = coords["colour_label"].astype(str)
    unique_labels = sorted(labels.unique(), key=lambda value: (len(value), value))
    code_map = {label: idx for idx, label in enumerate(unique_labels)}
    codes = labels.map(code_map).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    scatter = ax.scatter(
        coords["umap_x"],
        coords["umap_y"],
        c=codes,
        s=args.point_size,
        alpha=args.alpha,
        cmap="tab20" if len(unique_labels) <= 60 else "turbo",
        linewidths=0,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"iCAIRD test patches coloured by {args.colour_column}")

    if len(unique_labels) <= 60:
        handles, _ = scatter.legend_elements(num=len(unique_labels))
        ax.legend(
            handles,
            unique_labels,
            title=args.colour_column,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            markerscale=2.5,
            frameon=False,
        )
    else:
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("cluster label code")

    safe_colour_name = args.colour_column.replace("/", "_").replace(" ", "_")
    fig.savefig(output_dir / f"test_patches_umap_by_{safe_colour_name}.png", dpi=args.dpi)
    fig.savefig(output_dir / f"test_patches_umap_by_{safe_colour_name}.pdf")
    plt.close(fig)


def write_features(
    manifest: pd.DataFrame,
    features: np.ndarray,
    image_paths: pd.Series,
    output_dir: Path,
) -> None:
    feature_columns = [f"feature_{index:03d}" for index in range(features.shape[1])]
    feature_frame = manifest[PATCH_ID_COLUMNS].copy()
    feature_frame["resolved_patch_path"] = image_paths.astype(str).to_numpy()
    for column in ["majority_label", "cluster_id", "sample_split"]:
        if column in manifest.columns:
            feature_frame[column] = manifest[column].to_numpy()
    feature_values = pd.DataFrame(features, columns=feature_columns)
    feature_frame = pd.concat([feature_frame, feature_values], axis=1)
    feature_frame.to_parquet(output_dir / "test_patches_features.parquet", index=False)
    feature_frame.to_csv(output_dir / "test_patches_features.csv", index=False)
    np.save(output_dir / "test_patches_features.npy", features)


def write_outputs(
    manifest: pd.DataFrame,
    features: np.ndarray,
    coordinates: np.ndarray,
    image_paths: pd.Series,
    args: argparse.Namespace,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_features(manifest, features, image_paths, args.output_dir)
    coords = manifest.copy()
    coords["resolved_patch_path"] = image_paths.astype(str).to_numpy()
    coords["colour_label"] = add_colour_label(coords, args)
    coords["umap_x"] = coordinates[:, 0]
    coords["umap_y"] = coordinates[:, 1]

    coords.to_parquet(args.output_dir / "test_patches_umap_coordinates.parquet", index=False)
    coords.to_csv(args.output_dir / "test_patches_umap_coordinates.csv", index=False)
    plot_projection(coords, args.output_dir, args)

    summary = {
        "test_manifest": str(args.test_manifest),
        "output_dir": str(args.output_dir),
        "patch_count": int(len(coords)),
        "slide_count": int(coords["slide_id"].nunique()),
        "colour_column": args.colour_column,
        "colour_label_count": int(coords["colour_label"].nunique()),
        "cluster_column": args.cluster_column,
        "global_cluster_label": bool(args.global_cluster_label),
        "feature_count": int(features.shape[1]),
        "seed": int(args.seed),
        "n_neighbors": int(args.n_neighbors),
        "min_dist": float(args.min_dist),
        "metric": args.metric,
        "pca_components": int(args.pca_components),
        "max_patches": int(args.max_patches),
    }
    (args.output_dir / "umap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {path}. Pass --overwrite to replace files."
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    manifest = load_manifest(args.test_manifest, args)
    image_paths = resolve_image_paths(manifest, args)

    features = extract_features(image_paths, args.workers)
    print(f"Extracted features with shape {features.shape}", flush=True)
    coordinates = project_umap(features, args)
    print(f"Computed UMAP coordinates with shape {coordinates.shape}", flush=True)
    write_outputs(manifest, features, coordinates, image_paths, args)

    print(f"Projected {len(manifest)} test patches")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
