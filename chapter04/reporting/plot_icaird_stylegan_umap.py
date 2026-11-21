#!/usr/bin/env python3
"""Plot a joint UMAP of real and StyleGAN iCAIRD UNI embeddings."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

DEFAULT_REAL = Path(
    "/raid/dm236/synpath/cache/embeddings/icaird_cervical100k/"
    "train_uni_224.parquet"
)
DEFAULT_SYNTHETIC = Path(
    "/raid/dm236/synpath/outputs/generators/icaird_cervical100k/"
    "stylegan2ada/true_label/seed0/metrics/synthetic_uni_224.parquet"
)
DEFAULT_OUTPUT = Path("figures/synthetic_patches_umap")

LABEL_ORDER = ("normal", "low_grade", "high_grade", "malignant")
LABEL_DISPLAY = {
    "normal": "Normal",
    "low_grade": "Low grade",
    "high_grade": "High grade",
    "malignant": "Malignant",
}
LABEL_COLOURS = {
    "normal": "#009E73",
    "low_grade": "#0072B2",
    "high_grade": "#E69F00",
    "malignant": "#D55E00",
}
REAL_COLOUR = "#1F77B4"
SYNTHETIC_COLOUR = "#D62728"
DEFAULT_OUTPUT_STEM = "icaird_stylegan_vs_real_uni_umap"


def feature_columns(columns: Sequence[str]) -> list[str]:
    """Return contiguous f0..fN feature columns in numeric order."""
    parsed: list[tuple[int, str]] = []
    for column in columns:
        if column.startswith("f") and column[1:].isdigit():
            parsed.append((int(column[1:]), column))
    parsed.sort()
    if not parsed:
        raise ValueError("No embedding feature columns matching f0, f1, ... were found")
    indices = [index for index, _ in parsed]
    expected = list(range(indices[-1] + 1))
    if indices != expected:
        raise ValueError("Embedding feature columns are not contiguous from f0")
    return [column for _, column in parsed]


def validate_embedding_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    expected_features: Sequence[str] | None = None,
) -> list[str]:
    """Validate metadata and numeric embedding columns."""
    required = {"image_path", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if frame["image_path"].isna().any() or frame["image_path"].duplicated().any():
        raise ValueError(f"{name} image_path values must be non-null and unique")
    if frame["label"].isna().any():
        raise ValueError(f"{name} contains null labels")

    features = feature_columns(list(frame.columns))
    if expected_features is not None and features != list(expected_features):
        raise ValueError(f"{name} embedding features do not match the real embeddings")
    values = frame[features].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite embedding values")
    return features


def matched_real_sample(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Sample real rows to exactly match synthetic counts for every label."""
    wanted = synthetic["label"].value_counts().sort_index()
    available = real["label"].value_counts()
    missing = wanted.index.difference(available.index)
    if len(missing):
        raise ValueError(f"Real embeddings lack synthetic labels: {list(missing)}")
    short = {label: int(count) for label, count in wanted.items() if available[label] < count}
    if short:
        raise ValueError(f"Not enough real rows to match synthetic labels: {short}")

    rng = np.random.default_rng(seed)
    chosen: list[pd.DataFrame] = []
    for label, count in wanted.items():
        candidates = real.index[real["label"] == label].to_numpy()
        indices = rng.choice(candidates, size=int(count), replace=False)
        chosen.append(real.loc[indices])
    result = pd.concat(chosen, ignore_index=True)
    return result.sort_values("image_path", kind="stable").reset_index(drop=True)


def deterministic_subsample(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Select at most max_rows without replacement using a fixed seed."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(frame), size=max_rows, replace=False))
    return frame.iloc[indices].reset_index(drop=True)


def fit_umap(
    values: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> np.ndarray:
    """Fit GPU UMAP with cuML and return host-memory coordinates."""
    try:
        from cuml.manifold import UMAP
    except ImportError as exc:  # pragma: no cover - exercised on the DGX
        raise RuntimeError("cuML is required to fit this UMAP; run the CLI on the DGX") from exc

    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    coordinates = reducer.fit_transform(values)
    if hasattr(coordinates, "get"):
        coordinates = coordinates.get()
    result = np.asarray(coordinates, dtype=np.float32)
    if result.shape != (len(values), 2) or not np.isfinite(result).all():
        raise ValueError("UMAP returned invalid coordinates")
    return result


def convex_hull_polygon(frame: pd.DataFrame) -> np.ndarray:
    """Return a closed convex-hull polygon for UMAP coordinates."""
    coordinates = frame[["umap_1", "umap_2"]].to_numpy(dtype=np.float64, copy=False)
    if len(coordinates) < 3 or not np.isfinite(coordinates).all():
        raise ValueError("At least three finite UMAP points are required for a polygon")
    hull = ConvexHull(coordinates)
    polygon = coordinates[hull.vertices]
    return np.vstack([polygon, polygon[0]])


def plot_umap(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    dataset_name: str = "iCAIRD",
    model_name: str = "StyleGAN2-ADA",
    output_stem: str = DEFAULT_OUTPUT_STEM,
) -> None:
    """Render publication PNG and PDF outputs."""
    fig, ax = plt.subplots(figsize=(8.4, 7.2), constrained_layout=False)
    source_styles = {
        "Real": {"colour": REAL_COLOUR, "marker": "o", "label": "Real"},
        model_name: {
            "colour": SYNTHETIC_COLOUR,
            "marker": "^",
            "label": "Synthetic",
        },
    }
    sizes = {"Real": 6.0, model_name: 8.0}
    alphas = {"Real": 0.30, model_name: 0.38}

    for source in ("Real", model_name):
        subset = frame[frame["source"] == source]
        if subset.empty:
            raise ValueError(f"No coordinates found for source {source!r}")
        style = source_styles[source]
        polygon = convex_hull_polygon(subset)
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor=style["colour"],
            edgecolor=style["colour"],
            alpha=0.09,
            linewidth=2.0,
            zorder=1,
        )
        ax.plot(
            polygon[:, 0],
            polygon[:, 1],
            color=style["colour"],
            linewidth=1.8,
            alpha=0.9,
            zorder=4,
        )
        ax.scatter(
            subset["umap_1"],
            subset["umap_2"],
            c=style["colour"],
            marker=style["marker"],
            s=sizes[source],
            alpha=alphas[source],
            linewidths=0,
            rasterized=True,
            zorder=2 if source == "Real" else 3,
        )

    source_handles = [
        Line2D(
            [0],
            [0],
            marker=source_styles[source]["marker"],
            color=source_styles[source]["colour"],
            linestyle="-",
            linewidth=1.8,
            markersize=6,
            markerfacecolor=source_styles[source]["colour"],
            markeredgewidth=0,
            label=source_styles[source]["label"],
        )
        for source in source_styles
    ]
    ax.legend(
        handles=source_handles,
        title="Source and polygon",
        loc="upper left",
        frameon=False,
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )

    ax.set_title(
        f"Relationship between real and {model_name} {dataset_name} manifolds",
        loc="left",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(
        0.09,
        0.025,
        f"Joint UMAP of UNI embeddings; {len(frame) // 2:,} patches per source. "
        "Polygons show the full convex hull of each source.",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.08, right=0.76, top=0.91, bottom=0.10)
    fig.savefig(output_dir / f"{output_stem}.png", dpi=400)
    fig.savefig(output_dir / f"{output_stem}.pdf", dpi=400)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-parquet", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--synthetic-parquet", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--umap-seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--dataset-name", default="iCAIRD")
    parser.add_argument("--model-name", default="StyleGAN2-ADA")
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--max-synthetic", type=int, default=10_000)
    return parser
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real = pd.read_parquet(args.real_parquet)
    synthetic = pd.read_parquet(args.synthetic_parquet)
    real_features = validate_embedding_frame(real, name="real")
    validate_embedding_frame(synthetic, name="synthetic", expected_features=real_features)
    original_synthetic_points = len(synthetic)
    synthetic = deterministic_subsample(
        synthetic,
        max_rows=args.max_synthetic,
        seed=args.sample_seed,
    )

    real_labels = set(real["label"].astype(str))
    synthetic_labels = set(synthetic["label"].astype(str))
    if len(synthetic_labels) < 2:
        raise ValueError("Synthetic embeddings must contain at least two labels")
    if not synthetic_labels.issubset(real_labels):
        raise ValueError(
            "Synthetic labels must be present in the real embeddings; missing "
            f"{sorted(synthetic_labels.difference(real_labels))}"
        )
    if "split" in real.columns and not (real["split"].astype(str) == "train").all():
        raise ValueError("Real embeddings must contain only the training split")

    real_sample = matched_real_sample(real, synthetic, seed=args.sample_seed)
    real_sample = real_sample.assign(source="Real", original_split="train")
    synthetic = synthetic.assign(source=args.model_name, original_split="generated")
    combined = pd.concat([real_sample, synthetic], ignore_index=True)

    coordinates = fit_umap(
        combined[real_features].to_numpy(dtype=np.float32, copy=False),
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.umap_seed,
    )
    combined["umap_1"] = coordinates[:, 0]
    combined["umap_2"] = coordinates[:, 1]

    coordinate_columns = [
        column for column in ("image_path", "label", "label_id", "source", "original_split")
        if column in combined.columns
    ] + ["umap_1", "umap_2"]
    coordinate_path = args.output_dir / f"{args.output_stem}_coordinates.csv"
    combined[coordinate_columns].to_csv(coordinate_path, index=False)

    source_counts = combined["source"].value_counts().sort_index().to_dict()
    class_source_counts = (
        combined.groupby(["source", "label"], observed=True).size().unstack(fill_value=0)
    )
    metadata = {
        "real_parquet": str(args.real_parquet),
        "synthetic_parquet": str(args.synthetic_parquet),
        "dataset_name": args.dataset_name,
        "model_name": args.model_name,
        "embedding_dimensions": len(real_features),
        "original_synthetic_points": original_synthetic_points,
        "selected_synthetic_points": len(synthetic),
        "total_points": len(combined),
        "source_counts": {key: int(value) for key, value in source_counts.items()},
        "class_counts_by_source": {
            source: {label: int(value) for label, value in row.items()}
            for source, row in class_source_counts.to_dict(orient="index").items()
        },
        "sample_seed": args.sample_seed,
        "umap": {
            "implementation": "cuml.manifold.UMAP",
            "n_components": 2,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
            "random_state": args.umap_seed,
        },
        "visualisation": {
            "real_colour": REAL_COLOUR,
            "synthetic_colour": SYNTHETIC_COLOUR,
            "boundary": "full convex hull",
        },
    }
    (args.output_dir / f"{args.output_stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_umap(
        combined,
        args.output_dir,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        output_stem=args.output_stem,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
