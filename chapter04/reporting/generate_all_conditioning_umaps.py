#!/usr/bin/env python3
"""Generate real-vs-synthetic UNI UMAP coordinates for one experiment arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plot_icaird_stylegan_umap import (
    deterministic_subsample,
    feature_columns,
    fit_umap,
    matched_real_sample,
    validate_embedding_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--conditioning", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-synthetic", type=int, default=10_000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--umap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_path = (
        args.run_root
        / "cache"
        / "embeddings"
        / args.dataset
        / "train_uni_224.parquet"
    )
    arm_root = (
        args.run_root
        / "outputs"
        / "generators"
        / args.dataset
        / args.model
        / args.conditioning
        / "seed0"
    )
    synthetic_path = arm_root / "metrics" / "synthetic_uni_224.parquet"
    output_dir = (
        args.output_root / args.conditioning / f"{args.dataset}_{args.model}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinate_path = output_dir / "coordinates.csv"
    metadata_path = output_dir / "metadata.json"
    metrics_path = output_dir / "metrics.json"

    real = pd.read_parquet(real_path)
    synthetic = pd.read_parquet(synthetic_path)
    features = validate_embedding_frame(real, name="real")
    validate_embedding_frame(
        synthetic, name="synthetic", expected_features=features
    )
    synthetic = deterministic_subsample(
        synthetic, max_rows=args.max_synthetic, seed=args.sample_seed
    )
    real_sample = matched_real_sample(real, synthetic, seed=args.sample_seed)
    real_sample = real_sample.assign(source="Real")
    synthetic = synthetic.assign(source="Synthetic")
    combined = pd.concat([real_sample, synthetic], ignore_index=True)

    coordinates = fit_umap(
        combined[features].to_numpy(dtype=np.float32, copy=False),
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        seed=args.umap_seed,
    )
    combined["umap_1"] = coordinates[:, 0]
    combined["umap_2"] = coordinates[:, 1]
    columns = ["image_path", "label", "source", "umap_1", "umap_2"]
    combined[columns].to_csv(coordinate_path, index=False)

    fid = json.loads(
        (arm_root / "metrics" / "inception_fid.json").read_text()
    )["fid"]
    pr = json.loads(
        (arm_root / "metrics" / "uni_precision_recall.json").read_text()
    )
    metrics_path.write_text(
        json.dumps(
            {
                "fid": float(fid),
                "improved_precision": float(pr["precision"]),
                "improved_recall": float(pr["recall"]),
                "precision_recall_method": pr.get("method"),
            },
            indent=2,
        )
        + "\n"
    )
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "model": args.model,
                "conditioning": args.conditioning,
                "real_parquet": str(real_path),
                "synthetic_parquet": str(synthetic_path),
                "points_per_source": len(synthetic),
                "embedding_dimensions": len(feature_columns(synthetic.columns)),
                "sample_seed": args.sample_seed,
                "umap": {
                    "implementation": "cuml.manifold.UMAP",
                    "n_neighbors": 15,
                    "min_dist": 0.1,
                    "metric": "cosine",
                    "random_state": args.umap_seed,
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
