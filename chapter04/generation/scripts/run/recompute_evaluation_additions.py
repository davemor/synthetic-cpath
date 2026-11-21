#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from synpath.artifacts import write_json
from synpath.config import load_resolved_yaml
from synpath.embeddings.uni import load_embedding_frame
from synpath.metrics.quality import (
    attach_real_clusters,
    cluster_adherence,
    inception_precision_recall,
    metric_sample,
    preserve_legacy_cluster_coverage,
)
from synpath.paths import SynpathPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pooled Inception precision/recall and predicted-cluster "
            "adherence from cached experiment artifacts."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, num-shards)")

    cfg = load_resolved_yaml(args.config)
    paths = SynpathPaths.from_env()
    artifacts = sorted(
        (paths.outputs_root / "generators").glob(
            "*/*/*/seed*/metrics/synthetic_uni_224.parquet"
        )
    )
    selected = artifacts[args.shard_index :: args.num_shards]
    print(
        f"evaluation shard {args.shard_index + 1}/{args.num_shards}: "
        f"{len(selected)} of {len(artifacts)} artifacts"
    )
    for index, synthetic_path in enumerate(selected, start=1):
        recompute(synthetic_path, paths, cfg)
        print(f"completed {index}/{len(selected)}: {synthetic_path.parent.parent}")


def recompute(synthetic_path: Path, paths: SynpathPaths, cfg: dict) -> None:
    relative = synthetic_path.relative_to(paths.outputs_root / "generators")
    dataset, _model, conditioning, _seed, _metrics, _filename = relative.parts
    real_path = paths.cache_root / "embeddings" / dataset / "train_uni_224.parquet"
    if not real_path.exists():
        raise RuntimeError(f"Missing cached real embeddings: {real_path}")

    full_real = load_embedding_frame(real_path)
    full_real = attach_real_clusters(full_real, dataset, conditioning, paths)
    full_synthetic = load_embedding_frame(synthetic_path)
    real, synthetic = metric_sample(full_real, full_synthetic, cfg)
    metrics_root = synthetic_path.parent

    inception_result = inception_precision_recall(
        real,
        synthetic,
        dataset,
        paths,
        cfg,
        metrics_root / "synthetic_inception_clean.npy",
    )
    write_json(
        metrics_root / "inception_precision_recall.json",
        inception_result,
    )

    result, contingency, counts = cluster_adherence(
        full_real,
        full_synthetic,
        dataset,
        conditioning,
        paths,
    )
    coverage_path = metrics_root / "uni_cluster_coverage.json"
    preserve_legacy_cluster_coverage(coverage_path)
    write_json(coverage_path, result)
    if contingency is not None:
        contingency.to_csv(
            metrics_root / "uni_cluster_adherence_contingency.csv",
            index=False,
        )
    if counts is not None:
        counts.to_csv(metrics_root / "uni_cluster_counts.csv", index=False)


if __name__ == "__main__":
    main()
