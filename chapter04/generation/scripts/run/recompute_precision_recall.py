#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from synpath.artifacts import write_json
from synpath.config import load_resolved_yaml
from synpath.embeddings.uni import feature_matrix, load_embedding_frame
from synpath.metrics.quality import metric_sample, precision_recall
from synpath.paths import SynpathPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute improved precision/recall from cached UNI embeddings."
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
        f"precision/recall shard {args.shard_index + 1}/{args.num_shards}: "
        f"{len(selected)} of {len(artifacts)} artifacts"
    )
    for index, synthetic_path in enumerate(selected, start=1):
        recompute(synthetic_path, paths, cfg)
        print(
            f"completed {index}/{len(selected)}: {synthetic_path.parent.parent}"
        )


def recompute(synthetic_path: Path, paths: SynpathPaths, cfg: dict) -> None:
    relative = synthetic_path.relative_to(paths.outputs_root / "generators")
    dataset, _model, _conditioning, _seed, _metrics, _filename = relative.parts
    real_path = (
        paths.cache_root / "embeddings" / dataset / "train_uni_224.parquet"
    )
    if not real_path.exists():
        raise RuntimeError(f"Missing cached real embeddings: {real_path}")

    real_frame = load_embedding_frame(real_path)
    synthetic_frame = load_embedding_frame(synthetic_path)
    real_frame, synthetic_frame = metric_sample(
        real_frame, synthetic_frame, cfg
    )
    metrics_cfg = cfg.get("metrics", {})
    result = precision_recall(
        feature_matrix(real_frame),
        feature_matrix(synthetic_frame),
        nearest_neighbors=int(metrics_cfg.get("precision_recall_k", 3)),
        batch_size=int(metrics_cfg.get("precision_recall_batch_size", 512)),
    )

    output = synthetic_path.parent / "uni_precision_recall.json"
    preserve_legacy_artifact(output)
    write_json(output, result)


def preserve_legacy_artifact(output: Path) -> None:
    if not output.exists():
        return
    with output.open("r") as handle:
        existing = json.load(handle)
    if existing.get("method") == "improved_precision_recall":
        return
    backup = output.with_name("uni_precision_recall_legacy_global_radius.json")
    if not backup.exists():
        shutil.copy2(output, backup)


if __name__ == "__main__":
    main()
