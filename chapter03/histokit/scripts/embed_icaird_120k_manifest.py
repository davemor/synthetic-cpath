#!/usr/bin/env python3
"""Embed an exported iCAIRD cervical 120K image manifest."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from histokit.embeddings.encoder import EmbeddingResult, _tensor_from_output
from histokit.embeddings.io import read_embedding_result, write_embedding_result
from histokit.embeddings.manual_features import (
    extract_manual_features,
    manual_feature_names,
)
from histokit.embeddings.models import load_embedding_model


DEFAULT_DATASET_ROOT = Path(
    "/home/dm236/scratch/data/icaird_cervical/"
    "icaird_cervical_120k_trainonly_slide_split_uni2_labelbalanced_kcenter"
)
MODEL_OUTPUT_NAMES = {
    "uni": "uni",
    "uni2": "uni2_h",
    "uni2_h": "uni2_h",
    "dinov2": "dinov2_large",
    "dinov2-large": "dinov2_large",
    "dinov2_large": "dinov2_large",
    "hibou": "hibou_l",
    "hibou-l": "hibou_l",
    "hibou_l": "hibou_l",
    "manual_features": "manual_features",
}
INDEX_COLUMNS = [
    "manifest_row",
    "sample_id",
    "slide_id",
    "row",
    "column",
    "split",
    "category",
    "subcategory",
    "annotation_label",
    "annotation_label_name",
    "patch_png_path",
]


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.dataset_root / "embeddings" / output_name(
        args.model
    )
    manifest_path = args.manifest or args.dataset_root / "manifests/sample_manifest.parquet"

    if args.mode == "chunks":
        write_chunks(
            manifest_path=manifest_path,
            output_root=output_root,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )
        return
    if args.mode == "embed":
        embed_chunk(
            manifest_path=manifest_path,
            output_root=output_root,
            model_name=args.model,
            chunk_id=args.chunk_id,
            batch_size=args.batch_size,
            device=args.device,
            env_file=args.env_file,
            overwrite=args.overwrite,
            no_pretrained=args.no_pretrained,
            limit=args.limit,
        )
        return
    if args.mode == "summarize":
        merge_chunks(
            manifest_path=manifest_path,
            output_root=output_root,
            model_name=args.model,
            overwrite=args.overwrite,
        )
        return
    raise ValueError(f"Unknown mode: {args.mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("chunks", "embed", "summarize"),
        default="embed",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model", default="uni")
    parser.add_argument("--chunk-id", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def output_name(model: str) -> str:
    return MODEL_OUTPUT_NAMES.get(model, model.replace("-", "_"))


def load_manifest(path: Path, *, check_paths: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sample manifest not found: {path}")
    frame = pd.read_parquet(path)
    missing = {"sample_id", "patch_png_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame.insert(0, "manifest_row", np.arange(len(frame), dtype=np.int64))
    frame["patch_png_path"] = frame["patch_png_path"].astype(str)
    if check_paths:
        missing_paths = [
            path for path in frame["patch_png_path"].tolist() if not Path(path).exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                f"{len(missing_paths)} PNG(s) are missing; first: {missing_paths[0]}"
            )
    return frame


def chunk_index_path(output_root: Path) -> Path:
    return output_root / "chunk_index.csv"


def chunk_dir(output_root: Path, chunk_id: int) -> Path:
    return output_root / "chunks" / f"chunk_{chunk_id:05d}.embedding"


def write_chunks(
    *,
    manifest_path: Path,
    output_root: Path,
    chunk_size: int,
    overwrite: bool,
) -> None:
    if chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    frame = load_manifest(manifest_path)
    if chunk_index_path(output_root).exists() and not overwrite:
        print(f"Chunk index already exists: {chunk_index_path(output_root)}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "chunks").mkdir(parents=True, exist_ok=True)
    rows = []
    for chunk_id, start in enumerate(range(0, len(frame), chunk_size)):
        stop = min(start + chunk_size, len(frame))
        rows.append(
            {
                "chunk_id": chunk_id,
                "start": start,
                "stop": stop,
                "chunk_dir": str(chunk_dir(output_root, chunk_id)),
                "row_count": stop - start,
            }
        )
    with open(chunk_index_path(output_root), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_run_manifest(
        output_root=output_root,
        manifest_path=manifest_path,
        model_name=None,
        stage="chunks",
        extra={"chunk_size": chunk_size, "chunk_count": len(rows)},
    )
    print(f"Wrote {len(rows)} chunks to {chunk_index_path(output_root)}")


def read_chunk_row(output_root: Path, chunk_id: int | None) -> pd.Series:
    if chunk_id is None:
        raise ValueError("--chunk-id is required for --mode embed")
    chunks = pd.read_csv(chunk_index_path(output_root))
    row = chunks.loc[chunks["chunk_id"].astype(int) == int(chunk_id)]
    if row.empty:
        raise ValueError(f"Unknown chunk id {chunk_id}")
    return row.iloc[0]


def embed_chunk(
    *,
    manifest_path: Path,
    output_root: Path,
    model_name: str,
    chunk_id: int | None,
    batch_size: int,
    device: str,
    env_file: Path,
    overwrite: bool,
    no_pretrained: bool,
    limit: int | None,
) -> None:
    chunk = read_chunk_row(output_root, chunk_id)
    target = chunk_dir(output_root, int(chunk["chunk_id"]))
    if embedding_complete(target) and not overwrite:
        print(f"Skipping complete chunk {chunk_id}: {target}")
        return
    if target.exists() and not overwrite:
        print(f"Removing incomplete chunk {chunk_id}: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    frame = load_manifest(manifest_path, check_paths=False).iloc[
        int(chunk["start"]) : int(chunk["stop"])
    ]
    if limit is not None:
        frame = frame.head(limit)
    frame = frame.reset_index(drop=True)
    missing_paths = [
        path for path in frame["patch_png_path"].tolist() if not Path(path).exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} PNG(s) are missing; first: {missing_paths[0]}"
        )

    if model_name == "manual_features":
        features = encode_manual(frame["patch_png_path"].tolist())
        model_manifest = {
            "name": model_name,
            "backend": "manual",
            "source": "histokit.embeddings.manual_features",
            "feature_names": manual_feature_names(),
        }
    else:
        model = load_embedding_model(
            model_name,
            device=device,
            env_file=env_file,
            pretrained=not no_pretrained,
        )
        features = encode_model_pngs(
            frame["patch_png_path"].tolist(),
            model=model,
            batch_size=batch_size,
        )
        model_manifest = {
            "name": model_name,
            "backend": model.backend,
            "source": model.source,
            "device": model.device,
        }

    index = frame[index_columns(frame)].copy()
    result = EmbeddingResult(
        features=features,
        index=index,
        manifest={
            "created_at": utc_now(),
            "model": model_manifest,
            "dataset_manifest": str(manifest_path),
            "chunk_id": int(chunk["chunk_id"]),
            "chunk_start": int(chunk["start"]),
            "chunk_stop": int(chunk["stop"]),
            "num_embeddings": int(features.shape[0]),
            "embedding_dim": int(features.shape[1]) if features.ndim == 2 else 0,
            "histokit_git_commit": git_commit(Path(__file__).resolve().parents[1]),
        },
    )
    try:
        write_embedding_result(result, target, overwrite=overwrite)
    except FileExistsError:
        if embedding_complete(target) and not overwrite:
            print(f"Skipping concurrently completed chunk {chunk_id}: {target}")
            return
        raise
    print(f"Wrote {features.shape[0]} x {features.shape[1]} embeddings: {target}")


def index_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in INDEX_COLUMNS if column in frame.columns]
    columns.extend(column for column in frame.columns if column not in columns)
    return columns


def encode_manual(paths: list[str]) -> np.ndarray:
    features = []
    for path in tqdm(paths, desc="Extracting manual features"):
        with Image.open(path) as image:
            features.append(extract_manual_features(image))
    if not features:
        return np.empty((0, 260), dtype=np.float32)
    return np.stack(features, axis=0).astype(np.float32, copy=False)


def encode_model_pngs(
    patch_paths: list[str],
    *,
    model: Any,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    import torch

    batches: list[np.ndarray] = []
    total = int(np.ceil(len(patch_paths) / batch_size))
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(patch_paths), batch_size),
            total=total,
            desc="Embedding PNG patches",
        ):
            chunk = patch_paths[start : start + batch_size]
            tensors = []
            for patch_path in chunk:
                with Image.open(patch_path) as image:
                    tensors.append(model.preprocess(image.convert("RGB")))
            batch = torch.stack(tensors, dim=0)
            output = model.encode_tensor(batch)
            tensor = _tensor_from_output(output).detach().float().cpu()
            batches.append(tensor.numpy().astype(np.float32, copy=False))
    return np.concatenate(batches, axis=0)


def merge_chunks(
    *,
    manifest_path: Path,
    output_root: Path,
    model_name: str,
    overwrite: bool,
) -> None:
    final_features = output_root / "features.npz"
    final_index = output_root / "index.parquet"
    final_manifest = output_root / "manifest.json"
    if final_features.exists() and final_index.exists() and final_manifest.exists():
        if not overwrite:
            print(f"Final embedding already exists: {output_root}")
            return

    chunks = pd.read_csv(chunk_index_path(output_root)).sort_values("chunk_id")
    features = []
    indexes = []
    chunk_manifests = []
    for row in chunks.itertuples(index=False):
        path = chunk_dir(output_root, int(row.chunk_id))
        if not embedding_complete(path):
            raise FileNotFoundError(f"Incomplete chunk: {path}")
        result = read_embedding_result(path)
        features.append(result.features)
        indexes.append(result.index)
        chunk_manifests.append(result.manifest)

    merged_features = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    merged_index = pd.concat(indexes, ignore_index=True)
    merged_index = merged_index.sort_values("manifest_row").reset_index(drop=True)
    order = merged_index.index.to_numpy()
    if not np.array_equal(order, np.arange(len(order))):
        raise RuntimeError("Unexpected merge order calculation failure")
    merged_features = merged_features[
        np.argsort(pd.concat(indexes, ignore_index=True)["manifest_row"].to_numpy())
    ]

    manifest = load_manifest(manifest_path)
    if len(merged_index) != len(manifest):
        raise ValueError(f"Merged rows {len(merged_index)} != manifest rows {len(manifest)}")
    if not merged_index["sample_id"].reset_index(drop=True).equals(
        manifest["sample_id"].reset_index(drop=True)
    ):
        raise ValueError("Merged index sample_id order does not match manifest")

    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_root / "features.npz", features=merged_features)
    merged_index.to_parquet(output_root / "index.parquet", index=False)
    final_payload = {
        "created_at": utc_now(),
        "model": chunk_manifests[0].get("model", {"name": model_name}),
        "dataset_manifest": str(manifest_path),
        "num_embeddings": int(merged_features.shape[0]),
        "embedding_dim": int(merged_features.shape[1]),
        "chunk_count": int(len(chunks)),
        "chunks": "chunks/",
        "histokit_git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "storage": {
            "format": "npz+parquet",
            "features": "features.npz",
            "index": "index.parquet",
            "dtype": "float32",
        },
    }
    with open(output_root / "manifest.json", "w") as file:
        json.dump(final_payload, file, indent=2)
    print(f"Merged {merged_features.shape[0]} x {merged_features.shape[1]}: {output_root}")


def embedding_complete(path: Path) -> bool:
    required = [path / "features.npz", path / "index.parquet", path / "manifest.json"]
    if not all(item.exists() and item.stat().st_size > 0 for item in required):
        return False
    try:
        result = read_embedding_result(path)
    except Exception:
        return False
    return result.features.ndim == 2 and len(result.index) == result.features.shape[0]


def write_run_manifest(
    *,
    output_root: Path,
    manifest_path: Path,
    model_name: str | None,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "created_at": utc_now(),
        "stage": stage,
        "dataset_manifest": str(manifest_path),
        "model": model_name,
        "histokit_git_commit": git_commit(Path(__file__).resolve().parents[1]),
    }
    if extra:
        payload.update(extra)
    with open(output_root / "run_manifest.json", "w") as file:
        json.dump(payload, file, indent=2)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    main()
