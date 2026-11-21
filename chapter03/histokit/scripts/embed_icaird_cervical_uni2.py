#!/usr/bin/env python3
"""Embed exported iCAIRD cervical bag patches with UNI2-h.

This script targets the canonical bags dataset layout:

    <bags-root>/patchsets/{slide_id}.patchset/frame.parquet
    <bags-root>/patches/{slide_id}/r{row}_c{column}.png
    <bags-root>/embeddings/uni2_h/slides/{slide_id}.uni2_h.embedding/

Only exported bag patches are embedded: rows with ``keep == true`` and a
non-empty ``patch_png_path``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from histokit.embeddings.encoder import EmbeddingResult, _tensor_from_output
from histokit.embeddings.io import read_embedding_result, write_embedding_result
from histokit.embeddings.models import load_embedding_model


DEFAULT_BAGS_ROOT = Path(
    "/sharedscratch/dm236/data/icaird_cervical/bags/"
    "full_canny005_or_resnet_light_grandqc_artifact"
)
DEFAULT_MODEL = "uni2_h"
REQUIRED_INDEX_COLUMNS = [
    "slide_id",
    "row",
    "column",
    "frame_index",
    "patch_png_path",
    "annotation_label",
    "annotation_label_name",
    "category",
    "subcategory",
    "split",
    "canny_tissue_score",
    "official_grandqc_artifact_class_id",
    "official_grandqc_artifact_class",
]


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.bags_root / "embeddings" / "uni2_h"

    if args.mode == "chunks":
        write_chunks(
            bags_root=args.bags_root,
            output_root=output_root,
            chunk_size=args.chunk_size,
        )
        return

    if args.mode == "summarize":
        summarize_embeddings(
            bags_root=args.bags_root,
            output_root=output_root,
        )
        return

    slide_ids = resolve_slide_ids(args)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "slides").mkdir(parents=True, exist_ok=True)
    write_root_manifest(
        output_root=output_root,
        bags_root=args.bags_root,
        model=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )

    slide_ids = filter_pending_slide_ids(
        slide_ids=slide_ids,
        output_root=output_root,
        model_name=args.model,
        overwrite=args.overwrite,
    )
    if not slide_ids:
        return

    model = load_embedding_model(
        args.model,
        device=args.device,
        env_file=args.env_file,
        pretrained=not args.no_pretrained,
    )
    for slide_id in slide_ids:
        embed_slide(
            slide_id=slide_id,
            bags_root=args.bags_root,
            output_root=output_root,
            model_name=args.model,
            model=model,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            limit_patches=args.limit_patches,
        )

    if args.chunk_csv is None and not args.slide_id:
        summarize_embeddings(bags_root=args.bags_root, output_root=output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("chunks", "embed", "summarize"),
        default="embed",
        help="Workflow stage to run.",
    )
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_BAGS_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <bags-root>/embeddings/uni2_h",
    )
    parser.add_argument("--slide-id", action="append", default=[])
    parser.add_argument("--chunk-csv", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--limit-slides", type=int)
    parser.add_argument("--limit-patches", type=int)
    return parser.parse_args()


def write_chunks(
    *,
    bags_root: Path,
    output_root: Path,
    chunk_size: int,
) -> None:
    if chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    slide_ids = list(iter_all_slide_ids(bags_root))
    chunks_dir = output_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, len(slide_ids), chunk_size)):
        chunk_path = chunks_dir / f"chunk_{index:05d}.csv"
        with open(chunk_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["slide_id"])
            writer.writeheader()
            for slide_id in slide_ids[start : start + chunk_size]:
                writer.writerow({"slide_id": slide_id})
    write_slide_index(output_root=output_root, slide_ids=slide_ids)
    print(f"Wrote {len(slide_ids)} slide ids into {chunks_dir}")


def filter_pending_slide_ids(
    *,
    slide_ids: list[str],
    output_root: Path,
    model_name: str,
    overwrite: bool,
) -> list[str]:
    if overwrite:
        return slide_ids

    pending = []
    for slide_id in slide_ids:
        target = output_root / "slides" / f"{slide_id}.{model_name}.embedding"
        if target.exists() and embedding_complete(target):
            print(f"Skipping complete embedding: {slide_id}")
            continue
        pending.append(slide_id)
    return pending


def resolve_slide_ids(args: argparse.Namespace) -> list[str]:
    if args.slide_id:
        slide_ids = args.slide_id
    elif args.chunk_csv is not None:
        slide_ids = list(read_slide_ids_from_csv(args.chunk_csv))
    else:
        slide_ids = list(iter_all_slide_ids(args.bags_root))

    if args.limit_slides is not None:
        slide_ids = slide_ids[: args.limit_slides]
    if not slide_ids:
        raise ValueError("No slide ids selected for embedding.")
    return slide_ids


def iter_all_slide_ids(bags_root: Path) -> Iterable[str]:
    index_path = bags_root / "dataset_index_ometiff.csv"
    if index_path.exists():
        index = pd.read_csv(index_path)
        for column in ("slide_id", "id", "sample_id"):
            if column in index.columns:
                yield from index[column].astype(str).tolist()
                return
        raise ValueError(f"No slide id column found in {index_path}")

    patchsets_root = bags_root / "patchsets"
    if not patchsets_root.exists():
        raise FileNotFoundError(
            f"Could not find {index_path} or {patchsets_root}"
        )
    for patchset_dir in sorted(patchsets_root.glob("*.patchset")):
        yield patchset_dir.name.removesuffix(".patchset")


def read_slide_ids_from_csv(path: Path) -> Iterable[str]:
    frame = pd.read_csv(path)
    for column in ("slide_id", "id", "sample_id"):
        if column in frame.columns:
            yield from frame[column].astype(str).tolist()
            return
    raise ValueError(f"No slide id column found in {path}")


def embed_slide(
    *,
    slide_id: str,
    bags_root: Path,
    output_root: Path,
    model_name: str,
    model: Any,
    batch_size: int,
    overwrite: bool,
    limit_patches: int | None,
) -> None:
    target = output_root / "slides" / f"{slide_id}.{model_name}.embedding"
    if target.exists() and not overwrite and embedding_complete(target):
        print(f"Skipping complete embedding: {slide_id}")
        return

    selected = load_slide_embedding_index(
        slide_id=slide_id,
        bags_root=bags_root,
        limit_patches=limit_patches,
    )
    features = encode_patch_pngs(
        selected["patch_png_path"].tolist(),
        model=model,
        batch_size=batch_size,
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "encoder": {
            "name": model_name,
            "source": model.source,
            "backend": model.backend,
        },
        "model": {
            "name": model_name,
            "source": model.source,
            "backend": model.backend,
            "device": model.device,
        },
        "embedding_dim": int(features.shape[1]) if features.ndim == 2 else 0,
        "dtype": "float32",
        "batch_size": batch_size,
        "device": model.device,
        "histokit_git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "patchset_source_root": str(bags_root / "patchsets"),
        "candidate_rule": "keep == true and patch_png_path != ''",
        "slide_id": slide_id,
        "num_embeddings": int(features.shape[0]),
    }
    write_embedding_result(
        EmbeddingResult(features=features, index=selected, manifest=manifest),
        target,
        overwrite=overwrite,
    )
    print(f"Wrote {features.shape[0]} x {features.shape[1]} embeddings: {target}")


def load_slide_embedding_index(
    *,
    slide_id: str,
    bags_root: Path,
    limit_patches: int | None,
) -> pd.DataFrame:
    patchset_path = bags_root / "patchsets" / f"{slide_id}.patchset"
    frame_path = patchset_path / "frame.parquet"
    if not frame_path.exists():
        raise FileNotFoundError(f"Patchset frame not found: {frame_path}")

    frame = pd.read_parquet(frame_path)
    if "keep" not in frame.columns:
        raise ValueError(f"{frame_path} has no keep column.")
    if "patch_png_path" not in frame.columns:
        raise ValueError(f"{frame_path} has no patch_png_path column.")

    patch_paths = frame["patch_png_path"].fillna("").astype(str)
    selected = frame[frame["keep"].astype(bool) & (patch_paths != "")].copy()
    if selected.empty:
        raise ValueError(f"No exported kept patches found for {slide_id}")
    if "frame_index" in selected.columns:
        selected["frame_index"] = selected.index.astype(int)
    else:
        selected.insert(0, "frame_index", selected.index.astype(int))
    selected["slide_id"] = slide_id
    selected["patch_png_path"] = selected["patch_png_path"].map(
        lambda path: str(resolve_patch_path(path, bags_root=bags_root))
    )
    for column in REQUIRED_INDEX_COLUMNS:
        if column not in selected.columns:
            selected[column] = pd.NA

    selected = selected[
        REQUIRED_INDEX_COLUMNS
        + [column for column in selected.columns if column not in REQUIRED_INDEX_COLUMNS]
    ].reset_index(drop=True)
    if limit_patches is not None:
        selected = selected.head(limit_patches)

    missing_paths = [
        path for path in selected["patch_png_path"].tolist() if not Path(path).exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} exported PNG(s) are missing for {slide_id}; "
            f"first missing path: {missing_paths[0]}"
        )
    return selected


def resolve_patch_path(path: str, *, bags_root: Path) -> Path:
    patch_path = Path(path)
    if patch_path.is_absolute():
        return patch_path
    return bags_root / patch_path


def encode_patch_pngs(
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


def embedding_complete(path: Path) -> bool:
    required = [
        path / "features.npz",
        path / "index.parquet",
        path / "manifest.json",
    ]
    if not all(item.exists() and item.stat().st_size > 0 for item in required):
        return False

    try:
        index = pd.read_parquet(path / "index.parquet", columns=["frame_index"])
    except Exception:
        return False
    return len(index) > 0


def write_root_manifest(
    *,
    output_root: Path,
    bags_root: Path,
    model: str,
    batch_size: int,
    device: str,
) -> None:
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "encoder_name": model,
        "encoder_source": "timm:hf-hub:MahmoodLab/UNI2-h",
        "output_format": "features.npz + index.parquet",
        "candidate_rule": "keep == true and patch_png_path != ''",
        "bags_root": str(bags_root),
        "patchset_source_root": str(bags_root / "patchsets"),
        "batch_size": batch_size,
        "device": device,
        "histokit_git_commit": git_commit(Path(__file__).resolve().parents[1]),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "manifest.json", "w") as file:
        json.dump(manifest, file, indent=2)


def summarize_embeddings(*, bags_root: Path, output_root: Path) -> None:
    slide_ids = list(iter_all_slide_ids(bags_root))
    rows = []
    for slide_id in slide_ids:
        path = output_root / "slides" / f"{slide_id}.uni2_h.embedding"
        complete = embedding_complete(path) if path.exists() else False
        row = {
            "slide_id": slide_id,
            "embedding_path": str(path),
            "complete": complete,
            "num_embeddings": 0,
            "embedding_dim": 0,
        }
        if complete:
            result = read_embedding_result(path)
            row["num_embeddings"] = int(result.features.shape[0])
            row["embedding_dim"] = (
                int(result.features.shape[1])
                if result.features.ndim == 2
                else 0
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "slide_index.csv", index=False)
    print(
        f"Complete embeddings: {int(summary['complete'].sum())} / "
        f"{len(summary)}"
    )


def write_slide_index(*, output_root: Path, slide_ids: list[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"slide_id": slide_ids}).to_csv(
        output_root / "slide_index.csv", index=False
    )


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
