"""PatchSet image embedding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from histokit.io.slides.slide import Region
from histokit.patchset.patchset import PatchSet

from .models import LoadedEmbeddingModel, load_embedding_model


@dataclass
class EmbeddingResult:
    """Embeddings and metadata for a PatchSet."""

    features: np.ndarray
    index: pd.DataFrame
    manifest: dict[str, Any]


def embed_patchset(
    patchset: PatchSet,
    model: str | LoadedEmbeddingModel,
    *,
    batch_size: int = 64,
    device: str = "auto",
    only_kept: bool = True,
    env_file: Path | str = ".env",
    limit: int | None = None,
    progress: bool = True,
) -> EmbeddingResult:
    """Embed patches by reading images directly from the PatchSet contexts."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    loaded = (
        model
        if isinstance(model, LoadedEmbeddingModel)
        else load_embedding_model(model, device=device, env_file=env_file)
    )

    selected = _select_frame(patchset, only_kept=only_kept, limit=limit)
    if selected.empty:
        return EmbeddingResult(
            features=np.empty((0, 0), dtype=np.float32),
            index=selected,
            manifest=_build_manifest(
                patchset=patchset,
                model=loaded,
                embedding_dim=0,
                num_embeddings=0,
                only_kept=only_kept,
            ),
        )

    torch = _require_torch()
    batches: list[np.ndarray] = []
    total = int(np.ceil(len(selected) / batch_size))
    iterator = _iter_image_batches(
        patchset=patchset,
        selected=selected,
        preprocess=loaded.preprocess,
        batch_size=batch_size,
        torch=torch,
    )
    if progress:
        iterator = tqdm(iterator, total=total, desc="Embedding patches")

    with torch.inference_mode():
        for batch in iterator:
            output = loaded.encode_tensor(batch)
            tensor = _tensor_from_output(output)
            tensor = tensor.detach().float().cpu()
            batches.append(tensor.numpy().astype(np.float32, copy=False))

    features = np.concatenate(batches, axis=0)
    manifest = _build_manifest(
        patchset=patchset,
        model=loaded,
        embedding_dim=int(features.shape[1]) if features.ndim == 2 else 0,
        num_embeddings=int(features.shape[0]),
        only_kept=only_kept,
    )
    return EmbeddingResult(features=features, index=selected, manifest=manifest)


def _select_frame(
    patchset: PatchSet,
    *,
    only_kept: bool,
    limit: int | None,
) -> pd.DataFrame:
    frame = patchset.frame
    if only_kept and "keep" in frame.columns:
        frame = frame[frame["keep"]]

    selected = frame.copy()
    if limit is not None:
        selected = selected.head(limit)

    selected.insert(0, "frame_index", selected.index.astype(int))
    selected["sample_id"] = selected["context_id"].map(
        lambda idx: patchset.contexts[int(idx)].sample.id
    )
    return selected.reset_index(drop=True)


def _iter_image_batches(
    *,
    patchset: PatchSet,
    selected: pd.DataFrame,
    preprocess: Any,
    batch_size: int,
    torch: Any,
):
    for context_id, group in selected.groupby("context_id", sort=False):
        ctx = patchset.contexts[int(context_id)]
        with ctx.sample.open_slide() as slide:
            downsample = float(slide.level_downsamples()[ctx.level])
            for start in range(0, len(group), batch_size):
                chunk = group.iloc[start : start + batch_size]
                tensors = []
                for _, row in chunk.iterrows():
                    patch = _read_patch_image(row, ctx, slide, downsample)
                    tensors.append(preprocess(patch))
                yield torch.stack(tensors, dim=0)


def _read_patch_image(
    row: pd.Series,
    ctx: Any,
    slide: Any,
    downsample: float,
) -> Any:
    x0 = int(int(row["column"]) * ctx.patch_size * downsample)
    y0 = int(int(row["row"]) * ctx.patch_size * downsample)
    region = Region.patch(x0, y0, ctx.patch_size, ctx.level)
    return slide.read_region(region).convert("RGB")


def _tensor_from_output(output: Any) -> Any:
    tensor = None
    if isinstance(output, dict):
        for key in ("pooler_output", "last_hidden_state", "logits"):
            value = output.get(key)
            if value is not None:
                tensor = value
                break
    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        tensor = output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        tensor = output.last_hidden_state
    elif hasattr(output, "logits"):
        tensor = output.logits
    elif isinstance(output, (tuple, list)) and output:
        tensor = output[0]
    else:
        tensor = output

    if tensor is None:
        raise TypeError(
            "Could not extract an embedding tensor from model output."
        )

    if tensor.ndim == 3:
        tensor = tensor[:, 0]
    elif tensor.ndim > 3:
        tensor = tensor.reshape(tensor.shape[0], -1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _build_manifest(
    *,
    patchset: PatchSet,
    model: LoadedEmbeddingModel,
    embedding_dim: int,
    num_embeddings: int,
    only_kept: bool,
) -> dict[str, Any]:
    return {
        "model": {
            "name": model.name,
            "backend": model.backend,
            "source": model.source,
            "device": model.device,
        },
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "only_kept": only_kept,
        "patchset": {
            "num_patches": len(patchset.frame),
            "contexts": [
                {
                    "sample_id": ctx.sample.id,
                    "level": ctx.level,
                    "patch_size": ctx.patch_size,
                }
                for ctx in patchset.contexts
            ],
        },
    }


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Patch embedding requires torch. Install the embedding extras with "
            "`uv pip install -e '.[embed]'` or install torch manually."
        ) from exc
    return torch
