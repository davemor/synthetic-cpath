from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import pandas as pd

from histokit.patchset.patchset import PatchSet
from ..params import resolve_value
from ..runtime import RuntimeContext
from ..stage import Stage


@dataclass(frozen=True)
class EmbedPatches(Stage):
    """Embed patches at the end of a pipeline."""

    model: str | Any = "resnet18"
    batch_size: int | Any = 64
    device: str | Any = "auto"
    only_kept: bool | Any = True
    env_file: str | Path | Any = ".env"
    output_dir: str | Path | Any | None = None
    append_to_frame: bool | Any = True
    prefix: str | Any = "embedding"
    overwrite: bool | Any = False

    def resolve(self, runtime: RuntimeContext) -> "EmbedPatches":
        values = {
            f.name: resolve_value(getattr(self, f.name), runtime)
            for f in fields(self)
        }
        return replace(self, **values)

    def run(self, patchset: PatchSet, runtime: RuntimeContext) -> PatchSet:
        if not isinstance(patchset, PatchSet):
            raise TypeError("EmbedPatches expects a PatchSet input.")
        if not self.append_to_frame and self.output_dir is None:
            raise ValueError(
                "EmbedPatches needs append_to_frame=True or an output_dir."
            )

        from histokit.embeddings.encoder import embed_patchset
        from histokit.embeddings.io import (
            safe_model_name,
            write_embedding_result,
        )

        result = embed_patchset(
            patchset,
            self.model,
            batch_size=self.batch_size,
            device=self.device,
            only_kept=self.only_kept,
            env_file=self.env_file,
            progress=False,
        )

        frame = patchset.frame.copy()
        if self.append_to_frame:
            frame = _append_features_to_frame(frame, result, prefix=self.prefix)

        if self.output_dir is not None:
            sample_id = _patchset_output_stem(patchset)
            target = (
                Path(self.output_dir)
                / f"{sample_id}.{safe_model_name(self.model)}.embedding"
            )
            write_embedding_result(result, target, overwrite=self.overwrite)
            frame[f"{self.prefix}_path"] = str(target)
            frame[f"{self.prefix}_index"] = pd.NA
            encoded_index = result.index["frame_index"].astype(int).to_numpy()
            frame.loc[encoded_index, f"{self.prefix}_index"] = range(
                len(encoded_index)
            )

        return PatchSet(
            frame=frame,
            contexts=patchset.contexts,
            manifest=patchset.manifest,
        )


def _append_features_to_frame(
    frame: pd.DataFrame,
    result: Any,
    *,
    prefix: str,
) -> pd.DataFrame:
    if result.features.size == 0:
        return frame

    columns = [f"{prefix}_{idx:04d}" for idx in range(result.features.shape[1])]
    features = pd.DataFrame(
        result.features,
        columns=columns,
        index=result.index["frame_index"].astype(int),
    )
    for column in columns:
        frame[column] = float("nan")
    frame.loc[features.index, columns] = features
    return frame


def _patchset_output_stem(patchset: PatchSet) -> str:
    if len(patchset.contexts) == 1:
        return patchset.contexts[0].sample.id
    sample_ids = [ctx.sample.id for ctx in patchset.contexts]
    if len(set(sample_ids)) == 1:
        return sample_ids[0]
    return "patchset"
