"""Patch embedding utilities."""

from .encoder import EmbeddingResult, embed_patchset
from .io import discover_patchset_dirs, read_embedding_result, write_embedding_result
from .manual_features import extract_manual_features, manual_feature_names
from .models import available_model_aliases, load_embedding_model

__all__ = [
    "EmbeddingResult",
    "available_model_aliases",
    "discover_patchset_dirs",
    "embed_patchset",
    "extract_manual_features",
    "load_embedding_model",
    "manual_feature_names",
    "read_embedding_result",
    "write_embedding_result",
]
