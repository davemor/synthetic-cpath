import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
import pandas as pd

if TYPE_CHECKING:
    from .feature_space import FeatureSpace

load_dotenv()


class OutputPath:
    def mkdir(self):
        return self.root.mkdir(parents=True, exist_ok=True)


@dataclass
class Paths:
    embeddings_root: Path
    data_root: Path

    @classmethod
    def from_env(cls):
        return cls(
            embeddings_root=Path(os.getenv("LOWGEN_EMBEDDINGS_ROOT")),
            data_root=Path(os.getenv("LOWGEN_DATA_ROOT")),
        )

    def run(self, id: str) -> "RunPath":
        return RunPath(
            self.data_root / "runs" / id,
        )

    def embedding(self, dataset: str, encoder: str) -> "EmbeddingsPath":
        folder = f"cache-{dataset}-{encoder}"
        return EmbeddingsPath(
            self.embeddings_root / folder,
        )


# ============================================================================
# Embedding Paths
# ============================================================================


@dataclass
class EmbeddingsPath:
    root: Path

    def label_map(self) -> Path:
        return self.root / "label_map.json"

    def split(self, split: str) -> "EmbeddingsSplitPath":
        return EmbeddingsSplitPath(self.root / split)

    def load_splits(self) -> tuple["FeatureSpace", "FeatureSpace"]:
        from .feature_space import get_train_test

        return get_train_test(self)

    def load_label_map(self) -> dict[int, str]:
        with self.label_map().open() as f:
            return json.load(f)


@dataclass
class EmbeddingsSplitPath:
    root: Path

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def emb_stats(self) -> Path:
        return self.root / "emb_stats.pt"

    def emb(self, shard: int | str) -> Path:
        return self.root / f"emb_{shard}.pt"

    def labels(self, shard: int | str) -> Path:
        return self.root / f"labels_{shard}.pt"

    def paths_json(self, shard: int | str) -> Path:
        return self.root / f"paths_{shard}.json"

    def glob_embs(self) -> list[Path]:
        return list(self.root.glob("emb_[0-9][0-9][0-9][0-9].pt"))

    def glob_labels(self) -> list[Path]:
        return list(self.root.glob("labels_[0-9][0-9][0-9][0-9].pt"))

    def glob_paths_json(self) -> list[Path]:
        return list(self.root.glob("paths_[0-9][0-9][0-9][0-9].json"))

    def load(self) -> "FeatureSpace":
        from .feature_space import load_embeddings

        return load_embeddings(self)


# ============================================================================
# Run Paths
# ============================================================================


@dataclass
class RunPath(OutputPath):
    root: Path

    def config(self) -> Path:
        return self.root / "config.yml"

    def log_file(self) -> Path:
        return self.root / "log.txt"

    def results_csv(self) -> Path:
        return self.root / "results.csv"

    def reducer_run(
        self, dataset: str, encoder: str, reducer_key: str
    ) -> "ReducerRunPath":
        return ReducerRunPath(self.root / "experiments" / dataset / encoder / reducer_key)

    def synthesis(self) -> "SynthesisPaths":
        return SynthesisPaths(self.root / "synthesis")

    def report(self) -> "ReportPaths":
        return ReportPaths(self.root / "report")


@dataclass
class ReducerRunPath(OutputPath):
    root: Path

    def reducer_pkl(self) -> Path:
        return self.root / "reducer.pkl"

    def phase1_metrics(self) -> Path:
        return self.root / "phase1.json"

    def phase2_metrics(self) -> Path:
        return self.root / "phase2.json"

    def generator_run(self, generator_key: str, seed_idx: int) -> "GeneratorRunPath":
        return GeneratorRunPath(self.root / "generators" / generator_key / f"seed_{seed_idx}")


@dataclass
class GeneratorRunPath(OutputPath):
    root: Path

    def generator_pkl(self) -> Path:
        return self.root / "generator.pkl"

    def fidelity_metrics(self) -> Path:
        return self.root / "fidelity.json"

    def utility_metrics(self) -> Path:
        return self.root / "utility.json"


@dataclass
class ReportPaths(OutputPath):
    root: Path

    def plots_dir(self) -> Path:
        return self.root / "plots"


@dataclass
class SynthesisPaths(OutputPath):
    root: Path

    def dataset(self, dataset: str) -> Path:
        return self.root / "datasets" / dataset
