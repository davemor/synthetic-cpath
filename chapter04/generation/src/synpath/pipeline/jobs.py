from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    stage: str
    dataset: str | None = None
    model: str | None = None
    conditioning: str | None = None
    seed: int | None = None
    source: str | None = None
    synthetic_model: str | None = None

    @property
    def key(self) -> str:
        parts = [
            self.stage,
            self.dataset,
            self.model,
            self.conditioning,
            f"seed{self.seed}" if self.seed is not None else None,
            self.source,
            self.synthetic_model,
        ]
        return "/".join(str(part) for part in parts if part)


@dataclass(frozen=True)
class StageResult:
    job: Job
    outputs: list[Path]
    skipped: bool = False
