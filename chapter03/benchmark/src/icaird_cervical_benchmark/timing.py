from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Iterator


class TimerRecorder:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.records: list[dict] = []

    @contextmanager
    def time(self, stage: str, **metadata) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - started
            self.records.append({"stage": stage, "seconds": elapsed, **metadata})
            self.flush()

    def add(self, stage: str, seconds: float, **metadata) -> None:
        self.records.append({"stage": stage, "seconds": seconds, **metadata})
        self.flush()

    def flush(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "timings.jsonl"
        with path.open("w") as handle:
            for record in self.records:
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")

