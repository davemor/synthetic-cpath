from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def stable_hash(payload: Any, length: int = 24) -> str:
    text = json.dumps(make_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()[:length]


def make_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w") as handle:
        json.dump(make_jsonable(payload), handle, indent=2, sort_keys=True)
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(make_jsonable(row), sort_keys=True) + "\n")
    tmp.replace(path)


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    np.savez_compressed(tmp, **arrays)
    actual = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    actual.replace(path)

