from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
