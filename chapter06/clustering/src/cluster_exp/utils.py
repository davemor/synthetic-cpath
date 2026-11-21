from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=json_default)


def stable_hash(data: Any, length: int = 16) -> str:
    return hashlib.sha1(stable_json(data).encode("utf-8")).hexdigest()[:length]


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2, sort_keys=True, default=json_default)
            handle.write("\n")
        Path(tmp_name).replace(path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, default=json_default))
                handle.write("\n")
        Path(tmp_name).replace(path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ensure_int(value: Any) -> int:
    return int(float(value))


def path_fingerprint(path: Path) -> dict[str, Any]:
    """Cheap fingerprint for a file or tree without reading large embeddings."""
    path = Path(path)
    if path.is_file():
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    files = []
    if path.exists():
        for item in sorted(p for p in path.rglob("*") if p.is_file()):
            stat = item.stat()
            files.append(
                {
                    "relpath": str(item.relative_to(path)),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {"path": str(path), "files": files}
