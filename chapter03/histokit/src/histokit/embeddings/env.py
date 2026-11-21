"""Environment helpers for model downloads."""

from __future__ import annotations

import os
from pathlib import Path


HF_TOKEN_KEYS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
)


def read_dotenv(path: Path | str) -> dict[str, str]:
    """Read simple KEY=VALUE pairs from a dotenv file.

    This intentionally supports the common subset needed for access tokens
    without adding a runtime dependency on python-dotenv.
    """
    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def ensure_hf_token(env_file: Path | str = ".env") -> str | None:
    """Load a Hugging Face token from env or *env_file* and expose it to libs."""
    for key in HF_TOKEN_KEYS:
        token = os.environ.get(key)
        if token:
            _set_hf_token_env(token)
            return token

    values = read_dotenv(env_file)
    token = next(
        (values[key] for key in HF_TOKEN_KEYS if values.get(key)), None
    )
    if token:
        _set_hf_token_env(token)
    return token


def _set_hf_token_env(token: str) -> None:
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
