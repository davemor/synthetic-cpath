from pathlib import Path
from typing import Any
import os
import re

import yaml

ENV_PATTERN = re.compile(r"\$\{oc\.env:([^,}]+),?([^}]*)\}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return data


def resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_PATTERN.sub(
            lambda m: os.environ.get(m.group(1), m.group(2)),
            value,
        )
    if isinstance(value, list):
        return [resolve_env(item) for item in value]
    if isinstance(value, dict):
        return {key: resolve_env(item) for key, item in value.items()}
    return value


def load_resolved_yaml(path: Path) -> dict[str, Any]:
    return resolve_env(load_yaml(path))
