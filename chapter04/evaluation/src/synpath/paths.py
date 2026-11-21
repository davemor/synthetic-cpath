from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class SynpathPaths:
    data_root: Path
    cache_root: Path
    outputs_root: Path
    reports_root: Path
    external_root: Path

    @classmethod
    def from_env(cls) -> "SynpathPaths":
        return cls(
            data_root=Path(os.environ.get("SYNPATH_DATA_ROOT", "data")),
            cache_root=Path(os.environ.get("SYNPATH_CACHE_ROOT", "data/cache")),
            outputs_root=Path(os.environ.get("SYNPATH_OUTPUTS_ROOT", "outputs")),
            reports_root=Path(os.environ.get("SYNPATH_REPORTS_ROOT", "reports")),
            external_root=Path(
                os.environ.get("SYNPATH_EXTERNAL_ROOT", "external")
            ),
        )
