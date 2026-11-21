from pathlib import Path

import numpy as np

from synpath.paths import SynpathPaths


def calibration_root(paths: SynpathPaths, cfg: dict | None = None) -> Path:
    name = (cfg or {}).get("cache_name", "cluster_sd_index_calibration")
    return paths.cache_root / str(name)


def load_feature_array(path: Path, key: str | None) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    loaded = np.load(path, mmap_mode="r")
    if key is None or key not in loaded:
        raise RuntimeError(f"{path} does not contain feature key {key!r}")
    return loaded[key]
