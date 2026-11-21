from __future__ import annotations

import importlib
from typing import Any


def validate_runtime(require_cuml: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {}
    if require_cuml:
        cupy = importlib.import_module("cupy")
        cuml = importlib.import_module("cuml")
        report["cupy"] = getattr(cupy, "__version__", "unknown")
        report["cuml"] = getattr(cuml, "__version__", "unknown")
        cudf = importlib.import_module("cudf")
        cugraph = importlib.import_module("cugraph")
        report["cudf"] = getattr(cudf, "__version__", "unknown")
        report["cugraph"] = getattr(cugraph, "__version__", "unknown")
        device_count = int(cupy.cuda.runtime.getDeviceCount())
        report["cuda_device_count"] = device_count
        if device_count < 1:
            raise RuntimeError("No CUDA devices visible to CuPy.")
        importlib.import_module("cuml.manifold").UMAP
        importlib.import_module("cuml.cluster").KMeans
        hdbscan_mod = importlib.import_module("cuml.cluster.hdbscan")
        hdbscan_mod.HDBSCAN
        if not hasattr(hdbscan_mod, "approximate_predict"):
            raise RuntimeError("cuML HDBSCAN approximate_predict is unavailable.")
        report["hdbscan_approximate_predict"] = True
        if not hasattr(cugraph, "leiden"):
            raise RuntimeError("cuGraph Leiden is unavailable.")
        report["cugraph_leiden"] = True
    return report
