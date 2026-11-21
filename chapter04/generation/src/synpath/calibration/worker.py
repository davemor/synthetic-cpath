from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np

from synpath.artifacts import utc_now, write_json
from synpath.calibration.common import calibration_root, load_feature_array
from synpath.calibration.sd_index import sd_index_components
from synpath.calibration.validity import sampled_silhouette_scores
from synpath.config import load_resolved_yaml
from synpath.paths import SynpathPaths


def fit_k_values(
    config_path: Path,
    dataset: str,
    encoder: str,
    k_values: list[int],
    resume: bool,
    rescore_only: bool = False,
) -> None:
    cfg = load_resolved_yaml(config_path)
    kmeans_cfg = cfg["kmeans"]
    backend = str(kmeans_cfg.get("backend", "cuml"))
    if not rescore_only:
        if backend == "cuml":
            try:
                import cuml
                from cuml.cluster import KMeans
            except ImportError as exc:
                raise RuntimeError(
                    "RAPIDS cuML is required for this calibration; "
                    "no CPU fallback is used"
                ) from exc
            backend_version = cuml.__version__
        elif backend == "sklearn":
            import sklearn
            from sklearn.cluster import KMeans

            backend_version = sklearn.__version__
        else:
            raise RuntimeError(f"unsupported K-means backend: {backend}")

    paths = SynpathPaths.from_env()
    track_root = calibration_root(paths, cfg) / dataset / encoder
    metadata_path = track_root / "features_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"feature metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    feature_path = Path(metadata["feature_path"])
    features = load_feature_array(feature_path, metadata.get("feature_key"))
    if features.dtype != np.float32:
        features = np.asarray(features, dtype=np.float32)
    fit_feature_path = Path(metadata.get("fit_feature_path", feature_path))
    fit_features = load_feature_array(
        fit_feature_path, metadata.get("fit_feature_key")
    )
    if fit_features.dtype != np.float32:
        fit_features = np.asarray(fit_features, dtype=np.float32)
    if fit_features.shape != features.shape:
        raise RuntimeError("fit and scoring feature matrices do not align")
    scorer_cfg = cfg.get("sd_index_scorer", {"kind": "halkidi_dis_at_kmax"})
    validity_cfg = cfg.get("validity", {})
    silhouette_cfg = validity_cfg.get("silhouette")
    osorio_cfg = validity_cfg.get("osorio_s_dbw")
    scoring_space = str(scorer_cfg.get("scoring_space", "raw_embeddings"))
    scoring_features = (
        fit_features if scoring_space == "fit_features" else features
    )
    if scoring_space not in {"fit_features", "raw_embeddings"}:
        raise RuntimeError(f"unsupported SD-index scoring space: {scoring_space}")
    features64 = (
        np.asarray(scoring_features, dtype=np.float64)
        if rescore_only or scorer_cfg["kind"] == "halkidi_dis_at_kmax"
        else None
    )

    fit_root = track_root / "fits"
    fit_root.mkdir(parents=True, exist_ok=True)
    for k in k_values:
        result_path = fit_root / f"k_{k:02d}.json"
        labels_path = fit_root / f"k_{k:02d}_labels.npy"
        centers_path = fit_root / f"k_{k:02d}_centers.npy"
        if rescore_only:
            missing = [
                path
                for path in (result_path, labels_path, centers_path)
                if not path.is_file()
            ]
            if missing:
                raise RuntimeError(
                    f"cannot rescore {dataset}/{encoder}/k={k}; "
                    f"missing artifacts: {', '.join(map(str, missing))}"
                )
            started = time.perf_counter()
            labels = np.asarray(np.load(labels_path), dtype=np.int32).reshape(
                -1
            )
            if features64 is None:
                raise RuntimeError("rescore requires the direct Halkidi scorer")
            if len(labels) != len(features64):
                raise RuntimeError(
                    f"{dataset}/{encoder}/k={k} has {len(labels)} labels for "
                    f"{len(features64)} feature rows"
                )
            components = sd_index_components(features64, labels)
            counts = np.bincount(labels, minlength=k)
            if len(counts) != k or np.any(counts <= 0):
                raise RuntimeError(
                    f"{dataset}/{encoder}/k={k} contains an empty cluster"
                )
            result = json.loads(result_path.read_text())
            result.update(
                {
                    "scattering": components.scattering,
                    "separation": components.separation,
                    "cluster_sizes": counts.astype(int).tolist(),
                    "rescored_at": utc_now(),
                    "rescore_runtime_seconds": time.perf_counter() - started,
                    "scattering_dispersion": "population_variance",
                }
            )
            write_json(result_path, result)
            print(f"rescored {dataset}/{encoder}/k={k}", flush=True)
            continue
        if resume and fit_artifact_complete(
            result_path,
            labels_path,
            centers_path,
            silhouette_cfg,
            osorio_cfg,
            dataset,
            encoder,
        ):
            print(f"skipped {dataset}/{encoder}/k={k}", flush=True)
            continue

        started = time.perf_counter()
        model_kwargs = {
            "n_clusters": k,
            "init": kmeans_cfg["init"],
            "n_init": int(kmeans_cfg["n_init"]),
            "max_iter": int(kmeans_cfg["max_iter"]),
            "random_state": int(kmeans_cfg["random_state"]),
        }
        if backend == "cuml":
            model_kwargs["output_type"] = "numpy"
        elif "algorithm" in kmeans_cfg:
            model_kwargs["algorithm"] = str(kmeans_cfg["algorithm"])
        model = KMeans(
            **model_kwargs,
        )
        labels = np.asarray(
            model.fit_predict(fit_features), dtype=np.int32
        ).reshape(-1)
        centers = np.asarray(model.cluster_centers_, dtype=np.float32)
        if scorer_cfg["kind"] == "s_dbw_sd_0_4_0":
            score = score_with_s_dbw(features, labels, scorer_cfg)
            components = None
        elif scorer_cfg["kind"] == "halkidi_dis_at_kmax":
            score = None
            components = sd_index_components(features64, labels)
        else:
            raise RuntimeError(
                f"unsupported SD-index scorer: {scorer_cfg['kind']}"
            )
        silhouette_scores = None
        if silhouette_cfg:
            silhouette_scores = sampled_silhouette_scores(
                fit_features,
                labels,
                int(silhouette_cfg["sample_size"]),
                [int(seed) for seed in silhouette_cfg["seeds"]],
                silhouette_scorer(backend),
            )
        osorio_score = None
        if osorio_audit_enabled(osorio_cfg, dataset, encoder):
            osorio_score = score_with_s_dbw(features, labels, osorio_cfg)
        counts = np.bincount(labels, minlength=k)
        if len(counts) != k or np.any(counts <= 0):
            raise RuntimeError(
                f"{dataset}/{encoder}/k={k} produced an empty cluster"
            )

        np.save(labels_path, labels)
        np.save(centers_path, centers)
        result = {
            "completed_at": utc_now(),
            "dataset": dataset,
            "encoder": encoder,
            "k": k,
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
            "cluster_sizes": counts.astype(int).tolist(),
            "labels_path": str(labels_path),
            "centers_path": str(centers_path),
            "manifest_fingerprint": metadata["manifest_fingerprint"],
            "n_samples": int(len(features)),
            "runtime_seconds": time.perf_counter() - started,
            "backend": backend,
            "backend_version": backend_version,
            "python_version": platform.python_version(),
            "kmeans": kmeans_cfg,
            "scattering_dispersion": "population_variance",
            "fit_feature_path": str(fit_feature_path),
            "fit_transform": metadata.get("fit_transform", "none"),
            "scoring_feature_path": str(
                fit_feature_path
                if scoring_space == "fit_features"
                else feature_path
            ),
            "scoring_space": scoring_space,
            "sd_index_scorer": scorer_cfg,
        }
        if components is not None:
            result.update(
                {
                    "scattering": components.scattering,
                    "separation": components.separation,
                }
            )
        else:
            result["sd_index"] = score
            result["scattering_dispersion"] = None
        if silhouette_scores is not None:
            result.update(
                {
                    "silhouette_scores": silhouette_scores,
                    "silhouette_mean": float(np.mean(silhouette_scores)),
                    "silhouette_std": float(np.std(silhouette_scores)),
                    "silhouette_sample_size": min(
                        int(silhouette_cfg["sample_size"]), len(fit_features)
                    ),
                    "silhouette_seeds": [
                        int(seed) for seed in silhouette_cfg["seeds"]
                    ],
                    "silhouette_scoring_space": "fit_features",
                }
            )
        if osorio_score is not None:
            result["osorio_sd_index"] = osorio_score
            result["osorio_sd_scoring_space"] = "raw_embeddings"
        write_json(result_path, result)
        print(f"completed {dataset}/{encoder}/k={k}", flush=True)


def silhouette_scorer(backend: str):
    if backend == "cuml":
        from cuml.metrics.cluster import silhouette_score

        return lambda features, labels: float(
            silhouette_score(features, labels, metric="euclidean")
        )
    from sklearn.metrics import silhouette_score

    return lambda features, labels: float(
        silhouette_score(features, labels, metric="euclidean")
    )


def osorio_audit_enabled(
    cfg: dict | None, dataset: str, encoder: str
) -> bool:
    if not cfg:
        return False
    return f"{dataset}/{encoder}" in set(cfg.get("enabled_tracks", []))


def fit_artifact_complete(
    result_path: Path,
    labels_path: Path,
    centers_path: Path,
    silhouette_cfg: dict | None,
    osorio_cfg: dict | None,
    dataset: str,
    encoder: str,
) -> bool:
    if not all(path.is_file() for path in (result_path, labels_path, centers_path)):
        return False
    result = json.loads(result_path.read_text())
    if silhouette_cfg and "silhouette_scores" not in result:
        return False
    if osorio_audit_enabled(osorio_cfg, dataset, encoder):
        return "osorio_sd_index" in result
    return True


def score_with_s_dbw(
    features: np.ndarray,
    labels: np.ndarray,
    scorer_cfg: dict,
) -> float:
    from s_dbw import SD

    package_version = version("s-dbw")
    expected_version = str(scorer_cfg["package_version"])
    if package_version != expected_version:
        raise RuntimeError(
            f"s-dbw {expected_version} is required; found {package_version}"
        )
    return float(
        SD(
            features,
            labels,
            k=float(scorer_cfg.get("k", 1.0)),
            centers_id=None,
            alg_noise=str(scorer_cfg.get("alg_noise", "bind")),
            centr=str(scorer_cfg.get("centr", "mean")),
            nearest_centr=bool(scorer_cfg.get("nearest_centr", True)),
            metric=str(scorer_cfg.get("metric", "euclidean")),
        )
    )


def parse_k_values(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item.strip()})
    if not values:
        raise ValueError("at least one k value is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="RAPIDS SD-index fit worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--k-values", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rescore-only", action="store_true")
    args = parser.parse_args()
    fit_k_values(
        args.config,
        args.dataset,
        args.encoder,
        parse_k_values(args.k_values),
        args.resume,
        args.rescore_only,
    )


if __name__ == "__main__":
    main()
