from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from synpath.calibration.common import calibration_root
from synpath.calibration.curation import (
    OSORIO_DEFAULT_THRESHOLDS,
    apply_osorio_filters,
    curate_dataset,
)
from synpath.calibration.features import (
    FeatureArtifact,
    dino_processor_input,
    index_fingerprint,
    load_train_index,
    prepare_features,
    prepare_fit_transform,
)
from synpath.calibration.report import (
    build_report,
    collect_results,
    select_tracks,
)
from synpath.calibration.sd_index import (
    sd_index,
    sd_index_components,
    select_minimum,
)
from synpath.calibration.worker import score_with_s_dbw
from synpath.calibration.validity import (
    detect_elbow,
    sampled_silhouette_scores,
    select_maximum,
)
from synpath.paths import SynpathPaths


def test_sd_index_reference_calculation() -> None:
    features = np.asarray(
        [[0.0, 0.0], [0.0, 2.0], [10.0, 0.0], [10.0, 2.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1])
    components = sd_index_components(features, labels)

    assert components.scattering == pytest.approx(1.0 / np.sqrt(626.0))
    assert components.separation == pytest.approx(0.2)
    assert sd_index(components, alpha=0.2) == pytest.approx(
        0.2 / np.sqrt(626.0) + 0.2
    )


def test_select_minimum_uses_lower_k_for_tie() -> None:
    selected = select_minimum(
        [
            {"k": 4, "sd_index": 0.5},
            {"k": 2, "sd_index": 0.5},
            {"k": 3, "sd_index": 0.7},
        ]
    )
    assert selected["k"] == 2


def test_kneedle_detects_known_inertia_elbow() -> None:
    k_values = list(range(2, 11))
    inertias = [100.0, 61.0, 39.0, 31.0, 27.0, 24.5, 23.0, 22.0, 21.5]

    assert detect_elbow(k_values, inertias, sensitivity=1.0) == 4


def test_select_maximum_uses_lower_k_for_tie() -> None:
    selected = select_maximum(
        [
            {"k": 8, "silhouette_mean": 0.3},
            {"k": 4, "silhouette_mean": 0.3},
            {"k": 2, "silhouette_mean": 0.2},
        ],
        "silhouette_mean",
    )
    assert selected["k"] == 4


def test_sampled_silhouette_is_seeded_and_repeatable() -> None:
    features = np.arange(80, dtype=np.float32).reshape(40, 2)
    labels = np.repeat([0, 1], 20)

    def scorer(sample: np.ndarray, sample_labels: np.ndarray) -> float:
        return float(sample[:, 0].mean() + sample_labels.mean())

    first = sampled_silhouette_scores(
        features, labels, sample_size=10, seeds=[0, 1, 2], scorer=scorer
    )
    second = sampled_silhouette_scores(
        features, labels, sample_size=10, seeds=[0, 1, 2], scorer=scorer
    )

    assert first == second
    assert len(first) == 3


def test_training_index_excludes_held_out_rows(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    image_paths = []
    for name in ("train_a", "train_b", "val", "test"):
        path = tmp_path / f"{name}.png"
        path.write_bytes(b"image")
        image_paths.append(str(path))
    manifest = pd.DataFrame(
        {
            "image_path": image_paths,
            "label": ["a", "b", "a", "b"],
            "label_id": [0, 1, 0, 1],
            "split": ["train", "train", "val", "test"],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    cfg = {
        "datasets": {
            "tiny": {"manifest": str(manifest_path), "expected_train": 2}
        }
    }

    index, _ = load_train_index(cfg, "tiny", paths)

    assert set(index["split"]) == {"train"}
    assert set(index["sample_id"]) == {"train_a", "train_b"}


def test_split_local_filenames_do_not_trigger_false_leakage(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    train_root.mkdir()
    val_root.mkdir()
    train_path = train_root / "00000000.png"
    val_path = val_root / "00000000.png"
    train_path.write_bytes(b"train")
    val_path.write_bytes(b"validation")
    manifest = pd.DataFrame(
        {
            "image_path": [str(train_path), str(val_path)],
            "label": ["a", "a"],
            "label_id": [0, 0],
            "split": ["train", "val"],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    cfg = {
        "datasets": {
            "tiny": {"manifest": str(manifest_path), "expected_train": 1}
        }
    }

    index, _ = load_train_index(cfg, "tiny", paths)

    assert index.iloc[0]["image_path"] == str(train_path)
    assert set(index["split"]) == {"train"}


def test_osorio_dino_preprocessing_passes_scaled_tensor() -> None:
    image = Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8), mode="RGB")

    corrected = dino_processor_input(image, "model_creator_defaults")
    released = dino_processor_input(image, "osorio_released_tensor_rescale")

    assert corrected is image
    assert released.shape == (3, 8, 8)
    assert float(released.min()) == pytest.approx(1.0)
    assert float(released.max()) == pytest.approx(1.0)


def test_osorio_filters_use_strict_lower_thresholds() -> None:
    row = {
        "mean_V": 70.0,
        "mean_S": 3.0,
        "std_V": 4.0,
        "std_S": 4.0,
        "std_H": 1.5,
        "var_L": 100.0,
        "n_contour": 10,
    }
    boundary = apply_osorio_filters(
        pd.DataFrame([row]), OSORIO_DEFAULT_THRESHOLDS
    )
    assert not bool(boundary.iloc[0]["remove"])

    below = pd.DataFrame(
        [
            {**row, column: value - 0.01}
            for column, value in {
                "mean_V": 70.0,
                "mean_S": 3.0,
                "std_V": 4.0,
                "std_S": 4.0,
                "std_H": 1.5,
                "var_L": 100.0,
                "n_contour": 10,
            }.items()
        ]
    )
    filtered = apply_osorio_filters(below, OSORIO_DEFAULT_THRESHOLDS)
    assert filtered["remove"].all()


def test_osorio_curation_writes_aligned_cached_manifest(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path)
    image_root = tmp_path / "images"
    image_root.mkdir()
    rng = np.random.default_rng(0)
    images = {
        "white": np.full((96, 96, 3), 255, dtype=np.uint8),
        "noise": rng.integers(0, 256, (96, 96, 3), dtype=np.uint8),
        "test": rng.integers(0, 256, (96, 96, 3), dtype=np.uint8),
    }
    image_paths = {}
    for name, values in images.items():
        path = image_root / f"{name}.png"
        Image.fromarray(values).save(path)
        image_paths[name] = str(path)

    manifest = pd.DataFrame(
        {
            "image_path": [
                image_paths["white"],
                image_paths["noise"],
                image_paths["test"],
            ],
            "label": ["negative", "positive", "positive"],
            "label_id": [0, 1, 1],
            "split": ["train", "train", "test"],
            "dataset": ["pcam"] * 3,
        }
    )
    manifest_path = tmp_path / "pcam.csv"
    manifest.to_csv(manifest_path, index=False)
    cfg = {
        "cache_name": "curation-test",
        "datasets": {
            "pcam": {
                "manifest": str(manifest_path),
                "expected_train": 2,
                "curation": {
                    "kind": "osorio_pcam_foreground",
                    "workers": 1,
                },
            }
        },
    }

    artifact = curate_dataset(cfg, "pcam", paths, resume=False)
    assert artifact is not None
    curated = pd.read_csv(artifact.manifest_path)
    measured = pd.read_parquet(artifact.features_path)
    metadata = json.loads(artifact.metadata_path.read_text())
    assert list(curated["image_path"]) == [
        image_paths["noise"],
        image_paths["test"],
    ]
    assert bool(measured.loc[0, "remove_low_mean_S"])
    assert metadata["n_kept_train"] == 1
    assert metadata["n_removed_train"] == 1

    resumed = curate_dataset(cfg, "pcam", paths, resume=True)
    assert resumed == artifact


def test_ordered_uni2_reuse_requires_exact_alignment(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    images = []
    for index in range(4):
        image_path = tmp_path / f"sample_{index}.png"
        image_path.write_bytes(b"png")
        images.append(str(image_path))
    manifest = pd.DataFrame(
        {
            "image_path": images,
            "label": ["a", "a", "b", "b"],
            "label_id": [0, 0, 1, 1],
            "split": ["train"] * 4,
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    source_manifest = tmp_path / "source.csv"
    features_path = tmp_path / "features.npz"
    manifest.to_csv(manifest_path, index=False)
    manifest.to_csv(source_manifest, index=False)
    np.savez(
        features_path, features=np.arange(12, dtype=np.float32).reshape(4, 3)
    )
    cfg = {
        "datasets": {
            "tiny": {
                "manifest": str(manifest_path),
                "expected_train": 4,
                "uni2_reuse": {
                    "kind": "ordered_manifest",
                    "features": str(features_path),
                    "features_key": "features",
                    "manifest": str(source_manifest),
                },
            }
        }
    }

    artifact = prepare_features(cfg, "tiny", "uni2", paths, resume=False)
    metadata = json.loads(artifact.metadata_path.read_text())
    assert metadata["reused"] is True
    assert metadata["n_samples"] == 4

    source = pd.read_csv(source_manifest).iloc[::-1]
    source.to_csv(source_manifest, index=False)
    with pytest.raises(RuntimeError, match="does not align"):
        prepare_features(cfg, "tiny", "uni2", paths, resume=False)


def test_standard_scaler_fit_transform_is_global_and_cached(
    tmp_path: Path,
) -> None:
    feature_path = tmp_path / "train_features.npy"
    metadata_path = tmp_path / "features_metadata.json"
    index_path = tmp_path / "train_index.parquet"
    values = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32
    )
    np.save(feature_path, values)
    metadata_path.write_text(
        json.dumps({"feature_path": str(feature_path), "feature_key": None})
    )
    artifact = FeatureArtifact(feature_path, None, index_path, metadata_path)

    prepare_fit_transform(artifact, "standard_scaler", resume=False)
    metadata = json.loads(metadata_path.read_text())
    transformed = np.load(metadata["fit_feature_path"])
    scaler = np.load(metadata["standard_scaler_path"])
    assert transformed.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-7)
    assert transformed.std(axis=0) == pytest.approx([1.0, 1.0])
    assert scaler["mean"] == pytest.approx([2.0, 20.0])

    prepare_fit_transform(artifact, "standard_scaler", resume=True)


def test_s_dbw_score_matches_package_api() -> None:
    from s_dbw import SD

    features = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [9.0, 9.0], [9.0, 10.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
    scorer = {
        "package_version": "0.4.0",
        "k": 1.0,
        "alg_noise": "bind",
        "centr": "mean",
        "nearest_centr": True,
        "metric": "euclidean",
    }
    expected = SD(
        features,
        labels,
        k=1.0,
        centers_id=None,
        alg_noise="bind",
        centr="mean",
        nearest_centr=True,
        metric="euclidean",
    )
    assert score_with_s_dbw(features, labels, scorer) == pytest.approx(expected)


def test_report_selects_precomputed_s_dbw_minimum(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    cfg = {
        "k_min": 2,
        "k_max": 50,
        "datasets": {"tiny": {"display_name": "Tiny"}},
        "encoders": {"test": {"display_name": "Test"}},
        "sd_index_scorer": {"kind": "s_dbw_sd_0_4_0"},
    }
    fit_root = calibration_root(paths) / "tiny" / "test" / "fits"
    fit_root.mkdir(parents=True)
    for k in range(2, 51):
        (fit_root / f"k_{k:02d}.json").write_text(
            json.dumps(
                {
                    "k": k,
                    "sd_index": 1.0 + abs(k - 33) / 100.0,
                    "inertia": 1.0,
                    "cluster_sizes": [1] * k,
                    "n_samples": 100,
                    "runtime_seconds": 0.1,
                    "backend": "test",
                    "backend_version": "1",
                }
            )
        )

    results = pd.DataFrame(collect_results(cfg, paths))
    selected = select_tracks(cfg, results)
    assert selected.iloc[0]["k"] == 33
    assert np.isnan(selected.iloc[0]["alpha"])


def test_index_fingerprint_changes_with_order() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["one", "two"],
            "label": ["a", "b"],
            "label_id": [0, 1],
        }
    )
    assert index_fingerprint(frame) != index_fingerprint(frame.iloc[::-1])


def test_report_uses_dis_at_k_max_as_alpha(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    cfg = {
        "k_min": 2,
        "k_max": 50,
        "report_name": "calibration",
        "datasets": {"tiny": {"display_name": "Tiny"}},
        "encoders": {"test": {"display_name": "Test"}},
    }
    track_root = calibration_root(paths) / "tiny" / "test"
    fit_root = track_root / "fits"
    fit_root.mkdir(parents=True)
    index = pd.DataFrame(
        {
            "manifest_row": range(4),
            "sample_id": ["a", "b", "c", "d"],
            "image_path": ["a.png", "b.png", "c.png", "d.png"],
            "label": ["x"] * 4,
            "label_id": [0] * 4,
        }
    )
    index.to_parquet(track_root / "train_index.parquet", index=False)
    (track_root / "features_metadata.json").write_text(
        json.dumps({"manifest_fingerprint": "fingerprint"})
    )
    for k in range(2, 51):
        scattering = 0.1 + abs(k - 7) / 100.0
        separation = 2.0 if k == 50 else 0.5
        result = {
            "k": k,
            "scattering": scattering,
            "separation": separation,
            "inertia": 1.0,
            "cluster_sizes": [1] * k,
            "n_samples": 4,
            "runtime_seconds": 0.1,
            "backend": "test",
            "backend_version": "1",
        }
        (fit_root / f"k_{k:02d}.json").write_text(json.dumps(result))
        labels = np.asarray([index % k for index in range(4)], dtype=np.int32)
        if k > 4:
            labels = np.asarray([0, 1, 2, 3], dtype=np.int32)
        np.save(fit_root / f"k_{k:02d}_labels.npy", labels)
        np.save(fit_root / f"k_{k:02d}_centers.npy", np.zeros((k, 2)))

    # Make k=2 the valid selected result so assignment support can be checked.
    selected_result = json.loads((fit_root / "k_02.json").read_text())
    selected_result["scattering"] = 0.01
    (fit_root / "k_02.json").write_text(json.dumps(selected_result))
    outputs = build_report(cfg, paths)

    results = pd.read_csv(
        paths.reports_root / "calibration" / "sd_index_results.csv"
    )
    selected = pd.read_csv(
        paths.reports_root / "calibration" / "selected_k.csv"
    )
    assert set(results["alpha"]) == {2.0}
    assert selected.iloc[0]["k"] == 2
    assert all(path.is_file() for path in outputs)


def test_cluster_count_report_keeps_selectors_separate(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    cfg = {
        "cache_name": "cluster-validity",
        "report_name": "cluster-validity",
        "report_kind": "cluster_count_validity",
        "k_min": 2,
        "k_max": 50,
        "datasets": {
            "tiny": {
                "display_name": "Tiny training set",
                "expected_train": 60,
            }
        },
        "encoders": {"test": {"display_name": "Test encoder"}},
        "validity": {
            "elbow": {"sensitivity": 1.0},
            "silhouette": {"sample_size": 10, "seeds": [0, 1, 2]},
        },
        "sd_index_scorer": {
            "kind": "halkidi_dis_at_kmax",
            "scoring_space": "fit_features",
        },
    }
    track_root = calibration_root(paths, cfg) / "tiny" / "test"
    fit_root = track_root / "fits"
    fit_root.mkdir(parents=True)
    index = pd.DataFrame(
        {
            "manifest_row": range(60),
            "sample_id": [f"sample_{value}" for value in range(60)],
            "image_path": [f"sample_{value}.png" for value in range(60)],
            "label": ["unused"] * 60,
            "label_id": [0] * 60,
            "split": ["train"] * 60,
        }
    )
    index.to_parquet(track_root / "train_index.parquet", index=False)
    (track_root / "features_metadata.json").write_text(
        json.dumps(
            {
                "manifest_fingerprint": "training-only",
                "standard_scaler_path": "scaler.npz",
            }
        )
    )
    for k in range(2, 51):
        labels = np.arange(60, dtype=np.int32) % k
        counts = np.bincount(labels, minlength=k).astype(int).tolist()
        record = {
            "k": k,
            "scattering": 0.1 + abs(k - 9) / 100.0,
            "separation": 0.2,
            "inertia": 1000.0 / k + 5.0,
            "silhouette_scores": [
                0.3 - abs(k - 7) / 100.0,
                0.31 - abs(k - 7) / 100.0,
                0.29 - abs(k - 7) / 100.0,
            ],
            "silhouette_mean": 0.3 - abs(k - 7) / 100.0,
            "silhouette_std": 0.01,
            "silhouette_sample_size": 10,
            "cluster_sizes": counts,
            "n_samples": 60,
            "runtime_seconds": 0.1,
            "backend": "test",
            "backend_version": "1",
            "fit_transform": "standard_scaler",
            "scoring_space": "fit_features",
        }
        (fit_root / f"k_{k:02d}.json").write_text(json.dumps(record))
        np.save(fit_root / f"k_{k:02d}_labels.npy", labels)
        np.save(
            fit_root / f"k_{k:02d}_centers.npy",
            np.zeros((k, 2), dtype=np.float32),
        )

    outputs = build_report(cfg, paths)

    selected = pd.read_csv(paths.reports_root / "cluster-validity" / "selected_k.csv")
    assert int(selected.iloc[0]["silhouette_k"]) == 7
    assert int(selected.iloc[0]["sd_index_k"]) == 9
    assert all(path.is_file() for path in outputs)


def make_paths(root: Path) -> SynpathPaths:
    return SynpathPaths(
        data_root=root / "data",
        cache_root=root / "cache",
        outputs_root=root / "outputs",
        reports_root=root / "reports",
        external_root=root / "external",
    )
