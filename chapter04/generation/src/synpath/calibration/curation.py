from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from synpath.artifacts import file_sha256, utc_now, write_json
from synpath.calibration.common import calibration_root
from synpath.paths import SynpathPaths


OSORIO_SOURCE_REPOSITORY = "https://github.com/Bayer-Group/rods-biogen"
OSORIO_SOURCE_COMMIT = "09325c9be310f3dee2992b1164f7acdf310ff48b"
OSORIO_DEFAULT_THRESHOLDS = {
    "th_V_max": 70.0,
    "th_S_max": 3.0,
    "th_std_V": 4.0,
    "th_std_S": 4.0,
    "th_std_H": 1.5,
    "th_var_L": 100.0,
    "th_n_contour": 10,
}
FEATURE_COLUMNS = [
    "mean_V",
    "median_V",
    "std_V",
    "mean_S",
    "median_S",
    "std_S",
    "std_H",
    "var_L",
    "n_contour",
]
REMOVAL_COLUMNS = [
    "remove_low_mean_V",
    "remove_low_mean_S",
    "remove_low_std_V",
    "remove_low_std_S",
    "remove_low_std_H",
    "remove_low_var_L",
    "remove_low_n_contour",
]


@dataclass(frozen=True)
class CurationArtifact:
    manifest_path: Path
    features_path: Path
    metadata_path: Path


def curate_dataset(
    cfg: dict,
    dataset: str,
    paths: SynpathPaths,
    resume: bool,
) -> CurationArtifact | None:
    spec = cfg["datasets"][dataset]
    curation = spec.get("curation")
    if not curation:
        return None
    if curation.get("kind") != "osorio_pcam_foreground":
        raise RuntimeError(
            f"unsupported calibration curation kind: {curation.get('kind')}"
        )

    source_manifest = Path(spec["manifest"])
    source_sha256 = file_sha256(source_manifest)
    root = calibration_root(paths, cfg) / dataset / "curation"
    root.mkdir(parents=True, exist_ok=True)
    artifact = CurationArtifact(
        manifest_path=root / "curated_manifest.csv",
        features_path=root / "curation_features.parquet",
        metadata_path=root / "curation_metadata.json",
    )
    thresholds = resolved_thresholds(curation.get("thresholds", {}))
    image_mode = str(curation.get("image_mode", "rgb"))
    if image_mode not in {"rgb", "source_native"}:
        raise RuntimeError(
            f"unsupported Osorio curation image mode: {image_mode}"
        )
    if resume and artifact.metadata_path.is_file():
        validate_curation_artifact(
            artifact,
            source_manifest,
            source_sha256,
            thresholds,
            image_mode,
        )
        return artifact

    frame = pd.read_csv(source_manifest)
    required = {"image_path", "label", "label_id", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"{source_manifest} is missing columns: {', '.join(sorted(missing))}"
        )
    train = frame.loc[frame["split"] == "train"].copy()
    expected = int(spec["expected_train"])
    if len(train) != expected:
        raise RuntimeError(
            f"{dataset} has {len(train)} source train rows; expected {expected}"
        )
    missing_images = [
        image_path
        for image_path in train["image_path"]
        if not Path(image_path).is_file()
    ]
    if missing_images:
        raise RuntimeError(
            f"{dataset} has {len(missing_images)} missing source images; "
            f"first missing: {missing_images[0]}"
        )

    workers = int(curation.get("workers", 1))
    values = extract_osorio_features(
        train["image_path"].astype(str),
        workers=workers,
        image_mode=image_mode,
    )
    measured = train.reset_index().rename(columns={"index": "source_row"})
    measured = pd.concat(
        [
            measured.reset_index(drop=True),
            pd.DataFrame(values, columns=FEATURE_COLUMNS),
        ],
        axis=1,
    )
    measured = apply_osorio_filters(measured, thresholds)
    measured.to_parquet(artifact.features_path, index=False)

    kept_source_rows = set(
        measured.loc[~measured["remove"], "source_row"].astype(int)
    )
    curated = frame.loc[
        (frame["split"] != "train") | frame.index.isin(kept_source_rows)
    ].copy()
    curated.to_csv(artifact.manifest_path, index=False)

    reason_counts = {
        column: int(measured[column].sum()) for column in REMOVAL_COLUMNS
    }
    kept_by_label = (
        measured.loc[~measured["remove"]]
        .groupby("label", observed=True)
        .size()
        .astype(int)
        .to_dict()
    )
    removed_by_label = (
        measured.loc[measured["remove"]]
        .groupby("label", observed=True)
        .size()
        .astype(int)
        .to_dict()
    )
    write_json(
        artifact.metadata_path,
        {
            "created_at": utc_now(),
            "dataset": dataset,
            "kind": "osorio_pcam_foreground",
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": source_sha256,
            "source_train_count": len(train),
            "n_kept_train": int((~measured["remove"]).sum()),
            "n_removed_train": int(measured["remove"].sum()),
            "kept_by_label": kept_by_label,
            "removed_by_label": removed_by_label,
            "overlapping_removal_reason_counts": reason_counts,
            "thresholds": thresholds,
            "comparison_operator": "strictly_less_than",
            "pillow_hsv_and_grayscale": True,
            "image_mode": image_mode,
            "contour_threshold": 150,
            "contour_mode": "cv2.RETR_TREE",
            "contour_method": "cv2.CHAIN_APPROX_NONE",
            "laplacian_depth": "cv2.CV_64F",
            "opencv_version": cv2.__version__,
            "source_repository": OSORIO_SOURCE_REPOSITORY,
            "source_commit": OSORIO_SOURCE_COMMIT,
            "features_path": str(artifact.features_path),
            "curated_manifest_path": str(artifact.manifest_path),
        },
    )
    validate_curation_artifact(
        artifact,
        source_manifest,
        source_sha256,
        thresholds,
        image_mode,
    )
    return artifact


def extract_osorio_features(
    image_paths: Iterable[str], workers: int, image_mode: str = "rgb"
) -> list[tuple[float, ...]]:
    paths = list(image_paths)
    extractor = partial(extract_osorio_image_features, image_mode=image_mode)
    if workers <= 1:
        return [extractor(path) for path in tqdm(paths)]
    with Pool(processes=workers) as pool:
        return list(
            tqdm(
                pool.imap(extractor, paths, chunksize=64),
                total=len(paths),
                desc="Osorio PCam curation",
            )
        )


def extract_osorio_image_features(
    image_path: str, image_mode: str = "rgb"
) -> tuple[float, ...]:
    with Image.open(image_path) as source:
        image = (
            source if image_mode == "source_native" else source.convert("RGB")
        )
        hsv = np.asarray(image.convert("HSV"))
        grayscale = np.asarray(image.convert("L"))

    _, thresholded = cv2.threshold(grayscale, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresholded,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )
    laplacian_variance = cv2.Laplacian(grayscale, cv2.CV_64F).var()
    return (
        float(np.mean(hsv[:, :, 2])),
        float(np.median(hsv[:, :, 2])),
        float(np.std(hsv[:, :, 2])),
        float(np.mean(hsv[:, :, 1])),
        float(np.median(hsv[:, :, 1])),
        float(np.std(hsv[:, :, 1])),
        float(np.std(hsv[:, :, 0])),
        float(laplacian_variance),
        len(contours),
    )


def apply_osorio_filters(
    frame: pd.DataFrame, thresholds: dict[str, float | int]
) -> pd.DataFrame:
    output = frame.copy()
    output["remove_low_mean_V"] = output["mean_V"] < thresholds["th_V_max"]
    output["remove_low_mean_S"] = output["mean_S"] < thresholds["th_S_max"]
    output["remove_low_std_V"] = output["std_V"] < thresholds["th_std_V"]
    output["remove_low_std_S"] = output["std_S"] < thresholds["th_std_S"]
    output["remove_low_std_H"] = output["std_H"] < thresholds["th_std_H"]
    output["remove_low_var_L"] = output["var_L"] < thresholds["th_var_L"]
    output["remove_low_n_contour"] = (
        output["n_contour"] < thresholds["th_n_contour"]
    )
    output["remove"] = output[REMOVAL_COLUMNS].any(axis=1)
    return output


def resolved_thresholds(
    configured: dict,
) -> dict[str, float | int]:
    unknown = set(configured) - set(OSORIO_DEFAULT_THRESHOLDS)
    if unknown:
        raise RuntimeError(
            f"unknown Osorio curation thresholds: {', '.join(sorted(unknown))}"
        )
    return {**OSORIO_DEFAULT_THRESHOLDS, **configured}


def validate_curation_artifact(
    artifact: CurationArtifact,
    source_manifest: Path,
    source_sha256: str,
    thresholds: dict[str, float | int],
    image_mode: str = "rgb",
) -> None:
    for path in (
        artifact.manifest_path,
        artifact.features_path,
        artifact.metadata_path,
    ):
        if not path.is_file():
            raise RuntimeError(f"curation artifact is missing: {path}")
    metadata = json.loads(artifact.metadata_path.read_text())
    if metadata["source_manifest"] != str(source_manifest):
        raise RuntimeError(
            "cached curation source manifest path does not match"
        )
    if metadata["source_manifest_sha256"] != source_sha256:
        raise RuntimeError(
            "cached curation source manifest hash does not match"
        )
    if metadata["thresholds"] != thresholds:
        raise RuntimeError("cached curation thresholds do not match")
    if metadata.get("image_mode", "rgb") != image_mode:
        raise RuntimeError("cached curation image mode does not match")
    measured = pd.read_parquet(artifact.features_path)
    curated = pd.read_csv(artifact.manifest_path)
    if len(measured) != int(metadata["source_train_count"]):
        raise RuntimeError(
            "cached curation measurements have the wrong row count"
        )
    kept = int((~measured["remove"]).sum())
    curated_train = int((curated["split"] == "train").sum())
    if kept != int(metadata["n_kept_train"]) or curated_train != kept:
        raise RuntimeError(
            "cached curation manifest and measurements do not align"
        )


def curated_manifest_for_dataset(
    cfg: dict, dataset: str, paths: SynpathPaths
) -> tuple[Path, int] | None:
    if not cfg["datasets"][dataset].get("curation"):
        return None
    root = calibration_root(paths, cfg) / dataset / "curation"
    metadata_path = root / "curation_metadata.json"
    manifest_path = root / "curated_manifest.csv"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"curation is required for {dataset}; run the curate stage first"
        )
    metadata = json.loads(metadata_path.read_text())
    source_manifest = Path(cfg["datasets"][dataset]["manifest"])
    validate_curation_artifact(
        CurationArtifact(
            manifest_path,
            root / "curation_features.parquet",
            metadata_path,
        ),
        source_manifest,
        file_sha256(source_manifest),
        resolved_thresholds(
            cfg["datasets"][dataset]["curation"].get("thresholds", {})
        ),
        str(cfg["datasets"][dataset]["curation"].get("image_mode", "rgb")),
    )
    return manifest_path, int(metadata["n_kept_train"])
