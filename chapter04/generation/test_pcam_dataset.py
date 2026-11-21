from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from synpath.data.prepare import build_pcam_hdf5_manifest
from synpath.paths import SynpathPaths


def write_pcam_split(root: Path, split: str, labels: list[int]) -> None:
    images = np.zeros((len(labels), 96, 96, 3), dtype=np.uint8)
    for idx, label in enumerate(labels):
        images[idx, :, :, :] = 64 + label * 96 + idx
    with h5py.File(root / f"camelyonpatch_level_2_split_{split}_x.h5", "w") as handle:
        handle.create_dataset("x", data=images)
    with h5py.File(root / f"camelyonpatch_level_2_split_{split}_y.h5", "w") as handle:
        handle.create_dataset("y", data=np.asarray(labels).reshape(-1, 1, 1, 1))


def test_build_pcam_hdf5_manifest_exports_pngs_and_maps_splits(tmp_path):
    raw_root = tmp_path / "raw" / "pcam"
    raw_root.mkdir(parents=True)
    write_pcam_split(raw_root, "train", [0, 1, 0, 1])
    write_pcam_split(raw_root, "valid", [0, 1])
    write_pcam_split(raw_root, "test", [1, 0])
    paths = SynpathPaths(
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        outputs_root=tmp_path / "outputs",
        reports_root=tmp_path / "reports",
        external_root=tmp_path / "external",
    )

    rows = build_pcam_hdf5_manifest(
        "pcam",
        {"root": str(raw_root)},
        paths,
        {"limits": {"max_train_images": 2}},
    )

    assert len(rows) == 6
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert {row["label"] for row in rows} == {"non_metastatic", "metastatic"}
    assert {row["label_id"] for row in rows} == {0, 1}
    train_rows = [row for row in rows if row["split"] == "train"]
    assert len(train_rows) == 2
    assert {row["label_id"] for row in train_rows} == {0, 1}
    for row in rows:
        image_path = Path(str(row["image_path"]))
        assert image_path.exists()
        assert Image.open(image_path).size == (96, 96)
