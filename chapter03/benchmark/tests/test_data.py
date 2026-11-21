from pathlib import Path

import numpy as np
from PIL import Image

from icaird_cervical_benchmark.data import (
    discover_image_splits,
    load_feature_splits,
    validate_feature_splits,
)


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(path)


def test_discover_image_splits_requires_train_valid_test(tmp_path):
    for split in ("train", "valid", "test"):
        _image(tmp_path / split / "negative" / "a.png")
        _image(tmp_path / split / "positive" / "b.png")

    splits = discover_image_splits(tmp_path)

    assert splits.class_names == ["negative", "positive"]
    assert len(splits.train) == 2
    assert len(splits.valid) == 2
    assert len(splits.test) == 2


def test_discover_image_splits_can_use_patch_label_index(tmp_path):
    import pandas as pd

    rows = []
    for split in ("train", "valid", "test"):
        image_a = tmp_path / split / "folder_a" / f"{split}_a.png"
        image_b = tmp_path / split / "folder_b" / f"{split}_b.png"
        _image(image_a)
        _image(image_b)
        rows.extend(
            [
                {
                    "sample_id": f"{split}_a",
                    "split": split,
                    "patch_png_path": str(image_a),
                    "subcategory": "folder_a",
                    "annotation_label_name": "normal",
                },
                {
                    "sample_id": f"{split}_b",
                    "split": split,
                    "patch_png_path": str(image_b),
                    "subcategory": "folder_b",
                    "annotation_label_name": "malignant",
                },
            ]
        )
    index_path = tmp_path / "index.parquet"
    pd.DataFrame(rows).to_parquet(index_path)

    splits = discover_image_splits(
        tmp_path,
        label_index_path=index_path,
        label_column="annotation_label_name",
    )

    assert splits.class_names == ["malignant", "normal"]
    assert {record.class_name for record in splits.train} == {"malignant", "normal"}
    assert {record.path.parent.name for record in splits.train} == {"folder_a", "folder_b"}


def test_load_feature_split_directory_and_validate_dim(tmp_path):
    for split in ("train", "valid", "test"):
        np.savez(
            tmp_path / f"{split}.npz",
            features=np.ones((4, 3)),
            labels=np.array([0, 1, 0, 1]),
            sample_ids=np.array([f"{split}_{idx}" for idx in range(4)]),
        )

    splits = load_feature_splits(tmp_path)

    assert validate_feature_splits(splits) == 3
    assert splits.train.sample_ids[0] == "train_0"


def test_load_hproj_style_npz(tmp_path):
    path = tmp_path / "embeddings.npz"
    np.savez(
        path,
        train_features=np.ones((4, 5)),
        train_labels=np.array([0, 1, 0, 1]),
        valid_features=np.ones((4, 5)),
        valid_labels=np.array([0, 1, 0, 1]),
        test_features=np.ones((4, 5)),
        test_labels=np.array([0, 1, 0, 1]),
    )

    splits = load_feature_splits(path)

    assert validate_feature_splits(splits, expected_dim=5) == 5


def test_load_npz_with_parquet_index_directory(tmp_path):
    import pandas as pd

    features = np.arange(24).reshape(6, 4)
    np.savez(tmp_path / "features.npz", features=features)
    pd.DataFrame(
        {
            "sample_id": [f"id_{idx}" for idx in range(6)],
            "split": ["train", "train", "valid", "valid", "test", "test"],
            "subcategory": ["a", "b", "a", "b", "a", "b"],
            "annotation_label_name": ["x", "y", "x", "y", "x", "y"],
        }
    ).to_parquet(tmp_path / "index.parquet")

    splits = load_feature_splits(tmp_path)

    assert validate_feature_splits(splits) == 4
    assert splits.train.features.shape == (2, 4)
    assert splits.test.sample_ids == ["id_4", "id_5"]


def test_load_npz_with_parquet_index_uses_configured_label_column(tmp_path):
    import pandas as pd

    features = np.arange(24).reshape(6, 4)
    np.savez(tmp_path / "features.npz", features=features)
    pd.DataFrame(
        {
            "sample_id": [f"id_{idx}" for idx in range(6)],
            "split": ["train", "train", "valid", "valid", "test", "test"],
            "subcategory": ["a", "b", "a", "b", "a", "b"],
            "annotation_label_name": ["x", "y", "x", "y", "x", "y"],
        }
    ).to_parquet(tmp_path / "index.parquet")

    splits = load_feature_splits(tmp_path, label_column="annotation_label_name")

    assert splits.train.labels.tolist() == ["x", "y"]
    assert splits.test.labels.tolist() == ["x", "y"]
