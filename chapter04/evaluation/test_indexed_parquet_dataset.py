from pathlib import Path

import pandas as pd
from PIL import Image

from synpath.data.prepare import build_indexed_parquet_manifest


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(128, 64, 32)).save(path)


def test_build_indexed_parquet_manifest_uses_annotation_label_name(tmp_path):
    root = tmp_path / "dataset"
    image_root = root / "images"
    rows = []
    for split in ("train", "valid", "test"):
        for label in ("Negative", "Positive"):
            image_path = image_root / split / f"{label.lower()}.png"
            write_image(image_path)
            rows.append(
                {
                    "patch_png_path": str(image_path.relative_to(root)),
                    "annotation_label_name": label,
                    "split": split,
                    "slide_id": f"{split}_{label}",
                    "patient_id": f"patient_{label}",
                }
            )
    index_path = root / "index.parquet"
    pd.DataFrame(rows).to_parquet(index_path)

    manifest_rows = build_indexed_parquet_manifest(
        "demo",
        {
            "root": str(root),
            "image_root": str(image_root),
            "index_parquet": str(index_path),
            "label_column": "annotation_label_name",
        },
        {},
    )

    assert len(manifest_rows) == 6
    assert {row["label"] for row in manifest_rows} == {"negative", "positive"}
    assert {row["split"] for row in manifest_rows} == {"train", "val", "test"}
    assert {row["label_id"] for row in manifest_rows} == {0, 1}
    assert all(Path(str(row["image_path"])).exists() for row in manifest_rows)


def test_indexed_parquet_manifest_remaps_absolute_image_paths(tmp_path):
    root = tmp_path / "mounted"
    image_root = root / "images"
    image_path = image_root / "train" / "normal" / "patch.png"
    write_image(image_path)
    index_path = root / "embeddings" / "uni2_h" / "index.parquet"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "patch_png_path": "/home/dm236/scratch/data/example/images/train/normal/patch.png",
                "annotation_label_name": "Normal",
                "split": "train",
                "slide_id": "slide_a",
            }
        ]
    ).to_parquet(index_path)

    manifest_rows = build_indexed_parquet_manifest(
        "demo",
        {
            "root": str(root),
            "image_root": str(image_root),
            "index_parquet": str(index_path),
            "label_column": "annotation_label_name",
        },
        {},
    )

    assert len(manifest_rows) == 1
    assert manifest_rows[0]["image_path"] == str(image_path.resolve())
