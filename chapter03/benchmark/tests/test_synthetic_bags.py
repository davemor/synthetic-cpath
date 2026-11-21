from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from icaird_cervical_benchmark.config import SyntheticBagConfig
from icaird_cervical_benchmark.synthetic_bags import (
    finalize_synthetic_bag_shards,
    fit_bag_description_model,
    load_real_bag_patch_label_counts,
    sample_bag_descriptions,
)


def _write_slide(root: Path, slide_id: str, labels: list[str], *, split: str, category: str, subcategory: str) -> None:
    slide_dir = root / "embeddings" / "uni2_h" / "slides" / f"{slide_id}.uni2_h.embedding"
    slide_dir.mkdir(parents=True)
    np.savez_compressed(slide_dir / "features.npz", features=np.ones((len(labels), 6), dtype=np.float32))
    pd.DataFrame(
        {
            "annotation_label_name": labels,
            "frame_index": np.arange(len(labels)),
        }
    ).to_parquet(slide_dir / "index.parquet", index=False)
    (slide_dir / "manifest.json").write_text(
        json.dumps({"num_embeddings": len(labels), "embedding_dim": 6})
    )


def _bag_root(tmp_path: Path) -> Path:
    root = tmp_path / "bags"
    root.mkdir(parents=True)
    rows = [
        {
            "id": "slide_train_a",
            "split": "train",
            "category": "normal_inflammation",
            "subcategory": "normal_inflammation",
        },
        {
            "id": "slide_train_b",
            "split": "train",
            "category": "low_grade",
            "subcategory": "cin1",
        },
        {
            "id": "slide_valid",
            "split": "valid",
            "category": "low_grade",
            "subcategory": "cin1",
        },
        {
            "id": "slide_test",
            "split": "test",
            "category": "malignant",
            "subcategory": "squamous_carcinoma",
        },
    ]
    pd.DataFrame(rows).to_csv(root / "dataset_index_ometiff.csv", index=False)
    _write_slide(
        root,
        "slide_train_a",
        ["normal", "normal_inflammation", "normal"],
        split="train",
        category="normal_inflammation",
        subcategory="normal_inflammation",
    )
    _write_slide(
        root,
        "slide_train_b",
        ["normal", "low_grade", "low_grade", "normal"],
        split="train",
        category="low_grade",
        subcategory="cin1",
    )
    _write_slide(
        root,
        "slide_valid",
        ["normal", "low_grade"],
        split="valid",
        category="low_grade",
        subcategory="cin1",
    )
    _write_slide(
        root,
        "slide_test",
        ["normal", "malignant", "malignant"],
        split="test",
        category="malignant",
        subcategory="squamous_carcinoma",
    )
    return root


def _config(tmp_path: Path) -> SyntheticBagConfig:
    return SyntheticBagConfig(
        real_bag_root=_bag_root(tmp_path),
        output_dir=tmp_path / "synthetic_bags",
        embedding_dim=6,
        sample_multipliers=[1.0],
        seed=123,
    )


def test_load_real_bag_patch_label_counts_normalises_normal_inflammation(tmp_path):
    config = _config(tmp_path)
    counts = load_real_bag_patch_label_counts(config)
    row = counts[counts["slide_id"] == "slide_train_a"].iloc[0]

    assert int(row["normal"]) == 3
    assert int(row["low_grade"]) == 0
    assert int(row["high_grade"]) == 0
    assert int(row["malignant"]) == 0


def test_fit_bag_description_model_uses_train_split_only(tmp_path):
    config = _config(tmp_path)
    metadata_dir = fit_bag_description_model(config)

    count_model = pd.read_csv(metadata_dir / "count_model_lognormal.csv")
    composition = pd.read_csv(metadata_dir / "composition_model_multinomial.csv")
    subcategory_model = pd.read_csv(metadata_dir / "subcategory_model.csv")

    assert set(count_model["subcategory"]) == {"normal_inflammation", "cin1"}
    assert set(composition["subcategory"]) == {"normal_inflammation", "cin1"}
    assert set(subcategory_model["subcategory"]) == {"normal_inflammation", "cin1"}


def test_sample_bag_descriptions_writes_positive_label_counts(tmp_path):
    config = _config(tmp_path)
    descriptions_path = sample_bag_descriptions(config, 1.0)
    descriptions = pd.read_csv(descriptions_path)

    assert len(descriptions) == 2
    assert set(descriptions["split"]) == {"train"}
    label_sum = descriptions[["normal", "low_grade", "high_grade", "malignant"]].sum(axis=1)
    assert (label_sum == descriptions["patch_count"]).all()
    assert (descriptions["patch_count"] > 0).all()


def test_finalize_synthetic_bag_shards_combines_indices_and_materializes_roots(tmp_path):
    config = _config(tmp_path)
    descriptions_path = sample_bag_descriptions(config, 1.0)
    generated_root = descriptions_path.parent
    descriptions = pd.read_csv(descriptions_path)
    slide_root = generated_root / "embeddings" / "uni2_h" / "slides"

    for shard_index in range(2):
        shard_dir = generated_root / "shards" / f"shard_{shard_index:03d}_of_002"
        shard_dir.mkdir(parents=True)
        rows = []
        for row in descriptions.iloc[shard_index::2].itertuples(index=False):
            slide_id = str(row.slide_id)
            slide_dir = slide_root / f"{slide_id}.uni2_h.embedding"
            slide_dir.mkdir(parents=True)
            np.savez_compressed(slide_dir / "features.npz", features=np.ones((int(row.patch_count), 6), dtype=np.float32))
            pd.DataFrame({"annotation_label_name": ["normal"] * int(row.patch_count)}).to_parquet(
                slide_dir / "index.parquet",
                index=False,
            )
            rows.append(
                {
                    "slide": f"{slide_id}.synthetic.ome.tiff",
                    "annotation": f"{slide_id}.synthetic.geojson",
                    "category": row.category,
                    "subcategory": row.subcategory,
                    "split": "train",
                    "id": slide_id,
                    "slide_path": "",
                    "slide_exists": True,
                    "source_canny_patchset_path": "",
                    "source_canny_exists": True,
                    "synthetic": True,
                    "generator": "test",
                }
            )
        pd.DataFrame(rows).to_csv(shard_dir / "dataset_index_ometiff.csv", index=False)

    roots = finalize_synthetic_bag_shards(config, multiplier=1.0, num_shards=2)
    synthetic_index = pd.read_csv(generated_root / "dataset_index_ometiff.csv")

    assert len(synthetic_index) == len(descriptions)
    assert {path.name for path in roots} == {"synthetic_only_1x", "real_plus_synthetic_1x"}
    for root in roots:
        assert (root / "dataset_index_ometiff.csv").exists()
