from pathlib import Path

import numpy as np
import pandas as pd
import torch

from icaird_cervical_benchmark.config import MilConfig, load_config
from icaird_cervical_benchmark.mil import (
    TASK_CLASS_NAMES,
    TransformerMilClassifier,
    load_bag_records,
    make_loss,
)


def _write_slide(root: Path, slide_id: str, embedding_dim: int = 6, patch_count: int = 3) -> None:
    slide_dir = root / "embeddings" / "uni2_h" / "slides" / f"{slide_id}.uni2_h.embedding"
    slide_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        slide_dir / "features.npz",
        features=np.ones((patch_count, embedding_dim), dtype=np.float32),
        frame_index=np.arange(patch_count),
        row=np.arange(patch_count),
        column=np.arange(patch_count),
        context_id=np.array([slide_id] * patch_count),
    )


def _bag_root(tmp_path: Path) -> Path:
    rows = []
    specs = [
        ("train", "normal_inflammation", "normal_inflammation"),
        ("train", "low_grade", "hpv"),
        ("train", "high_grade", "cin1"),
        ("train", "malignant", "adenocarcinoma"),
        ("train", "malignant", "other"),
        ("valid", "normal_inflammation", "normal_inflammation"),
        ("valid", "low_grade", "cin2"),
        ("valid", "high_grade", "cin3"),
        ("valid", "malignant", "cgin"),
        ("valid", "malignant", "squamous_carcinoma"),
        ("test", "normal_inflammation", "normal_inflammation"),
        ("test", "low_grade", "hpv"),
        ("test", "high_grade", "cin1"),
        ("test", "malignant", "adenocarcinoma"),
        ("test", "malignant", "other"),
    ]
    # Ensure every subcategory class is present in every split.
    for split in ("train", "valid", "test"):
        for subcategory in TASK_CLASS_NAMES["subcategory"]:
            category = "malignant" if subcategory in {"adenocarcinoma", "squamous_carcinoma", "other"} else "low_grade"
            if subcategory in {"cin2", "cin3"}:
                category = "high_grade"
            if subcategory == "normal_inflammation":
                category = "normal_inflammation"
            specs.append((split, category, subcategory))
    for idx, (split, category, subcategory) in enumerate(specs):
        slide_id = f"IC-CX-{idx:05d}-01"
        _write_slide(tmp_path, slide_id)
        rows.append(
            {
                "id": slide_id,
                "slide": f"{slide_id}.ome.tiff",
                "annotation": f"{slide_id}.geojson",
                "split": split,
                "category": category,
                "subcategory": subcategory,
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "dataset_index_ometiff.csv", index=False)
    return tmp_path


def test_load_config_reads_mil_defaults(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
data:
  dataset_root: /tmp/example
mil:
  enabled: true
  bag_root: /tmp/bags
"""
    )

    config = load_config(config_path)

    assert config.mil.enabled is True
    assert config.mil.embedding_name == "uni2_h"
    assert config.mil.embedding_dim == 1536
    assert config.mil.tasks == ["malignant", "category", "subcategory"]


def test_load_bag_records_maps_malignant_task(tmp_path):
    root = _bag_root(tmp_path)
    config = MilConfig(bag_root=root, embedding_dim=6)

    records = load_bag_records(config, "malignant")

    assert {record.split for record in records} == {"train", "valid", "test"}
    assert {record.class_name for record in records} == {"benign", "malignant"}
    malignant = [record for record in records if record.class_name == "malignant"]
    assert all(record.label == 1 for record in malignant)
    assert records[0].patch_count == 3


def test_load_bag_records_preserves_class_orders(tmp_path):
    root = _bag_root(tmp_path)
    config = MilConfig(bag_root=root, embedding_dim=6)

    category = load_bag_records(config, "category")
    subcategory = load_bag_records(config, "subcategory")

    category_labels = {record.class_name: record.label for record in category}
    subcategory_labels = {record.class_name: record.label for record in subcategory}
    assert [category_labels[name] for name in TASK_CLASS_NAMES["category"]] == [0, 1, 2, 3]
    assert [subcategory_labels[name] for name in TASK_CLASS_NAMES["subcategory"]] == list(range(9))


def test_transformer_mil_forward_binary_and_multiclass():
    binary = TransformerMilClassifier(input_dim=6, num_classes=2, d_model=8, nhead=2, num_encoder_layers=1)
    multiclass = TransformerMilClassifier(input_dim=6, num_classes=4, d_model=8, nhead=2, num_encoder_layers=1)
    features = torch.ones(5, 6)

    assert binary(features).shape == (1,)
    assert multiclass(features).shape == (1, 4)


def test_make_loss_balances_binary_and_multiclass():
    records = [
        type("Record", (), {"label": 0})(),
        type("Record", (), {"label": 0})(),
        type("Record", (), {"label": 1})(),
        type("Record", (), {"label": 2})(),
    ]

    binary = make_loss(records[:3], num_classes=2, balance_classes=True, device=torch.device("cpu"))
    multiclass = make_loss(records, num_classes=3, balance_classes=True, device=torch.device("cpu"))

    assert torch.allclose(binary.pos_weight, torch.tensor([2.0]))
    assert torch.allclose(multiclass.weight, torch.tensor([4 / 6, 4 / 3, 4 / 3]))
