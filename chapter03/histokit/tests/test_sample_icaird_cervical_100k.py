from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "sample_icaird_cervical_100k.py"
SPEC = importlib.util.spec_from_file_location("sample_icaird_100k", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
sampler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sampler
SPEC.loader.exec_module(sampler)


def test_allocate_evenly_caps_and_redistributes() -> None:
    quotas = sampler.allocate_evenly({"a": 2, "b": 10, "c": 10}, 12)

    assert quotas == {"a": 2, "b": 5, "c": 5}
    assert sum(quotas.values()) == 12


def test_allocate_dataset_quotas_hits_split_totals() -> None:
    rows = []
    for split, count in [("train", 12), ("valid", 6)]:
        for subcategory in ["normal", "high_grade"]:
            for slide_id in ["s1", "s2"]:
                for i in range(count):
                    rows.append(
                        {
                            "split": split,
                            "category": "benign"
                            if subcategory == "normal"
                            else "abnormal",
                            "subcategory": subcategory,
                            "slide_id": f"{split}_{subcategory}_{slide_id}",
                            "row": i,
                            "column": 0,
                            "patch_png_path": f"{i}.png",
                        }
                    )
    candidates = pd.DataFrame(rows)

    quota_frame, _summary = sampler.allocate_dataset_quotas(
        candidates, {"train": 20, "valid": 8}
    )

    assert quota_frame.groupby("split")["slide_quota"].sum().to_dict() == {
        "train": 20,
        "valid": 8,
    }
    assert (
        quota_frame.groupby(["split", "subcategory"])["slide_quota"].sum().to_dict()
        == {
            ("train", "high_grade"): 10,
            ("train", "normal"): 10,
            ("valid", "high_grade"): 4,
            ("valid", "normal"): 4,
        }
    )


def test_prepare_candidates_filters_official_train_and_generates_slide_splits() -> None:
    rows = []
    for official_split in ["train", "test"]:
        for subcategory in ["cin1", "cin2"]:
            for slide_number in range(6):
                slide_id = f"{official_split}_{subcategory}_{slide_number}"
                for patch in range(2):
                    rows.append(
                        {
                            "slide_id": slide_id,
                            "row": patch,
                            "column": 0,
                            "split": official_split,
                            "category": "low_grade"
                            if subcategory == "cin1"
                            else "high_grade",
                            "subcategory": subcategory,
                            "patch_png_path": f"{slide_id}_{patch}.png",
                        }
                    )
    candidates = pd.DataFrame(rows)
    args = type(
        "Args",
        (),
        {
            "source_official_split": "train",
            "generated_slide_splits": True,
            "slide_split_stratify_columns": "category,subcategory",
            "seed": 236,
        },
    )()

    prepared = sampler.prepare_candidates_for_sampling(
        candidates,
        args,
        {"train": 4, "valid": 1, "test": 1},
    )

    assert set(prepared["official_split"]) == {"train"}
    assert not prepared["slide_id"].str.startswith("test_").any()
    assert set(prepared["split"]) == {"train", "valid", "test"}
    slide_split_counts = prepared.groupby("slide_id")["split"].nunique()
    assert slide_split_counts.max() == 1


def test_prepare_candidates_filters_grandqc_majority_tissue() -> None:
    rows = []
    for label, label_name in [
        (1, "normal"),
        (3, "low_grade"),
        (4, "high_grade"),
        (5, "malignant"),
    ]:
        for patch in range(4):
            rows.append(
                {
                    "slide_id": f"s{label}",
                    "row": patch,
                    "column": 0,
                    "split": "train",
                    "category": label_name,
                    "subcategory": label_name,
                    "annotation_label": label,
                    "annotation_label_name": label_name,
                    "official_grandqc_artifact_class_id": 1,
                    "official_grandqc_artifact_class": "normal_tissue",
                    "patch_png_path": f"s{label}_{patch}.png",
                }
            )
            rows.append(
                {
                    "slide_id": f"s{label}",
                    "row": patch,
                    "column": 1,
                    "split": "train",
                    "category": label_name,
                    "subcategory": label_name,
                    "annotation_label": label,
                    "annotation_label_name": label_name,
                    "official_grandqc_artifact_class_id": 7,
                    "official_grandqc_artifact_class": "background",
                    "patch_png_path": f"s{label}_{patch}_bg.png",
                }
            )
    candidates = pd.DataFrame(rows)
    args = type(
        "Args",
        (),
        {
            "source_official_split": "train",
            "grandqc_majority_tissue_only": True,
            "generated_slide_splits": False,
            "slide_split_stratify_columns": "category,subcategory",
            "seed": 236,
        },
    )()

    prepared = sampler.prepare_candidates_for_sampling(
        candidates,
        args,
        {"train": 8},
    )
    quota_frame, _summary = sampler.allocate_dataset_quotas(
        prepared,
        {"train": 8},
        quota_strategy="label_balanced_joint_subcategory",
    )

    assert set(prepared["official_grandqc_artifact_class_id"]) == {1}
    assert set(prepared["official_grandqc_artifact_class"]) == {"normal_tissue"}
    assert len(prepared) == 16
    assert quota_frame.groupby(
        ["annotation_label", "annotation_label_name"]
    )["slide_quota"].sum().to_dict() == {
        (1, "normal"): 2,
        (3, "low_grade"): 2,
        (4, "high_grade"): 2,
        (5, "malignant"): 2,
    }


def test_prepare_candidates_grandqc_tissue_filter_requires_columns() -> None:
    candidates = pd.DataFrame(
        {
            "slide_id": ["s1"],
            "row": [0],
            "column": [0],
            "split": ["train"],
            "category": ["normal"],
            "subcategory": ["normal"],
            "patch_png_path": ["s1_0.png"],
        }
    )
    args = type(
        "Args",
        (),
        {
            "source_official_split": "train",
            "grandqc_majority_tissue_only": True,
            "generated_slide_splits": False,
            "slide_split_stratify_columns": "category,subcategory",
            "seed": 236,
        },
    )()

    with pytest.raises(ValueError, match="GrandQC majority tissue filter requires"):
        sampler.prepare_candidates_for_sampling(candidates, args, {"train": 1})


def test_allocate_dataset_quotas_joint_label_subcategory_caps_and_redistributes() -> None:
    rows = []
    cells = [
        (1, "normal", "low_grade", "cin1", "s1", 10),
        (3, "low_grade", "low_grade", "cin1", "s2", 2),
        (4, "high_grade", "high_grade", "cin2", "s3", 10),
    ]
    for label, label_name, category, subcategory, slide_id, count in cells:
        for patch in range(count):
            rows.append(
                {
                    "slide_id": slide_id,
                    "row": patch,
                    "column": 0,
                    "split": "train",
                    "category": category,
                    "subcategory": subcategory,
                    "annotation_label": label,
                    "annotation_label_name": label_name,
                    "patch_png_path": f"{slide_id}_{patch}.png",
                }
            )
    candidates = pd.DataFrame(rows)

    quota_frame, summary = sampler.allocate_dataset_quotas(
        candidates,
        {"train": 12},
        quota_strategy="joint_label_subcategory",
    )

    assert summary["quota_columns"] == [
        "annotation_label",
        "annotation_label_name",
        "category",
        "subcategory",
    ]
    assert int(quota_frame["slide_quota"].sum()) == 12
    cell_counts = quota_frame.groupby(
        ["annotation_label", "annotation_label_name", "category", "subcategory"]
    )["slide_quota"].sum()
    assert cell_counts.to_dict() == {
        (1, "normal", "low_grade", "cin1"): 5,
        (3, "low_grade", "low_grade", "cin1"): 2,
        (4, "high_grade", "high_grade", "cin2"): 5,
    }


def test_label_balanced_joint_subcategory_keeps_exact_label_totals() -> None:
    rows = []
    cells = [
        (1, "normal", "benign", "normal_small", "s1", 2),
        (1, "normal", "benign", "normal_large", "s2", 20),
        (3, "low_grade", "low_grade", "cin1", "s3", 10),
        (3, "low_grade", "low_grade", "cin2", "s4", 10),
    ]
    for label, label_name, category, subcategory, slide_id, count in cells:
        for patch in range(count):
            rows.append(
                {
                    "slide_id": slide_id,
                    "row": patch,
                    "column": 0,
                    "split": "train",
                    "category": category,
                    "subcategory": subcategory,
                    "annotation_label": label,
                    "annotation_label_name": label_name,
                    "patch_png_path": f"{slide_id}_{patch}.png",
                }
            )
    candidates = pd.DataFrame(rows)

    quota_frame, summary = sampler.allocate_dataset_quotas(
        candidates,
        {"train": 20},
        quota_strategy="label_balanced_joint_subcategory",
    )

    assert summary["label_columns"] == ["annotation_label", "annotation_label_name"]
    assert int(quota_frame["slide_quota"].sum()) == 20
    label_counts = quota_frame.groupby(
        ["annotation_label", "annotation_label_name"]
    )["slide_quota"].sum()
    assert label_counts.to_dict() == {
        (1, "normal"): 10,
        (3, "low_grade"): 10,
    }
    cell_counts = quota_frame.groupby(
        ["annotation_label", "annotation_label_name", "category", "subcategory"]
    )["slide_quota"].sum()
    assert cell_counts.to_dict() == {
        (1, "normal", "benign", "normal_large"): 8,
        (1, "normal", "benign", "normal_small"): 2,
        (3, "low_grade", "low_grade", "cin1"): 5,
        (3, "low_grade", "low_grade", "cin2"): 5,
    }


def test_label_balanced_joint_subcategory_fails_when_label_capacity_is_low() -> None:
    rows = []
    cells = [
        (1, "normal", "benign", "normal", "s1", 20),
        (3, "low_grade", "low_grade", "cin1", "s2", 3),
    ]
    for label, label_name, category, subcategory, slide_id, count in cells:
        for patch in range(count):
            rows.append(
                {
                    "slide_id": slide_id,
                    "row": patch,
                    "column": 0,
                    "split": "train",
                    "category": category,
                    "subcategory": subcategory,
                    "annotation_label": label,
                    "annotation_label_name": label_name,
                    "patch_png_path": f"{slide_id}_{patch}.png",
                }
            )
    candidates = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="Strict even allocation exceeds capacity"):
        sampler.allocate_dataset_quotas(
            candidates,
            {"train": 10},
            quota_strategy="label_balanced_joint_subcategory",
        )


def test_farthest_first_positions_starts_near_centroid() -> None:
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.1, 0.0],
        ],
        dtype=np.float32,
    )

    positions = sampler.farthest_first_positions(features, 3)

    assert positions == [4, 2, 1]


def test_finalize_sample_manifest_writes_target_paths(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "slide_id": ["s1"],
            "row": [2],
            "column": [3],
            "split": ["train"],
            "category": ["normal"],
            "subcategory": ["normal"],
            "source_patch_png_path": ["/source/r2_c3.png"],
            "source_patchset_path": ["/source/s1.patchset"],
            "split_quota": [80_000],
            "subcategory_quota": [20_000],
            "slide_quota": [1],
        }
    )
    args = type("Args", (), {"output_root": tmp_path, "seed": 236})()

    manifest = sampler.finalize_sample_manifest(
        frame, args=args, sampling_method="uni2_kcenter"
    )

    assert manifest.loc[0, "sample_id"] == "icaird_cervical_000000"
    assert manifest.loc[0, "patch_png_path"] == str(
        tmp_path / "images/train/normal/icaird_cervical_000000.png"
    )
    assert manifest.loc[0, "sampling_method"] == "uni2_kcenter"
