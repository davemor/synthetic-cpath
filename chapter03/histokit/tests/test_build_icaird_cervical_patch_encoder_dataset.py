from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_icaird_cervical_patch_encoder_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("build_icaird_patch_encoder", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        output_root=tmp_path,
        source_canny_patchsets_root=tmp_path / "canny",
        canny_threshold=0.05,
        grandqc_threshold=0.372,
        resnet_threshold=0.5,
        patch_size=224,
        patch_level=1,
        labels=tmp_path / "labels.json",
        annotations_root=tmp_path / "annotations",
        render_annotations=True,
        overwrite=False,
        export_kept_pngs=False,
    )


def test_slide_id_from_name_handles_ome_tiff() -> None:
    assert builder.slide_id_from_name("example-slide-001.ome.tiff") == "example-slide-001"
    assert builder.slide_id_from_name("example-slide-002.tif") == "example-slide-002"


def test_load_index_adds_slide_ids_and_paths(tmp_path: Path) -> None:
    index_path = tmp_path / "index.csv"
    index_path.write_text("slide,label\ns1.ome.tiff,A\ns2.ome.tiff,B\n")

    index = builder.load_index(index_path, tmp_path / "slides")

    assert index["id"].tolist() == ["s1", "s2"]
    assert index["slide_path"].tolist() == [
        str(tmp_path / "slides" / "s1.ome.tiff"),
        str(tmp_path / "slides" / "s2.ome.tiff"),
    ]


def test_load_index_fails_on_duplicate_ids(tmp_path: Path) -> None:
    index_path = tmp_path / "index.csv"
    index_path.write_text("slide\ns1.ome.tiff\ns1.ome.tiff\n")

    with pytest.raises(ValueError, match="Duplicate slide ids"):
        builder.load_index(index_path, tmp_path / "slides")


def test_write_tiffslide_labels_rewrites_slide_schema(tmp_path: Path) -> None:
    source = tmp_path / "labels.json"
    target = tmp_path / "labels_tiffslide.json"
    source.write_text(
        json.dumps(
            {
                "slides": {"kind": "isyntax", "type": "isyntax"},
                "other": {"kind": "unchanged"},
            }
        )
    )

    builder.write_tiffslide_labels(source, target)

    labels = json.loads(target.read_text())
    assert labels["slides"]["kind"] == "tiffslide"
    assert labels["slides"]["type"] == "tiffslide"
    assert labels["other"]["kind"] == "unchanged"


def test_compose_patch_frame_uses_canny_and_light_artifact_qc(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    row = pd.Series(
        {
            "id": "s1",
            "slide": "s1.ome.tiff",
            "slide_path": str(tmp_path / "s1.ome.tiff"),
            "diagnosis": "x",
        }
    )
    canny = pd.DataFrame(
        {
            "row": [0, 0, 1, 1],
            "column": [0, 1, 0, 1],
            "tissue_score": [0.1, 0.01, 0.1, 0.01],
        }
    )
    resnet = pd.DataFrame(
        {
            "slide_id": ["s1", "s1", "s1", "s1"],
            "row": [0, 0, 1, 1],
            "column": [0, 1, 0, 1],
            "resnet19_background_probability": [0.8, 0.1, 0.8, 0.9],
            "resnet19_foreground_probability": [0.2, 0.9, 0.2, 0.1],
            "resnet19_prediction": ["background", "foreground", "background", "background"],
        }
    )
    artifact = pd.DataFrame(
        {
            "slide_id": ["s1", "s1", "s1", "s1"],
            "row": [0, 0, 1, 1],
            "column": [0, 1, 0, 1],
            "official_grandqc_artifact_class_id": [1, 2, 3, 5],
            "official_grandqc_artifact_class": [
                "normal_tissue",
                "fold",
                "darkspot_foreign_object",
                "air_bubble_slide_edge",
            ],
        }
    )

    frame = builder.compose_patch_frame(
        args=args,
        row=row,
        canny=canny,
        resnet=resnet,
        official_artifact=artifact,
    )

    assert frame["canny_export_keep"].tolist() == [True, False, True, False]
    assert frame["candidate_tissue_keep"].tolist() == [True, False, True, False]
    assert frame["official_grandqc_artifact_keep"].tolist() == [True, True, False, False]
    assert frame["primary_training_keep"].tolist() == [True, False, False, False]
    assert frame["keep"].tolist() == [True, False, False, False]
    assert frame["canny_class"].tolist() == [
        "foreground",
        "background",
        "foreground",
        "background",
    ]
    assert frame["official_grandqc_artifact_class"].tolist() == [
        "normal_tissue",
        "fold",
        "darkspot_foreign_object",
        "air_bubble_slide_edge",
    ]
    assert frame["diagnosis"].tolist() == ["x", "x", "x", "x"]


def test_render_annotation_labels_uses_histokit_grid_rendering(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.labels.write_text(
        json.dumps(
            {
                "slides": {"type": "tiffslide", "labellings": {}},
                "annotations": {
                    "type": "geojson",
                    "labels": {"background": 0, "high_grade": 4},
                    "cutout": "background",
                    "fill": "background",
                    "order": ["background", "high_grade"],
                },
            }
        )
    )
    args.annotations_root.mkdir()
    annotation_path = args.annotations_root / "s1.geojson"
    annotation_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"classification": {"name": "High grade"}},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[0, 0], [300, 0], [300, 300], [0, 300], [0, 0]]
                            ],
                        },
                    }
                ],
            }
        )
    )
    row = pd.Series({"id": "s1", "slide_path": str(tmp_path / "s1.ome.tiff")})
    canny = pd.DataFrame(
        {"row": [0, 0, 1], "column": [0, 1, 0], "tissue_score": [0.1, 0.2, 0.3]}
    )

    labels = builder.render_annotation_labels(args=args, row=row, canny=canny)

    assert labels["has_annotation"].tolist() == [True, True, True]
    assert labels["annotation_path"].tolist() == [str(annotation_path)] * 3
    assert labels["annotation_label"].tolist() == [4, 0, 0]
    assert labels["annotation_label_name"].tolist() == [
        "high_grade",
        "background",
        "background",
    ]


def test_compose_patch_frame_fails_on_missing_resnet_scores(tmp_path: Path) -> None:
    args = _args(tmp_path)
    row = pd.Series({"id": "s1"})
    canny = pd.DataFrame({"row": [0], "column": [0], "tissue_score": [0.1]})
    resnet = pd.DataFrame(
        {
            "row": [1],
            "column": [1],
            "resnet19_foreground_probability": [0.9],
        }
    )
    artifact = pd.DataFrame(
        {
            "row": [0],
            "column": [0],
            "official_grandqc_artifact_class_id": [1],
        }
    )

    with pytest.raises(ValueError, match="Missing ResNet scores"):
        builder.compose_patch_frame(
            args=args,
            row=row,
            canny=canny,
            resnet=resnet,
            official_artifact=artifact,
        )


def test_compose_patch_frame_fails_on_missing_official_artifact_scores(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    row = pd.Series({"id": "s1"})
    canny = pd.DataFrame({"row": [0], "column": [0], "tissue_score": [0.1]})
    resnet = pd.DataFrame(
        {
            "row": [0],
            "column": [0],
            "resnet19_foreground_probability": [0.9],
        }
    )
    artifact = pd.DataFrame(
        {
            "row": [1],
            "column": [1],
            "official_grandqc_artifact_class_id": [1],
        }
    )

    with pytest.raises(ValueError, match="Missing official GrandQC artifact scores"):
        builder.compose_patch_frame(
            args=args,
            row=row,
            canny=canny,
            resnet=resnet,
            official_artifact=artifact,
        )
