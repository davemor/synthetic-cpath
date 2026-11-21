from __future__ import annotations

import os
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from histokit.dataset.sample import Sample
from histokit.dataset.schema import SlideSchema
from histokit.embeddings.encoder import EmbeddingResult
from histokit.embeddings.env import ensure_hf_token, read_dotenv
from histokit.embeddings.io import (
    discover_patchset_dirs,
    read_embedding_result,
    safe_model_name,
    write_embedding_result,
)
from histokit.embeddings.manual_features import (
    extract_manual_features,
    manual_feature_names,
)
from histokit.embeddings.models import resolve_model_spec
from histokit.patchset.context import PatchContext
from histokit.patchset.manifest import PatchSetManifest
from histokit.patchset.patchset import PatchSet
from histokit.pipelines.runtime import RuntimeContext
from histokit.pipelines.stages import EmbedPatches


def test_read_dotenv_reads_common_token_key(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN='secret-token'\nOTHER=value\n")

    values = read_dotenv(env_file)

    assert values["HF_TOKEN"] == "secret-token"
    assert values["OTHER"] == "value"


def test_ensure_hf_token_sets_common_env_vars(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("HUGGING_FACE_HUB_TOKEN=secret-token\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    token = ensure_hf_token(env_file)

    assert token == "secret-token"
    assert os.environ["HF_TOKEN"] == "secret-token"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "secret-token"


def test_resolve_model_spec_aliases_and_inference():
    assert resolve_model_spec("resnet18").backend == "torchvision"
    assert resolve_model_spec("hf:facebook/dinov2-small").backend == "hf"
    assert resolve_model_spec("facebook/dinov2-small").backend == "hf"
    assert resolve_model_spec("hf-hub:org/model").backend == "timm"
    assert resolve_model_spec("uni").source == "hf-hub:MahmoodLab/UNI"
    assert resolve_model_spec("dinov2").source == "facebook/dinov2-large"
    assert resolve_model_spec("dinov2-large").source == "facebook/dinov2-large"
    assert resolve_model_spec("hibou").source == "histai/hibou-L"
    assert resolve_model_spec("hibou-l").source == "histai/hibou-L"
    uni2 = resolve_model_spec("uni2_h")
    assert uni2.backend == "timm"
    assert uni2.source == "hf-hub:MahmoodLab/UNI2-h"
    assert resolve_model_spec("uni2").source == uni2.source


def test_manual_features_are_260_dimensional_and_stable():
    image = Image.new("RGB", (224, 224), color=(128, 64, 32))

    features = extract_manual_features(image)
    names = manual_feature_names()

    assert features.shape == (260,)
    assert features.dtype == np.float32
    assert len(names) == 260
    assert names[:4] == [
        "rgb_r_mean",
        "rgb_g_mean",
        "rgb_b_mean",
        "rgb_r_std",
    ]
    assert names[-1] == "thumbnail_7_7_b"


def test_manifest_manual_embedding_chunks_and_merges(tmp_path):
    script_path = (
        Path(__file__).parents[1] / "scripts" / "embed_icaird_120k_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("embed_manifest", script_path)
    assert spec is not None
    assert spec.loader is not None
    embed_manifest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(embed_manifest)

    dataset_root = tmp_path / "dataset"
    image_dir = dataset_root / "images"
    image_dir.mkdir(parents=True)
    rows = []
    for index, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        path = image_dir / f"patch_{index}.png"
        Image.new("RGB", (224, 224), color=color).save(path)
        rows.append(
            {
                "sample_id": f"s{index}",
                "slide_id": "slide",
                "row": index,
                "column": 0,
                "split": "train",
                "category": "normal",
                "subcategory": "normal",
                "annotation_label": 1,
                "annotation_label_name": "normal",
                "patch_png_path": str(path),
            }
        )
    manifest_path = dataset_root / "manifests" / "sample_manifest.parquet"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(manifest_path, index=False)
    output_root = dataset_root / "embeddings" / "manual_features"

    embed_manifest.write_chunks(
        manifest_path=manifest_path,
        output_root=output_root,
        chunk_size=2,
        overwrite=False,
    )
    for chunk_id in [0, 1]:
        embed_manifest.embed_chunk(
            manifest_path=manifest_path,
            output_root=output_root,
            model_name="manual_features",
            chunk_id=chunk_id,
            batch_size=2,
            device="cpu",
            env_file=tmp_path / ".env",
            overwrite=False,
            no_pretrained=False,
            limit=None,
        )
    embed_manifest.merge_chunks(
        manifest_path=manifest_path,
        output_root=output_root,
        model_name="manual_features",
        overwrite=False,
    )

    result = read_embedding_result(output_root)
    assert result.features.shape == (3, 260)
    assert result.index["sample_id"].tolist() == ["s0", "s1", "s2"]
    assert result.manifest["model"]["name"] == "manual_features"


def test_safe_model_name_handles_hf_hub_reference():
    assert (
        safe_model_name("timm:hf-hub:MahmoodLab/UNI2-h")
        == "timm_hf-hub_MahmoodLab_UNI2-h"
    )


def test_discover_patchset_dirs_single_and_collection(tmp_path):
    single = tmp_path / "one.patchset"
    single.mkdir()
    (single / "frame.parquet").touch()
    (single / "manifest.json").touch()

    found_single = discover_patchset_dirs(single)

    assert found_single == [single]

    root = tmp_path / "collection"
    root.mkdir()
    child = root / "child.patchset"
    child.mkdir()
    (child / "frame.parquet").touch()
    (child / "manifest.json").touch()

    assert discover_patchset_dirs(root) == [child]


def test_write_embedding_result(tmp_path):
    result = EmbeddingResult(
        features=np.array([[1.0, 2.0]], dtype=np.float32),
        index=pd.DataFrame(
            {"frame_index": [0], "row": [0], "column": [1], "context_id": [0]}
        ),
        manifest={"model": {"name": "fake"}, "embedding_dim": 2},
    )
    out = tmp_path / "embeddings"

    write_embedding_result(result, out)

    assert (out / "features.npz").exists()
    assert (out / "index.parquet").exists()
    assert (out / "manifest.json").exists()
    with np.load(out / "features.npz") as archive:
        assert archive["features"].shape == (1, 2)
        assert archive["features"].dtype == np.float32
        assert archive["frame_index"].tolist() == [0]
        assert archive["row"].tolist() == [0]
        assert archive["column"].tolist() == [1]
        assert archive["context_id"].tolist() == [0]

    loaded = read_embedding_result(out)
    assert loaded.features.shape == (1, 2)
    assert loaded.features.dtype == np.float32
    pd.testing.assert_frame_equal(loaded.index, result.index)


def test_read_embedding_result_supports_legacy_npy(tmp_path):
    out = tmp_path / "legacy.embedding"
    out.mkdir()
    expected = np.array([[3.0, 4.0]], dtype=np.float32)
    np.save(out / "features.npy", expected)
    index = pd.DataFrame({"frame_index": [5], "row": [2], "column": [7]})
    index.to_parquet(out / "index.parquet", index=False)
    (out / "manifest.json").write_text('{"model": {"name": "legacy"}}')

    loaded = read_embedding_result(out)

    np.testing.assert_array_equal(loaded.features, expected)
    pd.testing.assert_frame_equal(loaded.index, index)
    assert loaded.manifest["model"]["name"] == "legacy"


def test_embed_patches_stage_can_append_features(monkeypatch):
    import histokit.embeddings.encoder as encoder

    def fake_embed_patchset(*args, **kwargs):
        return EmbeddingResult(
            features=np.array([[0.1, 0.2]], dtype=np.float32),
            index=pd.DataFrame({"frame_index": [0]}),
            manifest={},
        )

    monkeypatch.setattr(encoder, "embed_patchset", fake_embed_patchset)

    sample = Sample(
        id="s",
        slide_path=Path("/tmp/s.tiff"),
        slide_schema=SlideSchema(kind="tiffslide", label_schema={}),
    )
    patchset = PatchSet(
        frame=pd.DataFrame(
            {
                "row": [0, 1],
                "column": [0, 0],
                "context_id": [0, 0],
                "keep": [True, False],
            }
        ),
        contexts=[PatchContext(sample=sample, level=1, patch_size=224)],
        manifest=PatchSetManifest(created_at="2026-01-01T00:00:00Z"),
    )

    result = EmbedPatches(model="fake", prefix="feat").run(
        patchset,
        RuntimeContext(),
    )

    assert "feat_0000" in result.frame.columns
    assert result.frame.loc[0, "feat_0000"] == pytest.approx(0.1)
    assert pd.isna(result.frame.loc[1, "feat_0000"])
