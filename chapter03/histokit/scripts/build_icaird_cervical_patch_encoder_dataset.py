#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


DATA_ROOT = Path("/sharedscratch/dm236/data/icaird_cervical")
DEFAULT_INDEX = Path("/sharedscratch/dcchb-group/datasets/icaird_cervical/index.csv")
DEFAULT_LABELS = Path("/sharedscratch/dcchb-group/datasets/icaird_cervical/labels.json")
DEFAULT_SLIDES_ROOT = Path("/sharedscratch/dcchb-group/datasets/icaird_cervical/ometiff")
DEFAULT_ANNOTATIONS_ROOT = Path(
    "/sharedscratch/dcchb-group/datasets/icaird_cervical/annotations"
)
DEFAULT_CANNY_ROOT = Path(
    "/sharedscratch/dcchb-group/datasets/icaird_cervical/patchsets/"
    "research_default_shards"
)
DEFAULT_OUTPUT_ROOT = (
    DATA_ROOT / "bags/full_canny005_or_resnet_light_grandqc_artifact"
)
DEFAULT_TISSUE_MODEL = DATA_ROOT / "grandqc_weights/Tissue_Detection_MPP10.pth"
DEFAULT_ARTIFACT_MODEL = DATA_ROOT / "grandqc_weights/GrandQC_MPP15.pth"
DEFAULT_RESNET_DIR = (
    DATA_ROOT
    / "resnet_patch_subset/stratified_120_consensus_100k/resnet19_patch_classifier"
)
DEFAULT_RESNET_THRESHOLD = 0.5
DEFAULT_ARTIFACT_MODEL_URL = (
    "https://zenodo.org/records/14041538/files/GrandQC_MPP15.pth?download=1"
)
PATCH_ID_COLUMNS = ["slide_id", "row", "column"]
ARTIFACT_CLASSES = {
    0: "unlabelled",
    1: "normal_tissue",
    2: "fold",
    3: "darkspot_foreign_object",
    4: "pen_marking",
    5: "air_bubble_slide_edge",
    6: "out_of_focus",
    7: "background",
}
ARTIFACT_PROB_COLUMNS = {
    class_id: f"grandqc_artifact_probability_{name}"
    for class_id, name in ARTIFACT_CLASSES.items()
}
ARTIFACT_CHANNEL_COUNT = max(ARTIFACT_CLASSES) + 1
LIGHT_QC_REJECT_ARTIFACT_CLASS_IDS = frozenset({3, 4, 5})
OFFICIAL_GRANDQC_MPP = 1.5
OFFICIAL_GRANDQC_PATCH_SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the full iCAIRD cervical patch encoder dataset."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "inventory",
            "missing-chunks",
            "grandqc-score",
            "resnet-score",
            "official-artifact-score",
            "build-slide",
            "summarize",
        ],
        required=True,
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--slides-root", type=Path, default=DEFAULT_SLIDES_ROOT)
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=DEFAULT_ANNOTATIONS_ROOT,
        help="Directory containing per-slide annotation files, usually GeoJSON.",
    )
    parser.add_argument(
        "--no-render-annotations",
        dest="render_annotations",
        action="store_false",
        help="Disable patch-grid rendering of slide annotations into patchsets.",
    )
    parser.set_defaults(render_annotations=True)
    parser.add_argument(
        "--source-canny-patchsets-root",
        type=Path,
        default=DEFAULT_CANNY_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--slide-id", default=None)
    parser.add_argument("--chunk-csv", type=Path, default=None)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--patch-level", type=int, default=1)
    parser.add_argument("--canny-threshold", type=float, default=0.05)
    parser.add_argument("--grandqc-threshold", type=float, default=0.372)
    parser.add_argument("--artifact-variant", choices=["5x", "7x", "10x"], default="7x")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_TISSUE_MODEL)
    parser.add_argument(
        "--artifact-model-path",
        type=Path,
        default=DEFAULT_ARTIFACT_MODEL,
    )
    parser.add_argument("--artifact-model-url", default=DEFAULT_ARTIFACT_MODEL_URL)
    parser.add_argument("--download-artifact-model-if-missing", action="store_true")
    parser.add_argument("--resnet-dir", type=Path, default=DEFAULT_RESNET_DIR)
    parser.add_argument("--resnet-threshold", type=float, default=DEFAULT_RESNET_THRESHOLD)
    parser.add_argument("--resnet-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--official-grandqc-output-root",
        type=Path,
        default=None,
        help="Defaults to <output-root>/grandqc_artifact_masks.",
    )
    parser.add_argument("--official-grandqc-mpp", type=float, default=OFFICIAL_GRANDQC_MPP)
    parser.add_argument(
        "--official-grandqc-model-patch-size",
        type=int,
        default=OFFICIAL_GRANDQC_PATCH_SIZE,
    )
    parser.add_argument(
        "--export-kept-pngs",
        dest="export_kept_pngs",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-export-kept-pngs",
        dest="export_kept_pngs",
        action="store_false",
    )
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument(
        "--missing-stage",
        choices=["resnet", "grandqc-artifact", "artifact-score", "build"],
        default=None,
    )
    parser.add_argument("--missing-chunk-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-slides", type=int, default=None)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def slide_id_from_name(name: str) -> str:
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".isyntax"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def load_index(path: Path, slides_root: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Index not found: {path}")
    frame = pd.read_csv(path)
    if "Unnamed: 0" in frame.columns:
        frame = frame.drop(columns=["Unnamed: 0"])
    if "slide" not in frame.columns:
        raise ValueError(f"Index missing slide column: {path}")
    frame = frame.copy()
    frame["slide"] = frame["slide"].astype(str)
    frame["id"] = frame["slide"].map(slide_id_from_name)
    if frame["id"].duplicated().any():
        dupes = frame.loc[frame["id"].duplicated(), "id"].head(10).tolist()
        raise ValueError(f"Duplicate slide ids in index: {dupes}")
    frame["slide_path"] = frame["slide"].map(lambda name: str(slides_root / name))
    return frame


def selected_slide_ids(args: argparse.Namespace, index: pd.DataFrame) -> list[str]:
    if args.slide_id:
        return [args.slide_id]
    if args.chunk_csv:
        chunk = pd.read_csv(args.chunk_csv)
        column = "slide_id" if "slide_id" in chunk.columns else "id"
        return chunk[column].astype(str).tolist()
    ids = index["id"].astype(str).tolist()
    if args.limit_slides is not None:
        ids = ids[: args.limit_slides]
    return ids


def ensure_dirs(output_root: Path) -> None:
    for subdir in [
        "grandqc_scores",
        "grandqc_artifact_masks",
        "grandqc_artifact_scores",
        "resnet_scores",
        "source_patchsets/missing_canny_patchsets",
        "patchsets",
        "patches",
        "exports",
        "status",
        "chunks/grandqc",
        "chunks/grandqc_artifact",
        "chunks/resnet",
        "chunks/build",
    ]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)


def index_canny_patchsets(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}
    paths: dict[str, Path] = {}
    for frame_path in root.glob("**/*.patchset/frame.parquet"):
        slide_id = frame_path.parent.name.removesuffix(".patchset")
        paths.setdefault(slide_id, frame_path.parent)
    return paths


def write_chunks(index: pd.DataFrame, output_root: Path, chunk_size: int) -> None:
    ids = index["id"].astype(str).tolist()
    for chunk_dir_name in [
        "chunks/grandqc_artifact",
        "chunks/resnet",
        "chunks/build",
    ]:
        chunk_dir = output_root / chunk_dir_name
        for old in chunk_dir.glob("chunk_*.csv"):
            old.unlink()
        for i in range(0, len(ids), chunk_size):
            chunk_ids = ids[i : i + chunk_size]
            text = "slide_id\n" + "\n".join(chunk_ids) + "\n"
            (chunk_dir / f"chunk_{i // chunk_size:04d}.csv").write_text(text)


def write_inventory(args: argparse.Namespace) -> None:
    ensure_dirs(args.output_root)
    index = load_index(args.index, args.slides_root)
    canny_paths = index_canny_patchsets(args.source_canny_patchsets_root)
    index["slide_exists"] = index["slide_path"].map(lambda path: Path(path).exists())
    index["source_canny_patchset_path"] = index["id"].map(
        lambda slide_id: str(canny_paths.get(str(slide_id), ""))
    )
    index["source_canny_exists"] = index["source_canny_patchset_path"].astype(bool)
    if args.limit_slides is not None:
        index = index.head(args.limit_slides).copy()
    index.to_csv(args.output_root / "dataset_index_ometiff.csv", index=False)
    if args.labels.exists():
        write_tiffslide_labels(args.labels, args.output_root / "labels_tiffslide.json")
    write_chunks(index, args.output_root, args.chunk_size)
    summary = {
        "created_at": utc_now(),
        "source_index": str(args.index),
        "slides_root": str(args.slides_root),
        "source_canny_patchsets_root": str(args.source_canny_patchsets_root),
        "output_root": str(args.output_root),
        "slide_count": int(len(index)),
        "slide_exists_count": int(index["slide_exists"].sum()),
        "source_canny_exists_count": int(index["source_canny_exists"].sum()),
        "patch_size": args.patch_size,
        "patch_level": args.patch_level,
        "canny_threshold": args.canny_threshold,
        "grandqc_threshold": args.grandqc_threshold,
        "artifact_variant": args.artifact_variant,
        "resnet_dir": str(args.resnet_dir),
        "resnet_threshold": args.resnet_threshold,
        "keep_rule": (
            "(canny_tissue_score > canny_threshold) AND "
            "official_grandqc_artifact_class_id NOT IN [3, 4, 5]"
        ),
        "light_qc_reject_artifact_classes": {
            str(class_id): ARTIFACT_CLASSES[class_id]
            for class_id in sorted(LIGHT_QC_REJECT_ARTIFACT_CLASS_IDS)
        },
    }
    (args.output_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "exports" / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def official_artifact_mask_exists(args: argparse.Namespace, row: pd.Series) -> bool:
    root = official_artifact_mask_root(args)
    slide_id = str(row["id"])
    slide_name = str(row["slide"])
    return (root / slide_id / "mask_qc" / f"{slide_name}_mask.png").exists()


def missing_target_exists(args: argparse.Namespace, row: pd.Series, stage: str) -> bool:
    slide_id = str(row["id"])
    if stage == "resnet":
        return resnet_score_path(args.output_root, slide_id).exists()
    if stage == "grandqc-artifact":
        return official_artifact_mask_exists(args, row)
    if stage == "artifact-score":
        return official_artifact_score_path(args.output_root, slide_id).exists()
    if stage == "build":
        path = patchset_dir(args.output_root, slide_id)
        return (path / "frame.parquet").exists() and (path / "manifest.json").exists()
    raise ValueError(f"Unsupported missing stage: {stage}")


def write_missing_chunks(args: argparse.Namespace) -> None:
    ensure_dirs(args.output_root)
    if args.missing_stage is None:
        raise ValueError("--missing-stage is required with --mode missing-chunks")
    index_path = args.output_root / "dataset_index_ometiff.csv"
    index = pd.read_csv(index_path) if index_path.exists() else load_index(args.index, args.slides_root)
    missing_ids: list[str] = []
    for _, row in index.iterrows():
        if not missing_target_exists(args, row, args.missing_stage):
            missing_ids.append(str(row["id"]))
    chunk_dir = args.missing_chunk_dir or (
        args.output_root / "chunks" / f"{args.missing_stage}_missing"
    )
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for old in chunk_dir.glob("chunk_*.csv"):
        old.unlink()
    for i in range(0, len(missing_ids), args.chunk_size):
        chunk_ids = missing_ids[i : i + args.chunk_size]
        text = "slide_id\n" + "\n".join(chunk_ids) + "\n"
        (chunk_dir / f"chunk_{i // args.chunk_size:04d}.csv").write_text(text)
    summary = {
        "created_at": utc_now(),
        "stage": args.missing_stage,
        "output_root": str(args.output_root),
        "chunk_dir": str(chunk_dir),
        "chunk_size": args.chunk_size,
        "total_slides": int(len(index)),
        "missing_count": int(len(missing_ids)),
        "chunk_count": int((len(missing_ids) + args.chunk_size - 1) // args.chunk_size)
        if missing_ids
        else 0,
    }
    (chunk_dir / "missing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def write_tiffslide_labels(source: Path, target: Path) -> None:
    labels = json.loads(source.read_text())
    if isinstance(labels, dict):
        for section in ("slides", "slide"):
            value = labels.get(section)
            if isinstance(value, dict):
                if "kind" in value:
                    value["kind"] = "tiffslide"
                if "type" in value:
                    value["type"] = "tiffslide"
    target.write_text(json.dumps(labels, indent=2) + "\n")


def require_slide_row(index: pd.DataFrame, slide_id: str) -> pd.Series:
    matches = index.loc[index["id"].astype(str) == slide_id]
    if len(matches) == 0:
        raise ValueError(f"Slide id not found in index: {slide_id}")
    if len(matches) > 1:
        raise ValueError(f"Multiple index rows for slide id: {slide_id}")
    return matches.iloc[0]


def score_path(output_root: Path, slide_id: str) -> Path:
    return output_root / "grandqc_scores" / f"{slide_id}.grandqc_scores.parquet"


def resnet_score_path(output_root: Path, slide_id: str) -> Path:
    return output_root / "resnet_scores" / f"{slide_id}.resnet_scores.parquet"


def official_artifact_score_path(output_root: Path, slide_id: str) -> Path:
    return (
        output_root
        / "grandqc_artifact_scores"
        / f"{slide_id}.grandqc_artifact_scores.parquet"
    )


def missing_canny_patchset_dir(output_root: Path, slide_id: str) -> Path:
    return output_root / "source_patchsets/missing_canny_patchsets" / f"{slide_id}.patchset"


def status_path(output_root: Path, slide_id: str) -> Path:
    return output_root / "status" / f"{slide_id}.json"


def patchset_dir(output_root: Path, slide_id: str) -> Path:
    return output_root / "patchsets" / f"{slide_id}.patchset"


def patches_dir(output_root: Path, slide_id: str) -> Path:
    return output_root / "patches" / slide_id


def write_status(output_root: Path, slide_id: str, status: dict[str, Any]) -> None:
    target = status_path(output_root, slide_id)
    existing: dict[str, Any] = {}
    if target.exists():
        try:
            existing = json.loads(target.read_text())
        except json.JSONDecodeError:
            existing = {}
    status = {**existing, "slide_id": slide_id, "updated_at": utc_now(), **status}
    target.write_text(json.dumps(status, indent=2) + "\n")


def artifact_mpp_for_variant(variant: str) -> float:
    return {"5x": 2.0, "7x": 1.5, "10x": 1.0}[variant]


def artifact_model_name_for_variant(variant: str) -> str:
    return {"5x": "GrandQC_MPP2.pth", "7x": "GrandQC_MPP15.pth", "10x": "GrandQC_MPP1.pth"}[variant]


def ensure_artifact_model(args: argparse.Namespace) -> Path:
    path = args.artifact_model_path
    if path.name == DEFAULT_ARTIFACT_MODEL.name and args.artifact_variant != "7x":
        path = path.parent / artifact_model_name_for_variant(args.artifact_variant)
    if path.exists():
        return path
    if not args.download_artifact_model_if_missing:
        raise FileNotFoundError(f"GrandQC artifact model not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    url = args.artifact_model_url
    if path.name != DEFAULT_ARTIFACT_MODEL.name:
        url = (
            "https://zenodo.org/records/14041538/files/"
            f"{artifact_model_name_for_variant(args.artifact_variant)}?download=1"
        )
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"Downloading GrandQC artifact model to {path}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


def run_grandqc_scores(
    *,
    slide_path: Path,
    tissue_model_path: Path,
    artifact_model_path: Path,
    artifact_variant: str,
    patch_size: int,
    patch_level: int,
) -> pd.DataFrame:
    from openslide import open_slide
    from tiffslide import TiffSlide
    from histokit.segmentation.grandqc import GrandQCDetector
    from histokit.utils.geometry import Size

    Image.MAX_IMAGE_PIXELS = None

    class HybridOmeTiffSlide:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.tsr = TiffSlide(str(path))
            self.osr = open_slide(str(path))

        def close(self) -> None:
            self.osr.close()
            self.tsr.close()

        @property
        def mpp(self) -> float | None:
            props = self.tsr.properties
            for prefix in ("tiffslide", "openslide"):
                try:
                    mpp_x = float(props[f"{prefix}.mpp-x"])
                    mpp_y = float(props[f"{prefix}.mpp-y"])
                    return (mpp_x + mpp_y) / 2.0
                except (KeyError, TypeError, ValueError):
                    continue
            return None

        @property
        def dimensions(self) -> list[Size]:
            return [Size(*dim) for dim in self.tsr.level_dimensions]

        def level_downsamples(self) -> list[float]:
            return list(self.tsr.level_downsamples)

        def size_in_patches(self, patch_size: int, patch_level: int) -> Size:
            dims = self.dimensions[patch_level]
            return Size(
                (dims.width + patch_size - 1) // patch_size,
                (dims.height + patch_size - 1) // patch_size,
            )

        def get_thumbnail_for_size(
            self, width: int, height: int, white_background: bool = True
        ) -> np.ndarray:
            image = self.osr.get_thumbnail((width, height)).convert("RGB")
            image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
            array = np.asarray(image)
            if white_background:
                black_pixels = np.all(array == 0, axis=2)
                array = array.copy()
                array[black_pixels] = [255, 255, 255]
            return array

    slide = HybridOmeTiffSlide(slide_path)
    try:
        tissue = GrandQCDetector(
            patch_size=patch_size,
            patch_level=patch_level,
            model_path=tissue_model_path,
        )(slide)
        artifact = run_artifact_model_on_slide(
            slide=slide,
            model_path=artifact_model_path,
            artifact_mpp=artifact_mpp_for_variant(artifact_variant),
            patch_size=patch_size,
            patch_level=patch_level,
        )
    finally:
        slide.close()

    rows, columns = np.indices(tissue.shape)
    frame = pd.DataFrame(
        {
            "row": rows.reshape(-1).astype(int),
            "column": columns.reshape(-1).astype(int),
            "grandqc_tissue_probability": tissue.reshape(-1).astype(float),
        }
    )
    for class_id, column in ARTIFACT_PROB_COLUMNS.items():
        frame[column] = artifact[:, :, class_id - 1].reshape(-1).astype(float)
    return frame


def run_artifact_model_on_slide(
    *,
    slide: Any,
    model_path: Path,
    artifact_mpp: float,
    patch_size: int,
    patch_level: int,
) -> np.ndarray:
    import segmentation_models_pytorch as smp
    import torch

    install_timm_pickle_aliases()
    install_smp_pickle_aliases()
    device = torch.device(
        os.environ.get(
            "HISTOKIT_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    model = torch.load(model_path, map_location=device, weights_only=False)
    patch_loaded_artifact_model(model)
    model.to(device)
    model.eval()
    preprocessing_fn = smp.encoders.get_preprocessing_fn("timm-efficientnet-b0", "imagenet")
    slide_mpp = slide.mpp
    if slide_mpp is None:
        raise ValueError("Slide lacks MPP metadata required for GrandQC artifact model")
    level0 = slide.dimensions[0]
    width = max(1, int(round(level0.width * slide_mpp / artifact_mpp)))
    height = max(1, int(round(level0.height * slide_mpp / artifact_mpp)))
    thumb = slide.get_thumbnail_for_size(width, height, white_background=True)
    tile_size = 512
    pad_h = (-height) % tile_size
    pad_w = (-width) % tile_size
    if pad_h or pad_w:
        thumb = np.pad(
            thumb,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=255,
        )
    padded_h, padded_w = thumb.shape[:2]
    probs = np.zeros((padded_h, padded_w, ARTIFACT_CHANNEL_COUNT), dtype=np.float32)
    with torch.no_grad():
        for y in range(0, padded_h, tile_size):
            for x in range(0, padded_w, tile_size):
                tile = thumb[y : y + tile_size, x : x + tile_size]
                arr = preprocessing_fn(tile).transpose(2, 0, 1).astype("float32")
                tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
                output = (
                    model.predict(tensor) if hasattr(model, "predict") else model(tensor)
                )
                prob = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
                if prob.shape[0] != ARTIFACT_CHANNEL_COUNT:
                    raise ValueError(
                        "Unexpected GrandQC artifact channel count: "
                        f"{prob.shape[0]} != {ARTIFACT_CHANNEL_COUNT}"
                    )
                probs[y : y + tile_size, x : x + tile_size] = prob.transpose(1, 2, 0)
    probs = probs[:height, :width]
    grid = slide.size_in_patches(patch_size, patch_level)
    resized_channels = []
    for class_index in range(ARTIFACT_CHANNEL_COUNT):
        img = Image.fromarray(probs[:, :, class_index], mode="F")
        resized = img.resize((grid.width, grid.height), resample=Image.Resampling.BOX)
        resized_channels.append(np.asarray(resized, dtype=np.float32))
    stacked = np.stack(resized_channels, axis=2)
    sums = stacked.sum(axis=2, keepdims=True)
    return np.divide(stacked, sums, out=np.zeros_like(stacked), where=sums > 0)


def install_timm_pickle_aliases() -> None:
    import importlib
    import pkgutil
    import sys

    try:
        timm_layers = importlib.import_module("timm.layers")
        for module_info in pkgutil.iter_modules(timm_layers.__path__):
            old_name = f"timm.models.layers.{module_info.name}"
            new_name = f"timm.layers.{module_info.name}"
            if old_name in sys.modules:
                continue
            try:
                sys.modules[old_name] = importlib.import_module(new_name)
            except ModuleNotFoundError:
                continue
    except ModuleNotFoundError:
        pass

    aliases = {
        "timm.models.layers.activations": "timm.layers.activations",
        "timm.models.layers.conv2d_same": "timm.layers.conv2d_same",
        "timm.models.layers.padding": "timm.layers.padding",
        "timm.models.layers.pool2d_same": "timm.layers.pool2d_same",
        "timm.models.layers.norm_act": "timm.layers.norm_act",
        "timm.models.layers.create_act": "timm.layers.create_act",
        "timm.models.layers.helpers": "timm.layers.helpers",
        "timm.models.efficientnet_blocks": "timm.models._efficientnet_blocks",
        "timm.models.efficientnet_builder": "timm.models._efficientnet_builder",
        "timm.models.features": "timm.models._features",
        "timm.models.helpers": "timm.models._helpers",
    }
    for old_name, new_name in aliases.items():
        if old_name in sys.modules:
            continue
        try:
            sys.modules[old_name] = importlib.import_module(new_name)
        except ModuleNotFoundError:
            continue


def install_smp_pickle_aliases() -> None:
    try:
        from segmentation_models_pytorch.decoders.unet import decoder
    except ModuleNotFoundError:
        return
    if not hasattr(decoder, "DecoderBlock") and hasattr(decoder, "UnetDecoderBlock"):
        decoder.DecoderBlock = decoder.UnetDecoderBlock
    if not hasattr(decoder, "CenterBlock") and hasattr(decoder, "UnetCenterBlock"):
        decoder.CenterBlock = decoder.UnetCenterBlock


def patch_loaded_artifact_model(model: Any) -> None:
    import torch.nn as nn

    try:
        from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoderBlock
        from timm.models._efficientnet_blocks import (
            DepthwiseSeparableConv,
            InvertedResidual,
        )
    except ModuleNotFoundError:
        return
    for module in model.modules():
        if isinstance(module, (DepthwiseSeparableConv, InvertedResidual)):
            if not hasattr(module, "conv_s2d"):
                module.conv_s2d = None
            if not hasattr(module, "bn_s2d"):
                module.bn_s2d = None
            if not hasattr(module, "aa"):
                module.aa = nn.Identity()
            if not hasattr(module, "drop_path"):
                module.drop_path = nn.Identity()
            if not hasattr(module, "has_skip"):
                module.has_skip = bool(getattr(module, "has_residual", False))
        if isinstance(module, UnetDecoderBlock):
            if not hasattr(module, "interpolation_mode"):
                module.interpolation_mode = "nearest"


def run_grandqc_mode(args: argparse.Namespace) -> None:
    ensure_dirs(args.output_root)
    index = load_index(args.index, args.slides_root)
    tissue_model_path = args.model_path
    artifact_model_path = ensure_artifact_model(args)
    for slide_id in selected_slide_ids(args, index):
        target = score_path(args.output_root, slide_id)
        if target.exists() and not args.overwrite:
            print(f"Skipping existing GrandQC scores for {slide_id}: {target}")
            continue
        row = require_slide_row(index, slide_id)
        slide_path = Path(row["slide_path"])
        if not slide_path.exists():
            write_status(args.output_root, slide_id, {"grandqc_status": "missing_slide"})
            raise FileNotFoundError(f"Slide not found for {slide_id}: {slide_path}")
        frame = run_grandqc_scores(
            slide_path=slide_path,
            tissue_model_path=tissue_model_path,
            artifact_model_path=artifact_model_path,
            artifact_variant=args.artifact_variant,
            patch_size=args.patch_size,
            patch_level=args.patch_level,
        )
        frame.insert(0, "slide_id", slide_id)
        frame.to_parquet(target, index=False)
        write_status(
            args.output_root,
            slide_id,
            {"grandqc_status": "complete", "grandqc_score_path": str(target), "rows": int(len(frame))},
        )
        print(f"Wrote {len(frame)} GrandQC rows for {slide_id} -> {target}")


def make_sample(row: pd.Series) -> Any:
    from histokit.dataset.sample import Sample
    from histokit.dataset.schema import SlideSchema

    metadata = {key: value for key, value in row.to_dict().items() if key not in {"slide_path"}}
    return Sample(
        id=str(row["id"]),
        slide_path=Path(row["slide_path"]),
        slide_schema=SlideSchema(kind="tiffslide", label_schema={}),
        metadata=metadata,
    )


def annotation_path_for_slide(annotations_root: Path, slide_id: str) -> Path:
    return annotations_root / f"{slide_id}.geojson"


def load_annotation_schema(labels_path: Path) -> Any | None:
    if not labels_path.exists():
        return None
    from histokit.dataset.schema import DatasetSchema

    return DatasetSchema.from_json(labels_path).annotations


def render_annotation_labels(
    *,
    args: argparse.Namespace,
    row: pd.Series,
    canny: pd.DataFrame,
) -> pd.DataFrame:
    slide_id = str(row["id"])
    out = canny[["row", "column"]].copy()
    annotation_path = annotation_path_for_slide(args.annotations_root, slide_id)
    out["annotation_path"] = str(annotation_path) if annotation_path.exists() else ""
    out["has_annotation"] = bool(annotation_path.exists())
    out["annotation_label"] = np.nan
    out["annotation_label_name"] = ""

    if not args.render_annotations or not annotation_path.exists():
        return out

    annotation_schema = load_annotation_schema(args.labels)
    if annotation_schema is None:
        return out

    import histokit.io.annotations  # noqa: F401 - registers annotation loaders
    from histokit.dataset.sample import Sample
    from histokit.dataset.schema import SlideSchema
    from histokit.utils.geometry import Shape

    sample = Sample(
        id=slide_id,
        slide_path=Path(row["slide_path"]),
        slide_schema=SlideSchema(kind="tiffslide", label_schema={}),
        annotation_path=annotation_path,
        annotation_schema=annotation_schema,
        metadata={},
    )
    annotation_set = sample.make_annotations()
    grid_h = int(canny["row"].max()) + 1
    grid_w = int(canny["column"].max()) + 1
    if annotation_set is not None and len(annotation_set) > 0:
        label_grid = annotation_set.render_to_grid(
            Shape(grid_h, grid_w),
            args.patch_size,
            args.patch_level,
        )
        rows = out["row"].to_numpy(dtype=int)
        columns = out["column"].to_numpy(dtype=int)
        labels = label_grid[rows, columns].astype(int)
    else:
        labels = np.full(
            len(out),
            annotation_schema.label_map[annotation_schema.fill_label],
            dtype=int,
        )

    label_names = {value: key for key, value in annotation_schema.label_map.items()}
    out["annotation_label"] = labels
    out["annotation_label_name"] = [label_names.get(int(value), "") for value in labels]
    return out


def load_or_compute_canny(args: argparse.Namespace, row: pd.Series) -> pd.DataFrame:
    slide_id = str(row["id"])
    canny_path = row.get("source_canny_patchset_path")
    if isinstance(canny_path, str) and canny_path:
        frame_path = Path(canny_path) / "frame.parquet"
        if frame_path.exists():
            frame = pd.read_parquet(frame_path)
            if "tissue_score" in frame.columns:
                return frame[["row", "column", "tissue_score"]].copy()
    cached_missing = missing_canny_patchset_dir(args.output_root, slide_id)
    cached_frame_path = cached_missing / "frame.parquet"
    if cached_frame_path.exists() and not args.overwrite:
        frame = pd.read_parquet(cached_frame_path)
        if "tissue_score" in frame.columns:
            return frame[["row", "column", "tissue_score"]].copy()
    scores = compute_canny_with_ome_pyramid(
        slide_path=Path(row["slide_path"]),
        patch_size=args.patch_size,
        patch_level=args.patch_level,
    )
    rows, columns = np.indices(scores.shape)
    print(f"Computed Canny scores for missing source patchset: {slide_id}")
    frame = pd.DataFrame(
        {
            "row": rows.reshape(-1).astype(int),
            "column": columns.reshape(-1).astype(int),
            "tissue_score": scores.reshape(-1).astype(float),
        }
    )
    save_canny_patchset(args, row, frame)
    write_status(
        args.output_root,
        slide_id,
        {
            "canny_status": "computed_missing_source",
            "canny_patchset_path": str(missing_canny_patchset_dir(args.output_root, slide_id)),
            "canny_rows": int(len(frame)),
        },
    )
    return frame


def compute_canny_with_ome_pyramid(
    *,
    slide_path: Path,
    patch_size: int,
    patch_level: int,
) -> np.ndarray:
    from histokit.segmentation.transforms import CannyEdgeDetector

    pyramid = open_ome_pyramid(slide_path)
    level = pyramid[str(patch_level)]
    shape = level.shape
    if len(shape) == 2:
        height_pixels, width_pixels = shape
    elif len(shape) == 3 and shape[0] in (3, 4) and shape[-1] not in (3, 4):
        _, height_pixels, width_pixels = shape
    elif len(shape) == 3:
        height_pixels, width_pixels = shape[:2]
    else:
        raise ValueError(f"Unsupported OME-TIFF level shape: {shape}")

    width_patches = (int(width_pixels) + patch_size) // patch_size
    height_patches = (int(height_pixels) + patch_size) // patch_size
    transform = CannyEdgeDetector()
    scores = np.full((height_patches, width_patches), -1.0, dtype=float)
    for row_idx in range(height_patches):
        for column_idx in range(width_patches):
            patch = read_ome_pyramid_patch(
                pyramid=pyramid,
                patch_level=patch_level,
                row=row_idx,
                column=column_idx,
                patch_size=patch_size,
            )
            scores[row_idx, column_idx] = float(transform(patch))
    return scores


def save_canny_patchset(args: argparse.Namespace, row: pd.Series, frame: pd.DataFrame) -> None:
    from histokit.patchset.context import PatchContext
    from histokit.patchset.manifest import PatchSetManifest
    from histokit.patchset.patchset import PatchSet

    sample = make_sample(row)
    patchset = PatchSet(
        frame=frame.copy(),
        contexts=[PatchContext(sample=sample, level=args.patch_level, patch_size=args.patch_size)],
        manifest=PatchSetManifest(created_at=utc_now()),
    )
    patchset.save(missing_canny_patchset_dir(args.output_root, str(row["id"])))


def classify_binary(prob: pd.Series | np.ndarray, threshold: float) -> np.ndarray:
    return np.where(np.asarray(prob, dtype=float) > threshold, "foreground", "background")


def load_resnet_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    from train_resnet19_patch_classifier import (
        make_resnet19,
        require_training_dependencies,
        resolve_device,
    )

    torch, nn, functional, _ = require_training_dependencies()
    device = resolve_device(args.device, torch)
    checkpoint = args.resnet_dir / "resnet19_final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"ResNet checkpoint not found: {checkpoint}")
    model = make_resnet19(nn).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return torch, functional, (model, device)


def patch_tensor_from_slide(
    *,
    pyramid: Any,
    row: int,
    column: int,
    patch_size: int,
    patch_level: int,
) -> np.ndarray:
    from train_resnet19_patch_classifier import IMAGENET_MEAN, IMAGENET_STD

    image = Image.fromarray(
        read_ome_pyramid_patch(
            pyramid=pyramid,
            patch_level=patch_level,
            row=row,
            column=column,
            patch_size=patch_size,
        )
    ).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def open_ome_pyramid(slide_path: Path) -> Any:
    import zarr
    from tifffile import imread

    store = imread(slide_path, aszarr=True)
    return zarr.open(store, mode="r")


def normalize_rgb_patch(array: np.ndarray, patch_size: int) -> np.ndarray:
    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=2)
    elif array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.transpose(array[:3], (1, 2, 0))
    elif array.ndim == 3 and array.shape[-1] in (3, 4):
        array = array[..., :3]
    else:
        raise ValueError(f"Unsupported OME-TIFF patch shape: {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.integer):
            max_value = np.iinfo(array.dtype).max
            array = (array.astype(np.float32) / max_value * 255.0).clip(0, 255)
        else:
            array = (array.astype(np.float32) * 255.0).clip(0, 255)
        array = array.astype(np.uint8)
    height, width = array.shape[:2]
    if height == patch_size and width == patch_size:
        return array
    padded = np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)
    padded[: min(height, patch_size), : min(width, patch_size)] = array[
        :patch_size, :patch_size
    ]
    return padded


def read_ome_pyramid_patch(
    *,
    pyramid: Any,
    patch_level: int,
    row: int,
    column: int,
    patch_size: int,
) -> np.ndarray:
    level = pyramid[str(patch_level)]
    y0 = int(row * patch_size)
    x0 = int(column * patch_size)
    y1 = y0 + patch_size
    x1 = x0 + patch_size
    shape = level.shape
    if len(shape) == 2:
        patch = level[y0:y1, x0:x1]
    elif len(shape) == 3 and shape[0] in (3, 4) and shape[-1] not in (3, 4):
        patch = level[:3, y0:y1, x0:x1]
    elif len(shape) == 3:
        patch = level[y0:y1, x0:x1, :3]
    else:
        raise ValueError(f"Unsupported OME-TIFF level shape: {shape}")
    return normalize_rgb_patch(np.asarray(patch), patch_size)


def score_resnet_for_slide(
    *,
    args: argparse.Namespace,
    row: pd.Series,
    canny: pd.DataFrame,
    torch: Any,
    functional: Any,
    model_and_device: tuple[Any, Any],
) -> pd.DataFrame:
    model, device = model_and_device
    slide_path = Path(row["slide_path"])
    patch_rows = canny[["row", "column"]].copy()
    background_probs: list[float] = []
    foreground_probs: list[float] = []
    predictions: list[str] = []
    batch_arrays: list[np.ndarray] = []

    def flush_batch() -> None:
        if not batch_arrays:
            return
        tensor = torch.from_numpy(np.stack(batch_arrays)).to(device, non_blocking=True)
        with torch.inference_mode():
            probs = functional.softmax(model(tensor), dim=1).detach().cpu().numpy()
        background_probs.extend(float(value) for value in probs[:, 0])
        foreground_probs.extend(float(value) for value in probs[:, 1])
        predictions.extend("foreground" if int(idx) == 1 else "background" for idx in probs.argmax(axis=1))
        batch_arrays.clear()

    pyramid = open_ome_pyramid(slide_path)
    for patch in patch_rows.itertuples(index=False):
        batch_arrays.append(
            patch_tensor_from_slide(
                pyramid=pyramid,
                row=int(patch.row),
                column=int(patch.column),
                patch_size=args.patch_size,
                patch_level=args.patch_level,
            )
        )
        if len(batch_arrays) >= args.resnet_batch_size:
            flush_batch()
    flush_batch()
    if len(foreground_probs) != len(patch_rows):
        raise ValueError(
            "ResNet score count does not match patch rows for "
            f"{row['id']}: {len(foreground_probs)} != {len(patch_rows)}"
        )
    out = patch_rows.copy()
    out.insert(0, "slide_id", str(row["id"]))
    out["resnet19_background_probability"] = background_probs
    out["resnet19_foreground_probability"] = foreground_probs
    out["resnet19_prediction"] = predictions
    out["resnet19_threshold"] = args.resnet_threshold
    out["resnet19_threshold_prediction"] = classify_binary(
        out["resnet19_foreground_probability"], args.resnet_threshold
    )
    return out


def run_resnet_mode(args: argparse.Namespace) -> None:
    ensure_dirs(args.output_root)
    index_path = args.output_root / "dataset_index_ometiff.csv"
    index = pd.read_csv(index_path) if index_path.exists() else load_index(args.index, args.slides_root)
    torch, functional, model_and_device = load_resnet_model(args)
    for slide_id in selected_slide_ids(args, index):
        target = resnet_score_path(args.output_root, slide_id)
        if target.exists() and not args.overwrite:
            print(f"Skipping existing ResNet scores for {slide_id}: {target}")
            continue
        row = require_slide_row(index, slide_id).copy()
        if not row.get("source_canny_patchset_path"):
            canny_paths = index_canny_patchsets(args.source_canny_patchsets_root)
            row["source_canny_patchset_path"] = str(canny_paths.get(slide_id, ""))
        canny = load_or_compute_canny(args, row)
        frame = score_resnet_for_slide(
            args=args,
            row=row,
            canny=canny,
            torch=torch,
            functional=functional,
            model_and_device=model_and_device,
        )
        frame.to_parquet(target, index=False)
        write_status(
            args.output_root,
            slide_id,
            {
                "resnet_status": "complete",
                "resnet_score_path": str(target),
                "resnet_rows": int(len(frame)),
                "resnet_dir": str(args.resnet_dir),
                "resnet_threshold": args.resnet_threshold,
            },
        )
        print(f"Wrote {len(frame)} ResNet rows for {slide_id} -> {target}")


def slide_mpp(slide: Any) -> float:
    props = slide.properties
    for prefix in ("tiffslide", "openslide"):
        try:
            mpp_x = float(props[f"{prefix}.mpp-x"])
            mpp_y = float(props[f"{prefix}.mpp-y"])
            return (mpp_x + mpp_y) / 2.0
        except (KeyError, TypeError, ValueError):
            continue
    raise ValueError("Slide lacks MPP metadata required for official GrandQC mask mapping")


def official_artifact_mask_root(args: argparse.Namespace) -> Path:
    return args.official_grandqc_output_root or (args.output_root / "grandqc_artifact_masks")


def official_artifact_mask_candidates(args: argparse.Namespace, row: pd.Series) -> list[Path]:
    root = official_artifact_mask_root(args)
    slide_id = str(row["id"])
    slide_name = str(row["slide"])
    names = [
        f"{slide_name}_mask.png",
        f"{slide_id}.ome.tiff_mask.png",
        f"{slide_id}_mask.png",
    ]
    candidates: list[Path] = []
    for name in names:
        candidates.extend(
            [
                root / slide_id / "mask_qc" / name,
                root / "output" / "mask_qc" / name,
                root / "mask_qc" / name,
            ]
        )
    candidates.extend(root.glob(f"**/{slide_name}_mask.png"))
    candidates.extend(root.glob(f"**/{slide_id}*_mask.png"))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def official_artifact_mask_path(args: argparse.Namespace, row: pd.Series) -> Path:
    for path in official_artifact_mask_candidates(args, row):
        if path.exists():
            return path
    raise FileNotFoundError(
        "Official GrandQC artifact mask not found for "
        f"{row['id']} under {official_artifact_mask_root(args)}"
    )


def map_official_artifact_mask_to_patches(
    *,
    args: argparse.Namespace,
    row: pd.Series,
    canny: pd.DataFrame,
) -> pd.DataFrame:
    from tiffslide import TiffSlide

    mask_path = official_artifact_mask_path(args, row)
    Image.MAX_IMAGE_PIXELS = None
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    slide_path = Path(row["slide_path"])
    with TiffSlide(str(slide_path)) as slide:
        downsample = float(slide.level_downsamples[args.patch_level])
        mpp = slide_mpp(slide)
    p_s = int(args.official_grandqc_mpp / mpp * args.official_grandqc_model_patch_size)
    if p_s <= 0:
        raise ValueError(f"Invalid official GrandQC mapped patch size for {row['id']}: {p_s}")
    x_level0 = (
        canny["column"].to_numpy(dtype=float) + 0.5
    ) * args.patch_size * downsample
    y_level0 = (
        canny["row"].to_numpy(dtype=float) + 0.5
    ) * args.patch_size * downsample
    x_mask = np.clip(
        np.floor(x_level0 / p_s * args.official_grandqc_model_patch_size).astype(int),
        0,
        mask.shape[1] - 1,
    )
    y_mask = np.clip(
        np.floor(y_level0 / p_s * args.official_grandqc_model_patch_size).astype(int),
        0,
        mask.shape[0] - 1,
    )
    class_id = mask[y_mask, x_mask].astype(int)
    out = canny[["row", "column"]].copy()
    out.insert(0, "slide_id", str(row["id"]))
    out["official_grandqc_artifact_class_id"] = class_id
    out["official_grandqc_artifact_class"] = [
        ARTIFACT_CLASSES.get(int(value), f"unknown_{int(value)}") for value in class_id
    ]
    out["official_grandqc_artifact_mask_path"] = str(mask_path)
    out["official_grandqc_artifact_keep"] = ~out[
        "official_grandqc_artifact_class_id"
    ].isin(LIGHT_QC_REJECT_ARTIFACT_CLASS_IDS)
    return out


def run_official_artifact_score_mode(args: argparse.Namespace) -> None:
    ensure_dirs(args.output_root)
    index_path = args.output_root / "dataset_index_ometiff.csv"
    index = pd.read_csv(index_path) if index_path.exists() else load_index(args.index, args.slides_root)
    for slide_id in selected_slide_ids(args, index):
        target = official_artifact_score_path(args.output_root, slide_id)
        if target.exists() and not args.overwrite:
            print(f"Skipping existing official GrandQC artifact scores for {slide_id}: {target}")
            continue
        row = require_slide_row(index, slide_id).copy()
        if not row.get("source_canny_patchset_path"):
            canny_paths = index_canny_patchsets(args.source_canny_patchsets_root)
            row["source_canny_patchset_path"] = str(canny_paths.get(slide_id, ""))
        canny = load_or_compute_canny(args, row)
        frame = map_official_artifact_mask_to_patches(args=args, row=row, canny=canny)
        frame.to_parquet(target, index=False)
        write_status(
            args.output_root,
            slide_id,
            {
                "official_grandqc_artifact_status": "complete",
                "official_grandqc_artifact_score_path": str(target),
                "official_grandqc_artifact_rows": int(len(frame)),
                "official_grandqc_artifact_mask_path": str(
                    frame["official_grandqc_artifact_mask_path"].iloc[0]
                )
                if len(frame)
                else "",
            },
        )
        print(f"Wrote {len(frame)} official GrandQC artifact rows for {slide_id} -> {target}")


def compose_patch_frame(
    *,
    args: argparse.Namespace,
    row: pd.Series,
    canny: pd.DataFrame,
    resnet: pd.DataFrame,
    official_artifact: pd.DataFrame,
    annotations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    slide_id = str(row["id"])
    canny = canny.rename(columns={"tissue_score": "canny_tissue_score"})
    frame = canny.merge(
        resnet.drop(columns=["slide_id"], errors="ignore"),
        on=["row", "column"],
        how="left",
        validate="one_to_one",
    )
    missing = frame["resnet19_foreground_probability"].isna()
    if missing.any():
        raise ValueError(f"Missing ResNet scores for {int(missing.sum())} patches in {slide_id}")
    frame = frame.merge(
        official_artifact.drop(columns=["slide_id"], errors="ignore"),
        on=["row", "column"],
        how="left",
        validate="one_to_one",
    )
    missing = frame["official_grandqc_artifact_class_id"].isna()
    if missing.any():
        raise ValueError(
            f"Missing official GrandQC artifact scores for {int(missing.sum())} patches in {slide_id}"
        )
    if annotations is not None:
        frame = frame.merge(
            annotations,
            on=["row", "column"],
            how="left",
            validate="one_to_one",
        )
        missing = frame["has_annotation"].isna()
        if missing.any():
            raise ValueError(
                f"Missing rendered annotation labels for {int(missing.sum())} patches in {slide_id}"
            )
    frame.insert(0, "slide_id", slide_id)
    frame["context_id"] = 0
    frame["canny_threshold"] = args.canny_threshold
    frame["canny_keep"] = frame["canny_tissue_score"].astype(float) > args.canny_threshold
    frame["canny_class"] = np.where(frame["canny_keep"], "foreground", "background")
    frame["resnet19_keep"] = (
        frame["resnet19_foreground_probability"].astype(float) > args.resnet_threshold
    )
    frame["canny_export_keep"] = frame["canny_keep"]
    frame["candidate_tissue_keep"] = frame["canny_export_keep"]
    frame["candidate_tissue_rule"] = f"canny_tissue_score>{args.canny_threshold}"
    artifact_ids = frame["official_grandqc_artifact_class_id"].astype(int)
    frame["official_grandqc_artifact_light_qc_reject"] = artifact_ids.isin(
        LIGHT_QC_REJECT_ARTIFACT_CLASS_IDS
    )
    frame["official_grandqc_artifact_keep"] = (
        ~frame["official_grandqc_artifact_light_qc_reject"]
    )
    frame["official_grandqc_light_qc_reject_classes"] = ",".join(
        ARTIFACT_CLASSES[class_id] for class_id in sorted(LIGHT_QC_REJECT_ARTIFACT_CLASS_IDS)
    )
    frame["primary_training_keep"] = (
        frame["canny_export_keep"] & frame["official_grandqc_artifact_keep"]
    )
    frame["keep"] = frame["primary_training_keep"]
    for col, value in row.to_dict().items():
        if col not in frame.columns and col != "source_canny_patchset_path":
            frame[col] = value
    frame["patch_png_path"] = ""
    return frame


def build_slide(args: argparse.Namespace, slide_id: str, index: pd.DataFrame) -> None:
    output_patchset = patchset_dir(args.output_root, slide_id)
    if (
        output_patchset.joinpath("frame.parquet").exists()
        and output_patchset.joinpath("manifest.json").exists()
        and not args.overwrite
    ):
        print(f"Skipping existing final patchset for {slide_id}")
        return
    row = require_slide_row(index, slide_id)
    row = row.copy()
    if not row.get("source_canny_patchset_path"):
        canny_paths = index_canny_patchsets(args.source_canny_patchsets_root)
        row["source_canny_patchset_path"] = str(canny_paths.get(slide_id, ""))
    canny = load_or_compute_canny(args, row)
    resnet_path = resnet_score_path(args.output_root, slide_id)
    if not resnet_path.exists():
        raise FileNotFoundError(f"Missing ResNet scores for {slide_id}: {resnet_path}")
    artifact_path = official_artifact_score_path(args.output_root, slide_id)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing official GrandQC artifact scores for {slide_id}: {artifact_path}"
        )
    resnet = pd.read_parquet(resnet_path)
    official_artifact = pd.read_parquet(artifact_path)
    annotations = render_annotation_labels(args=args, row=row, canny=canny)
    frame = compose_patch_frame(
        args=args,
        row=row,
        canny=canny,
        resnet=resnet,
        official_artifact=official_artifact,
        annotations=annotations,
    )
    if args.export_kept_pngs:
        export_patch_pngs(args, row, frame)
    save_patchset(args, row, frame)
    write_status(
        args.output_root,
        slide_id,
        {
            "build_status": "complete",
            "patchset_path": str(output_patchset),
            "patch_count": int(len(frame)),
            "kept_count": int(frame["keep"].sum()),
        },
    )
    print(f"Built {slide_id}: patches={len(frame)} kept={int(frame['keep'].sum())}")


def export_patch_pngs(args: argparse.Namespace, row: pd.Series, frame: pd.DataFrame) -> None:
    slide_path = Path(row["slide_path"])
    out_dir = patches_dir(args.output_root, str(row["id"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    pyramid = open_ome_pyramid(slide_path)
    for idx, patch in frame.loc[frame["keep"]].iterrows():
        r = int(patch["row"])
        c = int(patch["column"])
        filename = f"r{r}_c{c}.png"
        target = out_dir / filename
        if not target.exists() or args.overwrite:
            image = Image.fromarray(
                read_ome_pyramid_patch(
                    pyramid=pyramid,
                    patch_level=args.patch_level,
                    row=r,
                    column=c,
                    patch_size=args.patch_size,
                )
            ).convert("RGB")
            image.save(target)
        frame.at[idx, "patch_png_path"] = str(target)


def save_patchset(args: argparse.Namespace, row: pd.Series, frame: pd.DataFrame) -> None:
    from histokit.patchset.context import PatchContext
    from histokit.patchset.manifest import PatchSetManifest
    from histokit.patchset.patchset import PatchSet

    sample = make_sample(row)
    patchset = PatchSet(
        frame=frame,
        contexts=[PatchContext(sample=sample, level=args.patch_level, patch_size=args.patch_size)],
        manifest=PatchSetManifest(created_at=utc_now()),
    )
    patchset.save(patchset_dir(args.output_root, str(row["id"])))


def run_build_mode(args: argparse.Namespace) -> None:
    ensure_dirs(args.output_root)
    index_path = args.output_root / "dataset_index_ometiff.csv"
    index = pd.read_csv(index_path) if index_path.exists() else load_index(args.index, args.slides_root)
    for slide_id in selected_slide_ids(args, index):
        build_slide(args, slide_id, index)


def is_failure_status(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "complete", "skipped", "computed_missing_source"}:
        return False
    return any(token in text for token in ("missing", "failed", "error"))


def summarize(args: argparse.Namespace) -> None:
    ensure_dirs(args.output_root)
    rows = []
    for status_file in sorted((args.output_root / "status").glob("*.json")):
        rows.append(json.loads(status_file.read_text()))
    status = pd.DataFrame(rows)
    summaries = []
    for patchset_frame in sorted((args.output_root / "patchsets").glob("*.patchset/frame.parquet")):
        slide_id = patchset_frame.parent.name.removesuffix(".patchset")
        frame = pd.read_parquet(patchset_frame)
        summaries.append(
            {
                "slide_id": slide_id,
                "patch_count": int(len(frame)),
                "kept_count": int(frame["keep"].sum()) if "keep" in frame.columns else 0,
                "candidate_tissue_count": int(frame["candidate_tissue_keep"].sum())
                if "candidate_tissue_keep" in frame.columns
                else 0,
                "canny_keep_count": int(frame["canny_keep"].sum()) if "canny_keep" in frame.columns else 0,
                "resnet_keep_count": int(frame["resnet19_keep"].sum())
                if "resnet19_keep" in frame.columns
                else 0,
                "artifact_qc_reject_count": int(
                    frame["official_grandqc_artifact_light_qc_reject"].sum()
                )
                if "official_grandqc_artifact_light_qc_reject" in frame.columns
                else 0,
                "png_count": int((frame.get("patch_png_path", "") != "").sum()),
            }
        )
    summary = pd.DataFrame(summaries)
    export_dir = args.output_root / "exports"
    if not summary.empty:
        summary.to_csv(export_dir / "slide_summary.csv", index=False)
    if not status.empty:
        status.to_csv(export_dir / "status_summary.csv", index=False)
    manifests = []
    for patchset_frame in sorted((args.output_root / "patchsets").glob("*.patchset/frame.parquet")):
        frame = pd.read_parquet(patchset_frame)
        if "keep" in frame.columns:
            manifests.append(frame.loc[frame["keep"]].copy())
    if manifests:
        kept = pd.concat(manifests, ignore_index=True)
        kept.to_parquet(export_dir / "kept_patch_manifest.parquet", index=False)
        kept.to_csv(export_dir / "kept_patch_manifest.csv", index=False)
    failed = pd.DataFrame()
    if not status.empty:
        failure_cols = [col for col in status.columns if col.endswith("_status")]
        if failure_cols:
            failed_mask = pd.Series(False, index=status.index)
            for col in failure_cols:
                failed_mask = failed_mask | status[col].map(is_failure_status)
            failed = status.loc[failed_mask].copy()
            failed.to_csv(export_dir / "failed_slides.csv", index=False)
    run_summary = {
        "created_at": utc_now(),
        "output_root": str(args.output_root),
        "slide_summary_rows": int(len(summary)),
        "status_rows": int(len(status)),
        "failed_status_rows": int(len(failed)),
        "kept_patch_rows": int(len(kept)) if manifests else 0,
        "keep_rule": (
            "(canny_tissue_score > canny_threshold) AND "
            "official_grandqc_artifact_class_id NOT IN [3, 4, 5]"
        ),
        "canny_threshold": args.canny_threshold,
        "resnet_threshold": args.resnet_threshold,
        "light_qc_reject_artifact_classes": {
            str(class_id): ARTIFACT_CLASSES[class_id]
            for class_id in sorted(LIGHT_QC_REJECT_ARTIFACT_CLASS_IDS)
        },
    }
    (export_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(
        json.dumps(run_summary, indent=2)
    )


def main() -> None:
    args = parse_args()
    if args.mode == "inventory":
        write_inventory(args)
    elif args.mode == "missing-chunks":
        write_missing_chunks(args)
    elif args.mode == "grandqc-score":
        run_grandqc_mode(args)
    elif args.mode == "resnet-score":
        run_resnet_mode(args)
    elif args.mode == "official-artifact-score":
        run_official_artifact_score_mode(args)
    elif args.mode == "build-slide":
        run_build_mode(args)
    elif args.mode == "summarize":
        summarize(args)


if __name__ == "__main__":
    main()
