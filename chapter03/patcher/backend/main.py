from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from pathlib import Path

import histokit.io.slides  # noqa: F401 - register slide backends
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from histokit.dataset.dataset import Dataset
from histokit.dataset.sample import Sample
from pydantic import BaseModel, Field

from backend.config import AppConfig, load_app_config_from_yaml


ANNOTATION_COLUMN = "human_tissue_annotation"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "icarid-cervical.yml"

app = FastAPI()

frontend_dist: Path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dist), name="static")

load_dotenv()


def load_runtime_config() -> AppConfig:
    config_path = Path(os.getenv("PATCHER_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    return load_app_config_from_yaml(str(config_path))


def require_path_env(name: str) -> Path:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            "Set it before starting patcher."
        )
    return Path(value).expanduser()


app_config: AppConfig = load_runtime_config()
DATA_ROOT = require_path_env("PATCHER_DATA_ROOT")
CACHE_ROOT = require_path_env("PATCHER_CACHE_ROOT")

dataset = Dataset.from_index(
    DATA_ROOT / app_config.dataset.index,
    DATA_ROOT / app_config.dataset.labels,
    slides_dir=app_config.dataset.slides_dir,
    annotations_dir=app_config.dataset.annotations_dir,
)
dataset.index["slide_id"] = dataset.index.apply(
    lambda row: dataset.sample_from_row(row).id, axis=1
)


class PatchAnnotation(BaseModel):
    row: int
    column: int
    human_tissue_annotation: bool


class PatchsetSaveRequest(BaseModel):
    annotations: list[PatchAnnotation] = Field(default_factory=list)
    annotation_threshold: float | None = None
    manual_edit_count: int = Field(default=0, ge=0)


def get_annotators() -> list[str]:
    return list(app_config.annotators)


def resolve_annotator_id(annotator_id: str | None) -> str:
    annotators = get_annotators()
    if not annotators:
        raise HTTPException(
            status_code=500,
            detail="No annotators are configured for this patcher instance",
        )

    resolved = annotator_id or annotators[0]
    if resolved not in annotators:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown annotator '{resolved}'",
        )

    return resolved


def get_sample_for_slide(slide_idx: int) -> Sample:
    if slide_idx < 0 or slide_idx >= len(dataset.index):
        raise HTTPException(status_code=404, detail="Slide index out of bounds")

    return dataset.sample_from_row(dataset.index.iloc[slide_idx])


def get_patchset_dir(sample: Sample) -> Path:
    return CACHE_ROOT / app_config.patchsets_dir / f"{sample.id}.patchset"


def get_patch_export_root() -> Path:
    return CACHE_ROOT / app_config.patches_dir


def label_name_for_patch(sample: Sample, annotation_label: object) -> str:
    if pd.isna(annotation_label):
        return "unlabelled"

    if sample.annotation_schema is None:
        return str(int(annotation_label))

    value_to_name = {
        value: name for name, value in sample.annotation_schema.label_map.items()
    }
    return value_to_name.get(int(annotation_label), str(int(annotation_label)))


def get_patch_path(sample: Sample, frame: pd.DataFrame, x: int, y: int) -> Path:
    patch_rows = frame[(frame["row"] == y) & (frame["column"] == x)]
    if patch_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Patch coordinates ({x}, {y}) not found for slide {sample.id}",
        )

    patch_row = patch_rows.iloc[0]
    patch_name = f"{sample.id}_{int(patch_row['row'])}_{int(patch_row['column'])}.png"
    label_name = label_name_for_patch(sample, patch_row.get("annotation_label"))
    patch_path = get_patch_export_root() / label_name / patch_name

    if patch_path.exists():
        return patch_path

    matches = sorted(get_patch_export_root().glob(f"*/{patch_name}"))
    if matches:
        return matches[0]

    raise HTTPException(
        status_code=404,
        detail=f"Patch not found for slide {sample.id} at ({x}, {y})",
    )


def get_thumbnail_path(sample: Sample) -> Path:
    thumbnail_name = (
        f"{sample.id}.ome_patchmap_L{app_config.patch_level}"
        f"_P{app_config.patch_size}_S{app_config.patch_display_size}.png"
    )
    return CACHE_ROOT / app_config.thumbnails_dir / thumbnail_name


def get_annotation_paths(sample: Sample, annotator_id: str) -> tuple[Path, Path]:
    annotation_dir = CACHE_ROOT / app_config.annotation_sets_dir / annotator_id
    return annotation_dir / f"{sample.id}.parquet", annotation_dir / f"{sample.id}.json"


def load_patchset_frame(slide_idx: int) -> tuple[Sample, Path, pd.DataFrame]:
    sample = get_sample_for_slide(slide_idx)
    patchset_dir = get_patchset_dir(sample)
    frame_path = patchset_dir / "frame.parquet"

    if not patchset_dir.exists() or not frame_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Patchset not found for slide {sample.id}",
        )

    frame = pd.read_parquet(frame_path)
    missing_columns = {"row", "column", "tissue_score", "keep"} - set(frame.columns)
    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=f"Patchset frame is missing required columns: {sorted(missing_columns)}",
        )

    return sample, patchset_dir, frame


def expected_patch_coordinates(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return {
        (int(row), int(column))
        for row, column in frame[["row", "column"]].itertuples(index=False, name=None)
    }


def load_annotation_frame(sample: Sample, annotator_id: str) -> pd.DataFrame | None:
    annotation_path, _ = get_annotation_paths(sample, annotator_id)
    if not annotation_path.exists():
        return None

    annotation_frame = pd.read_parquet(annotation_path)
    required_columns = {"row", "column", ANNOTATION_COLUMN}
    missing_columns = required_columns - set(annotation_frame.columns)
    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Annotation sidecar for {sample.id}/{annotator_id} is missing "
                f"required columns: {sorted(missing_columns)}"
            ),
        )

    return annotation_frame


def load_annotation_metadata(sample: Sample, annotator_id: str) -> dict:
    _, metadata_path = get_annotation_paths(sample, annotator_id)
    if not metadata_path.exists():
        return {}

    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Annotation metadata for {sample.id}/{annotator_id} is invalid JSON",
        ) from exc


def merge_annotation_frame(
    sample: Sample, annotator_id: str, frame: pd.DataFrame
) -> pd.DataFrame:
    merged_frame = frame.copy()
    merged_frame[ANNOTATION_COLUMN] = pd.NA

    annotation_frame = load_annotation_frame(sample, annotator_id)
    if annotation_frame is None:
        return merged_frame

    annotation_map = {
        (int(row), int(column)): bool(annotation)
        for row, column, annotation in annotation_frame[
            ["row", "column", ANNOTATION_COLUMN]
        ].itertuples(index=False, name=None)
    }
    expected_coordinates = expected_patch_coordinates(frame)

    if (
        len(annotation_map) != len(annotation_frame)
        or set(annotation_map) != expected_coordinates
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Annotation sidecar for {sample.id}/{annotator_id} does not "
                "match patchset coordinates"
            ),
        )

    merged_frame[ANNOTATION_COLUMN] = [
        annotation_map[(int(row), int(column))]
        for row, column in merged_frame[["row", "column"]].itertuples(
            index=False, name=None
        )
    ]
    return merged_frame


def save_annotation_frame(
    sample: Sample,
    annotator_id: str,
    frame: pd.DataFrame,
    annotation_map: dict[tuple[int, int], bool],
    annotation_threshold: float | None,
    manual_edit_count: int,
) -> None:
    annotation_path, metadata_path = get_annotation_paths(sample, annotator_id)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)

    annotation_frame = frame[["row", "column"]].copy()
    annotation_frame[ANNOTATION_COLUMN] = [
        annotation_map[(int(row), int(column))]
        for row, column in annotation_frame[["row", "column"]].itertuples(
            index=False, name=None
        )
    ]

    tmp_annotation_path = annotation_path.with_suffix(".tmp.parquet")
    annotation_frame.to_parquet(tmp_annotation_path, index=False)
    tmp_annotation_path.replace(annotation_path)

    metadata = {
        "slide_id": sample.id,
        "annotator_id": annotator_id,
        "patch_count": int(len(annotation_frame)),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "annotation_column": ANNOTATION_COLUMN,
        "annotation_threshold": annotation_threshold,
        "manual_edit_count": int(manual_edit_count),
    }
    tmp_metadata_path = metadata_path.with_suffix(".json.tmp")
    tmp_metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    tmp_metadata_path.replace(metadata_path)


def build_patchset_response(sample: Sample, frame: pd.DataFrame) -> dict:
    has_annotation_column = ANNOTATION_COLUMN in frame.columns
    patches = []

    for patch in frame.itertuples(index=False):
        annotation_value = getattr(patch, ANNOTATION_COLUMN, None)
        patches.append(
            {
                "row": int(patch.row),
                "column": int(patch.column),
                "tissue_score": float(patch.tissue_score),
                "keep": bool(patch.keep),
                ANNOTATION_COLUMN: (
                    bool(annotation_value)
                    if has_annotation_column and pd.notna(annotation_value)
                    else None
                ),
            }
        )

    with sample.open_slide() as slide:
        level_dimensions = slide.dimensions[app_config.patch_level]

    return {
        "slide_id": sample.id,
        "grid_rows": int(frame["row"].max()) + 1 if not frame.empty else 0,
        "grid_columns": int(frame["column"].max()) + 1 if not frame.empty else 0,
        "patch_count": len(patches),
        "default_tissue_threshold": float(app_config.default_tissue_threshold),
        "scrollable_thumbnail_viewport": bool(
            app_config.scrollable_thumbnail_viewport
        ),
        "patch_size": int(app_config.patch_size),
        "level_width": int(level_dimensions.width),
        "level_height": int(level_dimensions.height),
        "patches": patches,
    }


@app.get("/")
async def read_root() -> FileResponse:
    return FileResponse(frontend_dist / "index.html")


@app.get("/dataset")
async def get_dataset_index() -> JSONResponse:
    index_data = dataset.index.to_dict(orient="records")
    return JSONResponse(content=index_data)


@app.get("/annotators")
async def get_configured_annotators() -> JSONResponse:
    annotators = get_annotators()
    return JSONResponse(
        content={
            "annotators": annotators,
            "default_annotator_id": annotators[0] if annotators else None,
        }
    )


@app.get("/dataset/{slide_idx}/thumbnail")
async def get_slide_thumbnail(slide_idx: int) -> FileResponse:
    sample = get_sample_for_slide(slide_idx)
    thumbnail_path = get_thumbnail_path(sample)
    if not thumbnail_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Thumbnail not found for slide {sample.id}",
        )
    return FileResponse(thumbnail_path, media_type="image/png")


@app.get("/dataset/{slide_idx}/patch")
async def get_slide_patch(slide_idx: int, x: int, y: int) -> FileResponse:
    sample, _, frame = load_patchset_frame(slide_idx)
    patch_path = get_patch_path(sample, frame, x, y)
    return FileResponse(patch_path, media_type="image/png")


@app.get("/dataset/{slide_idx}/patchset")
async def get_slide_patchset(
    slide_idx: int, annotator_id: str | None = Query(default=None)
) -> JSONResponse:
    resolved_annotator_id = resolve_annotator_id(annotator_id)
    sample, _, frame = load_patchset_frame(slide_idx)
    annotated_frame = merge_annotation_frame(sample, resolved_annotator_id, frame)
    response = build_patchset_response(sample, annotated_frame)
    metadata = load_annotation_metadata(sample, resolved_annotator_id)
    response["annotator_id"] = resolved_annotator_id
    response["annotation_threshold"] = metadata.get("annotation_threshold")
    response["manual_edit_count"] = metadata.get("manual_edit_count", 0)
    return JSONResponse(content=response)


@app.post("/dataset/{slide_idx}/patchset")
async def save_slide_patchset(
    slide_idx: int,
    payload: PatchsetSaveRequest,
    annotator_id: str | None = Query(default=None),
) -> JSONResponse:
    resolved_annotator_id = resolve_annotator_id(annotator_id)
    sample, _, frame = load_patchset_frame(slide_idx)

    annotation_map = {
        (patch.row, patch.column): bool(patch.human_tissue_annotation)
        for patch in payload.annotations
    }
    expected_coordinates = expected_patch_coordinates(frame)

    if (
        len(annotation_map) != len(payload.annotations)
        or set(annotation_map) != expected_coordinates
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Annotation payload does not match patchset coordinates "
                f"(expected {len(expected_coordinates)} patches, got {len(annotation_map)})"
            ),
        )

    save_annotation_frame(
        sample,
        resolved_annotator_id,
        frame,
        annotation_map,
        payload.annotation_threshold,
        payload.manual_edit_count,
    )

    return JSONResponse(
        content={
            "detail": (
                f"Saved patch annotations for slide {sample.id} "
                f"as {resolved_annotator_id}"
            ),
            "slide_id": sample.id,
            "annotator_id": resolved_annotator_id,
            "patch_count": len(frame),
            "annotation_threshold": payload.annotation_threshold,
            "manual_edit_count": payload.manual_edit_count,
        }
    )
