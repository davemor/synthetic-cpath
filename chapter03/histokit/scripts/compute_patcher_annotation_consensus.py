#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd


ANNOTATION_COLUMN = "human_tissue_annotation"
DEFAULT_ANNOTATIONS_ROOT = (
    Path.home() / "Desktop" / "icaird_cervical_patcher_annotations_2026-05-29"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute patch-level majority labels and Cohen's kappa summaries "
            "from patcher sidecar annotations."
        )
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=DEFAULT_ANNOTATIONS_ROOT,
        help=(
            "Root containing annotator directories with {slide_id}.parquet and "
            "{slide_id}.json files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for derived outputs. Defaults to "
            "<annotations-root>/consensus."
        ),
    )
    parser.add_argument(
        "--annotators",
        nargs="+",
        default=None,
        help=(
            "Annotator directory names to include. Defaults to every directory "
            "under --annotations-root except the output directory."
        ),
    )
    return parser.parse_args()


def cohen_kappa(
    labels_a: pd.Series, labels_b: pd.Series
) -> dict[str, float | int | None]:
    frame = pd.DataFrame({"a": labels_a, "b": labels_b}).dropna()
    if frame.empty:
        return {
            "n_items": 0,
            "observed_agreement": None,
            "expected_agreement": None,
            "cohen_kappa": None,
        }

    frame["a"] = frame["a"].astype(bool)
    frame["b"] = frame["b"].astype(bool)
    n_items = len(frame)
    observed = float((frame["a"] == frame["b"]).mean())

    true_a = float(frame["a"].mean())
    true_b = float(frame["b"].mean())
    false_a = 1.0 - true_a
    false_b = 1.0 - true_b
    expected = true_a * true_b + false_a * false_b

    if expected == 1.0:
        kappa = 1.0 if observed == 1.0 else None
    else:
        kappa = (observed - expected) / (1.0 - expected)

    return {
        "n_items": int(n_items),
        "observed_agreement": observed,
        "expected_agreement": expected,
        "cohen_kappa": kappa,
    }


def fleiss_kappa_binary(consensus: pd.DataFrame, annotators: list[str]) -> dict[str, Any]:
    if consensus.empty:
        return {"n_items": 0, "n_raters": len(annotators), "fleiss_kappa": None}

    label_columns = [f"{annotator}_{ANNOTATION_COLUMN}" for annotator in annotators]
    labels = consensus[label_columns]
    complete = labels.notna().all(axis=1)
    labels = labels.loc[complete].astype(bool)

    if labels.empty:
        return {"n_items": 0, "n_raters": len(annotators), "fleiss_kappa": None}

    n_raters = len(label_columns)
    foreground_counts = labels.sum(axis=1).astype(float)
    background_counts = n_raters - foreground_counts
    item_agreement = (
        foreground_counts * (foreground_counts - 1)
        + background_counts * (background_counts - 1)
    ) / (n_raters * (n_raters - 1))

    mean_item_agreement = float(item_agreement.mean())
    foreground_proportion = float(foreground_counts.sum() / (len(labels) * n_raters))
    background_proportion = 1.0 - foreground_proportion
    expected_agreement = foreground_proportion**2 + background_proportion**2

    if expected_agreement == 1.0:
        kappa = 1.0 if mean_item_agreement == 1.0 else None
    else:
        kappa = (mean_item_agreement - expected_agreement) / (
            1.0 - expected_agreement
        )

    return {
        "n_items": int(len(labels)),
        "n_raters": int(n_raters),
        "mean_item_agreement": mean_item_agreement,
        "expected_agreement": expected_agreement,
        "fleiss_kappa": kappa,
    }


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def annotator_dirs(root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        dirs = [root / annotator for annotator in requested]
    else:
        dirs = sorted(path for path in root.iterdir() if path.is_dir())

    dirs = [path for path in dirs if path.name != "consensus"]
    missing = [path for path in dirs if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing annotator directories: {joined}")
    if len(dirs) < 2:
        raise ValueError("At least two annotator directories are required")
    return dirs


def load_annotator_frames(
    annotator_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotator = annotator_dir.name
    frames = []
    metadata_rows = []

    for parquet_path in sorted(annotator_dir.glob("*.parquet")):
        slide_id = parquet_path.stem
        frame = pd.read_parquet(parquet_path)
        required_columns = {"row", "column", ANNOTATION_COLUMN}
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{parquet_path} is missing columns: {sorted(missing_columns)}"
            )

        frame = frame.copy()
        frame["slide_id"] = slide_id
        frame = frame.rename(
            columns={
                ANNOTATION_COLUMN: f"{annotator}_{ANNOTATION_COLUMN}",
            }
        )
        frames.append(
            frame[
                [
                    "slide_id",
                    "row",
                    "column",
                    f"{annotator}_{ANNOTATION_COLUMN}",
                ]
            ]
        )

        metadata = load_metadata(parquet_path.with_suffix(".json"))
        metadata_rows.append(
            {
                "annotator": annotator,
                "slide_id": slide_id,
                "parquet_path": str(parquet_path),
                "json_path": str(parquet_path.with_suffix(".json")),
                **metadata,
            }
        )

    if not frames:
        raise ValueError(f"No parquet files found under {annotator_dir}")

    return pd.concat(frames, ignore_index=True), pd.DataFrame(metadata_rows)


def merge_annotator_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    consensus = frames[0]
    for frame in frames[1:]:
        consensus = consensus.merge(
            frame,
            on=["slide_id", "row", "column"],
            how="outer",
            validate="one_to_one",
        )
    return consensus.sort_values(["slide_id", "row", "column"]).reset_index(drop=True)


def add_majority_columns(consensus: pd.DataFrame, annotators: list[str]) -> pd.DataFrame:
    label_columns = [f"{annotator}_{ANNOTATION_COLUMN}" for annotator in annotators]
    labels = consensus[label_columns]
    foreground_votes = labels.eq(True).sum(axis=1)
    background_votes = labels.eq(False).sum(axis=1)
    n_annotators = labels.notna().sum(axis=1)

    consensus = consensus.copy()
    consensus["n_annotators"] = n_annotators.astype(int)
    consensus["foreground_votes"] = foreground_votes.astype(int)
    consensus["background_votes"] = background_votes.astype(int)
    consensus[f"majority_{ANNOTATION_COLUMN}"] = foreground_votes > background_votes
    consensus["majority_label"] = consensus[f"majority_{ANNOTATION_COLUMN}"].map(
        {True: "foreground", False: "background"}
    )
    consensus["unanimous"] = (
        (foreground_votes == n_annotators) | (background_votes == n_annotators)
    ) & (n_annotators > 0)
    consensus["any_missing_annotation"] = n_annotators < len(annotators)
    return consensus


def pairwise_kappas(
    consensus: pd.DataFrame, annotators: list[str], by_slide: bool
) -> pd.DataFrame:
    rows = []
    groups: list[tuple[str, pd.DataFrame]]
    if by_slide:
        groups = list(consensus.groupby("slide_id", sort=True))
    else:
        groups = [("ALL_SLIDES", consensus)]

    for slide_id, group in groups:
        for annotator_a, annotator_b in combinations(annotators, 2):
            column_a = f"{annotator_a}_{ANNOTATION_COLUMN}"
            column_b = f"{annotator_b}_{ANNOTATION_COLUMN}"
            stats = cohen_kappa(group[column_a], group[column_b])
            rows.append(
                {
                    "slide_id": slide_id,
                    "annotator_a": annotator_a,
                    "annotator_b": annotator_b,
                    **stats,
                }
            )

    return pd.DataFrame(rows)


def write_outputs(
    consensus: pd.DataFrame,
    metadata: pd.DataFrame,
    overall_kappas: pd.DataFrame,
    slide_kappas: pd.DataFrame,
    fleiss: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    consensus.to_parquet(output_dir / "patch_consensus.parquet", index=False)
    consensus.to_csv(output_dir / "patch_consensus.csv", index=False)

    metadata.to_parquet(output_dir / "annotation_metadata.parquet", index=False)
    metadata.to_csv(output_dir / "annotation_metadata.csv", index=False)

    overall_kappas.to_csv(output_dir / "overall_pairwise_cohen_kappa.csv", index=False)
    slide_kappas.to_csv(output_dir / "per_slide_pairwise_cohen_kappa.csv", index=False)

    summary = {
        "patch_count": int(len(consensus)),
        "complete_patch_count": int((~consensus["any_missing_annotation"]).sum()),
        "unanimous_patch_count": int(consensus["unanimous"].sum()),
        "foreground_majority_count": int(
            consensus[f"majority_{ANNOTATION_COLUMN}"].sum()
        ),
        "background_majority_count": int(
            (~consensus[f"majority_{ANNOTATION_COLUMN}"]).sum()
        ),
        "overall_pairwise_cohen_kappa": overall_kappas.to_dict(orient="records"),
        "overall_fleiss_kappa": fleiss,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    annotations_root = args.annotations_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else annotations_root / "consensus"
    )

    dirs = annotator_dirs(annotations_root, args.annotators)
    annotators = [path.name for path in dirs]

    frames = []
    metadata_frames = []
    for annotator_dir in dirs:
        frame, metadata = load_annotator_frames(annotator_dir)
        frames.append(frame)
        metadata_frames.append(metadata)

    consensus = merge_annotator_frames(frames)
    consensus = add_majority_columns(consensus, annotators)
    metadata = pd.concat(metadata_frames, ignore_index=True)
    overall_kappas = pairwise_kappas(consensus, annotators, by_slide=False)
    slide_kappas = pairwise_kappas(consensus, annotators, by_slide=True)
    fleiss = fleiss_kappa_binary(consensus, annotators)

    write_outputs(
        consensus=consensus,
        metadata=metadata,
        overall_kappas=overall_kappas,
        slide_kappas=slide_kappas,
        fleiss=fleiss,
        output_dir=output_dir,
    )

    print(f"Annotators: {', '.join(annotators)}")
    print(f"Patch rows: {len(consensus)}")
    print(f"Complete rows: {(~consensus['any_missing_annotation']).sum()}")
    print(f"Output directory: {output_dir}")
    print()
    print(overall_kappas.to_string(index=False))
    print()
    print("Fleiss kappa:", fleiss)


if __name__ == "__main__":
    main()
