#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from histokit.embeddings.io import read_embedding_result


DEFAULT_BAGS_ROOT = Path(
    "/home/dm236/scratch/data/icaird_cervical/bags/"
    "full_canny005_or_resnet_light_grandqc_artifact"
)
DEFAULT_EMBEDDINGS_ROOT = DEFAULT_BAGS_ROOT / "embeddings/uni2_h"
DEFAULT_OUTPUT_ROOT = Path(
    "/home/dm236/scratch/data/icaird_cervical/"
    "icaird_cervical_100k_uni2_kcenter"
)
PATCH_ID_COLUMNS = ["slide_id", "row", "column"]
SPLIT_QUOTAS = {"train": 80_000, "valid": 10_000, "test": 10_000}
DEFAULT_SLIDE_SPLIT_STRATIFY_COLUMNS = "category,subcategory"
QUOTA_STRATEGIES = (
    "subcategory",
    "joint_label_subcategory",
    "label_balanced_joint_subcategory",
)
REQUIRED_COLUMNS = {
    "slide_id",
    "row",
    "column",
    "split",
    "category",
    "subcategory",
    "patch_png_path",
}
GRANDQC_TISSUE_CLASS_ID = 1
GRANDQC_TISSUE_CLASS_NAME = "normal_tissue"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the iCAIRD cervical 100K UNI2 k-center patch dataset from "
            "existing bag PNGs and completed per-slide UNI2 embeddings."
        )
    )
    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "build-candidate-manifest",
            "select-kcenter",
            "export-images",
            "summarize",
        ),
        default="all",
    )
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_BAGS_ROOT)
    parser.add_argument(
        "--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=236)
    parser.add_argument("--train-quota", type=int, default=SPLIT_QUOTAS["train"])
    parser.add_argument("--valid-quota", type=int, default=SPLIT_QUOTAS["valid"])
    parser.add_argument("--test-quota", type=int, default=SPLIT_QUOTAS["test"])
    parser.add_argument(
        "--source-official-split",
        type=str,
        default=None,
        help="Filter candidates to this original split and preserve it as official_split.",
    )
    parser.add_argument(
        "--generated-slide-splits",
        action="store_true",
        help="Create new slide-level train/valid/test splits before patch sampling.",
    )
    parser.add_argument(
        "--slide-split-stratify-columns",
        type=str,
        default=DEFAULT_SLIDE_SPLIT_STRATIFY_COLUMNS,
        help="Comma-separated slide-level columns used to stratify generated splits.",
    )
    parser.add_argument(
        "--quota-strategy",
        choices=QUOTA_STRATEGIES,
        default="subcategory",
        help="Patch quota hierarchy within each split.",
    )
    parser.add_argument(
        "--grandqc-majority-tissue-only",
        action="store_true",
        help=(
            "Keep only patches where the GrandQC majority artifact class is "
            "1 | normal_tissue."
        ),
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=128,
        help="Use 0 to run k-center in the original embedding dimension.",
    )
    parser.add_argument("--limit-slides", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quotas = {
        "train": args.train_quota,
        "valid": args.valid_quota,
        "test": args.test_quota,
    }

    if args.mode in ("all", "build-candidate-manifest"):
        candidates = build_candidate_manifest(args, quotas)
    else:
        candidates = read_candidate_manifest(args.output_root)
        candidates = prepare_candidates_for_sampling(candidates, args, quotas)

    if args.mode in ("all", "select-kcenter"):
        sample, random_baseline, quota_frame, summary = select_kcenter_sample(
            candidates=candidates,
            split_quotas=quotas,
            args=args,
        )
        write_selection_outputs(
            sample=sample,
            random_baseline=random_baseline,
            quota_frame=quota_frame,
            summary=summary,
            args=args,
        )

    if args.mode in ("all", "export-images"):
        export_images(args)

    if args.mode in ("all", "summarize"):
        summarize(args)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stable_seed(*parts: object, seed: int) -> int:
    value = seed & 0xFFFFFFFF
    for part in parts:
        for byte in str(part).encode("utf-8"):
            value = ((value * 131) + byte) & 0xFFFFFFFF
    return value


def slide_embedding_dirs(root: Path) -> list[Path]:
    slides_root = root / "slides"
    if not slides_root.exists():
        raise FileNotFoundError(f"Embedding slides directory not found: {slides_root}")
    return sorted(slides_root.glob("*.embedding"))


def build_candidate_manifest(
    args: argparse.Namespace, split_quotas: dict[str, int]
) -> pd.DataFrame:
    paths = slide_embedding_dirs(args.embeddings_root)
    if args.limit_slides is not None:
        paths = paths[: args.limit_slides]
    if not paths:
        raise ValueError(f"No embedding directories found under {args.embeddings_root}")

    manifests_dir(args.output_root).mkdir(parents=True, exist_ok=True)
    rows = []
    for path in tqdm(paths, desc="Reading embedding indices"):
        index = pd.read_parquet(path / "index.parquet")
        validate_candidate_columns(index, path)
        index = index.copy()
        index["embedding_path"] = str(path)
        index["source_patchset_path"] = index.get(
            "source_patchset_path",
            str(args.bags_root / "patchsets" / f"{slide_id_from_embedding_path(path)}.patchset"),
        )
        index["source_patch_png_path"] = index["patch_png_path"].astype(str)
        rows.append(index)

    candidates = pd.concat(rows, ignore_index=True)
    candidates["slide_id"] = candidates["slide_id"].astype(str)
    candidates["split"] = candidates["split"].astype(str)
    candidates["category"] = candidates["category"].astype(str)
    candidates["subcategory"] = candidates["subcategory"].astype(str)
    candidates["row"] = candidates["row"].astype(int)
    candidates["column"] = candidates["column"].astype(int)
    candidates = candidates.sort_values(PATCH_ID_COLUMNS).reset_index(drop=True)
    candidates = prepare_candidates_for_sampling(candidates, args, split_quotas)

    candidate_path = manifests_dir(args.output_root) / "candidate_manifest.parquet"
    if candidate_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Candidate manifest already exists: {candidate_path}. Use --overwrite."
        )
    candidates.to_parquet(candidate_path, index=False)
    candidates.to_csv(
        manifests_dir(args.output_root) / "candidate_manifest.csv", index=False
    )
    return candidates


def validate_candidate_columns(frame: pd.DataFrame, path: Path) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} index is missing columns: {sorted(missing)}")


def slide_id_from_embedding_path(path: Path) -> str:
    name = path.name
    if name.endswith(".uni2_h.embedding"):
        return name.removesuffix(".uni2_h.embedding")
    return name.removesuffix(".embedding").split(".", 1)[0]


def read_candidate_manifest(output_root: Path) -> pd.DataFrame:
    path = manifests_dir(output_root) / "candidate_manifest.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate manifest not found: {path}. Run build-candidate-manifest."
        )
    return pd.read_parquet(path)


def prepare_candidates_for_sampling(
    candidates: pd.DataFrame, args: argparse.Namespace, split_quotas: dict[str, int]
) -> pd.DataFrame:
    candidates = candidates.copy()
    if "official_split" not in candidates.columns:
        candidates["official_split"] = candidates["split"].astype(str)

    if args.source_official_split:
        candidates = candidates.loc[
            candidates["official_split"].astype(str) == str(args.source_official_split)
        ].copy()
        if candidates.empty:
            raise ValueError(
                f"No candidates remain after filtering to official split "
                f"{args.source_official_split!r}."
            )

    if getattr(args, "grandqc_majority_tissue_only", False):
        candidates = filter_grandqc_majority_tissue(candidates)

    if args.generated_slide_splits:
        stratify_columns = parse_column_list(args.slide_split_stratify_columns)
        slide_splits = generate_slide_split_assignments(
            candidates,
            split_quotas=split_quotas,
            stratify_columns=stratify_columns,
            seed=args.seed,
        )
        candidates["split"] = candidates["slide_id"].map(slide_splits)
        if candidates["split"].isna().any():
            raise ValueError("Generated slide split assignment missing for candidates.")

    candidates["split"] = candidates["split"].astype(str)
    return candidates.sort_values(PATCH_ID_COLUMNS).reset_index(drop=True)


def filter_grandqc_majority_tissue(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "official_grandqc_artifact_class_id",
        "official_grandqc_artifact_class",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(
            "GrandQC majority tissue filter requires columns: "
            f"{sorted(required)}. Missing: {sorted(missing)}"
        )

    filtered = candidates.loc[
        candidates["official_grandqc_artifact_class_id"].astype(str)
        == str(GRANDQC_TISSUE_CLASS_ID)
    ].copy()
    filtered = filtered.loc[
        filtered["official_grandqc_artifact_class"].astype(str)
        == GRANDQC_TISSUE_CLASS_NAME
    ].copy()
    if filtered.empty:
        raise ValueError(
            "No candidates remain after filtering to GrandQC majority tissue "
            f"({GRANDQC_TISSUE_CLASS_ID} | {GRANDQC_TISSUE_CLASS_NAME})."
        )
    return filtered


def parse_column_list(value: str) -> list[str]:
    columns = [column.strip() for column in value.split(",") if column.strip()]
    if not columns:
        raise ValueError("At least one column is required.")
    return columns


def generate_slide_split_assignments(
    candidates: pd.DataFrame,
    *,
    split_quotas: dict[str, int],
    stratify_columns: list[str],
    seed: int,
) -> dict[str, str]:
    missing = set(stratify_columns) - set(candidates.columns)
    if missing:
        raise ValueError(f"Slide split stratify columns missing: {sorted(missing)}")

    per_slide_counts = candidates.groupby("slide_id")[stratify_columns].nunique(
        dropna=False
    )
    multi_value = per_slide_counts.gt(1).any(axis=1)
    if multi_value.any():
        examples = multi_value.loc[multi_value].head(10).index.tolist()
        raise ValueError(
            "Generated slide splits require one stratify value per slide; "
            f"examples with multiple values: {examples}"
        )

    slide_frame = (
        candidates[["slide_id", *stratify_columns]]
        .drop_duplicates()
        .sort_values(["slide_id"])
        .reset_index(drop=True)
    )
    split_order = [split for split, quota in split_quotas.items() if int(quota) > 0]
    if not split_order:
        raise ValueError("At least one split quota must be positive.")

    assignments: dict[str, str] = {}
    groupby_key = stratify_columns[0] if len(stratify_columns) == 1 else stratify_columns
    for values, group in slide_frame.groupby(groupby_key, dropna=False, sort=True):
        slides = group["slide_id"].astype(str).to_numpy()
        rng = np.random.default_rng(
            stable_seed("slide-split", values, len(slides), seed=seed)
        )
        shuffled = slides[rng.permutation(len(slides))]
        counts = allocate_split_counts(len(shuffled), split_quotas)
        offset = 0
        for split in split_order:
            count = counts.get(split, 0)
            for slide_id in shuffled[offset : offset + count]:
                assignments[str(slide_id)] = split
            offset += count

    if len(assignments) != slide_frame["slide_id"].nunique():
        raise ValueError("Failed to assign every slide to a generated split.")
    return assignments


def allocate_split_counts(total: int, split_quotas: dict[str, int]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    positive = {split: int(quota) for split, quota in split_quotas.items() if int(quota) > 0}
    counts = {split: 0 for split in split_quotas}
    if total == 0 or not positive:
        return counts

    split_order = list(positive)
    if total >= len(split_order):
        for split in split_order:
            counts[split] = 1
        remaining = total - len(split_order)
    else:
        for split in sorted(split_order, key=lambda split: (-positive[split], split))[:total]:
            counts[split] = 1
        return counts

    if remaining == 0:
        return counts

    total_weight = sum(positive.values())
    ideals = {split: remaining * positive[split] / total_weight for split in split_order}
    additions = {split: int(np.floor(value)) for split, value in ideals.items()}
    leftover = remaining - sum(additions.values())
    for split, value in sorted(
        ideals.items(),
        key=lambda item: (-(item[1] - np.floor(item[1])), -positive[item[0]], item[0]),
    )[:leftover]:
        additions[split] += 1
    for split, count in additions.items():
        counts[split] += count
    return counts


def select_kcenter_sample(
    *,
    candidates: pd.DataFrame,
    split_quotas: dict[str, int],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    quota_frame, summary = allocate_dataset_quotas(
        candidates, split_quotas, quota_strategy=args.quota_strategy
    )
    selected_parts = []
    random_parts = []

    grouped_quotas = quota_frame.loc[quota_frame["slide_quota"] > 0]
    iterator = grouped_quotas.itertuples(index=False)
    for quota in tqdm(list(iterator), desc="Selecting k-center patches"):
        group = candidates.loc[
            quota_mask(candidates, quota, summary["quota_columns"])
        ].copy()
        group = group.sort_values(["row", "column"]).reset_index(drop=True)
        selected = kcenter_select_group(
            group=group,
            quota=int(quota.slide_quota),
            seed=args.seed,
            pca_components=args.pca_components,
        )
        selected_parts.append(add_quota_columns(selected, quota))

        rng = np.random.default_rng(
            stable_seed(*quota_seed_parts(quota, summary["quota_columns"]), seed=args.seed)
        )
        random_positions = np.sort(
            rng.choice(len(group), size=int(quota.slide_quota), replace=False)
        )
        random_parts.append(add_quota_columns(group.iloc[random_positions].copy(), quota))

    sample = finalize_sample_manifest(
        pd.concat(selected_parts, ignore_index=True),
        args=args,
        sampling_method="uni2_kcenter",
    )
    random_baseline = finalize_sample_manifest(
        pd.concat(random_parts, ignore_index=True),
        args=args,
        sampling_method="random_quota_matched",
    )
    summary["selected_count"] = int(len(sample))
    summary["random_baseline_count"] = int(len(random_baseline))
    return sample, random_baseline, quota_frame, summary


def allocate_dataset_quotas(
    candidates: pd.DataFrame,
    split_quotas: dict[str, int],
    *,
    quota_strategy: str = "subcategory",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    quota_columns = quota_strategy_columns(quota_strategy)
    if quota_strategy == "label_balanced_joint_subcategory":
        return allocate_label_balanced_joint_subcategory_quotas(
            candidates, split_quotas, quota_columns=quota_columns
        )

    capacities = (
        candidates.groupby(["split", *quota_columns, "slide_id"], dropna=False)
        .size()
        .rename("available_patches")
        .reset_index()
    )
    rows = []
    split_summaries = {}
    for split, requested in split_quotas.items():
        split_capacities = capacities.loc[capacities["split"] == split].copy()
        total_available = int(split_capacities["available_patches"].sum())
        if total_available < requested:
            raise ValueError(
                f"Split {split!r} has {total_available} candidates, "
                f"below requested quota {requested}."
            )
        cell_caps = (
            split_capacities.groupby(quota_columns, dropna=False)["available_patches"]
            .sum()
            .to_dict()
        )
        cell_quotas = allocate_evenly(cell_caps, requested)
        split_summaries[split] = {
            "requested_quota": int(requested),
            "available_patches": total_available,
            "quota_cell_count": int(len(cell_caps)),
        }

        for cell_key, cell_quota in cell_quotas.items():
            cell_values = tuple(cell_key if isinstance(cell_key, tuple) else (cell_key,))
            mask = split_capacities["split"] == split
            for column, value in zip(quota_columns, cell_values, strict=True):
                mask &= split_capacities[column].eq(value)
            slide_caps = split_capacities.loc[mask].set_index("slide_id")[
                "available_patches"
            ].to_dict()
            slide_quotas = allocate_evenly(slide_caps, cell_quota)
            for slide_id, slide_quota in slide_quotas.items():
                available = int(slide_caps[slide_id])
                row = {
                    "split": split,
                    "slide_id": slide_id,
                    "available_patches": available,
                    "split_quota": int(requested),
                    "cell_quota": int(cell_quota),
                    "subcategory_quota": int(cell_quota),
                    "slide_quota": int(slide_quota),
                }
                row.update(dict(zip(quota_columns, cell_values, strict=True)))
                rows.append(row)

    quota_frame = pd.DataFrame(rows).sort_values(["split", *quota_columns, "slide_id"])
    summary = {
        "created_at": utc_now(),
        "quota_policy": quota_policy_text(quota_strategy),
        "quota_strategy": quota_strategy,
        "quota_columns": quota_columns,
        "split_summaries": split_summaries,
    }
    return quota_frame.reset_index(drop=True), summary


def quota_strategy_columns(quota_strategy: str) -> list[str]:
    if quota_strategy == "subcategory":
        return ["category", "subcategory"]
    if quota_strategy in (
        "joint_label_subcategory",
        "label_balanced_joint_subcategory",
    ):
        return ["annotation_label", "annotation_label_name", "category", "subcategory"]
    raise ValueError(f"Unknown quota strategy: {quota_strategy}")


def quota_policy_text(quota_strategy: str) -> str:
    if quota_strategy == "subcategory":
        return (
            "Equal split quotas as requested; within each split, quotas are "
            "allocated as evenly as possible across subcategories, then as "
            "evenly as possible across slides, capped by available patches."
        )
    if quota_strategy == "joint_label_subcategory":
        return (
            "Equal split quotas as requested; within each split, quotas are "
            "allocated as evenly as possible across annotation-label and "
            "subcategory cells, then as evenly as possible across slides, "
            "capped by available patches."
        )
    if quota_strategy == "label_balanced_joint_subcategory":
        return (
            "Equal split quotas as requested; within each split, quotas are "
            "allocated as evenly as possible across annotation labels first. "
            "Within each label, quotas are allocated as evenly as possible "
            "across subcategory cells, then as evenly as possible across "
            "slides, capped by available patches."
        )
    raise ValueError(f"Unknown quota strategy: {quota_strategy}")


def allocate_label_balanced_joint_subcategory_quotas(
    candidates: pd.DataFrame,
    split_quotas: dict[str, int],
    *,
    quota_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_columns = ["annotation_label", "annotation_label_name"]
    cell_columns = quota_columns
    capacities = (
        candidates.groupby(["split", *cell_columns, "slide_id"], dropna=False)
        .size()
        .rename("available_patches")
        .reset_index()
    )

    rows = []
    split_summaries = {}
    for split, requested in split_quotas.items():
        split_capacities = capacities.loc[capacities["split"] == split].copy()
        total_available = int(split_capacities["available_patches"].sum())
        if total_available < requested:
            raise ValueError(
                f"Split {split!r} has {total_available} candidates, "
                f"below requested quota {requested}."
            )

        label_caps = (
            split_capacities.groupby(label_columns, dropna=False)["available_patches"]
            .sum()
            .to_dict()
        )
        label_quotas = allocate_strict_evenly(label_caps, requested)
        split_summaries[split] = {
            "requested_quota": int(requested),
            "available_patches": total_available,
            "label_count": int(len(label_caps)),
            "quota_cell_count": int(
                split_capacities.groupby(cell_columns, dropna=False).ngroups
            ),
            "label_quotas": {
                str(key): int(value) for key, value in label_quotas.items()
            },
        }

        for label_key, label_quota in label_quotas.items():
            label_values = tuple(
                label_key if isinstance(label_key, tuple) else (label_key,)
            )
            label_mask = split_capacities["split"] == split
            for column, value in zip(label_columns, label_values, strict=True):
                label_mask &= split_capacities[column].eq(value)
            label_capacities = split_capacities.loc[label_mask]
            cell_caps = (
                label_capacities.groupby(cell_columns, dropna=False)[
                    "available_patches"
                ]
                .sum()
                .to_dict()
            )
            cell_quotas = allocate_evenly(cell_caps, label_quota)

            for cell_key, cell_quota in cell_quotas.items():
                cell_values = tuple(
                    cell_key if isinstance(cell_key, tuple) else (cell_key,)
                )
                cell_mask = split_capacities["split"] == split
                for column, value in zip(cell_columns, cell_values, strict=True):
                    cell_mask &= split_capacities[column].eq(value)
                slide_caps = split_capacities.loc[cell_mask].set_index("slide_id")[
                    "available_patches"
                ].to_dict()
                slide_quotas = allocate_evenly(slide_caps, cell_quota)
                for slide_id, slide_quota in slide_quotas.items():
                    available = int(slide_caps[slide_id])
                    row = {
                        "split": split,
                        "slide_id": slide_id,
                        "available_patches": available,
                        "split_quota": int(requested),
                        "label_quota": int(label_quota),
                        "cell_quota": int(cell_quota),
                        "subcategory_quota": int(cell_quota),
                        "slide_quota": int(slide_quota),
                    }
                    row.update(dict(zip(cell_columns, cell_values, strict=True)))
                    rows.append(row)

    quota_frame = pd.DataFrame(rows).sort_values(["split", *cell_columns, "slide_id"])
    summary = {
        "created_at": utc_now(),
        "quota_policy": quota_policy_text("label_balanced_joint_subcategory"),
        "quota_strategy": "label_balanced_joint_subcategory",
        "quota_columns": quota_columns,
        "label_columns": label_columns,
        "split_summaries": split_summaries,
    }
    return quota_frame.reset_index(drop=True), summary


def quota_mask(candidates: pd.DataFrame, quota: Any, quota_columns: list[str]) -> pd.Series:
    values = quota._asdict()
    mask = (candidates["split"] == values["split"]) & (
        candidates["slide_id"] == values["slide_id"]
    )
    for column in quota_columns:
        mask &= candidates[column].eq(values[column])
    return mask


def quota_seed_parts(quota: Any, quota_columns: list[str]) -> list[object]:
    values = quota._asdict()
    return [values["split"], *(values[column] for column in quota_columns), values["slide_id"]]


def allocate_evenly(capacities: dict[Any, int], target: int) -> dict[Any, int]:
    if target < 0:
        raise ValueError("target must be non-negative")
    clean = {key: int(value) for key, value in capacities.items() if int(value) > 0}
    if sum(clean.values()) < target:
        raise ValueError("Total capacity is below requested target")
    quotas = {key: 0 for key in clean}
    remaining = int(target)
    active = sorted(clean, key=lambda key: str(key))

    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        next_active = []
        changed = False
        for key in active:
            spare = clean[key] - quotas[key]
            add = min(spare, share, remaining)
            if add > 0:
                quotas[key] += add
                remaining -= add
                changed = True
            if quotas[key] < clean[key]:
                next_active.append(key)
            if remaining == 0:
                break
        if not changed:
            break
        active = next_active

    if remaining != 0:
        raise ValueError(f"Failed to allocate {remaining} remaining items")
    return quotas


def allocate_strict_evenly(capacities: dict[Any, int], target: int) -> dict[Any, int]:
    if target < 0:
        raise ValueError("target must be non-negative")
    clean = {key: int(value) for key, value in capacities.items() if int(value) > 0}
    if not clean and target > 0:
        raise ValueError("No capacity available for target allocation")

    keys = sorted(clean, key=lambda key: str(key))
    base = target // len(keys) if keys else 0
    remainder = target % len(keys) if keys else 0
    quotas = {
        key: int(base + (position < remainder)) for position, key in enumerate(keys)
    }
    insufficient = {
        key: {"available": clean[key], "requested": quota}
        for key, quota in quotas.items()
        if clean[key] < quota
    }
    if insufficient:
        raise ValueError(
            "Strict even allocation exceeds capacity for one or more groups: "
            f"{insufficient}"
        )
    return quotas


def kcenter_select_group(
    *,
    group: pd.DataFrame,
    quota: int,
    seed: int,
    pca_components: int,
) -> pd.DataFrame:
    if quota <= 0:
        return group.iloc[[]].copy()
    if len(group) <= quota:
        selected = group.copy()
        selected["kcenter_rank"] = np.arange(len(selected), dtype=int)
        return selected

    features = load_group_features(group)
    features = l2_normalize(features)
    if pca_components > 0 and features.shape[1] > pca_components:
        features = pca_reduce(features, pca_components, seed=seed)
        features = l2_normalize(features)

    positions = farthest_first_positions(features, quota)
    selected = group.iloc[positions].copy()
    selected["kcenter_rank"] = np.arange(len(selected), dtype=int)
    return selected


def load_group_features(group: pd.DataFrame) -> np.ndarray:
    embedding_paths = group["embedding_path"].astype(str).unique()
    if len(embedding_paths) != 1:
        raise ValueError("K-center groups must come from exactly one embedding path")
    result = read_embedding_result(Path(embedding_paths[0]))
    index = result.index.copy()
    index["_feature_position"] = np.arange(len(index), dtype=int)
    merged = group[PATCH_ID_COLUMNS].merge(
        index[PATCH_ID_COLUMNS + ["_feature_position"]],
        on=PATCH_ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if merged["_feature_position"].isna().any():
        raise ValueError(f"Embedding alignment failed for {embedding_paths[0]}")
    positions = merged["_feature_position"].astype(int).to_numpy()
    return np.asarray(result.features[positions], dtype=np.float32)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (features / norms).astype(np.float32, copy=False)


def pca_reduce(features: np.ndarray, components: int, *, seed: int) -> np.ndarray:
    if features.shape[0] <= components:
        return features
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(features).astype(np.float32, copy=False)


def farthest_first_positions(features: np.ndarray, quota: int) -> list[int]:
    centroid = features.mean(axis=0, dtype=np.float64)
    distances = np.sum((features - centroid.astype(np.float32)) ** 2, axis=1)
    first = int(np.argmin(distances))
    selected = [first]
    min_distances = np.sum((features - features[first]) ** 2, axis=1)
    min_distances[first] = -1.0

    while len(selected) < quota:
        position = int(np.argmax(min_distances))
        selected.append(position)
        new_distances = np.sum((features - features[position]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[selected] = -1.0
    return selected


def add_quota_columns(frame: pd.DataFrame, quota: Any) -> pd.DataFrame:
    frame = frame.copy()
    values = quota._asdict()
    frame["split_quota"] = int(quota.split_quota)
    if "label_quota" in values:
        frame["label_quota"] = int(values["label_quota"])
    frame["subcategory_quota"] = int(quota.subcategory_quota)
    frame["slide_quota"] = int(quota.slide_quota)
    return frame


def finalize_sample_manifest(
    frame: pd.DataFrame,
    *,
    args: argparse.Namespace,
    sampling_method: str,
) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame.sort_values(["split", "subcategory", "slide_id", "row", "column"])
    frame = frame.reset_index(drop=True)
    frame["sampling_method"] = sampling_method
    frame["sampling_seed"] = int(args.seed)
    frame["source_patch_png_path"] = frame["source_patch_png_path"].astype(str)
    frame["sample_id"] = [
        f"icaird_cervical_{i:06d}" for i in range(len(frame))
    ]
    frame["patch_png_path"] = [
        str(
            args.output_root
            / "images"
            / str(split)
            / str(subcategory)
            / f"{sample_id}.png"
        )
        for split, subcategory, sample_id in frame[
            ["split", "subcategory", "sample_id"]
        ].itertuples(index=False, name=None)
    ]
    first_columns = [
        "sample_id",
        "slide_id",
        "row",
        "column",
        "split",
        "official_split",
        "category",
        "subcategory",
        "annotation_label",
        "annotation_label_name",
        "patch_png_path",
        "source_patch_png_path",
        "source_patchset_path",
        "sampling_method",
        "sampling_seed",
        "split_quota",
        "label_quota",
        "subcategory_quota",
        "slide_quota",
    ]
    ordered = [column for column in first_columns if column in frame.columns]
    ordered.extend(column for column in frame.columns if column not in ordered)
    return frame[ordered]


def write_selection_outputs(
    *,
    sample: pd.DataFrame,
    random_baseline: pd.DataFrame,
    quota_frame: pd.DataFrame,
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output_root = args.output_root
    manifests_dir(output_root).mkdir(parents=True, exist_ok=True)
    sample_path = manifests_dir(output_root) / "sample_manifest.parquet"
    if sample_path.exists() and not args.overwrite:
        raise FileExistsError(f"Sample manifest exists: {sample_path}. Use --overwrite.")

    sample.to_parquet(sample_path, index=False)
    sample.to_csv(manifests_dir(output_root) / "sample_manifest.csv", index=False)
    random_baseline.to_parquet(
        manifests_dir(output_root) / "random_baseline_manifest.parquet",
        index=False,
    )
    quota_frame.to_csv(
        manifests_dir(output_root) / "quota_by_split_subcategory_slide.csv",
        index=False,
    )
    summary["grandqc_majority_tissue_only"] = bool(
        getattr(args, "grandqc_majority_tissue_only", False)
    )
    if summary["grandqc_majority_tissue_only"]:
        summary["grandqc_majority_tissue_filter"] = {
            "official_grandqc_artifact_class_id": GRANDQC_TISSUE_CLASS_ID,
            "official_grandqc_artifact_class": GRANDQC_TISSUE_CLASS_NAME,
        }
    for split, split_frame in sample.groupby("split", sort=True):
        split_frame.to_parquet(
            manifests_dir(output_root) / f"sample_manifest_{split}.parquet",
            index=False,
        )
    with open(output_root / "sampling_summary.json", "w") as file:
        json.dump(summary, file, indent=2)


def export_images(args: argparse.Namespace) -> None:
    sample = read_sample_manifest(args.output_root)
    if export_complete(sample) and not args.overwrite:
        print("Image export already complete; skipping.")
        return

    for row in tqdm(sample.itertuples(index=False), total=len(sample), desc="Copying PNGs"):
        source = Path(row.source_patch_png_path)
        target = Path(row.patch_png_path)
        if not source.exists():
            raise FileNotFoundError(f"Source PNG not found: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not args.overwrite:
            continue
        shutil.copy2(source, target)


def export_complete(sample: pd.DataFrame) -> bool:
    paths = [Path(path) for path in sample["patch_png_path"].astype(str)]
    return len(paths) == len(sample) and all(path.exists() for path in paths)


def summarize(args: argparse.Namespace) -> None:
    sample = read_sample_manifest(args.output_root)
    output_root = args.output_root
    counts = {
        "created_at": utc_now(),
        "output_root": str(output_root),
        "sample_manifest_rows": int(len(sample)),
        "png_count": int(
            sum(Path(path).exists() for path in sample["patch_png_path"].astype(str))
        ),
        "duplicate_patch_ids": int(sample.duplicated(PATCH_ID_COLUMNS).sum()),
        "counts_by_split": nested_counts(sample, ["split"]),
        "counts_by_split_category_subcategory": nested_counts(
            sample, ["split", "category", "subcategory"]
        ),
        "counts_by_split_annotation_label_subcategory": nested_counts(
            sample,
            [
                "split",
                "annotation_label",
                "annotation_label_name",
                "category",
                "subcategory",
            ],
        )
        if {
            "annotation_label",
            "annotation_label_name",
            "category",
            "subcategory",
        }
        <= set(sample.columns)
        else {},
        "counts_by_official_split": nested_counts(sample, ["official_split"])
        if "official_split" in sample.columns
        else {},
        "counts_by_annotation_label": nested_counts(
            sample, ["annotation_label", "annotation_label_name"]
        )
        if {"annotation_label", "annotation_label_name"} <= set(sample.columns)
        else {},
        "counts_by_grandqc_artifact_class": nested_counts(
            sample,
            [
                "official_grandqc_artifact_class_id",
                "official_grandqc_artifact_class",
            ],
        )
        if {
            "official_grandqc_artifact_class_id",
            "official_grandqc_artifact_class",
        }
        <= set(sample.columns)
        else {},
        "histokit_git_commit": git_commit(Path(__file__).resolve().parents[1]),
    }
    with open(output_root / "manifest.json", "w") as file:
        json.dump(
            {
                "created_at": utc_now(),
                "dataset": output_root.name,
                "sampling_method": "uni2_kcenter",
                "sample_count": int(len(sample)),
                "source_bags_root": str(args.bags_root),
                "source_embeddings_root": str(args.embeddings_root),
                "grandqc_majority_tissue_only": bool(
                    getattr(args, "grandqc_majority_tissue_only", False)
                ),
                "files": {
                    "sample_manifest": "manifests/sample_manifest.parquet",
                    "dataset_index": "dataset_index.csv",
                    "sampling_summary": "sampling_summary.json",
                },
            },
            file,
            indent=2,
        )
    with open(output_root / "dataset_summary.json", "w") as file:
        json.dump(counts, file, indent=2)
    sample.to_csv(output_root / "dataset_index.csv", index=False)


def nested_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    counts = frame.groupby(columns, dropna=False).size().sort_index()
    return {" | ".join(map(str, key if isinstance(key, tuple) else (key,))): int(value)
            for key, value in counts.items()}


def read_sample_manifest(output_root: Path) -> pd.DataFrame:
    path = manifests_dir(output_root) / "sample_manifest.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Sample manifest not found: {path}")
    return pd.read_parquet(path)


def manifests_dir(output_root: Path) -> Path:
    return output_root / "manifests"


def git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    main()
