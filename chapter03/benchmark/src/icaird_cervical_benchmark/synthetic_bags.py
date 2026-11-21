from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import SyntheticBagConfig, SyntheticEmbeddingConfig
from .synthetic_embeddings import EDMLatentDiffusion, EMA, _make_diffusion


SPLITS = ("train", "valid", "test")
GENERATOR_NAME = "conditional_edm_mlp_lognormal_multinomial"


def fit_bag_description_model(config: SyntheticBagConfig) -> Path:
    """Fit train-only bag-size and patch-label composition models."""
    _validate_config(config)
    output_dir = config.output_dir
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    counts = load_real_bag_patch_label_counts(config)
    counts.to_csv(metadata_dir / "real_bag_patch_label_counts.csv", index=False)

    train = counts[counts["split"] == "train"].copy()
    if train.empty:
        raise ValueError("Cannot fit synthetic bag model: no train bags were found.")
    label_cols = list(config.patch_class_order)
    count_rows = []
    composition_rows = []
    category_rows = []
    for subcategory, group in train.groupby("subcategory", sort=True):
        patch_counts = group["patch_count"].to_numpy(dtype=np.float64)
        if np.any(patch_counts <= 0):
            raise ValueError(f"Subcategory {subcategory!r} has non-positive patch counts.")
        log_counts = np.log(patch_counts)
        count_rows.append(
            {
                "subcategory": subcategory,
                "n_bags": int(len(group)),
                "min_count": int(patch_counts.min()),
                "max_count": int(patch_counts.max()),
                "median_count": float(np.median(patch_counts)),
                "mean_count": float(patch_counts.mean()),
                "log_count_mean": float(log_counts.mean()),
                "log_count_std": float(max(log_counts.std(ddof=0), 1.0e-6)),
            }
        )
        totals = group[label_cols].sum(axis=0).astype(float)
        total = float(totals.sum())
        probs = totals / total if total > 0 else pd.Series(1.0 / len(label_cols), index=label_cols)
        composition_rows.append(
            {
                "subcategory": subcategory,
                **{f"p_{label}": float(probs[label]) for label in label_cols},
                **{f"count_{label}": int(totals[label]) for label in label_cols},
                "total_patches": int(total),
            }
        )
        category_rows.append(
            {
                "subcategory": subcategory,
                "category": str(group["category"].mode(dropna=False).iloc[0]),
            }
        )

    subcategory_counts = train["subcategory"].value_counts().sort_index()
    subcategory_probs = (
        subcategory_counts / subcategory_counts.sum()
    ).rename_axis("subcategory").reset_index(name="probability")
    subcategory_probs["n_bags"] = subcategory_probs["subcategory"].map(subcategory_counts).astype(int)

    pd.DataFrame(count_rows).to_csv(metadata_dir / "count_model_lognormal.csv", index=False)
    pd.DataFrame(composition_rows).to_csv(metadata_dir / "composition_model_multinomial.csv", index=False)
    pd.DataFrame(category_rows).to_csv(metadata_dir / "subcategory_to_category.csv", index=False)
    subcategory_probs.to_csv(metadata_dir / "subcategory_model.csv", index=False)
    (metadata_dir / "config_snapshot.json").write_text(json.dumps(_jsonable_config(config), indent=2))
    (metadata_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generator": GENERATOR_NAME,
                "real_bag_root": str(config.real_bag_root),
                "fit_split": "train",
                "count_model": config.count_model,
                "composition_model": config.composition_model,
                "patch_class_order": config.patch_class_order,
                "normal_label_aliases": config.normal_label_aliases,
                "n_train_bags": int(len(train)),
                "n_all_bags": int(len(counts)),
            },
            indent=2,
        )
    )
    return metadata_dir


def load_real_bag_patch_label_counts(config: SyntheticBagConfig) -> pd.DataFrame:
    _validate_config(config)
    assert config.real_bag_root is not None
    index_path = config.real_bag_root / "dataset_index_ometiff.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Expected bag index at {index_path}.")
    slide_index = pd.read_csv(index_path)
    required = {"id", "split", "category", "subcategory"}
    missing = sorted(required - set(slide_index.columns))
    if missing:
        raise ValueError(f"{index_path} is missing columns {missing}.")
    rows = []
    for row in tqdm(slide_index.itertuples(index=False), total=len(slide_index), desc="count real bags", unit="bag"):
        row_data = row._asdict()
        slide_id = str(row_data["id"])
        patch_index_path = _embedding_slide_dir(config.real_bag_root, config.embedding_name, slide_id) / "index.parquet"
        if not patch_index_path.exists():
            raise FileNotFoundError(f"Missing patch index for {slide_id}: {patch_index_path}")
        patch_index = pd.read_parquet(patch_index_path, columns=[config.patch_label_column])
        labels = patch_index[config.patch_label_column].astype(str).map(lambda value: _normalise_patch_label(value, config))
        value_counts = labels.value_counts().to_dict()
        count_row = {
            "slide_id": slide_id,
            "split": str(row_data["split"]),
            "category": str(row_data["category"]),
            "subcategory": str(row_data["subcategory"]),
            "patch_count": int(len(labels)),
        }
        for label in config.patch_class_order:
            count_row[label] = int(value_counts.get(label, 0))
        unexpected = sorted(set(value_counts) - set(config.patch_class_order))
        if unexpected:
            raise ValueError(f"{slide_id} has unexpected patch labels after normalization: {unexpected}")
        rows.append(count_row)
    return pd.DataFrame(rows)


def sample_bag_descriptions(config: SyntheticBagConfig, multiplier: float) -> Path:
    metadata_dir = config.output_dir / "metadata"
    if not (metadata_dir / "manifest.json").exists():
        metadata_dir = fit_bag_description_model(config)
    output_dir = _generated_output_dir(config, multiplier)
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptions = _sample_bag_description_frame(config, metadata_dir, multiplier)
    descriptions.to_csv(output_dir / "bag_descriptions.csv", index=False)
    return output_dir / "bag_descriptions.csv"


def generate_synthetic_bag_datasets(
    bag_config: SyntheticBagConfig,
    embedding_config: SyntheticEmbeddingConfig,
    checkpoint_path: Path | None = None,
) -> list[Path]:
    """Generate synthetic slide bags and MIL-ready roots for configured sample multipliers."""
    metadata_dir = fit_bag_description_model(bag_config)
    checkpoint_path = checkpoint_path or embedding_config.output_dir / "checkpoints" / "best.pt"
    sampler = _EDMPatchSampler(embedding_config, checkpoint_path)
    written_roots = []
    for multiplier in bag_config.sample_multipliers:
        generated_root = _generated_output_dir(bag_config, multiplier)
        generated_root.mkdir(parents=True, exist_ok=True)
        descriptions = _sample_bag_description_frame(bag_config, metadata_dir, multiplier)
        descriptions.to_csv(generated_root / "bag_descriptions.csv", index=False)
        _write_generated_bags(generated_root, descriptions, bag_config, sampler, checkpoint_path)
        if bag_config.materialize_synthetic_only:
            written_roots.append(
                materialize_mil_root(
                    bag_config,
                    generated_root,
                    root_name=f"synthetic_only_{_ratio_label(multiplier)}x",
                    include_real_train=False,
                )
            )
        if bag_config.materialize_real_plus_synthetic:
            written_roots.append(
                materialize_mil_root(
                    bag_config,
                    generated_root,
                    root_name=f"real_plus_synthetic_{_ratio_label(multiplier)}x",
                    include_real_train=True,
                )
            )
        (generated_root / "manifest.json").write_text(
            json.dumps(
                {
                    "generator": GENERATOR_NAME,
                    "multiplier": float(multiplier),
                    "num_synthetic_bags": int(len(descriptions)),
                    "num_synthetic_patches": int(descriptions["patch_count"].sum()),
                    "checkpoint": str(checkpoint_path),
                    "mil_roots": [str(path) for path in written_roots if _ratio_label(multiplier) in path.name],
                },
                indent=2,
            )
        )
    return written_roots


def generate_synthetic_bag_shard(
    bag_config: SyntheticBagConfig,
    embedding_config: SyntheticEmbeddingConfig,
    *,
    multiplier: float,
    num_shards: int,
    shard_index: int,
    checkpoint_path: Path | None = None,
) -> Path:
    """Generate one deterministic shard of synthetic slide bags."""
    _validate_shard_args(num_shards, shard_index)
    metadata_dir = bag_config.output_dir / "metadata"
    if not (metadata_dir / "manifest.json").exists():
        metadata_dir = fit_bag_description_model(bag_config)
    generated_root = _generated_output_dir(bag_config, multiplier)
    generated_root.mkdir(parents=True, exist_ok=True)

    description_path = generated_root / "bag_descriptions.csv"
    if description_path.exists():
        descriptions = pd.read_csv(description_path)
    else:
        descriptions = _sample_bag_description_frame(bag_config, metadata_dir, multiplier)
        descriptions.to_csv(description_path, index=False)
    shard_descriptions = descriptions.iloc[shard_index::num_shards].copy()

    shard_dir = _shard_dir(generated_root, num_shards, shard_index)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_descriptions.to_csv(shard_dir / "bag_descriptions.csv", index=False)

    checkpoint_path = checkpoint_path or embedding_config.output_dir / "checkpoints" / "best.pt"
    sampler = _EDMPatchSampler(embedding_config, checkpoint_path)
    _write_generated_bags(
        generated_root,
        shard_descriptions,
        bag_config,
        sampler,
        checkpoint_path,
        index_output_path=shard_dir / "dataset_index_ometiff.csv",
    )
    (shard_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generator": GENERATOR_NAME,
                "multiplier": float(multiplier),
                "num_shards": int(num_shards),
                "shard_index": int(shard_index),
                "num_synthetic_bags": int(len(shard_descriptions)),
                "num_synthetic_patches": int(shard_descriptions["patch_count"].sum()),
                "checkpoint": str(checkpoint_path),
            },
            indent=2,
        )
    )
    return shard_dir


def finalize_synthetic_bag_shards(
    bag_config: SyntheticBagConfig,
    *,
    multiplier: float,
    num_shards: int,
    checkpoint_path: Path | None = None,
) -> list[Path]:
    """Combine shard indices and materialize MIL roots for one multiplier."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    generated_root = _generated_output_dir(bag_config, multiplier)
    description_path = generated_root / "bag_descriptions.csv"
    if not description_path.exists():
        raise FileNotFoundError(f"Expected full bag descriptions at {description_path}.")
    descriptions = pd.read_csv(description_path)

    shard_indices = []
    for shard_index in range(num_shards):
        shard_index_path = _shard_dir(generated_root, num_shards, shard_index) / "dataset_index_ometiff.csv"
        if not shard_index_path.exists():
            raise FileNotFoundError(f"Missing generated shard index: {shard_index_path}")
        shard_indices.append(pd.read_csv(shard_index_path))
    synthetic_index = pd.concat(shard_indices, ignore_index=True)
    expected_ids = set(descriptions["slide_id"].astype(str))
    actual_ids = set(synthetic_index["id"].astype(str))
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(
            "Shard finalization found mismatched slide IDs: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    order = {slide_id: idx for idx, slide_id in enumerate(descriptions["slide_id"].astype(str))}
    synthetic_index["_order"] = synthetic_index["id"].astype(str).map(order)
    synthetic_index = synthetic_index.sort_values("_order").drop(columns=["_order"])
    synthetic_index.to_csv(generated_root / "dataset_index_ometiff.csv", index=False)

    written_roots = []
    if bag_config.materialize_synthetic_only:
        written_roots.append(
            materialize_mil_root(
                bag_config,
                generated_root,
                root_name=f"synthetic_only_{_ratio_label(multiplier)}x",
                include_real_train=False,
            )
        )
    if bag_config.materialize_real_plus_synthetic:
        written_roots.append(
            materialize_mil_root(
                bag_config,
                generated_root,
                root_name=f"real_plus_synthetic_{_ratio_label(multiplier)}x",
                include_real_train=True,
            )
        )

    (generated_root / "manifest.json").write_text(
        json.dumps(
            {
                "generator": GENERATOR_NAME,
                "multiplier": float(multiplier),
                "num_shards": int(num_shards),
                "num_synthetic_bags": int(len(descriptions)),
                "num_synthetic_patches": int(descriptions["patch_count"].sum()),
                "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
                "mil_roots": [str(path) for path in written_roots],
            },
            indent=2,
        )
    )
    return written_roots


def materialize_mil_root(
    config: SyntheticBagConfig,
    generated_root: Path,
    *,
    root_name: str,
    include_real_train: bool,
) -> Path:
    _validate_config(config)
    assert config.real_bag_root is not None
    root = config.output_dir / "mil_roots" / root_name
    slide_dir = root / "embeddings" / config.embedding_name / "slides"
    slide_dir.mkdir(parents=True, exist_ok=True)

    real_index = pd.read_csv(config.real_bag_root / "dataset_index_ometiff.csv")
    rows = []
    real_splits = ["valid", "test"] if config.include_real_valid_test else []
    if include_real_train:
        real_splits = ["train", *real_splits]
    for row in real_index.itertuples(index=False):
        row_data = row._asdict()
        split = str(row_data["split"])
        if split not in real_splits:
            continue
        slide_id = str(row_data["id"])
        source = _embedding_slide_dir(config.real_bag_root, config.embedding_name, slide_id)
        destination = slide_dir / source.name
        _ensure_link_or_copy(source, destination)
        rows.append(row_data)

    synthetic_index = pd.read_csv(generated_root / "dataset_index_ometiff.csv")
    generated_slide_dir = generated_root / "embeddings" / config.embedding_name / "slides"
    for row in synthetic_index.itertuples(index=False):
        row_data = row._asdict()
        slide_id = str(row_data["id"])
        source = generated_slide_dir / f"{slide_id}.{config.embedding_name}.embedding"
        destination = slide_dir / source.name
        _ensure_link_or_copy(source, destination)
        rows.append(row_data)

    pd.DataFrame(rows).to_csv(root / "dataset_index_ometiff.csv", index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "generator": GENERATOR_NAME,
                "source_generated_root": str(generated_root),
                "source_real_bag_root": str(config.real_bag_root),
                "include_real_train": include_real_train,
                "include_real_valid_test": config.include_real_valid_test,
                "num_rows": int(len(rows)),
            },
            indent=2,
        )
    )
    return root


class _EDMPatchSampler:
    def __init__(self, config: SyntheticEmbeddingConfig, checkpoint_path: Path) -> None:
        if config.model_mode != "edm_conditional":
            raise ValueError(
                "Synthetic bag generation is configured for the Conditional EDM MLP; "
                f"got synthetic_embeddings.model_mode={config.model_mode!r}."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.class_names = [str(value) for value in checkpoint["class_names"]]
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = _make_diffusion(config, len(self.class_names)).to(self.device)
        self.diffusion.load_state_dict(checkpoint["model"])
        if "ema" in checkpoint:
            ema = EMA(self.diffusion, config.ema_decay)
            ema.load_state_dict(checkpoint["ema"])
            ema.copy_to(self.diffusion)
        self.diffusion.eval()
        stats = np.load(config.output_dir / "metadata" / "train_stats.npz")
        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)

    def sample_bag(
        self,
        label_counts: dict[str, int],
        *,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        features_by_label = []
        labels_by_label = []
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.no_grad():
            for label_name in self.class_names:
                count = int(label_counts.get(label_name, 0))
                if count <= 0:
                    continue
                label_id = self.class_to_id[label_name]
                chunks = []
                remaining = count
                while remaining > 0:
                    batch_size = min(self.config.batch_size, remaining)
                    labels = torch.full((batch_size,), label_id, dtype=torch.long, device=self.device)
                    batch_z = self.diffusion.sample(
                        labels,
                        steps=self.config.ddim_steps,
                        device=self.device,
                        generator=generator,
                        guidance_scale=self.config.guidance_scale
                        if isinstance(self.diffusion, EDMLatentDiffusion)
                        else 1.0,
                    )
                    chunks.append(batch_z.cpu().numpy())
                    remaining -= batch_size
                z = np.vstack(chunks).astype(np.float32)
                features_by_label.append((z * self.std + self.mean).astype(np.float32))
                labels_by_label.append(np.full(count, label_id, dtype=np.int64))
        if not features_by_label:
            raise ValueError(f"Cannot sample empty bag from label_counts={label_counts}.")
        features = np.vstack(features_by_label).astype(np.float32)
        labels = np.concatenate(labels_by_label)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(labels))
        return features[order], labels[order]


def _write_generated_bags(
    generated_root: Path,
    descriptions: pd.DataFrame,
    config: SyntheticBagConfig,
    sampler: _EDMPatchSampler,
    checkpoint_path: Path,
    *,
    index_output_path: Path | None = None,
) -> None:
    slide_root = generated_root / "embeddings" / config.embedding_name / "slides"
    slide_root.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for row in tqdm(descriptions.itertuples(index=False), total=len(descriptions), desc="sample synthetic bags", unit="bag"):
        row_data = row._asdict()
        slide_id = str(row_data["slide_id"])
        label_counts = {label: int(row_data[label]) for label in config.patch_class_order}
        features, labels = sampler.sample_bag(label_counts, seed=int(row_data["seed"]))
        slide_dir = slide_root / f"{slide_id}.{config.embedding_name}.embedding"
        slide_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(slide_dir / "features.npz", features=features)
        patch_rows = []
        for patch_idx, label_id in enumerate(labels.astype(int)):
            label_name = sampler.class_names[label_id]
            patch_rows.append(
                {
                    "slide_id": slide_id,
                    "row": -1,
                    "column": -1,
                    "frame_index": patch_idx,
                    "patch_png_path": "",
                    config.patch_label_column: label_name,
                    "annotation_label_name": label_name,
                    "category": row_data["category"],
                    "subcategory": row_data["subcategory"],
                    "split": "train",
                    "synthetic": True,
                    "generator": GENERATOR_NAME,
                    "checkpoint": str(checkpoint_path),
                    "seed": int(row_data["seed"]),
                }
            )
        pd.DataFrame(patch_rows).to_parquet(slide_dir / "index.parquet", index=False)
        (slide_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "generator": GENERATOR_NAME,
                    "slide_id": slide_id,
                    "num_embeddings": int(features.shape[0]),
                    "embedding_dim": int(features.shape[1]),
                    "label_counts": label_counts,
                    "checkpoint": str(checkpoint_path),
                    "seed": int(row_data["seed"]),
                },
                indent=2,
            )
        )
        index_rows.append(_synthetic_slide_index_row(slide_id, row_data["category"], row_data["subcategory"]))
    index_output_path = index_output_path or generated_root / "dataset_index_ometiff.csv"
    index_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(index_rows).to_csv(index_output_path, index=False)


def _sample_bag_description_frame(
    config: SyntheticBagConfig,
    metadata_dir: Path,
    multiplier: float,
) -> pd.DataFrame:
    if multiplier <= 0:
        raise ValueError("sample multiplier must be positive.")
    rng = np.random.default_rng(config.seed + int(round(multiplier * 1000)))
    counts = pd.read_csv(metadata_dir / "real_bag_patch_label_counts.csv")
    train = counts[counts["split"] == "train"]
    n_bags = max(1, int(round(len(train) * multiplier)))
    if config.max_bags_per_multiplier is not None:
        n_bags = min(n_bags, int(config.max_bags_per_multiplier))
    subcategory_model = pd.read_csv(metadata_dir / "subcategory_model.csv")
    count_model = pd.read_csv(metadata_dir / "count_model_lognormal.csv").set_index("subcategory")
    composition_model = pd.read_csv(metadata_dir / "composition_model_multinomial.csv").set_index("subcategory")
    category_model = pd.read_csv(metadata_dir / "subcategory_to_category.csv").set_index("subcategory")["category"]

    subcategories = subcategory_model["subcategory"].astype(str).to_numpy()
    probabilities = subcategory_model["probability"].to_numpy(dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    sampled_subcategories = rng.choice(subcategories, size=n_bags, replace=True, p=probabilities)
    rows = []
    for idx, subcategory in enumerate(sampled_subcategories):
        count_row = count_model.loc[subcategory]
        total = int(round(math.exp(rng.normal(count_row["log_count_mean"], count_row["log_count_std"]))))
        total = max(1, total)
        if config.clamp_total_count_to_train_range:
            total = int(np.clip(total, int(count_row["min_count"]), int(count_row["max_count"])))
        probs = np.asarray(
            [composition_model.loc[subcategory, f"p_{label}"] for label in config.patch_class_order],
            dtype=np.float64,
        )
        probs = probs / probs.sum()
        label_counts = rng.multinomial(total, probs)
        slide_id = f"SYN-CX-{_ratio_label(multiplier)}x-{idx:06d}"
        rows.append(
            {
                "slide_id": slide_id,
                "split": "train",
                "category": str(category_model.loc[subcategory]),
                "subcategory": str(subcategory),
                "patch_count": int(total),
                "seed": int(config.seed + idx + int(round(multiplier * 10_000))),
                **{label: int(label_counts[label_idx]) for label_idx, label in enumerate(config.patch_class_order)},
            }
        )
    return pd.DataFrame(rows)


def _synthetic_slide_index_row(slide_id: str, category: str, subcategory: str) -> dict[str, object]:
    return {
        "slide": f"{slide_id}.synthetic.ome.tiff",
        "annotation": f"{slide_id}.synthetic.geojson",
        "category": category,
        "subcategory": subcategory,
        "split": "train",
        "id": slide_id,
        "slide_path": "",
        "slide_exists": True,
        "source_canny_patchset_path": "",
        "source_canny_exists": True,
        "synthetic": True,
        "generator": GENERATOR_NAME,
    }


def _normalise_patch_label(value: str, config: SyntheticBagConfig) -> str:
    if value in set(config.normal_label_aliases):
        return "normal"
    return value


def _embedding_slide_dir(root: Path, embedding_name: str, slide_id: str) -> Path:
    return root / "embeddings" / embedding_name / "slides" / f"{slide_id}.{embedding_name}.embedding"


def _generated_output_dir(config: SyntheticBagConfig, multiplier: float) -> Path:
    return config.output_dir / "generated" / f"{GENERATOR_NAME}_{_ratio_label(multiplier)}x"


def _shard_dir(generated_root: Path, num_shards: int, shard_index: int) -> Path:
    return generated_root / "shards" / f"shard_{shard_index:03d}_of_{num_shards:03d}"


def _validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}); got {shard_index}.")


def _ratio_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _ensure_link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _validate_config(config: SyntheticBagConfig) -> None:
    if config.real_bag_root is None:
        raise ValueError("synthetic_bags.real_bag_root is required.")
    if config.count_model != "lognormal":
        raise ValueError("Only count_model='lognormal' is currently implemented.")
    if config.composition_model != "multinomial":
        raise ValueError("Only composition_model='multinomial' is currently implemented.")
    if not config.patch_class_order:
        raise ValueError("synthetic_bags.patch_class_order cannot be empty.")


def _jsonable_config(config: SyntheticBagConfig) -> dict[str, object]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data
