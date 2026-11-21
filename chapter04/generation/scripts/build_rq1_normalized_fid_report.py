from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import linalg


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "icaird_cervical100k": "iCAIRD",
    "pathmnist224": "Kather",
    "pcam": "PCam",
}
ARCHITECTURES = {
    "stylegan2ada": "SG2",
    "sd_lora": "SD-LoRA",
    "dcgan": "DCGAN",
}
CONDITIONING = {
    "unconditioned": "Unconditioned",
    "pseudo_label": "Global cluster",
}
FEATURE_DIMENSION = 1024
FONT_SIZE = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the normalized-FID RQ1 table and heatmap."
    )
    parser.add_argument(
        "--quality-csv",
        type=Path,
        default=ROOT / "final-results" / "quality_results_fid_ip_ir_authenticity.csv",
    )
    parser.add_argument(
        "--long-results-csv",
        type=Path,
        default=ROOT / "utility-report-sources" / "pcam.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("SYNPATH_DATA_ROOT", "/raid/dm236/synpath/data")),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("SYNPATH_CACHE_ROOT", "/raid/dm236/synpath/cache")),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(os.environ.get("SYNPATH_OUTPUTS_ROOT", "/raid/dm236/synpath/outputs")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "final-results",
    )
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fid-device", default="cuda")
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--fid-workers", type=int, default=16)
    parser.add_argument(
        "--inception-cache-dir",
        type=Path,
        help="Resume cache for matched CleanFID features.",
    )
    parser.add_argument(
        "--skip-inception-normalization",
        action="store_true",
        help="Useful for testing; thesis outputs require normalized Inception-FID.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_id_hash(values: pd.Series) -> str:
    payload = "\n".join(values.astype(str).tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_order(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    if "image_path" not in frame:
        raise ValueError("Embedding frame has no image_path column.")
    keys = frame["image_path"].astype(str).map(
        lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    )
    return frame.assign(_sample_key=keys).sort_values(
        ["_sample_key", "image_path"], kind="stable"
    ).drop(columns="_sample_key").reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame if column.startswith("f")]
    columns.sort(key=lambda column: int(column[1:]))
    return columns


def validate_embedding_frame(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    path: Path,
    expected_dimension: int = FEATURE_DIMENSION,
) -> None:
    if "image_path" not in frame or "image_path" not in manifest:
        raise ValueError(f"Missing image_path in cache or manifest for {path}")
    if frame["image_path"].duplicated().any():
        raise ValueError(f"Duplicate image_path values in {path}")
    columns = feature_columns(frame)
    if len(columns) != expected_dimension:
        raise ValueError(
            f"Expected {expected_dimension} UNI features in {path}; found {len(columns)}"
        )
    values = frame[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite UNI features in {path}")
    cache_ids = set(frame["image_path"].astype(str))
    manifest_ids = set(manifest["image_path"].astype(str))
    if cache_ids != manifest_ids:
        missing = len(manifest_ids - cache_ids)
        extra = len(cache_ids - manifest_ids)
        raise ValueError(
            f"Cache/manifest mismatch for {path}: {missing} missing, {extra} extra"
        )


def frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Fréchet inputs must be 2D with equal feature dimensions.")
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Fréchet distance requires at least two samples per population.")
    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    covariance_x = np.cov(x, rowvar=False)
    covariance_y = np.cov(y, rowvar=False)
    covariance_mean = linalg.sqrtm(covariance_x @ covariance_y)
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(covariance_mean.imag, 0.0, atol=1e-5):
            raise ValueError("Fréchet covariance product has a material imaginary part.")
        covariance_mean = covariance_mean.real
    value = (
        np.sum((mean_x - mean_y) ** 2)
        + np.trace(covariance_x + covariance_y - 2.0 * covariance_mean)
    )
    return float(max(value, 0.0))


def load_raw_results(quality_csv: Path, long_results_csv: Path) -> pd.DataFrame:
    quality = pd.read_csv(quality_csv)
    required = {
        "Architecture",
        "Dataset",
        "Conditioning",
        "Seed",
        "FID",
        "IP",
        "IR",
        "Authenticity",
    }
    missing = required - set(quality)
    if missing:
        raise ValueError(f"Quality CSV is missing columns: {sorted(missing)}")
    if len(quality) != 45:
        raise ValueError(f"Expected 45 canonical quality rows; found {len(quality)}")

    long = pd.read_csv(long_results_csv)
    uni = long[(long["family"] == "quality") & (long["metric"] == "uni_fid")].copy()
    uni["seed"] = pd.to_numeric(uni["seed"], errors="raise").astype(int)
    uni["value"] = pd.to_numeric(uni["value"], errors="raise")
    key = ["dataset", "model", "conditioning", "seed"]
    duplicates = uni.groupby(key)["value"].nunique()
    if (duplicates > 1).any():
        raise ValueError("Conflicting duplicate UNI-FID values in long-form results.")
    uni = uni.drop_duplicates(key).rename(
        columns={
            "dataset": "Dataset",
            "model": "Architecture",
            "conditioning": "Conditioning",
            "seed": "Seed",
            "value": "UNI-FID",
        }
    )
    merged = quality.merge(
        uni[["Dataset", "Architecture", "Conditioning", "Seed", "UNI-FID"]],
        on=["Dataset", "Architecture", "Conditioning", "Seed"],
        how="left",
        validate="one_to_one",
    )
    if merged["UNI-FID"].isna().any() or len(merged) != 45:
        raise ValueError("The canonical table does not have complete UNI-FID coverage.")
    rq1 = merged[merged["Conditioning"].isin(CONDITIONING)].copy()
    if len(rq1) != 18:
        raise ValueError(f"Expected 18 RQ1 rows; found {len(rq1)}")
    return rq1


def load_populations(
    rq1: pd.DataFrame,
    data_root: Path,
    cache_root: Path,
    outputs_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[tuple[str, str, str, int], pd.DataFrame], dict]:
    real_frames: dict[str, pd.DataFrame] = {}
    synthetic_frames: dict[tuple[str, str, str, int], pd.DataFrame] = {}
    provenance: dict = {"real": {}, "synthetic": {}}

    for dataset in DATASETS:
        embedding_path = cache_root / "embeddings" / dataset / "train_uni_224.parquet"
        manifest_path = data_root / "manifests" / f"{dataset}.csv"
        if not embedding_path.exists():
            raise FileNotFoundError(
                f"Missing real UNI cache: {embedding_path}. Run the SynPath UNI embedding stage."
            )
        manifest = pd.read_csv(manifest_path)
        if "split" not in manifest:
            raise ValueError(f"Dataset manifest has no split column: {manifest_path}")
        manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)
        frame = pd.read_parquet(embedding_path)
        validate_embedding_frame(frame, manifest, embedding_path)
        real_frames[dataset] = frame
        provenance["real"][dataset] = {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "embeddings": str(embedding_path),
            "embeddings_sha256": sha256_file(embedding_path),
            "count": len(frame),
        }

    for row in rq1.itertuples(index=False):
        key = (row.Dataset, row.Architecture, row.Conditioning, int(row.Seed))
        arm_root = (
            outputs_root
            / "generators"
            / row.Dataset
            / row.Architecture
            / row.Conditioning
            / f"seed{row.Seed}"
        )
        manifest_path = arm_root / "synthetic_manifest.csv"
        embedding_path = arm_root / "metrics" / "synthetic_uni_224.parquet"
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing synthetic UNI cache: {embedding_path}")
        manifest = pd.read_csv(manifest_path)
        frame = pd.read_parquet(embedding_path)
        validate_embedding_frame(frame, manifest, embedding_path)
        synthetic_frames[key] = frame
        provenance["synthetic"]["|".join(map(str, key))] = {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "embeddings": str(embedding_path),
            "embeddings_sha256": sha256_file(embedding_path),
            "count": len(frame),
        }
    return real_frames, synthetic_frames, provenance


def common_sample_count(
    real_frames: dict[str, pd.DataFrame],
    synthetic_frames: dict[tuple[str, str, str, int], pd.DataFrame],
    maximum: int,
) -> int:
    candidates = [maximum]
    candidates.extend(len(frame) // 2 for frame in real_frames.values())
    candidates.extend(len(frame) for frame in synthetic_frames.values())
    sample_count = min(candidates)
    if sample_count < 2:
        raise ValueError("Insufficient samples for matched normalized FID.")
    return sample_count


def build_clean_fid_extractor(args: argparse.Namespace):
    try:
        from cleanfid import fid
    except ImportError as exc:
        raise RuntimeError("Install clean-fid to calculate normalized Inception-FID.") from exc
    import torch

    device = torch.device(args.fid_device)
    return fid, fid.build_feature_extractor(
        "clean", device, use_dataparallel=True
    )


def clean_features(paths: pd.Series, fid_module, model, args: argparse.Namespace) -> np.ndarray:
    import torch

    return fid_module.get_files_features(
        paths.astype(str).tolist(),
        model,
        num_workers=args.fid_workers,
        batch_size=args.fid_batch_size,
        device=torch.device(args.fid_device),
        mode="clean",
        description="CleanFID features",
        verbose=True,
    )


def cached_clean_features(
    paths: pd.Series,
    cache_path: Path,
    fid_module,
    model,
    args: argparse.Namespace,
) -> np.ndarray:
    ids_hash = ordered_id_hash(paths)
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("ordered_ids_sha256") == ids_hash:
            values = np.load(cache_path, mmap_mode="r")
            if len(values) == len(paths) and values.ndim == 2:
                return np.asarray(values)
    values = clean_features(paths, fid_module, model, args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, values)
    metadata_path.write_text(
        json.dumps(
            {
                "ordered_ids_sha256": ids_hash,
                "count": len(values),
                "feature_dimension": values.shape[1],
                "mode": "clean",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return values


def calculate_normalized_metrics(
    rq1: pd.DataFrame,
    real_frames: dict[str, pd.DataFrame],
    synthetic_frames: dict[tuple[str, str, str, int], pd.DataFrame],
    sample_count: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    output = rq1.copy()
    output["nFID"] = np.nan
    output["nUFD"] = np.nan
    details: dict = {"datasets": {}}
    features = [f"f{index}" for index in range(FEATURE_DIMENSION)]
    fid_module = None
    fid_model = None
    if not args.skip_inception_normalization:
        fid_module, fid_model = build_clean_fid_extractor(args)
        if args.inception_cache_dir is None:
            raise ValueError("inception_cache_dir must be resolved before calculation")

    for dataset in DATASETS:
        ordered_real = deterministic_order(real_frames[dataset], args.seed)
        real_a = ordered_real.iloc[:sample_count]
        real_b = ordered_real.iloc[sample_count : 2 * sample_count]
        real_a_x = real_a[features].to_numpy(dtype=np.float32)
        real_b_x = real_b[features].to_numpy(dtype=np.float32)
        uni_floor = frechet_distance(real_a_x, real_b_x)
        if uni_floor <= 0:
            raise ValueError(f"Non-positive UNI-FID floor for {dataset}: {uni_floor}")
        inception_floor = None
        if not args.skip_inception_normalization:
            real_a_inception = cached_clean_features(
                real_a["image_path"],
                args.inception_cache_dir / dataset / "real-a.npy",
                fid_module,
                fid_model,
                args,
            )
            real_b_inception = cached_clean_features(
                real_b["image_path"],
                args.inception_cache_dir / dataset / "real-b.npy",
                fid_module,
                fid_model,
                args,
            )
            inception_floor = frechet_distance(real_a_inception, real_b_inception)
            if inception_floor <= 0:
                raise ValueError(f"Non-positive Inception-FID floor for {dataset}")

        details["datasets"][dataset] = {
            "real_a_ids_sha256": ordered_id_hash(real_a["image_path"]),
            "real_b_ids_sha256": ordered_id_hash(real_b["image_path"]),
            "uni_fid_floor": uni_floor,
            "inception_fid_floor": inception_floor,
            "arms": {},
        }
        selected = output["Dataset"] == dataset
        for index, row in output[selected].iterrows():
            key = (dataset, row["Architecture"], row["Conditioning"], int(row["Seed"]))
            synthetic = deterministic_order(synthetic_frames[key], args.seed).iloc[:sample_count]
            synthetic_x = synthetic[features].to_numpy(dtype=np.float32)
            uni_numerator = frechet_distance(real_a_x, synthetic_x)
            output.at[index, "nUFD"] = uni_numerator / uni_floor
            inception_numerator = None
            if not args.skip_inception_normalization:
                synthetic_inception = cached_clean_features(
                    synthetic["image_path"],
                    args.inception_cache_dir
                    / dataset
                    / row["Architecture"]
                    / f"{row['Conditioning']}-seed{row['Seed']}.npy",
                    fid_module,
                    fid_model,
                    args,
                )
                inception_numerator = frechet_distance(
                    real_a_inception, synthetic_inception
                )
                output.at[index, "nFID"] = inception_numerator / inception_floor
            arm_name = "|".join(map(str, key))
            details["datasets"][dataset]["arms"][arm_name] = {
                "synthetic_ids_sha256": ordered_id_hash(synthetic["image_path"]),
                "uni_fid_numerator": uni_numerator,
                "normalized_ufd": output.at[index, "nUFD"],
                "inception_fid_numerator": inception_numerator,
                "normalized_inception_fid": (
                    None if inception_floor is None else output.at[index, "nFID"]
                ),
            }
    return output, details


def paired_differences(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["nFID", "nUFD", "IP", "IR", "Authenticity"]
    key = ["Dataset", "Architecture"]
    baseline = results[results["Conditioning"] == "unconditioned"].set_index(key)
    clustered = results[results["Conditioning"] == "pseudo_label"].set_index(key)
    if not baseline.index.equals(clustered.index):
        baseline = baseline.sort_index()
        clustered = clustered.sort_index()
    if not baseline.index.equals(clustered.index):
        raise ValueError("Unconditioned and global-cluster RQ1 cells are not aligned.")
    delta = clustered[metrics] - baseline[metrics]
    return delta.reset_index()


def aggregate_differences(differences: pd.DataFrame) -> pd.DataFrame:
    metrics = ["nFID", "nUFD", "IP", "IR", "Authenticity"]
    rows = []
    for dataset, group in differences.groupby("Dataset", sort=False):
        row = {"Dataset": dataset, "n_cells": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        rows.append(row)
    row = {"Dataset": "Overall", "n_cells": len(differences)}
    for metric in metrics:
        row[f"{metric}_mean"] = differences[metric].mean()
        row[f"{metric}_sd"] = differences[metric].std(ddof=1)
    rows.append(row)
    return pd.DataFrame(rows)


def format_value(value: float, decimals: int, bold: bool) -> str:
    rendered = f"{value:.{decimals}f}"
    return rf"\textbf{{{rendered}}}" if bold else rendered


def latex_table(results: pd.DataFrame) -> str:
    directions = {
        "FID": "min",
        "nUFD": "min",
        "IP": "max",
        "IR": "max",
        "Authenticity": "max",
    }
    decimals = {
        "FID": 2,
        "nUFD": 2,
        "IP": 4,
        "IR": 4,
        "Authenticity": 4,
    }
    lines = [
        "% Requires \\usepackage{booktabs,multirow}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "Dataset & Architecture & Conditioning & FID $\\downarrow$ & nUFD $\\downarrow$ & IP $\\uparrow$ & IR $\\uparrow$ & Auth. $\\uparrow$ \\\\",
        "\\midrule",
    ]
    dataset_order = list(DATASETS)
    architecture_order = list(ARCHITECTURES)
    condition_order = list(CONDITIONING)
    for dataset_index, dataset in enumerate(dataset_order):
        dataset_rows = results[results["Dataset"] == dataset]
        for architecture_index, architecture in enumerate(architecture_order):
            group = dataset_rows[dataset_rows["Architecture"] == architecture]
            best = {}
            for metric, direction in directions.items():
                target = group[metric].min() if direction == "min" else group[metric].max()
                best[metric] = target if int((group[metric] == target).sum()) == 1 else None
            for condition_index, condition in enumerate(condition_order):
                row = group[group["Conditioning"] == condition].iloc[0]
                dataset_cell = (
                    rf"\multirow{{6}}{{*}}{{{DATASETS[dataset]}}}" if architecture_index == 0 and condition_index == 0 else ""
                )
                architecture_cell = (
                    rf"\multirow{{2}}{{*}}{{{ARCHITECTURES[architecture]}}}" if condition_index == 0 else ""
                )
                values = [
                    format_value(row[metric], decimals[metric], best[metric] == row[metric])
                    for metric in directions
                ]
                lines.append(
                    " & ".join(
                        [dataset_cell, architecture_cell, CONDITIONING[condition], *values]
                    )
                    + r" \\"
                )
            if architecture_index < len(architecture_order) - 1:
                lines.append(r"\cmidrule(lr){2-8}")
        if dataset_index < len(dataset_order) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{RQ1 quality results comparing unconditioned generation with global-cluster pseudo-label conditioning. FID is retained for comparison with previous studies. nUFD denotes normalized UNI Fr\\'echet distance, $\\mathrm{nUFD}=\\mathrm{FD}^{\\mathrm{test}}_{\\mathrm{UNI}}/\\mathrm{FD}^{\\mathrm{floor}}_{\\mathrm{UNI}}$, following the floor-normalization convention of HistoFID and using MahmoodLab/UNI, not UNI2. FID and nUFD are lower-is-better; improved precision (IP), improved recall (IR), and Authenticity (Auth.) are higher-is-better. nUFD is an exploratory calibrated measure and does not establish universal cross-dataset comparability. Bold denotes the unique better value within each matched dataset--architecture pair.}",
            "\\label{tab:rq1-normalized-quality}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def plot_heatmap(differences: pd.DataFrame, output_stem: Path) -> None:
    metrics = [
        ("nUFD", r"nUFD $\downarrow$", 2, -1.0),
        ("IP", r"IP $\uparrow$", 4, 1.0),
        ("IR", r"IR $\uparrow$", 4, 1.0),
        ("Authenticity", r"Authenticity $\uparrow$", 4, 1.0),
    ]
    dataset_labels = list(DATASETS.values())
    architecture_labels = list(ARCHITECTURES.values())
    display = differences.replace(
        {"Dataset": DATASETS, "Architecture": ARCHITECTURES}
    )
    plt.rcParams.update(
        {"font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold"}
    )
    cmap = LinearSegmentedColormap.from_list(
        "rq1_delta", ["#D55E00", "#F7F7F7", "#0072B2"]
    )
    figure, axes = plt.subplots(1, 4, figsize=(24, 8), constrained_layout=True)
    for panel, (axis, (metric, title, precision, improvement_sign)) in enumerate(
        zip(axes, metrics, strict=True)
    ):
        values = display.pivot(index="Dataset", columns="Architecture", values=metric).loc[
            dataset_labels, architecture_labels
        ].to_numpy(float)
        improvement = improvement_sign * values
        limit = max(float(np.max(np.abs(improvement))), np.finfo(float).eps)
        axis.imshow(
            improvement,
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            aspect="equal",
        )
        axis.set_title(rf"$\Delta$ {title}", fontsize=FONT_SIZE, pad=12)
        axis.set_xticks(range(3), architecture_labels, fontsize=FONT_SIZE)
        axis.set_yticks(range(3))
        axis.set_yticklabels(dataset_labels if panel == 0 else [], fontsize=FONT_SIZE)
        axis.tick_params(length=0, pad=7)
        axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        axis.grid(which="minor", color="#B8B8B8", linewidth=1.0)
        axis.tick_params(which="minor", bottom=False, left=False)
        for row in range(3):
            for column in range(3):
                value = values[row, column]
                annotation = "0.0" if abs(value) < 0.5 * 10**-precision else f"{value:.{precision}f}"
                colour = "white" if abs(improvement[row, column]) / limit > 0.62 else "#202020"
                axis.text(
                    column,
                    row,
                    annotation,
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=FONT_SIZE,
                    fontweight="bold",
                )
        for spine in axis.spines.values():
            spine.set_visible(False)
    legend = figure.colorbar(
        ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=cmap),
        ax=axes,
        orientation="horizontal",
        fraction=0.08,
        pad=0.13,
        shrink=0.5,
        aspect=55,
        ticks=[-1, 0, 1],
    )
    legend.ax.set_xticklabels(["Deterioration", "No change", "Improvement"])
    legend.ax.tick_params(labelsize=FONT_SIZE, length=0, pad=8)
    for label in legend.ax.get_xticklabels():
        label.set_fontweight("bold")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.inception_cache_dir is None:
        args.inception_cache_dir = (
            args.output_dir / "rq1-normalized-fid-cache" / "inception"
        )
    rq1 = load_raw_results(args.quality_csv, args.long_results_csv)
    real_frames, synthetic_frames, provenance = load_populations(
        rq1, args.data_root, args.cache_root, args.outputs_root
    )
    sample_count = common_sample_count(real_frames, synthetic_frames, args.max_samples)
    results, calculation = calculate_normalized_metrics(
        rq1, real_frames, synthetic_frames, sample_count, args
    )
    results = results.sort_values(
        ["Dataset", "Architecture", "Conditioning"], kind="stable"
    ).reset_index(drop=True)
    differences = paired_differences(results)
    aggregates = aggregate_differences(differences)

    stem = args.output_dir / "rq1-global-vs-unconditioned-normalized-quality"
    results.to_csv(stem.with_suffix(".csv"), index=False)
    differences.to_csv(args.output_dir / "rq1-normalized-quality-deltas.csv", index=False)
    aggregates.to_csv(args.output_dir / "rq1-normalized-quality-aggregates.csv", index=False)
    (args.output_dir / "rq1-normalized-quality.tex").write_text(
        latex_table(results), encoding="utf-8"
    )
    plot_heatmap(differences, stem)

    metadata = {
        "metric": "floor-normalized Frechet distance ratio",
        "encoder": "hf-hub:MahmoodLab/UNI",
        "encoder_feature_dimension": FEATURE_DIMENSION,
        "reference_population": "training split used by the canonical quality evaluation",
        "sample_count_per_population": sample_count,
        "maximum_requested_samples": args.max_samples,
        "selection_seed": args.seed,
        "strictly_disjoint_real_partitions": True,
        "quality_csv": str(args.quality_csv),
        "quality_csv_sha256": sha256_file(args.quality_csv),
        "long_results_csv": str(args.long_results_csv),
        "long_results_csv_sha256": sha256_file(args.long_results_csv),
        "provenance": provenance,
        "calculation": calculation,
    }
    (args.output_dir / "rq1-normalized-quality-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
