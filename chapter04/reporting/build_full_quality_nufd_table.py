from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_rq1_normalized_fid_report import (
    DATASETS,
    deterministic_order,
    feature_columns,
    frechet_distance,
)


ARCHITECTURES = {
    "stylegan2ada": "StyleGAN2-ADA",
    "sd_lora": "SD-LoRA",
    "dcgan": "DCGAN",
}
CONDITIONING = {
    "unconditioned": "Unconditioned",
    "pseudo_label": "Global cluster",
    "true_label": "Class label",
    "label_global_cluster": "Class + global",
    "label_local_cluster": "Class + local",
}
METRICS = {
    "FID": ("min", 2),
    "nUFD": ("min", 2),
    "IP": ("max", 4),
    "IR": ("max", 4),
    "Authenticity": ("max", 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def calculate_nufd(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    output = frame.copy()
    output["nUFD"] = np.nan
    for dataset in DATASETS:
        real = pd.read_parquet(
            args.cache_root / "embeddings" / dataset / "train_uni_224.parquet"
        )
        ordered_real = deterministic_order(real, args.seed)
        count = min(args.max_samples, len(ordered_real) // 2)
        columns = feature_columns(real)
        real_a = ordered_real.iloc[:count][columns].to_numpy(dtype=np.float32)
        real_b = ordered_real.iloc[count : 2 * count][columns].to_numpy(
            dtype=np.float32
        )
        floor = frechet_distance(real_a, real_b)
        selected = output["Dataset"] == dataset
        for index, row in output[selected].iterrows():
            path = (
                args.outputs_root
                / "generators"
                / dataset
                / row["Architecture"]
                / row["Conditioning"]
                / f"seed{int(row['Seed'])}"
                / "metrics"
                / "synthetic_uni_224.parquet"
            )
            synthetic = deterministic_order(pd.read_parquet(path), args.seed)
            sample_count = min(count, len(synthetic))
            if sample_count != count:
                raise ValueError(f"{path} has only {sample_count} embeddings")
            synthetic_values = synthetic.iloc[:count][columns].to_numpy(
                dtype=np.float32
            )
            output.at[index, "nUFD"] = frechet_distance(
                real_a, synthetic_values
            ) / floor
    if output["nUFD"].isna().any():
        raise ValueError("nUFD calculation is incomplete")
    return output


def configuration(row: pd.Series) -> str:
    return f"{ARCHITECTURES[row['Architecture']]} / {CONDITIONING[row['Conditioning']]}"


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, display_dataset in DATASETS.items():
        group = frame[frame["Dataset"] == dataset]
        for metric, (direction, _) in METRICS.items():
            best_index = group[metric].idxmin() if direction == "min" else group[metric].idxmax()
            worst_index = group[metric].idxmax() if direction == "min" else group[metric].idxmin()
            best = group.loc[best_index]
            worst = group.loc[worst_index]
            rows.append(
                {
                    "Dataset": display_dataset,
                    "Metric": metric,
                    "Minimum": group[metric].min(),
                    "Maximum": group[metric].max(),
                    "Best configuration": configuration(best),
                    "Best value": best[metric],
                    "Worst configuration": configuration(worst),
                    "Worst value": worst[metric],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    quality = pd.read_csv(args.quality_csv)
    if len(quality) != 45:
        raise ValueError(f"Expected 45 rows, found {len(quality)}")
    output = calculate_nufd(quality, args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    build_summary(output).to_csv(
        args.output_csv.with_name("quality-metric-range-best-worst.csv"), index=False
    )


if __name__ == "__main__":
    main()
