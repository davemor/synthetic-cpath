from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cluster_exp.dimension_plots import select_best_dimension_configs, summarise_dimension_trends


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/Users/dm236/Documents/Research/AllResults/Chapter6-clustering/res"),
    )
    parser.add_argument("--encoder", type=str, default="uni2")
    parser.add_argument("--min-dimension", type=int, default=1)
    parser.add_argument("--max-dimension", type=int, default=50)
    args = parser.parse_args()

    stability = pd.read_csv(args.run_dir / "clustering_stability.csv")
    datasets = sorted(stability.loc[stability["encoder"].eq(args.encoder), "dataset"].dropna().unique())
    rows = []
    for dataset in datasets:
        trends = []
        for projector in ["pca", "umap"]:
            selected = select_best_dimension_configs(
                stability,
                projector=projector,
                dataset=str(dataset),
                encoder=args.encoder,
                min_dimension=args.min_dimension,
                max_dimension=args.max_dimension,
            )
            trends.append(summarise_dimension_trends(selected))
        combined = pd.concat(trends, ignore_index=True)
        rows.append(
            {
                "dataset": dataset,
                "mean_ami": combined["ami_mean"].mean(),
                "mean_ari": combined["ari_mean"].mean(),
                "mean_v_measure": combined["v_measure_mean"].mean(),
                "mean_stability": combined["stability_mean"].mean(),
            }
        )

    ranking = pd.DataFrame(rows).sort_values("mean_ami", ascending=False)
    print(ranking.to_csv(index=False, float_format="%.6f"))
    print(f"BEST,{ranking.iloc[0]['dataset']}")
    print(f"WORST,{ranking.iloc[-1]['dataset']}")


if __name__ == "__main__":
    main()
