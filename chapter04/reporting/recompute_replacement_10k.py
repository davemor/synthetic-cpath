from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "final-results"

BASELINES = {
    "icaird_cervical100k": {
        0: {"Accuracy": 0.6052, "ROC-AUC": 0.8350226733333334},
        1: {"Accuracy": 0.6052, "ROC-AUC": 0.8332594533333333},
        2: {"Accuracy": 0.6198, "ROC-AUC": 0.8409529466666666},
    },
    "pathmnist224": {
        0: {"Accuracy": 0.9246518105849583, "ROC-AUC": 0.9907588242378801},
        1: {"Accuracy": 0.9250696378830083, "ROC-AUC": 0.9932070095187551},
        2: {"Accuracy": 0.918941504178273, "ROC-AUC": 0.9919530058937541},
    },
    "pcam": {
        0: {"Accuracy": 0.808013916015625, "ROC-AUC": 0.8904804294315766},
        1: {"Accuracy": 0.83990478515625, "ROC-AUC": 0.9275298526471957},
        2: {"Accuracy": 0.85028076171875, "ROC-AUC": 0.9332543024028123},
    },
}


def summarize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby([*columns, "Metric"], sort=False)
        .agg(
            synthetic_mean=("Synthetic-only", "mean"),
            synthetic_sd=("Synthetic-only", "std"),
            baseline_mean=("Real-only", "mean"),
            baseline_sd=("Real-only", "std"),
            delta_mean=("Delta", "mean"),
            delta_sd=("Delta", "std"),
            n=("Delta", "size"),
        )
        .reset_index()
    )


def main() -> None:
    old = pd.read_csv(ROOT / "replacement-vs-real-paired-deltas.csv")
    new = old.copy()
    new["Real-only"] = [
        BASELINES[dataset][int(seed)][metric]
        for dataset, seed, metric in zip(new.Dataset, new.Seed, new.Metric)
    ]
    new["Delta"] = new["Synthetic-only"] - new["Real-only"]
    new.to_csv(ROOT / "replacement-vs-real-10k-paired-deltas.csv", index=False)

    sections = []
    for level, columns in (
        ("Overall", []),
        ("Dataset", ["Dataset"]),
        ("Architecture", ["Architecture"]),
        ("Conditioning", ["Conditioning"]),
    ):
        old_summary = summarize(old, columns)
        new_summary = summarize(new, columns)
        keys = [*columns, "Metric"]
        comparison = old_summary.merge(new_summary, on=keys, suffixes=("_old", "_new"))
        comparison.insert(0, "Level", level)
        comparison["delta_mean_change"] = comparison.delta_mean_new - comparison.delta_mean_old
        comparison["delta_sd_change"] = comparison.delta_sd_new - comparison.delta_sd_old
        sections.append(comparison)

    comparison = pd.concat(sections, ignore_index=True)
    comparison.to_csv(ROOT / "replacement-real-baseline-comparison.csv", index=False)

    print("OVERALL")
    print(summarize(new, []).to_string(index=False))
    print("\nBY DATASET")
    print(summarize(new, ["Dataset"]).to_string(index=False))
    print("\nBY ARCHITECTURE")
    print(summarize(new, ["Architecture"]).to_string(index=False))
    print("\nBY CONDITIONING")
    print(summarize(new, ["Conditioning"]).to_string(index=False))
    print("\nNON-NEGATIVE DELTAS")
    print(new.groupby("Metric").Delta.apply(lambda values: (values >= 0).sum()).to_string())


if __name__ == "__main__":
    main()
