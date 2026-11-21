import json
from pathlib import Path

import pandas as pd


base = Path("/raid/dm236/synpath/outputs/utility")
rows = []
for path in sorted(base.glob("*/resnet18/*/*/seed*/metrics.json")):
    dataset, _, architecture, source, seed_name = path.relative_to(base).parts[:5]
    with path.open() as file:
        metrics = json.load(file)
    if source.startswith("synthetic_only_"):
        regime = "Synthetic-only"
        conditioning = source.removeprefix("synthetic_only_")
    elif source.startswith("real_plus_synthetic_"):
        regime = "Real + synthetic"
        conditioning = source.removeprefix("real_plus_synthetic_")
    else:
        continue
    rows.append(
        {
            "Dataset": dataset,
            "Architecture": architecture,
            "Conditioning": conditioning,
            "Regime": regime,
            "Seed": int(seed_name.removeprefix("seed")),
            "Accuracy": metrics["accuracy"],
            "ROC-AUC": metrics["macro_roc_auc_ovr"],
        }
    )

frame = pd.DataFrame(rows)
group_columns = ["Dataset", "Architecture", "Conditioning", "Regime"]
counts = frame.groupby(group_columns)["Seed"].nunique()
if len(frame) != 162 or not counts.eq(3).all():
    raise RuntimeError("Unexpected canonical utility-result inventory")
summary = (
    frame.groupby(group_columns)
    .agg(
        Accuracy_mean=("Accuracy", "mean"),
        Accuracy_sd=("Accuracy", "std"),
        ROC_AUC_mean=("ROC-AUC", "mean"),
        ROC_AUC_sd=("ROC-AUC", "std"),
    )
    .reset_index()
)
wide = summary.pivot(
    index=["Dataset", "Architecture", "Conditioning"],
    columns="Regime",
).reset_index()
wide.columns = [
    "_".join(value for value in column if value).replace(" ", "_")
    if isinstance(column, tuple)
    else column
    for column in wide.columns
]
output = Path("/raid/dm236/synpath/reports/final_original_experiment")
output.mkdir(parents=True, exist_ok=True)
wide.to_csv(output / "utility_results_accuracy_roc_auc.csv", index=False)
print(wide.to_csv(index=False))
