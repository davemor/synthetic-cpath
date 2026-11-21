from pathlib import Path

import pandas as pd


root = Path("/raid/dm236/synpath/outputs/generators")
rows = []
for manifest in sorted(root.glob("*/*/*/seed0/synthetic_manifest.csv")):
    dataset, architecture, conditioning, seed = manifest.relative_to(root).parts[:4]
    frame = pd.read_csv(manifest)
    counts = frame.groupby("condition", dropna=False).size()
    for condition, count in counts.items():
        rows.append(
            {
                "dataset": dataset,
                "architecture": architecture,
                "conditioning": conditioning,
                "seed": int(seed.removeprefix("seed")),
                "condition": condition,
                "generated_count": count,
                "generated_total": len(frame),
            }
        )

output = Path("/raid/dm236/synpath/reports/final_original_experiment")
output.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(output / "generation_condition_counts.csv", index=False)
