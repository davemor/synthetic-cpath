import json
from pathlib import Path

import pandas as pd


base = Path("/raid/dm236/synpath/outputs/generators")
rows = []
for fid_path in sorted(base.glob("*/*/*/seed0/metrics/inception_fid.json")):
    seed_root = fid_path.parent.parent
    relative = seed_root.relative_to(base)
    dataset, architecture, conditioning, seed_name = relative.parts
    metrics = fid_path.parent
    with fid_path.open() as file:
        fid = json.load(file)["fid"]
    with (metrics / "uni_precision_recall.json").open() as file:
        precision_recall = json.load(file)
    with (metrics / "uni_authenticity.json").open() as file:
        authenticity = json.load(file)
    rows.append(
        {
            "Architecture": architecture,
            "Dataset": dataset,
            "Conditioning": conditioning,
            "Seed": int(seed_name.removeprefix("seed")),
            "FID": fid,
            "IP": precision_recall["precision"],
            "IR": precision_recall["recall"],
            "Authenticity": authenticity["authenticity"],
        }
    )

frame = pd.DataFrame(rows).sort_values(
    ["Dataset", "Architecture", "Conditioning"]
)
if len(frame) != 45 or frame["Seed"].unique().tolist() != [0]:
    raise RuntimeError("Unexpected canonical quality-result inventory")
output = Path("/raid/dm236/synpath/reports/final_original_experiment")
output.mkdir(parents=True, exist_ok=True)
frame.to_csv(output / "quality_results_fid_ip_ir_authenticity.csv", index=False)
print(frame.to_csv(index=False))
