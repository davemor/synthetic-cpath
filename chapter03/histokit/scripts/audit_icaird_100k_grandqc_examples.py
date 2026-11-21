#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(
    "/home/dm236/scratch/data/icaird_cervical/"
    "icaird_cervical_100k_uni2_kcenter"
)
CLASSES = [(2, "fold"), (6, "out_of_focus"), (7, "background")]


def main() -> None:
    out = ROOT / "audit_grandqc_examples"
    out.mkdir(exist_ok=True)
    frame = pd.read_parquet(ROOT / "manifests/sample_manifest.parquet")
    rows = []
    for class_id, class_name in CLASSES:
        group = frame[
            frame["official_grandqc_artifact_class_id"].astype(int) == class_id
        ].copy()
        sample = group.sample(n=min(24, len(group)), random_state=236).reset_index(
            drop=True
        )
        csv_path = out / f"sample_{class_id}_{class_name}.csv"
        sheet_path = out / f"examples_{class_id}_{class_name}.jpg"
        sample.to_csv(csv_path, index=False)
        write_sheet(sample, sheet_path)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "count": len(group),
                "sheet": str(sheet_path),
                "csv": str(csv_path),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def write_sheet(sample: pd.DataFrame, path: Path) -> None:
    cols = 6
    thumb = 112
    label_h = 42
    rows = (len(sample) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in sample.iterrows():
        x = (i % cols) * thumb
        y = (i // cols) * (thumb + label_h)
        image = Image.open(row["patch_png_path"]).convert("RGB").resize((thumb, thumb))
        sheet.paste(image, (x, y))
        label = (
            f"{row['sample_id']}\n"
            f"{row['split']} {row['subcategory']}\n"
            f"r{int(row['row'])} c{int(row['column'])}"
        )
        draw.text((x + 2, y + thumb + 2), label, fill=(0, 0, 0))
    sheet.save(path, quality=92)


if __name__ == "__main__":
    main()
