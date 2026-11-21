from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "summarize_icaird_patch_annotation_labels.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_icaird_labels", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


def _write_patchset(root: Path, slide_id: str, frame: pd.DataFrame) -> None:
    path = root / "patchsets" / f"{slide_id}.patchset"
    path.mkdir(parents=True)
    frame.to_parquet(path / "frame.parquet", index=False)


def test_summarize_patch_annotation_labels_writes_background_row(tmp_path: Path) -> None:
    bags_root = tmp_path / "bags"
    _write_patchset(
        bags_root,
        "s1",
        pd.DataFrame(
            {
                "annotation_label_name": ["normal", "normal_inflammation", "low_grade"],
                "category": ["normal_inflammation"] * 3,
                "subcategory": ["normal_inflammation"] * 3,
                "keep": [True, False, False],
            }
        ),
    )
    _write_patchset(
        bags_root,
        "s2",
        pd.DataFrame(
            {
                "annotation_label_name": ["malignant", "high_grade"],
                "category": ["malignant"] * 2,
                "subcategory": ["squamous_carcinoma"] * 2,
                "keep": [True, False],
            }
        ),
    )

    result = summary.summarize_patchsets(bags_root)
    summary.write_outputs(result, output_dir=bags_root / "exports")

    all_counts = pd.read_csv(bags_root / "exports/patch_annotation_label_counts_all.csv")
    kept_counts = pd.read_csv(bags_root / "exports/patch_annotation_label_counts_kept.csv")
    all_tex = (bags_root / "exports/patch_annotation_label_counts_all.tex").read_text()
    kept_tex = (bags_root / "exports/patch_annotation_label_counts_kept.tex").read_text()

    normal_all = all_counts.loc[all_counts["annotation_label"] == "Normal"].iloc[0]
    background = all_counts.loc[all_counts["annotation_label"] == "Background"].iloc[0]
    all_total = all_counts.loc[all_counts["annotation_label"] == "Total"].iloc[0]
    normal_kept = kept_counts.loc[kept_counts["annotation_label"] == "Normal"].iloc[0]
    kept_total = kept_counts.loc[kept_counts["annotation_label"] == "Total"].iloc[0]

    assert int(normal_all["Normal/Normal inflammation"]) == 2
    assert int(normal_kept["Normal/Normal inflammation"]) == 1
    assert int(background["Normal/Normal inflammation"]) == 2
    assert int(background["Malignant/Squamous carcinoma"]) == 1
    assert int(all_total["total"]) == 8
    assert int(kept_total["total"]) == 2
    assert "Background denotes patches not retained" in all_tex
    assert "Background" not in kept_tex
