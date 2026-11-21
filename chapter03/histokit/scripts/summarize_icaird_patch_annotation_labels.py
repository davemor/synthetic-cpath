#!/usr/bin/env python3
"""Summarize rendered patch annotation labels for iCAIRD cervical bags."""

from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


DEFAULT_BAGS_ROOT = Path(
    "/sharedscratch/dm236/data/icaird_cervical/bags/"
    "full_canny005_or_resnet_light_grandqc_artifact"
)
COLUMN_GROUPS = OrderedDict(
    [
        ("Normal", ["Normal inflammation"]),
        ("Low-grade", ["CIN1", "HPV"]),
        ("High-grade", ["CIN2", "CIN3"]),
        (
            "Malignant",
            ["Adenocarcinoma", "CGIN", "Other", "Squamous carcinoma"],
        ),
    ]
)
ROW_ORDER = ["Normal", "Low-grade", "High-grade", "Malignant"]
ALL_ROW_ORDER = ROW_ORDER + ["Background"]


def main() -> None:
    args = parse_args()
    result = summarize_patchsets(args.bags_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(result, output_dir=args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_BAGS_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <bags-root>/exports",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.bags_root / "exports"
    return args


def summarize_patchsets(bags_root: Path) -> dict[str, object]:
    patchsets = sorted((bags_root / "patchsets").glob("*.patchset/frame.parquet"))
    if not patchsets:
        raise FileNotFoundError(f"No patchset frames found under {bags_root / 'patchsets'}")

    all_counts: Counter[tuple[str, str, str]] = Counter()
    kept_counts: Counter[tuple[str, str, str]] = Counter()
    background_counts: Counter[tuple[str, str]] = Counter()
    slide_counts: Counter[tuple[str, str]] = Counter()

    for frame_path in patchsets:
        frame = pd.read_parquet(
            frame_path,
            columns=["annotation_label_name", "category", "subcategory", "keep"],
        )
        if frame.empty:
            continue

        category, subcategory = slide_category_subcategory(frame)
        slide_counts[(category, subcategory)] += 1
        labels = frame["annotation_label_name"].fillna("unlabelled").astype(str)
        coarse_labels = labels.map(coarse_annotation_label)
        keep = frame["keep"].astype(bool)

        for label, count in coarse_labels.value_counts().items():
            all_counts[(label, category, subcategory)] += int(count)
        for label, count in coarse_labels[keep].value_counts().items():
            kept_counts[(label, category, subcategory)] += int(count)
        background_counts[(category, subcategory)] += int((~keep).sum())

    return {
        "bags_root": str(bags_root),
        "created_at": datetime.now(UTC).isoformat(),
        "num_patchsets": len(patchsets),
        "all_counts": all_counts,
        "kept_counts": kept_counts,
        "background_counts": background_counts,
        "slide_counts": slide_counts,
    }


def slide_category_subcategory(frame: pd.DataFrame) -> tuple[str, str]:
    category = coarse_slide_category(str(frame["category"].iloc[0]))
    subcategory = format_subcategory(str(frame["subcategory"].iloc[0]))
    return category, subcategory


def coarse_slide_category(category: str) -> str:
    normalized = category.strip().lower()
    if normalized in {"normal", "normal_inflammation", "normal inflammation"}:
        return "Normal"
    if normalized in {"low_grade", "low-grade", "low grade"}:
        return "Low-grade"
    if normalized in {"high_grade", "high-grade", "high grade"}:
        return "High-grade"
    if normalized == "malignant":
        return "Malignant"
    return category.replace("_", " ").title()


def format_subcategory(subcategory: str) -> str:
    normalized = subcategory.strip().lower()
    mapping = {
        "normal": "Normal",
        "normal_inflammation": "Normal inflammation",
        "normal inflammation": "Normal inflammation",
        "cin1": "CIN1",
        "cin2": "CIN2",
        "cin3": "CIN3",
        "hpv": "HPV",
        "cgin": "CGIN",
        "adenocarcinoma": "Adenocarcinoma",
        "other": "Other",
        "squamous_carcinoma": "Squamous carcinoma",
        "squamous carcinoma": "Squamous carcinoma",
    }
    return mapping.get(normalized, subcategory.replace("_", " ").title())


def coarse_annotation_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized in {"normal", "normal_inflammation", "normal inflammation"}:
        return "Normal"
    if normalized in {"low_grade", "low-grade", "low grade"}:
        return "Low-grade"
    if normalized in {"high_grade", "high-grade", "high grade"}:
        return "High-grade"
    if normalized == "malignant":
        return "Malignant"
    if normalized in {"", "nan"}:
        return "Unlabelled"
    return label.replace("_", " ").title()


def write_outputs(result: dict[str, object], *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_counts = result["all_counts"]
    kept_counts = result["kept_counts"]
    background_counts = result["background_counts"]
    slide_counts = result["slide_counts"]
    assert isinstance(all_counts, Counter)
    assert isinstance(kept_counts, Counter)
    assert isinstance(background_counts, Counter)
    assert isinstance(slide_counts, Counter)

    all_table = make_count_table(
        all_counts,
        row_order=ALL_ROW_ORDER,
        background_counts=background_counts,
    )
    kept_table = make_count_table(kept_counts, row_order=ROW_ORDER)
    slide_table = make_slide_count_table(slide_counts)

    all_table.to_csv(output_dir / "patch_annotation_label_counts_all.csv", index=False)
    kept_table.to_csv(output_dir / "patch_annotation_label_counts_kept.csv", index=False)
    slide_table.to_csv(output_dir / "patch_annotation_slide_counts.csv", index=False)
    (output_dir / "patch_annotation_label_counts_all.tex").write_text(
        make_latex_table(
            all_table,
            caption=(
                "All patch annotation labels by slide category and subcategory. "
                "Normal and normal-inflammation patch annotations are combined. "
                "Background denotes patches not retained by the final bag "
                "filtering rule ($\\texttt{keep} = \\texttt{false}$)."
            ),
        )
    )
    (output_dir / "patch_annotation_label_counts_kept.tex").write_text(
        make_latex_table(
            kept_table,
            caption=(
                "Kept patch annotation labels by slide category and subcategory. "
                "Normal and normal-inflammation patch annotations are combined."
            ),
        )
    )

    all_background_total = int(
        all_table.loc[all_table["annotation_label"] == "Background", "total"].iloc[0]
    )
    all_diagnostic_total = int(
        all_table.loc[all_table["annotation_label"].isin(ROW_ORDER), "total"].sum()
    )
    all_table_total = int(all_table.loc[all_table["annotation_label"] == "Total", "total"].iloc[0])
    kept_patch_total = int(
        kept_table.loc[kept_table["annotation_label"] == "Total", "total"].iloc[0]
    )

    summary = {
        "created_at": result["created_at"],
        "bags_root": result["bags_root"],
        "num_patchsets": result["num_patchsets"],
        "all_diagnostic_annotation_total": all_diagnostic_total,
        "all_table_total_including_background_row": all_table_total,
        "kept_patch_total": kept_patch_total,
        "background_patch_total": all_background_total,
        "normal_annotation_labels_collapsed": True,
        "background_definition": "keep == false",
        "outputs": [
            "patch_annotation_label_counts_all.csv",
            "patch_annotation_label_counts_kept.csv",
            "patch_annotation_slide_counts.csv",
            "patch_annotation_label_counts_all.tex",
            "patch_annotation_label_counts_kept.tex",
        ],
    }
    (output_dir / "patch_annotation_label_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("Wrote:")
    for name in summary["outputs"] + ["patch_annotation_label_summary.json"]:
        print(output_dir / name)


def make_count_table(
    counts: Counter[tuple[str, str, str]],
    *,
    row_order: list[str],
    background_counts: Counter[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    columns = ordered_columns()
    rows = []
    for label in row_order:
        row = {"annotation_label": label}
        total = 0
        for category, subcategory in columns:
            if label == "Background":
                value = int((background_counts or Counter()).get((category, subcategory), 0))
            else:
                value = int(counts.get((label, category, subcategory), 0))
            row[column_name(category, subcategory)] = value
            total += value
        row["total"] = total
        rows.append(row)
    total_row = {"annotation_label": "Total"}
    for category, subcategory in columns:
        name = column_name(category, subcategory)
        total_row[name] = int(sum(row[name] for row in rows))
    total_row["total"] = int(sum(row["total"] for row in rows))
    rows.append(total_row)
    return pd.DataFrame(rows)


def make_slide_count_table(slide_counts: Counter[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for category, subcategory in ordered_columns():
        rows.append(
            {
                "category": category,
                "subcategory": subcategory,
                "slides": int(slide_counts.get((category, subcategory), 0)),
            }
        )
    rows.append(
        {
            "category": "Total",
            "subcategory": "",
            "slides": int(sum(slide_counts.values())),
        }
    )
    return pd.DataFrame(rows)


def ordered_columns() -> list[tuple[str, str]]:
    return [
        (category, subcategory)
        for category, subcategories in COLUMN_GROUPS.items()
        for subcategory in subcategories
    ]


def column_name(category: str, subcategory: str) -> str:
    return f"{category}/{subcategory}"


def make_latex_table(table: pd.DataFrame, *, caption: str) -> str:
    columns = ordered_columns()
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "r" * (len(columns) + 1) + "}",
        r"\toprule",
    ]
    header = [r"\multirow{2}{*}{Patch annotation}"]
    for category, subcategories in COLUMN_GROUPS.items():
        header.append(rf"\multicolumn{{{len(subcategories)}}}{{c}}{{{category}}}")
    header.append(r"\multirow{2}{*}{Total}")
    lines.append(" & ".join(header) + r" \\")
    lines.append(
        r"\cmidrule(lr){2-2}\cmidrule(lr){3-4}"
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-10}"
    )
    lines.append(
        " & "
        + " & ".join(latex_escape(subcategory) for _, subcategory in columns)
        + r" &  \\"
    )
    lines.append(r"\midrule")
    for _, row in table.iterrows():
        label = row["annotation_label"]
        values = [
            row[column_name(category, subcategory)]
            for category, subcategory in columns
        ]
        total = row["total"]
        if label == "Total":
            lines.append(r"\midrule")
        lines.append(
            latex_escape(str(label))
            + " & "
            + " & ".join(format_int(value) for value in values)
            + f" & {format_int(total)}"
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("_", r"\_")


def format_int(value: object) -> str:
    return f"{int(value):,}"


if __name__ == "__main__":
    main()
