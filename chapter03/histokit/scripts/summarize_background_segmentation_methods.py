#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DATA_ROOT = Path("/sharedscratch/dm236/data/icaird_cervical")
DEFAULT_COMPARISON_DIR = DATA_ROOT / "fgbg_method_comparison"
DEFAULT_OUTPUT_DIR = DEFAULT_COMPARISON_DIR / "methods_summary"

METHOD_DESCRIPTIONS = {
    "canny": (
        "Cached histokit per-patch Canny edge tissue score; foreground if "
        "score is greater than the selected threshold."
    ),
    "grandqc": (
        "Histokit GrandQC tissue detector score; foreground if tissue "
        "probability is greater than the selected threshold."
    ),
    "clam_segmentation_otsu": (
        "Histokit CLAM-style slide-level Otsu tissue mask, converted to a "
        "patch-level 0/1 score and thresholded."
    ),
    "otsu_hs_segmentation": (
        "Histokit hue/saturation Otsu slide-level tissue mask, converted to a "
        "patch-level 0/1 score and thresholded."
    ),
    "resnet19": (
        "Supervised ResNet-19 foreground/background patch classifier trained "
        "on the training split; prediction is the argmax class."
    ),
}

THRESHOLD_METHODS = {
    "canny",
    "grandqc",
    "clam_segmentation_otsu",
    "otsu_hs_segmentation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a reproducible methods summary for iCAIRD cervical "
            "foreground/background segmentation experiments."
        )
    )
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_inputs(comparison_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = comparison_dir / "summary_metrics.csv"
    thresholds_path = comparison_dir / "selected_thresholds.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing summary metrics: {metrics_path}")
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Missing selected thresholds: {thresholds_path}")
    return pd.read_csv(metrics_path), pd.read_csv(thresholds_path)


def check_consistency(metrics: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    methods = sorted(metrics["method"].dropna().unique().tolist())
    threshold_methods = sorted(thresholds["method"].dropna().unique().tolist())
    expected_methods = sorted(METHOD_DESCRIPTIONS)
    checks.append(
        {
            "check": "expected_methods_present",
            "passed": methods == expected_methods and threshold_methods == expected_methods,
            "details": f"metrics={methods}; thresholds={threshold_methods}",
        }
    )
    split_counts = metrics.groupby("split")["support"].nunique().to_dict()
    split_supports = (
        metrics.groupby("split")["support"]
        .first()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    checks.append(
        {
            "check": "same_patch_count_per_split",
            "passed": all(value == 1 for value in split_counts.values()),
            "details": json.dumps(split_supports, sort_keys=True),
        }
    )
    method_split_counts = metrics.groupby("method")["split"].nunique().to_dict()
    checks.append(
        {
            "check": "every_method_has_train_and_test",
            "passed": all(value == 2 for value in method_split_counts.values()),
            "details": json.dumps(method_split_counts, sort_keys=True),
        }
    )
    threshold_rows = thresholds.loc[thresholds["method"].isin(THRESHOLD_METHODS)]
    checks.append(
        {
            "check": "threshold_methods_selected_on_train_macro_f1",
            "passed": bool(
                (threshold_rows["selection_split"].astype(str) == "train").all()
                and (threshold_rows["selection_metric"].astype(str) == "macro_f1").all()
            ),
            "details": threshold_rows[
                ["method", "selected_threshold", "selection_split", "selection_metric"]
            ].to_json(orient="records"),
        }
    )
    resnet = thresholds.loc[thresholds["method"] == "resnet19"]
    checks.append(
        {
            "check": "resnet_uses_argmax_not_threshold",
            "passed": bool(
                len(resnet) == 1
                and str(resnet.iloc[0]["selection_metric"]) == "argmax logits"
            ),
            "details": resnet.to_json(orient="records"),
        }
    )
    return pd.DataFrame(checks)


def format_float(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = frame.columns.tolist()
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(rows)


def build_method_table(metrics: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    threshold_map = thresholds.set_index("method").to_dict(orient="index")
    for method in sorted(METHOD_DESCRIPTIONS):
        method_metrics = metrics.loc[metrics["method"] == method]
        train = method_metrics.loc[method_metrics["split"] == "train"].iloc[0]
        test = method_metrics.loc[method_metrics["split"] == "test"].iloc[0]
        threshold = threshold_map.get(method, {}).get("selected_threshold", "")
        rows.append(
            {
                "method": method,
                "decision_rule": (
                    f"score > {format_float(threshold, 3)}"
                    if method in THRESHOLD_METHODS
                    else "argmax logits"
                ),
                "train_macro_f1": format_float(train["macro_f1"]),
                "test_macro_f1": format_float(test["macro_f1"]),
                "test_foreground_recall": format_float(test["foreground_recall"]),
                "test_background_recall": format_float(test["background_recall"]),
                "test_accuracy": format_float(test["accuracy"]),
            }
        )
    return pd.DataFrame(rows)


def build_markdown(
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    checks: pd.DataFrame,
    method_table: pd.DataFrame,
) -> str:
    split_supports = (
        metrics.groupby("split")["support"]
        .first()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    threshold_table = thresholds.copy()
    threshold_table["selected_threshold"] = threshold_table["selected_threshold"].map(
        lambda value: format_float(value, 3)
    )
    check_table = checks.copy()
    check_table["passed"] = check_table["passed"].map(lambda value: "yes" if value else "no")

    return "\n".join(
        [
            "# Background Segmentation Methods Summary",
            "",
            "## Consistency",
            "",
            (
                "All methods were evaluated as patch-level foreground/background "
                "classifiers against the same 3-annotator majority-label consensus "
                "patches. Thresholded methods selected their operating point on the "
                "training split by macro F1, then applied that fixed threshold to "
                "both train and held-out test splits. ResNet-19 used argmax over "
                "foreground/background logits rather than a scalar threshold."
            ),
            "",
            f"Patch counts: train={split_supports.get('train')}, test={split_supports.get('test')}.",
            "",
            markdown_table(check_table[["check", "passed"]]),
            "",
            "## Method Definitions",
            "",
            *[
                f"- `{method}`: {description}"
                for method, description in METHOD_DESCRIPTIONS.items()
            ],
            "",
            "## Selected Thresholds",
            "",
            markdown_table(
                threshold_table[
                    ["method", "selected_threshold", "selection_split", "selection_metric"]
                ]
            ),
            "",
            "## Performance Summary",
            "",
            markdown_table(method_table),
            "",
            "## Methods Text",
            "",
            (
                "The iCAIRD cervical foreground/background experiments used a "
                "slide-level split of the 120 stratified slides into 100 training "
                "slides and 20 held-out test slides. Patch labels were derived from "
                "the three annotators by majority vote. All compared methods were "
                "evaluated on the same patch coordinates and labels within each "
                "split. For Canny, GrandQC, CLAM-Otsu, and Otsu-HS, scalar "
                "patch-level scores were thresholded as foreground when "
                "`score > threshold`; thresholds were selected on the training split "
                "to maximize macro F1, with foreground recall and then lower "
                "threshold as deterministic tie-breakers. The selected thresholds "
                "were then applied unchanged to the held-out test split. ResNet-19 "
                "was trained on the training split and evaluated on both train and "
                "test splits using the argmax foreground/background prediction. "
                "Performance was reported using scikit-learn classification reports "
                "and confusion matrices for both splits."
            ),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    metrics, thresholds = read_inputs(args.comparison_dir)
    checks = check_consistency(metrics, thresholds)
    method_table = build_method_table(metrics, thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks.to_csv(args.output_dir / "methods_consistency_checks.csv", index=False)
    method_table.to_csv(args.output_dir / "methods_performance_summary.csv", index=False)
    markdown = build_markdown(metrics, thresholds, checks, method_table)
    (args.output_dir / "methods_summary.md").write_text(markdown)
    print(markdown)
    if not checks["passed"].all():
        raise SystemExit("One or more consistency checks failed")


if __name__ == "__main__":
    main()
