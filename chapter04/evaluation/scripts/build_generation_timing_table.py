from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


DATASETS = {
    "icaird_cervical100k": "iCAIRD",
    "pathmnist224": "Kather 100K",
    "pcam": "PCam",
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def measure_manifest(path: Path) -> dict[str, object]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    timestamps = sorted(Path(row["image_path"]).stat().st_mtime_ns / 1e9 for row in rows)
    if len(timestamps) < 2:
        raise ValueError(f"Need at least two generated patches: {path}")
    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    return {
        "num_patches": len(rows),
        "patch_time_mean_s": statistics.mean(intervals),
        "patch_time_sd_s": statistics.stdev(intervals),
        "total_time_s": timestamps[-1] - timestamps[0],
        "maximum_gap_s": max(intervals),
        "gaps_over_60_s": sum(interval > 60 for interval in intervals),
    }


def collect(outputs_root: Path) -> list[dict[str, object]]:
    rows = []
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            for conditioning in CONDITIONING:
                manifest = (
                    outputs_root
                    / "generators"
                    / dataset
                    / architecture
                    / conditioning
                    / "seed0"
                    / "synthetic_manifest.csv"
                )
                if not manifest.exists():
                    raise FileNotFoundError(manifest)
                rows.append(
                    {
                        "dataset": dataset,
                        "architecture": architecture,
                        "conditioning": conditioning,
                        **measure_manifest(manifest),
                        "manifest": str(manifest),
                    }
                )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "dataset",
        "architecture",
        "conditioning",
        "patch_time_mean_s",
        "patch_time_sd_s",
        "num_patches",
        "total_time_s",
        "maximum_gap_s",
        "gaps_over_60_s",
        "manifest",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latex_table(rows: list[dict[str, object]]) -> str:
    by_key = {
        (row["dataset"], row["architecture"], row["conditioning"]): row
        for row in rows
    }
    lines = [
        "% Requires \\usepackage{booktabs,multirow}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "Dataset & Architecture & Conditioning & Patch generation time (s) & Patches & Total time (s) \\\\",
        "\\midrule",
    ]
    for dataset_index, (dataset, dataset_label) in enumerate(DATASETS.items()):
        for architecture_index, (architecture, architecture_label) in enumerate(
            ARCHITECTURES.items()
        ):
            for conditioning_index, (conditioning, conditioning_label) in enumerate(
                CONDITIONING.items()
            ):
                row = by_key[(dataset, architecture, conditioning)]
                dataset_cell = (
                    f"\\multirow{{15}}{{*}}{{{dataset_label}}}"
                    if architecture_index == 0 and conditioning_index == 0
                    else ""
                )
                architecture_cell = (
                    f"\\multirow{{5}}{{*}}{{{architecture_label}}}"
                    if conditioning_index == 0
                    else ""
                )
                interrupted = int(row["gaps_over_60_s"]) > 0
                marker = r"\textsuperscript{\dagger}" if interrupted else ""
                patch_time = (
                    f"${float(row['patch_time_mean_s']):.4f} \\pm "
                    f"{float(row['patch_time_sd_s']):.4f}$"
                )
                total = f"{float(row['total_time_s']):.1f}{marker}"
                lines.append(
                    " & ".join(
                        [
                            dataset_cell,
                            architecture_cell,
                            conditioning_label,
                            patch_time,
                            f"{int(row['num_patches']):,}",
                            total,
                        ]
                    )
                    + r" \\"
                )
            if architecture_index < len(ARCHITECTURES) - 1:
                lines.append(r"\cmidrule(lr){2-6}")
        if dataset_index < len(DATASETS) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Observed synthetic-patch sampling throughput by dataset, architecture, and conditioning regime. Patch generation time is the mean $\\pm$ SD interval between successive completed PNG files; total time is the elapsed interval from the first to the final completed patch. These throughput-derived measurements include image encoding and file writing but exclude generator training and process start-up. They are not single-image inference latencies because generation was parallelised and batched. Rows marked $\\dagger$ contain an interruption longer than 60 seconds and should not be used for direct runtime comparison.}",
            "\\label{tab:patch-generation-times}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(args.outputs_root)
    write_csv(rows, args.output_dir / "patch-generation-times.csv")
    (args.output_dir / "patch-generation-times.tex").write_text(
        latex_table(rows), encoding="utf-8"
    )
    metadata = {
        "definition": "interval between successive PNG completion timestamps",
        "excludes": ["generator training", "process startup before first PNG"],
        "includes": ["sampling", "PNG encoding", "file writing", "parallel execution"],
        "interruption_threshold_seconds": 60,
        "rows_with_interruptions": [
            {
                "dataset": row["dataset"],
                "architecture": row["architecture"],
                "conditioning": row["conditioning"],
                "maximum_gap_s": row["maximum_gap_s"],
                "gaps_over_60_s": row["gaps_over_60_s"],
            }
            for row in rows
            if int(row["gaps_over_60_s"]) > 0
        ],
    }
    (args.output_dir / "patch-generation-times-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
