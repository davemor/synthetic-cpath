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
    active_intervals = [interval for interval in intervals if interval <= 60]
    return {
        "num_patches": len(rows),
        "patch_time_mean_s": statistics.mean(intervals),
        "patch_time_sd_s": statistics.stdev(intervals),
        "total_time_s": timestamps[-1] - timestamps[0],
        "maximum_gap_s": max(intervals),
        "gaps_over_60_s": sum(interval > 60 for interval in intervals),
        "active_patch_time_mean_s": statistics.mean(active_intervals),
        "standardized_10k_time_s": statistics.mean(active_intervals) * 9_999,
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
        "active_patch_time_mean_s",
        "standardized_10k_time_s",
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


def compact_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    real_counts = {
        "icaird_cervical100k": 100_000,
        "pathmnist224": 89_996,
        "pcam": 50_000,
    }
    output = []
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            group = [
                row
                for row in rows
                if row["dataset"] == dataset and row["architecture"] == architecture
                and int(row["gaps_over_60_s"]) == 0
            ]
            patch_times = [float(row["patch_time_mean_s"]) for row in group]
            total_times = [float(row["patch_time_mean_s"]) * 9_999 for row in group]
            output.append(
                {
                    "dataset": dataset,
                    "architecture": architecture,
                    "patch_time_mean_s": statistics.mean(patch_times),
                    "patch_time_sd_s": statistics.stdev(patch_times),
                    "num_synthetic_patches": 10_000,
                    "total_time_mean_s": statistics.mean(total_times),
                    "total_time_sd_s": statistics.stdev(total_times),
                    "real_plus_synthetic_size": real_counts[dataset] + 10_000,
                    "conditioning_runs": len(group),
                }
            )
    return output


def duration(seconds: float, decimals: int = 0) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds % 60
    if hours:
        return f"{hours}h {minutes}m {remainder:.0f}s"
    if minutes:
        return f"{minutes}m {remainder:.0f}s"
    return f"{remainder:.{decimals}f} s"


def compact_latex_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "% Requires \\usepackage{booktabs,multirow}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Architecture & Patch time & Synthetic patches & Total time & Real + synthetic \\\\",
        "\\midrule",
    ]
    for dataset_index, (dataset, dataset_label) in enumerate(DATASETS.items()):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for architecture_index, row in enumerate(dataset_rows):
            dataset_cell = (
                f"\\multirow{{4}}{{*}}{{{dataset_label}}}"
                if architecture_index == 0
                else ""
            )
            lines.append(
                " & ".join(
                    [
                        dataset_cell,
                        ARCHITECTURES[str(row["architecture"])],
                        f"${duration(float(row['patch_time_mean_s']), 4)} \\pm {duration(float(row['patch_time_sd_s']), 4)}$",
                        f"{int(row['num_synthetic_patches']):,}",
                        f"${duration(float(row['total_time_mean_s']))} \\pm {duration(float(row['total_time_sd_s']), 1)}$",
                        f"{int(row['real_plus_synthetic_size']):,}",
                    ]
                )
                + r" \\"
            )
        dataset_total = sum(float(row["total_time_mean_s"]) * 5 for row in dataset_rows)
        lines.extend(
            [
                r"\cmidrule(lr){2-6}",
                f" & \\multicolumn{{2}}{{l}}{{\\textit{{5 regimes; 15 generation runs}}}} & 150,000 & \\textbf{{{duration(dataset_total)}}} & -- \\\\",
            ]
        )
        if dataset_index < len(DATASETS) - 1:
            lines.append(r"\midrule")
    overall_total = sum(float(row["total_time_mean_s"]) * 5 for row in rows)
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{3}{l}{\textbf{Overall: 5 regimes $\times$ 3 architectures $\times$ 3 datasets}}"
            f" & \\textbf{{450,000}} & \\textbf{{{duration(overall_total)}}} & -- \\\\",
        ]
    )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Synthetic-patch generation throughput and augmentation-set sizes. Generation times are mean $\\pm$ SD across the five conditioning regimes. Patch time is the observed interval between successive completed patches, and total time is standardized to generating 10,000 patches. The Kather 100K StyleGAN2-ADA mean and SD use three conditioning runs because two runs containing operational interruptions longer than 60 seconds were excluded; its standardized total extrapolates that valid-run mean across all five regimes. Dataset and overall totals are cumulative standardized generation times, not parallel wall-clock runtimes. Measurements include parallel sampling, PNG encoding, and file writing, but exclude generator training and process start-up.}",
            "\\label{tab:compact-generation-utility-sizes}",
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
    compact = compact_rows(rows)
    write_csv_fields = list(compact[0])
    with (args.output_dir / "compact-generation-utility-sizes.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=write_csv_fields)
        writer.writeheader()
        writer.writerows(compact)
    (args.output_dir / "compact-generation-utility-sizes.tex").write_text(
        compact_latex_table(compact), encoding="utf-8"
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
