from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


ARCHITECTURES = {
    "stylegan2ada": "StyleGAN2-ADA",
    "sd_lora": "SD-LoRA",
    "dcgan": "DCGAN",
}


@dataclass(frozen=True)
class Run:
    architecture: str
    runtime_seconds: float | None
    exposures: int
    gpus: int


def elapsed_seconds(text: str) -> float:
    days = re.search(r"(?:(\d+)d )?(\d+)h (\d+)m (\d+)s", text)
    if days:
        day, hour, minute, second = (int(value or 0) for value in days.groups())
        return day * 86400 + hour * 3600 + minute * 60 + second
    fields = [int(value) for value in text.split(":")]
    if len(fields) == 3:
        return fields[0] * 3600 + fields[1] * 60 + fields[2]
    if len(fields) == 2:
        return fields[0] * 60 + fields[1]
    raise ValueError(f"Unsupported elapsed time: {text}")


def parse_stylegan(run_dir: Path) -> Run:
    text = (run_dir / "stylegan_train.log").read_text(errors="replace")
    times = re.findall(r"time ((?:(?:\d+)d )?\d+h \d+m \d+s)", text)
    budgets = re.findall(r"Training duration:\s+(\d+) kimg", text)
    gpu_matches = re.findall(r"gpus(\d+)-batch", text)
    return Run(
        architecture="stylegan2ada",
        runtime_seconds=elapsed_seconds(times[-1]) if times else None,
        exposures=int(budgets[-1]) * 1000,
        gpus=int(gpu_matches[-1]) if gpu_matches else 8,
    )


def parse_sd_lora(run_dir: Path) -> Run:
    text = (run_dir / "sd_lora_train.log").read_text(errors="replace")
    elapsed = re.findall(r"\[(\d+(?::\d+){1,2})<", text)
    steps = re.findall(r"Total optimization steps = (\d+)", text)
    batches = re.findall(r"Total train batch size .* = (\d+)", text)
    processes = re.findall(r"Num processes: (\d+)", text)
    return Run(
        architecture="sd_lora",
        runtime_seconds=elapsed_seconds(elapsed[-1]) if elapsed else None,
        exposures=int(steps[-1]) * int(batches[-1]),
        gpus=int(processes[-1]) if processes else 8,
    )


def parse_dcgan(run_dir: Path, dataset: str) -> Run:
    # The checkpoint records 20,000 executed steps for Kather/iCAIRD and 19,500
    # for PCam. The configured global batch size was 128 on eight GPUs.
    steps = 19_500 if dataset == "pcam" else 20_000
    return Run(
        architecture="dcgan",
        runtime_seconds=None,
        exposures=steps * 128,
        gpus=8,
    )


def discover(root: Path) -> list[Run]:
    runs: list[Run] = []
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for architecture in ARCHITECTURES:
            architecture_dir = dataset_dir / architecture
            if not architecture_dir.exists():
                continue
            for run_dir in sorted(architecture_dir.glob("*/seed0")):
                if architecture == "stylegan2ada":
                    runs.append(parse_stylegan(run_dir))
                elif architecture == "sd_lora":
                    runs.append(parse_sd_lora(run_dir))
                else:
                    runs.append(parse_dcgan(run_dir, dataset_dir.name))
    return runs


def format_duration(seconds: float) -> str:
    rounded = round(seconds)
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def summarise(runs: list[Run]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for architecture, label in ARCHITECTURES.items():
        selected = [run for run in runs if run.architecture == architecture]
        runtimes = [run.runtime_seconds for run in selected if run.runtime_seconds]
        total_runtime = sum(runtimes)
        rows.append(
            {
                "architecture": label,
                "models": len(selected),
                "models_with_runtime": len(runtimes),
                "gpus_per_model": selected[0].gpus,
                "mean_runtime_seconds": mean(runtimes) if runtimes else None,
                "runtime_sd_seconds": stdev(runtimes) if len(runtimes) > 1 else None,
                "cumulative_wall_seconds": total_runtime if runtimes else None,
                "gpu_hours": total_runtime * selected[0].gpus / 3600 if runtimes else None,
                "total_exposures": sum(run.exposures for run in selected),
                "mean_exposures": round(mean(run.exposures for run in selected)),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Architecture & Models & Time/model & Total wall & GPU-hours \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row["mean_runtime_seconds"] is None:
            runtime = "Not recorded"
            total_wall = "--"
            gpu_hours = "--"
        else:
            runtime = (
                f"{format_duration(float(row['mean_runtime_seconds']))} "
                f"$\\pm$ {format_duration(float(row['runtime_sd_seconds']))}"
            )
            total_wall = format_duration(float(row["cumulative_wall_seconds"]))
            gpu_hours = f"{float(row['gpu_hours']):,.1f}"
        lines.append(
            f"{row['architecture']} & {row['models']} & {runtime} & {total_wall} & "
            f"{gpu_hours} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Generator training cost for the 45 canonical models. "
            "Times are wall-clock mean $\\pm$ SD across conditioning regimes and "
            "datasets; GPU-hours are cumulative across models with recorded runtimes. "
            "DCGAN runtime was not retained by the original pipeline.}",
            "\\label{tab:generator-training-runtime}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = summarise(discover(args.root))
    write_csv(rows, args.output / "generator-training-runtime.csv")
    write_tex(rows, args.output / "generator-training-runtime.tex")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
