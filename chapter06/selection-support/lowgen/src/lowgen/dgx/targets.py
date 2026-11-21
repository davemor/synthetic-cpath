from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig


@dataclass(frozen=True)
class Target:
    dataset: str
    encoder: str
    projector: str
    dimension: int | None
    target_name: str
    seed: int
    source_metric: str = ""
    condition: str = ""


def import_dimension_sources(cfg: ExperimentConfig) -> dict[str, Path]:
    cfg.paths.run_dir.mkdir(parents=True, exist_ok=True)
    thresholds = load_threshold_dimensions(cfg)
    intrinsic = load_intrinsic_dimensions(cfg)
    threshold_path = cfg.paths.run_dir / "threshold_dimensions_copied.csv"
    intrinsic_path = cfg.paths.run_dir / "intrinsic_dimensions_copied.csv"
    thresholds.to_csv(threshold_path, index=False)
    intrinsic.to_csv(intrinsic_path, index=False)
    cfg.write_run_config()
    return {"threshold_dimensions": threshold_path, "intrinsic_dimensions": intrinsic_path}


def load_threshold_dimensions(cfg: ExperimentConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.paths.threshold_source)
    if "encoder" not in df.columns:
        parts = df["dataset"].astype(str).str.rsplit("_", n=1, expand=True)
        df["dataset"] = parts[0]
        df["encoder"] = parts[1]
    filtered = df[
        (df["classifier"].astype(str) == cfg.projection.threshold_classifier)
        & (df["projector"].astype(str).isin(cfg.projection.projectors))
        & (df["metric"].astype(str).isin(cfg.projection.threshold_metrics))
        & (df["condition"].astype(float).isin([float(x) for x in cfg.projection.threshold_conditions]))
        & (df["dataset"].astype(str).isin(cfg.datasets))
        & (df["encoder"].astype(str).isin(cfg.encoders))
    ].copy()
    return filtered.sort_values(["dataset", "encoder", "projector", "metric", "condition"])


def load_intrinsic_dimensions(cfg: ExperimentConfig) -> pd.DataFrame:
    if not cfg.paths.intrinsic_source.exists():
        return pd.DataFrame(columns=["dataset", "encoder", "intrinsic_dim_mle"])
    df = pd.read_csv(cfg.paths.intrinsic_source)
    filtered = df[df["dataset"].astype(str).isin(cfg.datasets) & df["encoder"].astype(str).isin(cfg.encoders)]
    return filtered.copy().sort_values(["dataset", "encoder"])


def build_targets(cfg: ExperimentConfig) -> tuple[list[Target], pd.DataFrame]:
    threshold_df = _copied_or_loaded_thresholds(cfg)
    intrinsic_df = _copied_or_loaded_intrinsic(cfg)
    audit: list[dict[str, Any]] = []
    targets: list[Target] = []
    keyed: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}

    if cfg.projection.include_raw_highdim:
        for dataset in cfg.datasets:
            for encoder in cfg.encoders:
                for seed in cfg.seeds:
                    targets.append(Target(dataset, encoder, "raw_highdim", None, "highdim_baseline", int(seed)))

    for dataset in cfg.datasets:
        for encoder in cfg.encoders:
            for projector in cfg.projection.projectors:
                for dimension in cfg.projection.fixed_dimensions:
                    dim = int(dimension)
                    for seed in cfg.seeds:
                        _add_keyed(
                            keyed,
                            dataset,
                            encoder,
                            projector,
                            dim,
                            int(seed),
                            f"dim_{dim}",
                            "fixed_dimension",
                            "",
                        )

                for metric in cfg.projection.threshold_metrics:
                    for condition in cfg.projection.threshold_conditions:
                        rows = threshold_df[
                            (threshold_df["dataset"].astype(str) == dataset)
                            & (threshold_df["encoder"].astype(str) == encoder)
                            & (threshold_df["projector"].astype(str) == projector)
                            & (threshold_df["metric"].astype(str) == metric)
                            & (threshold_df["condition"].astype(float) == float(condition))
                        ]
                        if rows.empty or pd.isna(rows.iloc[0].get("achieved_dim")):
                            audit.append(
                                {
                                    "dataset": dataset,
                                    "encoder": encoder,
                                    "projector": projector,
                                    "metric": metric,
                                    "condition": float(condition),
                                    "reason": "missing_threshold_dimension",
                                }
                            )
                            continue
                        dim = int(float(rows.iloc[0]["achieved_dim"]))
                        name = f"{metric}_{int(round(float(condition) * 100))}"
                        for seed in cfg.seeds:
                            _add_keyed(keyed, dataset, encoder, projector, dim, int(seed), name, metric, str(condition))

                if cfg.projection.include_intrinsic:
                    rows = intrinsic_df[
                        (intrinsic_df["dataset"].astype(str) == dataset)
                        & (intrinsic_df["encoder"].astype(str) == encoder)
                    ]
                    if rows.empty or pd.isna(rows.iloc[0].get("intrinsic_dim_mle")):
                        audit.append(
                            {
                                "dataset": dataset,
                                "encoder": encoder,
                                "projector": projector,
                                "metric": "intrinsic_mle",
                                "condition": "",
                                "reason": "missing_intrinsic_dimension",
                            }
                        )
                        continue
                    dim = int(float(rows.iloc[0]["intrinsic_dim_mle"]))
                    for seed in cfg.seeds:
                        _add_keyed(keyed, dataset, encoder, projector, dim, int(seed), "intrinsic_mle", "intrinsic_mle", "")

    targets.extend(
        Target(
            dataset=row["dataset"],
            encoder=row["encoder"],
            projector=row["projector"],
            dimension=row["dimension"],
            target_name="+".join(sorted(row["target_names"])),
            seed=row["seed"],
            source_metric="+".join(sorted(row["source_metrics"])),
            condition="+".join(sorted(row["conditions"])),
        )
        for row in keyed.values()
    )
    targets.sort(key=lambda x: (x.dataset, x.encoder, x.projector, x.dimension or -1, x.seed))
    return targets, pd.DataFrame(audit)


def _copied_or_loaded_thresholds(cfg: ExperimentConfig) -> pd.DataFrame:
    copied = cfg.paths.run_dir / "threshold_dimensions_copied.csv"
    return pd.read_csv(copied) if copied.exists() else load_threshold_dimensions(cfg)


def _copied_or_loaded_intrinsic(cfg: ExperimentConfig) -> pd.DataFrame:
    copied = cfg.paths.run_dir / "intrinsic_dimensions_copied.csv"
    return pd.read_csv(copied) if copied.exists() else load_intrinsic_dimensions(cfg)


def _add_keyed(
    keyed: dict[tuple[str, str, str, int, int], dict[str, Any]],
    dataset: str,
    encoder: str,
    projector: str,
    dimension: int,
    seed: int,
    target_name: str,
    source_metric: str,
    condition: str,
) -> None:
    key = (dataset, encoder, projector, int(dimension), int(seed))
    payload = keyed.setdefault(
        key,
        {
            "dataset": dataset,
            "encoder": encoder,
            "projector": projector,
            "dimension": int(dimension),
            "seed": int(seed),
            "target_names": set(),
            "source_metrics": set(),
            "conditions": set(),
        },
    )
    payload["target_names"].add(target_name)
    payload["source_metrics"].add(source_metric)
    if condition:
        payload["conditions"].add(str(condition))
