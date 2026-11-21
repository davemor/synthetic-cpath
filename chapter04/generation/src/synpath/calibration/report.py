from __future__ import annotations

import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from synpath.artifacts import utc_now, write_json
from synpath.calibration.common import calibration_root
from synpath.calibration.sd_index import (
    SdIndexComponents,
    sd_index,
    select_minimum,
)
from synpath.paths import SynpathPaths


def build_report(cfg: dict, paths: SynpathPaths) -> list[Path]:
    if cfg.get("report_kind") == "cluster_count_validity":
        from synpath.calibration.validity_report import build_validity_report

        return build_validity_report(cfg, paths)
    rows = collect_results(cfg, paths)
    results = pd.DataFrame(rows)
    selections = select_tracks(cfg, results)
    report_root = paths.reports_root / cfg["report_name"]
    report_root.mkdir(parents=True, exist_ok=True)

    results_path = report_root / "sd_index_results.csv"
    selected_csv = report_root / "selected_k.csv"
    selected_yaml = report_root / "selected_k.yaml"
    figure_path = report_root / "sd_index_curves.png"
    markdown_path = report_root / "report.md"
    results.to_csv(results_path, index=False)
    selections.to_csv(selected_csv, index=False)
    selected_payload: dict[str, dict[str, int]] = {}
    for row in selections.itertuples(index=False):
        selected_payload.setdefault(row.dataset, {})[row.encoder] = int(row.k)
    selected_yaml.write_text(yaml.safe_dump(selected_payload, sort_keys=False))
    write_selected_artifacts(cfg, paths, selections)
    plot_curves(cfg, results, selections, figure_path)
    markdown_path.write_text(report_markdown(cfg, selections))
    metadata_path = report_root / "metadata.json"
    write_json(
        metadata_path,
        {
            "created_at": utc_now(),
            "python_version": platform.python_version(),
            "datasets": list(cfg["datasets"]),
            "encoders": list(cfg["encoders"]),
            "k_min": int(cfg["k_min"]),
            "k_max": int(cfg["k_max"]),
            "selection": "minimum_sd_index_with_lower_k_tie_break",
            "sd_index_scorer": cfg.get(
                "sd_index_scorer", {"kind": "halkidi_dis_at_kmax"}
            ),
            "production_clusters_modified": False,
        },
    )
    return [
        results_path,
        selected_csv,
        selected_yaml,
        figure_path,
        markdown_path,
        metadata_path,
    ]


def collect_results(cfg: dict, paths: SynpathPaths) -> list[dict[str, object]]:
    rows = []
    expected_k = list(range(int(cfg["k_min"]), int(cfg["k_max"]) + 1))
    for dataset in cfg["datasets"]:
        for encoder in cfg["encoders"]:
            fit_root = calibration_root(paths, cfg) / dataset / encoder / "fits"
            records = []
            for k in expected_k:
                path = fit_root / f"k_{k:02d}.json"
                if not path.is_file():
                    raise RuntimeError(f"missing SD-index fit result: {path}")
                record = json.loads(path.read_text())
                if int(record["k"]) != k:
                    raise RuntimeError(f"incorrect k recorded in {path}")
                records.append(record)
            if len(records) != 49:
                raise RuntimeError(
                    f"{dataset}/{encoder} does not contain 49 k values"
                )
            scorer_cfg = cfg.get(
                "sd_index_scorer", {"kind": "halkidi_dis_at_kmax"}
            )
            package_scorer = scorer_cfg["kind"] == "s_dbw_sd_0_4_0"
            alpha = (
                np.nan if package_scorer else float(records[-1]["separation"])
            )
            for record in records:
                if package_scorer:
                    value = float(record["sd_index"])
                    scattering = np.nan
                    separation = np.nan
                else:
                    scattering = float(record["scattering"])
                    separation = float(record["separation"])
                    value = sd_index(
                        SdIndexComponents(scattering, separation),
                        alpha,
                    )
                if not np.isfinite(value):
                    raise RuntimeError(
                        f"non-finite SD-index for {dataset}/{encoder}"
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "encoder": encoder,
                        "k": int(record["k"]),
                        "scattering": scattering,
                        "separation": separation,
                        "alpha": alpha,
                        "sd_index": value,
                        "inertia": float(record["inertia"]),
                        "min_cluster_size": min(record["cluster_sizes"]),
                        "max_cluster_size": max(record["cluster_sizes"]),
                        "cluster_sizes": json.dumps(record["cluster_sizes"]),
                        "n_samples": int(record["n_samples"]),
                        "runtime_seconds": float(record["runtime_seconds"]),
                        "backend": record["backend"],
                        "backend_version": record["backend_version"],
                    }
                )
    return rows


def select_tracks(cfg: dict, results: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for dataset in cfg["datasets"]:
        for encoder in cfg["encoders"]:
            rows = results.loc[
                (results["dataset"] == dataset)
                & (results["encoder"] == encoder)
            ].to_dict("records")
            selected.append(select_minimum(rows))
    columns = [
        "dataset",
        "encoder",
        "k",
        "sd_index",
        "scattering",
        "separation",
        "alpha",
        "n_samples",
        "min_cluster_size",
        "max_cluster_size",
    ]
    return pd.DataFrame(selected)[columns]


def write_selected_artifacts(
    cfg: dict,
    paths: SynpathPaths,
    selections: pd.DataFrame,
) -> None:
    for selected in selections.itertuples(index=False):
        track_root = (
            calibration_root(paths, cfg) / selected.dataset / selected.encoder
        )
        fit_root = track_root / "fits"
        labels = np.load(fit_root / f"k_{selected.k:02d}_labels.npy")
        index = pd.read_parquet(track_root / "train_index.parquet")
        if len(labels) != len(index):
            raise RuntimeError(
                "selected labels do not align with feature index"
            )
        counts = np.bincount(labels, minlength=int(selected.k))
        if len(counts) != int(selected.k) or np.any(counts <= 0):
            raise RuntimeError("selected clustering contains an empty cluster")
        output = index.copy()
        output["cluster_id"] = labels.astype(int)
        output.to_csv(track_root / "selected_assignments.csv", index=False)
        metadata = json.loads(
            (track_root / "features_metadata.json").read_text()
        )
        write_json(
            track_root / "selected_metadata.json",
            {
                "created_at": utc_now(),
                "dataset": selected.dataset,
                "encoder": selected.encoder,
                "k": int(selected.k),
                "sd_index": float(selected.sd_index),
                "alpha": (
                    None if pd.isna(selected.alpha) else float(selected.alpha)
                ),
                "selection": "minimum_sd_index_with_lower_k_tie_break",
                "sd_index_scorer": cfg.get(
                    "sd_index_scorer", {"kind": "halkidi_dis_at_kmax"}
                ),
                "global_clustering": True,
                "class_labels_used": False,
                "projection": "none",
                "manifest_fingerprint": metadata["manifest_fingerprint"],
                "assignments_path": str(
                    track_root / "selected_assignments.csv"
                ),
                "centers_path": str(
                    fit_root / f"k_{selected.k:02d}_centers.npy"
                ),
                "production_clusters_modified": False,
            },
        )


def plot_curves(
    cfg: dict,
    results: pd.DataFrame,
    selections: pd.DataFrame,
    output: Path,
) -> None:
    datasets = list(cfg["datasets"])
    encoders = list(cfg["encoders"])
    figure, axes = plt.subplots(
        len(encoders),
        len(datasets),
        figsize=(15, 8),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, encoder in enumerate(encoders):
        for column_index, dataset in enumerate(datasets):
            axis = axes[row_index, column_index]
            curve = results.loc[
                (results["dataset"] == dataset)
                & (results["encoder"] == encoder)
            ].sort_values("k")
            selected = selections.loc[
                (selections["dataset"] == dataset)
                & (selections["encoder"] == encoder)
            ].iloc[0]
            axis.plot(
                curve["k"], curve["sd_index"], color="#2563eb", linewidth=2
            )
            axis.scatter(
                [selected["k"]],
                [selected["sd_index"]],
                color="#dc2626",
                s=55,
                zorder=3,
            )
            axis.annotate(
                f"k={int(selected['k'])}",
                (selected["k"], selected["sd_index"]),
                xytext=(7, 8),
                textcoords="offset points",
                fontsize=10,
            )
            dataset_name = cfg["datasets"][dataset]["display_name"]
            encoder_name = cfg["encoders"][encoder]["display_name"]
            axis.set_title(f"{dataset_name} - {encoder_name}")
            axis.set_xlabel("Number of clusters (k)")
            axis.set_ylabel("SD-index (lower is better)")
            axis.grid(alpha=0.25)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def report_markdown(cfg: dict, selections: pd.DataFrame) -> str:
    table = selections[
        ["dataset", "encoder", "k", "sd_index", "n_samples"]
    ].copy()
    table["sd_index"] = table["sd_index"].map(lambda value: f"{value:.6f}")
    scorer_cfg = cfg.get("sd_index_scorer", {"kind": "halkidi_dis_at_kmax"})
    if scorer_cfg["kind"] == "s_dbw_sd_0_4_0":
        method = (
            "Assignments were scored with s-dbw 0.4.0's SD function using "
            "its package-default k=1 weighting, mean centres snapped to the "
            "nearest observed feature vector, and Euclidean distance. Lower "
            "values are better."
        )
    else:
        method = (
            "The selected k minimizes the Halkidi SD-index, using feature-wise "
            "population variances for Scat and Dis(50) as alpha."
        )
    lines = [
        "# SD-Index Cluster Calibration",
        "",
        "Global K-means was fitted independently for every k from 2 through 50. Class labels were not used.",
        "",
        method,
        "",
        table.to_markdown(index=False),
        "",
        "DINO is the literature-aligned track; UNI2 is a histopathology-specific sensitivity analysis. Absolute SD-index values are not compared between encoders.",
        "",
        "Clustering uses each dataset's complete training split before any experiment cap is applied. Validation and test patches are excluded.",
        "",
        "These calibration outputs are isolated and have not replaced any production cluster assignments.",
        "",
    ]
    for note in cfg.get("report_notes", []):
        lines.extend([str(note), ""])
    return "\n".join(lines)
