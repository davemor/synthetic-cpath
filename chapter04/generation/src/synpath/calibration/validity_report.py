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
from synpath.calibration.sd_index import SdIndexComponents, sd_index
from synpath.calibration.validity import detect_elbow, select_maximum
from synpath.paths import SynpathPaths


def build_validity_report(cfg: dict, paths: SynpathPaths) -> list[Path]:
    results = pd.DataFrame(collect_validity_results(cfg, paths))
    selections = select_validity_tracks(cfg, results)
    report_root = paths.reports_root / cfg["report_name"]
    report_root.mkdir(parents=True, exist_ok=True)

    results_path = report_root / "cluster_validity_results.csv"
    selected_csv = report_root / "selected_k.csv"
    selected_yaml = report_root / "selected_k.yaml"
    markdown_path = report_root / "report.md"
    metadata_path = report_root / "metadata.json"
    inertia_path = report_root / "inertia_elbow_curves.png"
    silhouette_path = report_root / "silhouette_curves.png"
    sd_path = report_root / "sd_index_curves.png"
    track_plot_root = report_root / "tracks"
    track_plot_root.mkdir(exist_ok=True)

    results.to_csv(results_path, index=False)
    selections.to_csv(selected_csv, index=False)
    selected_yaml.write_text(
        yaml.safe_dump(selection_payload(selections), sort_keys=False)
    )
    write_elbow_artifacts(cfg, paths, selections)
    plot_overview(cfg, results, selections, "inertia", inertia_path)
    plot_overview(cfg, results, selections, "silhouette", silhouette_path)
    plot_overview(cfg, results, selections, "sd_index", sd_path)
    track_paths = plot_track_panels(
        cfg, results, selections, track_plot_root
    )

    audit_path = write_external_audit(cfg, report_root)
    markdown_path.write_text(
        validity_report_markdown(cfg, selections, audit_path)
    )
    write_json(
        metadata_path,
        {
            "created_at": utc_now(),
            "python_version": platform.python_version(),
            "datasets": list(cfg["datasets"]),
            "encoders": list(cfg["encoders"]),
            "k_min": int(cfg["k_min"]),
            "k_max": int(cfg["k_max"]),
            "primary_selection": "offline_kneedle_inertia_elbow",
            "training_only": True,
            "silhouette": cfg["validity"]["silhouette"],
            "sd_index_scorer": cfg["sd_index_scorer"],
            "production_clusters_modified": False,
        },
    )
    outputs = [
        results_path,
        selected_csv,
        selected_yaml,
        inertia_path,
        silhouette_path,
        sd_path,
        markdown_path,
        metadata_path,
        *track_paths,
    ]
    if audit_path is not None:
        outputs.append(audit_path)
    return outputs


def collect_validity_results(
    cfg: dict, paths: SynpathPaths
) -> list[dict[str, object]]:
    rows = []
    expected_k = list(range(int(cfg["k_min"]), int(cfg["k_max"]) + 1))
    for dataset in cfg["datasets"]:
        for encoder in configured_encoders(cfg, dataset):
            track_root = calibration_root(paths, cfg) / dataset / encoder
            validate_training_index(cfg, dataset, track_root)
            records = []
            for k in expected_k:
                path = track_root / "fits" / f"k_{k:02d}.json"
                if not path.is_file():
                    raise RuntimeError(f"missing cluster fit result: {path}")
                record = json.loads(path.read_text())
                if int(record["k"]) != k:
                    raise RuntimeError(f"incorrect k recorded in {path}")
                required = {
                    "scattering",
                    "separation",
                    "silhouette_scores",
                    "silhouette_mean",
                    "silhouette_std",
                }
                missing = required - set(record)
                if missing:
                    raise RuntimeError(
                        f"{path} is missing validity fields: "
                        f"{', '.join(sorted(missing))}"
                    )
                records.append(record)
            if len(records) != len(expected_k):
                raise RuntimeError(
                    f"{dataset}/{encoder} does not contain "
                    f"{len(expected_k)} k values"
                )
            alpha = float(records[-1]["separation"])
            for record in records:
                counts = [int(value) for value in record["cluster_sizes"]]
                if (
                    len(counts) != int(record["k"])
                    or any(value <= 0 for value in counts)
                    or sum(counts) != int(record["n_samples"])
                ):
                    raise RuntimeError(
                        f"invalid cluster support for {dataset}/{encoder}/"
                        f"k={record['k']}"
                    )
                components = SdIndexComponents(
                    float(record["scattering"]),
                    float(record["separation"]),
                )
                halkidi = sd_index(components, alpha)
                values = {
                    "dataset": dataset,
                    "encoder": encoder,
                    "k": int(record["k"]),
                    "n_samples": int(record["n_samples"]),
                    "inertia": float(record["inertia"]),
                    "silhouette_mean": float(record["silhouette_mean"]),
                    "silhouette_std": float(record["silhouette_std"]),
                    "silhouette_scores": json.dumps(
                        record["silhouette_scores"]
                    ),
                    "silhouette_sample_size": int(
                        record["silhouette_sample_size"]
                    ),
                    "scattering": components.scattering,
                    "separation": components.separation,
                    "alpha": alpha,
                    "sd_index": halkidi,
                    "osorio_sd_index": record.get("osorio_sd_index"),
                    "min_cluster_size": min(counts),
                    "max_cluster_size": max(counts),
                    "cluster_sizes": json.dumps(counts),
                    "runtime_seconds": float(record["runtime_seconds"]),
                    "backend": record["backend"],
                    "backend_version": record["backend_version"],
                    "fit_transform": record.get("fit_transform", "none"),
                    "sd_scoring_space": record.get("scoring_space"),
                    "training_only": True,
                }
                if not all(
                    np.isfinite(float(values[column]))
                    for column in (
                        "inertia",
                        "silhouette_mean",
                        "silhouette_std",
                        "sd_index",
                    )
                ):
                    raise RuntimeError(
                        f"non-finite validity score for {dataset}/{encoder}"
                    )
                rows.append(values)
    return rows


def validate_training_index(
    cfg: dict, dataset: str, track_root: Path
) -> None:
    path = track_root / "train_index.parquet"
    if not path.is_file():
        raise RuntimeError(f"training index is missing: {path}")
    index = pd.read_parquet(path)
    if "split" not in index or set(index["split"].astype(str)) != {"train"}:
        raise RuntimeError(f"non-training rows found in {path}")
    expected = int(cfg["datasets"][dataset].get("expected_after_curation", 0))
    if not expected:
        expected = int(cfg["datasets"][dataset]["expected_train"])
    if len(index) != expected:
        raise RuntimeError(
            f"{dataset} training index has {len(index)} rows; expected {expected}"
        )
    if index["sample_id"].duplicated().any():
        raise RuntimeError(f"duplicate training sample IDs found in {path}")


def configured_encoders(cfg: dict, dataset: str) -> list[str]:
    return list(cfg["datasets"][dataset].get("encoders", cfg["encoders"]))


def select_validity_tracks(cfg: dict, results: pd.DataFrame) -> pd.DataFrame:
    selected = []
    sensitivity = float(cfg["validity"]["elbow"]["sensitivity"])
    for dataset in cfg["datasets"]:
        for encoder in configured_encoders(cfg, dataset):
            curve = results.loc[
                (results["dataset"] == dataset)
                & (results["encoder"] == encoder)
            ].sort_values("k")
            records = curve.to_dict("records")
            elbow_k = detect_elbow(
                curve["k"].tolist(),
                curve["inertia"].tolist(),
                sensitivity,
            )
            silhouette = select_maximum(records, "silhouette_mean")
            halkidi = min(
                records,
                key=lambda row: (float(row["sd_index"]), int(row["k"])),
            )
            osorio_records = [
                row
                for row in records
                if pd.notna(row.get("osorio_sd_index"))
            ]
            osorio = (
                min(
                    osorio_records,
                    key=lambda row: (
                        float(row["osorio_sd_index"]),
                        int(row["k"]),
                    ),
                )
                if osorio_records
                else None
            )
            selected.append(
                {
                    "dataset": dataset,
                    "encoder": encoder,
                    "n_samples": int(curve.iloc[0]["n_samples"]),
                    "elbow_k": elbow_k,
                    "silhouette_k": int(silhouette["k"]),
                    "silhouette_mean": float(
                        silhouette["silhouette_mean"]
                    ),
                    "silhouette_std": float(silhouette["silhouette_std"]),
                    "sd_index_k": int(halkidi["k"]),
                    "sd_index": float(halkidi["sd_index"]),
                    "osorio_sd_k": None if osorio is None else int(osorio["k"]),
                    "osorio_sd_index": (
                        None
                        if osorio is None
                        else float(osorio["osorio_sd_index"])
                    ),
                }
            )
    return pd.DataFrame(selected)


def selection_payload(selections: pd.DataFrame) -> dict[str, dict[str, dict]]:
    payload: dict[str, dict[str, dict]] = {}
    for row in selections.itertuples(index=False):
        payload.setdefault(row.dataset, {})[row.encoder] = {
            "primary_elbow_k": nullable_int(row.elbow_k),
            "silhouette_max_k": int(row.silhouette_k),
            "halkidi_sd_min_k": int(row.sd_index_k),
            "osorio_sd_min_k": nullable_int(row.osorio_sd_k),
        }
    return payload


def nullable_int(value: object) -> int | None:
    return None if pd.isna(value) else int(value)


def write_elbow_artifacts(
    cfg: dict, paths: SynpathPaths, selections: pd.DataFrame
) -> None:
    for selected in selections.itertuples(index=False):
        elbow_k = nullable_int(selected.elbow_k)
        if elbow_k is None:
            continue
        track_root = (
            calibration_root(paths, cfg) / selected.dataset / selected.encoder
        )
        labels = np.load(track_root / "fits" / f"k_{elbow_k:02d}_labels.npy")
        index = pd.read_parquet(track_root / "train_index.parquet")
        if len(labels) != len(index):
            raise RuntimeError("elbow assignments do not align with training index")
        if set(index["split"].astype(str)) != {"train"}:
            raise RuntimeError("elbow assignments contain held-out rows")
        output = index.copy()
        output["cluster_id"] = labels.astype(int)
        assignment_path = track_root / "elbow_assignments.csv"
        output.to_csv(assignment_path, index=False)
        feature_metadata = json.loads(
            (track_root / "features_metadata.json").read_text()
        )
        write_json(
            track_root / "elbow_metadata.json",
            {
                "created_at": utc_now(),
                "dataset": selected.dataset,
                "encoder": selected.encoder,
                "k": elbow_k,
                "selection": "offline_kneedle_inertia_elbow",
                "training_only": True,
                "global_clustering": True,
                "class_labels_used": False,
                "manifest_fingerprint": feature_metadata[
                    "manifest_fingerprint"
                ],
                "assignments_path": str(assignment_path),
                "centers_path": str(
                    track_root / "fits" / f"k_{elbow_k:02d}_centers.npy"
                ),
                "standard_scaler_path": feature_metadata.get(
                    "standard_scaler_path"
                ),
                "production_clusters_modified": False,
            },
        )


def plot_overview(
    cfg: dict,
    results: pd.DataFrame,
    selections: pd.DataFrame,
    metric: str,
    output: Path,
) -> None:
    datasets = list(cfg["datasets"])
    encoders = list(cfg["encoders"])
    figure, axes = plt.subplots(
        len(encoders),
        len(datasets),
        figsize=(4.2 * len(datasets), 3.7 * len(encoders)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, encoder in enumerate(encoders):
        for column_index, dataset in enumerate(datasets):
            axis = axes[row_index, column_index]
            if encoder not in configured_encoders(cfg, dataset):
                axis.set_visible(False)
                continue
            curve = results.loc[
                (results["dataset"] == dataset)
                & (results["encoder"] == encoder)
            ].sort_values("k")
            selected = selections.loc[
                (selections["dataset"] == dataset)
                & (selections["encoder"] == encoder)
            ].iloc[0]
            plot_metric(axis, curve, selected, metric)
            axis.set_title(
                f"{cfg['datasets'][dataset]['display_name']}\n"
                f"{cfg['encoders'][encoder]['display_name']}"
            )
            axis.set_xlabel("Number of clusters (k)")
            axis.grid(alpha=0.25)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def plot_metric(axis, curve: pd.DataFrame, selected: pd.Series, metric: str):
    if metric == "inertia":
        y = curve["inertia"]
        selected_k = nullable_int(selected["elbow_k"])
        label = "K-means inertia"
    elif metric == "silhouette":
        y = curve["silhouette_mean"]
        axis.fill_between(
            curve["k"],
            y - curve["silhouette_std"],
            y + curve["silhouette_std"],
            color="#2563eb",
            alpha=0.16,
        )
        selected_k = int(selected["silhouette_k"])
        label = "Mean silhouette"
    elif metric == "sd_index":
        y = curve["sd_index"]
        selected_k = int(selected["sd_index_k"])
        label = "Halkidi SD-index"
    else:
        raise ValueError(f"unknown plot metric: {metric}")
    axis.plot(curve["k"], y, color="#2563eb", linewidth=1.8)
    axis.set_ylabel(label)
    if selected_k is not None:
        value = float(curve.loc[curve["k"] == selected_k, y.name].iloc[0])
        axis.scatter([selected_k], [value], color="#c2410c", s=36, zorder=3)
        axis.annotate(
            f"k={selected_k}",
            (selected_k, value),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
        )


def plot_track_panels(
    cfg: dict,
    results: pd.DataFrame,
    selections: pd.DataFrame,
    output_root: Path,
) -> list[Path]:
    outputs = []
    for selected in selections.itertuples(index=False):
        curve = results.loc[
            (results["dataset"] == selected.dataset)
            & (results["encoder"] == selected.encoder)
        ].sort_values("k")
        selected_series = pd.Series(selected._asdict())
        figure, axes = plt.subplots(
            1, 3, figsize=(12, 3.6), constrained_layout=True
        )
        for axis, metric in zip(
            axes, ("inertia", "silhouette", "sd_index"), strict=True
        ):
            plot_metric(axis, curve, selected_series, metric)
            axis.set_xlabel("Number of clusters (k)")
            axis.grid(alpha=0.25)
        figure.suptitle(
            f"{cfg['datasets'][selected.dataset]['display_name']} - "
            f"{cfg['encoders'][selected.encoder]['display_name']}"
        )
        output = output_root / f"{selected.dataset}_{selected.encoder}.png"
        figure.savefig(output, dpi=300)
        plt.close(figure)
        outputs.append(output)
    return outputs


def write_external_audit(cfg: dict, report_root: Path) -> Path | None:
    audit = cfg.get("external_osorio_audit")
    if not audit:
        return None
    source = Path(audit["results_path"])
    if not source.is_file():
        raise RuntimeError(f"configured Osorio audit is missing: {source}")
    frame = pd.read_csv(source)
    expected = int(cfg["k_max"]) - int(cfg["k_min"]) + 1
    if len(frame) != expected or set(frame["k"]) != set(
        range(int(cfg["k_min"]), int(cfg["k_max"]) + 1)
    ):
        raise RuntimeError("external Osorio audit does not contain k=2..50")
    output = report_root / "osorio_released_audit.csv"
    frame.to_csv(output, index=False)
    return output


def validity_report_markdown(
    cfg: dict, selections: pd.DataFrame, audit_path: Path | None
) -> str:
    table = selections.copy()
    table["elbow_k"] = table["elbow_k"].map(
        lambda value: "none" if pd.isna(value) else str(int(value))
    )
    columns = [
        "dataset",
        "encoder",
        "n_samples",
        "elbow_k",
        "silhouette_k",
        "sd_index_k",
        "osorio_sd_k",
    ]
    lines = [
        "# Training-Set Cluster-Count Calibration",
        "",
        "Global K-means was fitted for every k from 2 through 50 using only each dataset's complete training split. Validation and test patches were excluded from standardisation, fitting, scoring, and selection.",
        "",
        table[columns].to_markdown(index=False),
        "",
        "The primary selection is the offline Kneedle elbow in the inertia curve. Maximum mean silhouette and minimum Halkidi SD-index are reported independently and are not combined into a composite score.",
        "",
        "Silhouette values are mean and standard deviation over three seeded uniform samples of 10,000 training patches. The Halkidi SD-index and silhouette are calculated in the globally standardised training feature space.",
        "",
        "DINO is the literature-comparison representation; UNI2 is a histopathology-specific sensitivity analysis. Absolute scores are not compared between encoders.",
        "",
        "Clusters are feature-space pseudo-labels or candidate morphological groups, not biologically validated categories.",
        "",
        "These isolated artifacts have not replaced production assignments.",
        "",
    ]
    if audit_path is not None:
        audit = pd.read_csv(audit_path)
        selected = audit.sort_values(["sd_index", "k"]).iloc[0]
        lines.extend(
            [
                "## Osorio Released-Code Audit",
                "",
                f"The separately retained scikit-learn/released-preprocessing audit selects k={int(selected['k'])} by its raw-space s-dbw 0.4.0 score. This audit is kept separate because its preprocessing and scoring geometry differ from the primary tracks.",
                "",
            ]
        )
    for note in cfg.get("report_notes", []):
        lines.extend([str(note), ""])
    return "\n".join(lines)
