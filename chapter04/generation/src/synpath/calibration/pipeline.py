from __future__ import annotations

from pathlib import Path

from synpath.calibration.curation import curate_dataset
from synpath.calibration.features import prepare_features
from synpath.calibration.worker import fit_k_values
from synpath.config import load_resolved_yaml
from synpath.paths import SynpathPaths


STAGES = ("curate", "embed", "fit", "report")


def run_calibration(
    config_path: Path,
    stage: str,
    dataset: str | None,
    encoder: str | None,
    resume: bool,
) -> list[Path]:
    if stage not in (*STAGES, "all"):
        raise ValueError(f"stage must be one of: all, {', '.join(STAGES)}")
    cfg = load_resolved_yaml(config_path)
    paths = SynpathPaths.from_env()
    datasets = selected_values(dataset, cfg["datasets"], "dataset")
    encoders = selected_values(encoder, cfg["encoders"], "encoder")
    outputs = []
    stages = STAGES if stage == "all" else (stage,)
    for current in stages:
        if current == "curate":
            for dataset_name in datasets:
                artifact = curate_dataset(
                    cfg,
                    dataset_name,
                    paths,
                    resume,
                )
                if artifact is not None:
                    outputs.extend(
                        [
                            artifact.manifest_path,
                            artifact.features_path,
                            artifact.metadata_path,
                        ]
                    )
        elif current == "embed":
            for dataset_name in datasets:
                for encoder_name in encoders:
                    artifact = prepare_features(
                        cfg,
                        dataset_name,
                        encoder_name,
                        paths,
                        resume,
                    )
                    outputs.extend(
                        [
                            artifact.features_path,
                            artifact.index_path,
                            artifact.metadata_path,
                        ]
                    )
        elif current == "fit":
            k_values = list(range(int(cfg["k_min"]), int(cfg["k_max"]) + 1))
            for dataset_name in datasets:
                for encoder_name in encoders:
                    fit_k_values(
                        config_path,
                        dataset_name,
                        encoder_name,
                        k_values,
                        resume,
                    )
        elif current == "report":
            if dataset is not None or encoder is not None:
                raise ValueError("report stage requires all configured tracks")
            from synpath.calibration.report import build_report

            outputs.extend(build_report(cfg, paths))
    return outputs


def selected_values(
    selected: str | None,
    configured: dict,
    kind: str,
) -> list[str]:
    if selected is None:
        return list(configured)
    if selected not in configured:
        raise ValueError(f"unknown calibration {kind}: {selected}")
    return [selected]
