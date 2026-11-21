from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from synpath.generation import wrappers
from synpath.reports.latex import (
    ARCHITECTURES,
    DATASETS,
    QUALITY_CONDITIONING,
    QUALITY_METRICS,
    UTILITY_CONDITIONING,
    UTILITY_METRICS,
    render_appendix_tables,
    summarise_quality,
    summarise_utility,
)


def test_appendix_tables_use_multirows_rules_and_standard_deviations():
    quality = summarise_quality(quality_frame(), (0, 1, 2))
    utility = summarise_utility(utility_frame(), (0, 1, 2))

    latex = render_appendix_tables(quality, utility)

    assert "\\usepackage{booktabs,longtable,multirow}" in latex
    assert "\\multirow{15}{*}{SG2}" in latex
    assert "\\multirow{3}{*}{Unconditioned}" in latex
    assert "\\multirow{9}{*}{SG2}" in latex
    assert "\\cmidrule(lr){2-7}" in latex
    assert "\\midrule" in latex
    assert "\\pm" in latex


def test_quality_summary_rejects_missing_generator_seed():
    frame = quality_frame()
    frame = frame[~((frame["seed"] == 2) & (frame.index == 2))]

    with pytest.raises(
        ValueError, match="required metric replicates are missing"
    ):
        summarise_quality(frame, (0, 1, 2))


def test_training_commands_receive_generator_seed(tmp_path, monkeypatch):
    stylegan = wrappers.stylegan_train_command(
        tmp_path / "repo",
        tmp_path / "python",
        tmp_path / "root",
        tmp_path / "data.zip",
        {"stylegan_gpus": 1, "generation": {"stylegan2ada": {}}},
        {},
        True,
        7,
    )
    assert stylegan[-2:] == ["--seed", "7"]

    script = tmp_path / "train_text_to_image_lora.py"
    script.write_text("")
    captured = []
    monkeypatch.setattr(wrappers, "write_metadata_jsonl", lambda *_: None)
    monkeypatch.setattr(
        wrappers,
        "run_command",
        lambda command, _: captured.append(command),
    )
    wrappers.train_sd_lora(
        tmp_path / "root",
        tmp_path / "images",
        tmp_path / "conditions.csv",
        SimpleNamespace(external_root=tmp_path),
        {"sd_lora_train_script": str(script)},
        seed=11,
    )
    seed_index = captured[0].index("--seed")
    assert captured[0][seed_index + 1] == "11"
    rank_index = captured[0].index("--rank")
    assert captured[0][rank_index + 1] == "4"


def quality_frame() -> pd.DataFrame:
    rows = []
    for dataset, _ in DATASETS:
        for model, _ in ARCHITECTURES:
            for conditioning, _ in QUALITY_CONDITIONING:
                for spec in QUALITY_METRICS:
                    for seed in (0, 1, 2):
                        rows.append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "conditioning": conditioning,
                                "seed": seed,
                                "metric": spec.metric,
                                "value": seed + 1,
                                "source_file": spec.source_file,
                            }
                        )
    return pd.DataFrame(rows)


def utility_frame() -> pd.DataFrame:
    rows = []
    for dataset, _ in DATASETS:
        for model, _ in ARCHITECTURES:
            for conditioning, _ in UTILITY_CONDITIONING:
                for source in ("synthetic_only", "real_plus_synthetic"):
                    for spec in UTILITY_METRICS:
                        for seed in (0, 1, 2):
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "model": "resnet18",
                                    "generator_model": model,
                                    "conditioning": f"{source}_{conditioning}",
                                    "seed": seed,
                                    "metric": spec.metric,
                                    "value": seed + 1,
                                }
                            )
    return pd.DataFrame(rows)
