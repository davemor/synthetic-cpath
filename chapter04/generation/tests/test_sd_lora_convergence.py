from __future__ import annotations

import pandas as pd
import pytest

from synpath.convergence.sd_lora import assess_plateau, validate_config
from synpath.generation.wrappers import sd_lora_optional_training_args


def test_convergence_config_requires_rank_16_and_final_checkpoint():
    cfg = {"generation": {"sd_lora": {"rank": 16, "max_train_steps": 20000}}}
    audit = {
        "dataset": "pathmnist224",
        "model": "sd_lora",
        "conditioning": "true_label",
        "checkpoints": [5000, 10000, 20000],
    }
    validate_config(cfg, audit)

    cfg["generation"]["sd_lora"]["rank"] = 4
    with pytest.raises(ValueError, match="rank 16"):
        validate_config(cfg, audit)


def test_sd_lora_checkpoint_arguments_are_explicit():
    args = sd_lora_optional_training_args(
        {"checkpointing_steps": 2500, "checkpoints_total_limit": 8}
    )
    assert args == [
        "--checkpointing_steps",
        "2500",
        "--checkpoints_total_limit",
        "8",
    ]


def test_plateau_assessment_uses_last_three_checkpoints():
    frame = pd.DataFrame(
        {
            "step": [5000, 10000, 15000, 20000],
            "fid": [100.0, 90.0, 89.5, 89.0],
            "uni_fid": [1000.0, 900.0, 895.0, 890.0],
            "precision": [0.01, 0.02, 0.025, 0.027],
            "recall": [0.01, 0.02, 0.025, 0.027],
        }
    )
    result = assess_plateau(
        frame,
        {
            "distance_relative_change": 0.02,
            "precision_recall_absolute_change": 0.01,
        },
    )
    assert result["overall_plateau"] is True
