from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from icaird_cervical_benchmark.config import load_config
from icaird_cervical_benchmark.mil import (
    TASK_CLASS_NAMES,
    TransformerMilClassifier,
    _evaluate_split,
    _mil_config_dict,
    load_bag_records,
    make_loss,
)
from icaird_cervical_benchmark.metrics import write_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=sorted(TASK_CLASS_NAMES))
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    config = load_config(args.config).mil
    task_dir = args.run_dir / "mil" / config.embedding_name / args.task / f"seed_{args.seed}"
    checkpoint_path = task_dir / "best_model_val_auroc.pt"
    started = perf_counter()

    records = load_bag_records(config, args.task)
    class_names = TASK_CLASS_NAMES[args.task]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerMilClassifier(
        input_dim=config.embedding_dim,
        num_classes=len(class_names),
        d_model=config.d_model,
        nhead=config.heads,
        num_encoder_layers=config.encoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        activation=config.activation,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    criterion = make_loss(
        [record for record in records if record.split == "train"],
        len(class_names),
        config.balance_classes,
        device,
    )

    valid_summary = _evaluate_split(model, records, "valid", criterion, config, device, task_dir, class_names)
    test_summary = _evaluate_split(model, records, "test", criterion, config, device, task_dir, class_names)
    elapsed = perf_counter() - started
    enriched = {
        **test_summary,
        "task": "mil",
        "mil_task": args.task,
        "embedding": config.embedding_name,
        "model_id": f"{config.embedding_name}:transformer-mil:{args.task}",
        "seed": args.seed,
        "feature_dim": config.embedding_dim,
        "best_val_auc": checkpoint.get("best_val_auc"),
        "valid_accuracy": valid_summary["accuracy"],
        "valid_auc": valid_summary["auc"],
        "runtime_seconds": elapsed,
    }
    pd.DataFrame([enriched]).to_csv(task_dir / "metrics.csv", index=False)
    write_summary(args.run_dir)
    print(enriched)
    print(_mil_config_dict(config))


if __name__ == "__main__":
    main()
