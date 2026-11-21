#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

REPOS = {
    "skin": "histai/SPIDER-skin",
    "colorectal": "histai/SPIDER-colorectal",
    "thorax": "histai/SPIDER-thorax",
    "breast": "histai/SPIDER-breast",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SPIDER organ datasets from Hugging Face Hub."
    )
    parser.add_argument(
        "--organ",
        choices=sorted(REPOS),
        required=True,
    )
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface-hub is not installed. Install project data "
            "dependencies before running this downloader."
        ) from exc

    target = args.root / args.organ
    path = snapshot_download(
        repo_id=REPOS[args.organ],
        repo_type="dataset",
        local_dir=target,
        local_dir_use_symlinks=False,
    )
    print(f"downloaded organ={args.organ} repo={REPOS[args.organ]} path={path}")
    print("If the dataset contains tar shards, extract them before prepare-data.")


if __name__ == "__main__":
    main()
