#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PathMNIST+ through the official medmnist API."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    try:
        from medmnist import PathMNIST
    except ImportError as exc:
        raise SystemExit(
            "medmnist is not installed. Install project data dependencies "
            "before running this downloader."
        ) from exc

    for split in ("train", "val", "test"):
        PathMNIST(
            split=split,
            download=True,
            size=args.size,
            root=str(args.root),
        )
        print(f"downloaded split={split} size={args.size} root={args.root}")


if __name__ == "__main__":
    main()
