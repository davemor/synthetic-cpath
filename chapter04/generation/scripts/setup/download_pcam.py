#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
from pathlib import Path

FILES = {
    "camelyonpatch_level_2_split_train_x.h5.gz": (
        "1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2",
        "1571f514728f59376b705fc836ff4b63",
    ),
    "camelyonpatch_level_2_split_train_y.h5.gz": (
        "1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG",
        "35c2d7259d906cfc8143347bb8e05be7",
    ),
    "camelyonpatch_level_2_split_valid_x.h5.gz": (
        "1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3",
        "d5b63470df7cfa627aeec8b9dc0c066e",
    ),
    "camelyonpatch_level_2_split_valid_y.h5.gz": (
        "1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO",
        "2b85f58b927af9964a4c15b8f7e8f179",
    ),
    "camelyonpatch_level_2_split_test_x.h5.gz": (
        "1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_",
        "d8c2d60d490dbd479f8199bdfa0cf6ec",
    ),
    "camelyonpatch_level_2_split_test_y.h5.gz": (
        "17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP",
        "60a7035772fbdb7f34eb86d4420cf66a",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and decompress the official PCam HDF5 files."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--decompress-only",
        action="store_true",
        help="Skip downloads and only decompress existing .h5.gz files.",
    )
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="Skip MD5 verification of downloaded .gz files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if not args.decompress_only:
        download_files(args.root, args.skip_md5)
    decompress_files(args.root)


def download_files(root: Path, skip_md5: bool) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "gdown is required for PCam downloads. Install it or place the "
            "official .h5/.h5.gz files in the requested root and rerun with "
            "--decompress-only."
        ) from exc

    for filename, (file_id, expected_md5) in FILES.items():
        path = root / filename
        if path.exists():
            print(f"exists: {path}")
        else:
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"downloading: {filename}")
            gdown.download(url, str(path), quiet=False)
        if not skip_md5:
            actual_md5 = md5(path)
            if actual_md5 != expected_md5:
                raise SystemExit(
                    f"MD5 mismatch for {path}: expected {expected_md5}, "
                    f"got {actual_md5}"
                )


def decompress_files(root: Path) -> None:
    for filename in FILES:
        gz_path = root / filename
        h5_path = root / filename.removesuffix(".gz")
        if h5_path.exists():
            print(f"exists: {h5_path}")
            continue
        if not gz_path.exists():
            raise SystemExit(f"missing: {gz_path}")
        print(f"decompressing: {gz_path} -> {h5_path}")
        with gzip.open(gz_path, "rb") as source, h5_path.open("wb") as target:
            shutil.copyfileobj(source, target)


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
