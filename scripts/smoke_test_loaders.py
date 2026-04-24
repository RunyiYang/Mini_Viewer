"""Tiny loader smoke test for local development."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.splat import SplatData


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ply", type=Path)
    group.add_argument("--folder-npy", "--folder_npy", dest="folder_npy", type=Path)
    parser.add_argument("--language-feature", "--language_feature", dest="language_feature", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-splats", "--max_splats", dest="max_splats", type=int, default=1000)
    parser.add_argument("--npy-scale-log", action="store_true")
    args = parser.parse_args()
    splats = SplatData(args, device=args.device)
    data = splats.get_data()
    print({key: tuple(value.shape) for key, value in data.items()})
    large = splats.get_large()
    print("language_feature_large", None if large is None else tuple(large.shape))


if __name__ == "__main__":
    main()
