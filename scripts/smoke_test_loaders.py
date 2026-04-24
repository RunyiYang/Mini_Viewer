"""Smoke-test Mini Viewer data and feature loaders without launching the GUI."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from core.splat import SplatData, load_feature_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Mini Viewer loaders.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ply", type=Path)
    group.add_argument("--folder-npy", "--folder_npy", dest="folder_npy", type=Path)
    parser.add_argument("--feature-file", "--language-feature", "--language_feature", dest="feature_file", type=Path)
    parser.add_argument("--query-feature", "--query_feature", dest="query_feature", type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-splats", "--max_splats", dest="max_splats", type=int, default=None)
    parser.add_argument("--npy-scale-log", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    loader_args = SimpleNamespace(
        ply=args.ply,
        folder_npy=args.folder_npy,
        language_feature=args.feature_file,
        max_splats=args.max_splats,
        npy_scale_log=args.npy_scale_log,
    )
    splats = SplatData(loader_args, device=args.device)
    data = splats.get_data()
    print(f"[smoke] splats: {len(data['means']):,}")
    print(f"[smoke] means: {tuple(data['means'].shape)}")
    large = splats.get_large()
    if large is not None:
        print(f"[smoke] aligned features: {tuple(large.shape)}")

    if args.query_feature is not None:
        query = load_feature_array(args.query_feature)
        print(f"[smoke] query feature: {tuple(query.shape)}")


if __name__ == "__main__":
    main()
