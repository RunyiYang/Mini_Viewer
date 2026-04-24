"""Pre-download/cache the default SigLIP2 text-query model.

Usage:
    python scripts/download_siglip2.py
    python scripts/download_siglip2.py --cache-dir /path/to/hf_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache SigLIP2 for Mini Viewer.")
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    cache_dir = str(args.cache_dir) if args.cache_dir is not None else None
    dtype = torch.float16 if args.device == "cuda" and torch.cuda.is_available() else torch.float32

    print(f"[download] processor: {args.model}")
    AutoProcessor.from_pretrained(args.model, cache_dir=cache_dir)

    print(f"[download] model: {args.model}")
    model_kwargs = {"cache_dir": cache_dir, "torch_dtype": dtype, "attn_implementation": "sdpa"}
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
    try:
        model = AutoModel.from_pretrained(args.model, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModel.from_pretrained(args.model, **model_kwargs)
    model.eval()
    print("[download] done")


if __name__ == "__main__":
    main()
