"""Pre-download/cache the default DINOv2 image encoder used by Mini Viewer."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache DINOv2 weights for Mini Viewer.")
    parser.add_argument("--model", default="facebook/dinov2-base", help="Hugging Face DINO/DINOv2 model id.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional Hugging Face cache directory.")
    args = parser.parse_args()

    from transformers import AutoImageProcessor, AutoModel

    kwargs = {"cache_dir": str(args.cache_dir)} if args.cache_dir is not None else {}
    print(f"[download-dino] downloading processor/model: {args.model}")
    AutoImageProcessor.from_pretrained(args.model, **kwargs)
    AutoModel.from_pretrained(args.model, **kwargs)
    print("[download-dino] done")


if __name__ == "__main__":
    main()
