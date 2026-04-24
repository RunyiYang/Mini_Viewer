"""Headless camera-path renderer for Mini Viewer.

Example:
    python scripts/render_camera_path.py \
        --ply scene.ply \
        --camera-path outputs/camera_path.json \
        --output outputs/render.mp4 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import nerfview
import numpy as np
import torch

from core.renderer import image_to_uint8_numpy, viewer_render_fn
from core.splat import SplatData


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[render] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return device


def _resolve_backend(backend: str, device: str) -> str:
    if backend == "auto":
        return "gsplat" if device == "cuda" else "torch"
    if backend == "gsplat" and device != "cuda":
        return "torch"
    return backend


def _fov_to_radians(fov: float) -> float:
    return math.radians(fov) if fov > math.pi else fov


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a Mini Viewer/Nerfstudio camera path.")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--ply", type=Path)
    data.add_argument("--folder-npy", "--folder_npy", dest="folder_npy", type=Path)
    parser.add_argument("--language-feature", "--language_feature", dest="language_feature", type=Path, default=None)
    parser.add_argument("--camera-path", "--camera_path", dest="camera_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/render.mp4"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--backend", choices=["auto", "gsplat", "torch"], default="auto")
    parser.add_argument("--sh-degree", "--sh_degree", dest="sh_degree", type=int, default=3)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--max-splats", "--max_splats", dest="max_splats", type=int, default=None)
    parser.add_argument("--max-cpu-splats", "--max_cpu_splats", dest="max_cpu_splats", type=int, default=180_000)
    parser.add_argument("--cpu-fallback-splats", "--cpu_fallback_splats", dest="cpu_fallback_splats", type=int, default=80_000)
    parser.add_argument(
        "--cpu-render-fallback",
        "--fallback-cpu-render",
        dest="cpu_render_fallback",
        action="store_true",
        default=True,
        help="Retry failed CUDA/gsplat frames with the CPU renderer. Enabled by default.",
    )
    parser.add_argument("--no-cpu-render-fallback", "--no-fallback-cpu-render", dest="cpu_render_fallback", action="store_false")
    parser.add_argument("--force-cpu-render", "--rerender-on-cpu", dest="force_cpu_render", action="store_true")
    parser.add_argument("--npy-scale-log", action="store_true")
    parser.add_argument("--prune", default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.device = _resolve_device(args.device)
    args.backend = _resolve_backend(args.backend, args.device)
    spec = json.loads(args.camera_path.read_text(encoding="utf8"))
    width = int(args.width or spec.get("render_width", 1280))
    height = int(args.height or spec.get("render_height", 720))
    fps = int(args.fps or spec.get("fps", 30))

    render_device = "cpu" if args.force_cpu_render else args.device
    render_backend = "torch" if args.force_cpu_render else args.backend

    splats = SplatData(args, device=args.device)
    data = splats.get_data()
    cpu_data = splats.get_data("cpu")
    render_data = cpu_data if args.force_cpu_render else data
    frames: list[np.ndarray] = []
    camera_path = spec["camera_path"]

    for i, item in enumerate(camera_path):
        c2w = np.asarray(item["camera_to_world"], dtype=np.float64).reshape(4, 4)
        fov = _fov_to_radians(float(item.get("fov", 45.0)))
        aspect = float(item.get("aspect", width / max(height, 1)))
        camera_state = nerfview.CameraState(c2w=c2w, fov=fov, aspect=aspect)
        image = viewer_render_fn(
            camera_state,
            (width, height),
            means=render_data["means"],
            quats=render_data["quats"],
            scales=render_data["scales"],
            opacities=render_data["opacities"],
            colors=render_data["colors"],
            sh_degree=args.sh_degree,
            device=render_device,
            backend=render_backend,
            max_cpu_splats=args.max_cpu_splats,
            fallback_to_cpu=args.cpu_render_fallback,
            cpu_fallback_splats=args.cpu_fallback_splats,
            cpu_means=cpu_data["means"],
            cpu_scales=cpu_data["scales"],
            cpu_opacities=cpu_data["opacities"],
            cpu_colors=cpu_data["colors"],
        )
        frames.append(image_to_uint8_numpy(image))
        if (i + 1) % 30 == 0 or i + 1 == len(camera_path):
            print(f"[render] {i + 1}/{len(camera_path)} frames")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=fps, macro_block_size=1)
    print(f"[render] Saved {args.output}")


if __name__ == "__main__":
    main()
