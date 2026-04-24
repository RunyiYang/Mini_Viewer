"""Mini Viewer entry point.

This version is intentionally explicit about CPU/CUDA behavior:
- CUDA uses gsplat when available.
- CPU uses a lightweight torch/numpy point-splat fallback.
- Language encoders are loaded lazily, and only when CUDA is available unless
  --enable-language-on-cpu is passed.
"""

from __future__ import annotations

import argparse
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
import viser

from actions.base import BasicFeature
from actions.camera_path import CameraPathFeature
from actions.language_feature import LanguageFeature
from core.renderer import viewer_render_fn
from core.splat import SplatData
from core.viewer import ViewerEditor


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_device(device: str) -> str:
    device = device.lower().strip()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[viewer] CUDA requested but torch.cuda.is_available() is false; falling back to CPU.")
        return "cpu"
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device '{device}'. Use auto, cuda, or cpu.")
    return device


def _resolve_backend(backend: str, device: str) -> str:
    backend = backend.lower().strip()
    if backend == "auto":
        return "gsplat" if device == "cuda" else "torch"
    if backend == "gsplat" and device != "cuda":
        print("[viewer] gsplat backend requested on CPU; using torch fallback instead.")
        return "torch"
    if backend not in {"gsplat", "torch"}:
        raise ValueError(f"Unsupported backend '{backend}'. Use auto, gsplat, or torch.")
    return backend


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini Viewer")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--ply", type=Path, default=None, help="Inria/3DGS style .ply file.")
    data.add_argument(
        "--folder-npy",
        "--folder_npy",
        dest="folder_npy",
        type=Path,
        default=None,
        help="Folder containing coord/quat/scale/opacity/color .npy files.",
    )

    parser.add_argument(
        "--feature-file",
        "--feature",
        dest="feature_file",
        type=Path,
        default=None,
        help="Generic aligned feature tensor (.npy/.npz/.pt/.pth). Works for SigLIP/CLIP/DINO features.",
    )
    parser.add_argument(
        "--language-feature",
        "--language_feature",
        dest="language_feature",
        type=Path,
        default=None,
        help="Backward-compatible alias for --feature-file.",
    )
    parser.add_argument(
        "--dino-feature",
        "--dino_feature",
        dest="dino_feature",
        type=Path,
        default=None,
        help="Backward-compatible visual-feature alias for --feature-file.",
    )
    parser.add_argument(
        "--feature-type",
        "--feature_type",
        dest="feature_type",
        default="siglip2",
        choices=["siglip2", "siglip", "clip", "dino", "dinov2", "dino2"],
        help=(
            "Query encoder/feature family. siglip/siglip2/clip use text queries; "
            "dino/dinov2 use image-path queries or --query-feature vectors."
        ),
    )
    parser.add_argument(
        "--query-feature",
        "--query_feature",
        dest="query_feature",
        type=Path,
        default=None,
        help="Optional .npy/.pt/.pth query embedding. If provided, no query encoder is required.",
    )
    parser.add_argument(
        "--query-image",
        "--query_image",
        dest="query_image",
        type=Path,
        default=None,
        help="Optional image path used as the initial DINO/DINOv2 query.",
    )
    parser.add_argument(
        "--siglip-model",
        "--siglip_model",
        dest="siglip_model",
        default="google/siglip2-so400m-patch16-512",
        help="Hugging Face model id for SigLIP/SigLIP2 text queries.",
    )
    parser.add_argument(
        "--dino-model",
        "--dino_model",
        dest="dino_model",
        default="facebook/dinov2-base",
        help="Hugging Face model id for DINO/DINOv2 image queries.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        "--hf_cache_dir",
        dest="hf_cache_dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory for downloaded query encoder weights.",
    )
    parser.add_argument(
        "--enable-feature-model-on-cpu",
        "--enable_feature_model_on_cpu",
        "--enable-language-on-cpu",
        dest="enable_feature_model_on_cpu",
        action="store_true",
        help="Allow text/image query encoders on CPU. This is slow but useful for debugging.",
    )

    parser.add_argument("--bbox-script", "--bbox_script", dest="bbox_script", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--backend", default="auto", choices=["auto", "gsplat", "torch"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sh-degree", "--sh_degree", dest="sh_degree", type=int, default=3)
    parser.add_argument("--max-splats", "--max_splats", dest="max_splats", type=int, default=None)
    parser.add_argument(
        "--max-cpu-splats",
        "--max_cpu_splats",
        dest="max_cpu_splats",
        type=int,
        default=180_000,
        help="CPU/torch render downsample cap. Interactive CPU rendering is approximate.",
    )
    parser.add_argument(
        "--cpu-fallback-splats",
        "--cpu_fallback_splats",
        dest="cpu_fallback_splats",
        type=int,
        default=80_000,
        help="Downsample cap used only when CUDA/gsplat rerendering falls back to CPU.",
    )
    parser.add_argument(
        "--cpu-render-fallback",
        "--fallback-cpu-render",
        dest="cpu_render_fallback",
        action="store_true",
        default=True,
        help="Retry failed CUDA/gsplat rerenders with the CPU torch renderer. Enabled by default.",
    )
    parser.add_argument(
        "--no-cpu-render-fallback",
        "--no-fallback-cpu-render",
        dest="cpu_render_fallback",
        action="store_false",
        help="Disable automatic CPU fallback after CUDA/gsplat render failures.",
    )
    parser.add_argument(
        "--force-cpu-render",
        "--rerender-on-cpu",
        dest="force_cpu_render",
        action="store_true",
        help="Load the scene normally but force all viewer rerenders through the CPU torch renderer.",
    )
    parser.add_argument(
        "--npy-scale-log",
        action="store_true",
        help="Treat NumPy scale arrays as log-scales and exponentiate them.",
    )
    parser.add_argument("--prune", default=None, help="Optional pruning mode/string kept for backwards compatibility.")

    # Camera path/video defaults used by the GUI and the headless script.
    parser.add_argument("--camera-path", "--camera_path", dest="camera_path", type=Path, default=Path("outputs/camera_path.json"))
    parser.add_argument("--video-output", "--video_output", dest="video_output", type=Path, default=Path("outputs/render.mp4"))
    parser.add_argument("--render-width", "--render_width", dest="render_width", type=int, default=1280)
    parser.add_argument("--render-height", "--render_height", dest="render_height", type=int, default=720)
    parser.add_argument("--render-fps", "--render_fps", dest="render_fps", type=int, default=30)
    parser.add_argument("--render-seconds", "--render_seconds", dest="render_seconds", type=float, default=5.0)
    return parser


def _draw_bbox_script(server: viser.ViserServer, bbox_script: Path | None) -> None:
    if bbox_script is None:
        return
    if not bbox_script.exists():
        print(f"[bbox] Missing script: {bbox_script}")
        return
    try:
        from viser_bbox import add_script_bboxes
    except Exception as exc:  # pragma: no cover - optional package path.
        print(f"[bbox] Could not import viser_bbox. Install with 'pip install -e ./viser_bbox'. Error: {exc}")
        return
    try:
        add_script_bboxes(server, str(bbox_script))
        print(f"[bbox] Loaded {bbox_script}")
    except Exception as exc:
        print(f"[bbox] Failed to draw {bbox_script}: {exc}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.device = _resolve_device(args.device)
    args.backend = _resolve_backend(args.backend, args.device)
    args.prune = _boolish(args.prune) if args.prune is not None else False
    feature_path = None
    for candidate in (args.feature_file, args.language_feature, args.dino_feature):
        if candidate is not None:
            feature_path = candidate
            break
    args.feature_file = feature_path
    # SplatData still uses this historical attribute internally.
    args.language_feature = feature_path
    # Backward-compatible attribute expected by older action code.
    args.enable_language_on_cpu = bool(args.enable_feature_model_on_cpu)

    server = viser.ViserServer(host=args.host, port=args.port, verbose=False)
    _draw_bbox_script(server, args.bbox_script)

    splatdata = SplatData(args, device=args.device)
    data = splatdata.get_data()
    cpu_data = splatdata.get_data("cpu")

    render_device = "cpu" if args.force_cpu_render else args.device
    render_backend = "torch" if args.force_cpu_render else args.backend
    render_data = cpu_data if args.force_cpu_render else data
    render_fn = partial(
        viewer_render_fn,
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

    viewer = ViewerEditor(args, splatdata, server=server, render_fn=render_fn, mode="rendering")
    BasicFeature(viewer, splatdata)
    LanguageFeature(
        viewer,
        splatdata,
        feature_type=args.feature_type,
        query_feature_path=args.query_feature,
        query_image_path=args.query_image,
    )
    CameraPathFeature(viewer, splatdata, args)

    try:
        server.scene.add_grid("/grid", width=10.0, height=10.0, cell_size=0.5)
    except Exception:
        pass
    try:
        server.scene.add_frame("/world", axes_length=0.5, axes_radius=0.02)
    except Exception:
        pass

    print(f"[viewer] Open http://localhost:{args.port}")
    print(
        f"[viewer] device={args.device}, backend={args.backend}, "
        f"render_device={render_device}, render_backend={render_backend}, splats={len(data['means']):,}"
    )
    print(
        f"[viewer] cpu_render_fallback={args.cpu_render_fallback}, "
        f"cpu_fallback_splats={args.cpu_fallback_splats:,}, force_cpu_render={args.force_cpu_render}"
    )

    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    # Keep CUDA allocations deterministic enough for interactive debugging.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
