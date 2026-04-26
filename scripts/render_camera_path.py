"""Headless camera-pose and camera-path renderer for Mini Viewer.

This script can render RGB, depth, normal maps, aligned feature PCA maps, and
query-score/mask maps from the same camera poses exported by the viewer.

Examples:
    # Render RGB video from a camera path.
    python scripts/render_camera_path.py \
        --folder-npy scene_folder \
        --camera-path outputs/camera_path.json \
        --output outputs/render.mp4 \
        --device cuda

    # Render aligned feature map video from the same camera path.
    python scripts/render_camera_path.py \
        --folder-npy scene_folder \
        --feature-file features.npy \
        --render-layer feature \
        --camera-path outputs/camera_path.json \
        --output outputs/feature_map.mp4 \
        --device cuda

    # Render one feature-map PNG from the current saved viewer camera pose.
    python scripts/render_camera_path.py \
        --folder-npy scene_folder \
        --feature-file features.npy \
        --render-layer feature \
        --camera-state .tmp/camera_state.json \
        --output outputs/feature_map.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.renderer import image_to_uint8_numpy, viewer_render_fn
from core.splat import SplatData, load_feature_array
from models.clip_query import (
    CosineClassifier,
    DINOv2Network,
    DINOv2NetworkConfig,
    OpenCLIPNetwork,
    OpenCLIPNetworkConfig,
    SigLIPNetwork,
    SigLIPNetworkConfig,
)

DINO_TYPES = {"dino", "dinov2", "dino2"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".gif"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def _resolve_device(device: str) -> str:
    device = str(device).lower().strip()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[render] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return device


def _resolve_backend(backend: str, device: str) -> str:
    backend = str(backend).lower().strip()
    if backend == "auto":
        return "gsplat" if device == "cuda" else "torch"
    if backend == "gsplat" and device != "cuda":
        return "torch"
    return backend


def _normalize_layer(layer: str) -> str:
    value = str(layer or "rgb").strip().lower().replace("_", "-")
    aliases = {
        "color": "rgb",
        "colors": "rgb",
        "pca": "feature",
        "features": "feature",
        "feature-map": "feature",
        "featuremap": "feature",
        "normal-map": "normal",
        "normals": "normal",
        "score": "query-score",
        "scores": "query-score",
        "query": "query-score",
        "queryscore": "query-score",
        "query-score-map": "query-score",
        "heatmap": "query-score",
        "mask": "query-mask",
        "querymask": "query-mask",
        "query-mask-map": "query-mask",
    }
    value = aliases.get(value, value)
    allowed = {"rgb", "depth", "normal", "feature", "query-score", "query-mask"}
    if value not in allowed:
        raise ValueError(f"Unsupported render layer '{layer}'. Choose one of {sorted(allowed)}.")
    return value


def _fov_to_radians(fov: float) -> float:
    fov = float(fov)
    return math.radians(fov) if fov > math.pi else fov


def _fov_to_degrees(fov: float) -> float:
    fov = float(fov)
    return fov if fov > math.pi else math.degrees(fov)


def _as_4x4_matrix(value: Any) -> np.ndarray:
    mat = np.asarray(value, dtype=np.float64)
    if mat.shape == (4, 4):
        return mat
    if mat.size == 16:
        return mat.reshape(4, 4)
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = mat
        return out
    raise ValueError(f"Expected 4x4, 3x4, or flat 16 camera matrix; got shape {mat.shape}.")


def _load_single_camera_state(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    spec = json.loads(path.read_text(encoding="utf8"))
    if "camera_path" in spec:
        # Be permissive: a camera-path JSON can also be passed to --camera-state.
        return list(spec["camera_path"])

    if "c2w" in spec:
        c2w = _as_4x4_matrix(spec["c2w"])
    elif "camera_to_world" in spec:
        c2w = _as_4x4_matrix(spec["camera_to_world"])
    elif "matrix" in spec:
        value = spec["matrix"]
        if isinstance(value, str):
            import ast

            value = ast.literal_eval(value)
        c2w = _as_4x4_matrix(value)
    else:
        raise ValueError(f"{path} does not contain c2w/camera_to_world/matrix.")

    fov = float(spec.get("fov", spec.get("fov_degrees", 45.0)))
    if "fov_degrees" in spec and "fov" not in spec:
        fov = float(spec["fov_degrees"])
    aspect = float(spec.get("aspect", width / max(height, 1)))
    return [
        {
            "camera_to_world": c2w.reshape(-1).tolist(),
            "fov": fov,
            "aspect": aspect,
        }
    ]


def _load_cameras(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int, int, int]:
    width = int(args.width or 1280)
    height = int(args.height or 720)
    fps = int(args.fps or 30)

    if args.camera_path is not None:
        spec = json.loads(Path(args.camera_path).read_text(encoding="utf8"))
        width = int(args.width or spec.get("render_width", width))
        height = int(args.height or spec.get("render_height", height))
        fps = int(args.fps or spec.get("fps", fps))
        if "camera_path" not in spec:
            raise ValueError(f"Camera-path JSON {args.camera_path} has no 'camera_path' field.")
        return list(spec["camera_path"]), width, height, fps

    if args.camera_state is not None:
        return _load_single_camera_state(Path(args.camera_state), width, height), width, height, fps

    raise ValueError("Pass either --camera-path outputs/camera_path.json or --camera-state .tmp/camera_state.json.")


def _normalize_colors(colors: torch.Tensor) -> torch.Tensor:
    colors = colors.detach().float().cpu()
    if colors.ndim == 3:
        colors = colors[:, 0, :]
    if colors.shape[-1] < 3:
        colors = F.pad(colors, (0, 3 - colors.shape[-1]))
    colors = colors[:, :3]
    if colors.numel() > 0 and float(torch.nan_to_num(colors).max()) > 1.5:
        colors = colors / 255.0
    return colors.clamp(0.0, 1.0).contiguous()


def _score_to_heatmap(scores: torch.Tensor) -> torch.Tensor:
    """Small dependency-free blue->cyan->yellow->red heat map."""
    x = scores.detach().float().cpu().flatten().clamp(0.0, 1.0)
    r = torch.clamp(1.8 * x - 0.35, 0.0, 1.0)
    g = torch.clamp(1.6 - torch.abs(2.0 * x - 1.0) * 1.6, 0.0, 1.0)
    b = torch.clamp(1.35 - 1.8 * x, 0.0, 1.0)
    return torch.stack([r, g, b], dim=-1).contiguous()


def _load_query_feature(path: Path, device: str) -> torch.Tensor:
    arr = np.asarray(load_feature_array(path), dtype=np.float32)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim == 1:
        arr = arr[None]
    query = torch.from_numpy(arr).float().to(device)
    return F.normalize(query, dim=-1, eps=1e-6)


def _build_model_query(args: argparse.Namespace, device: str) -> torch.Tensor | None:
    feature_type = str(args.feature_type or "siglip2").lower().strip()
    cache_dir = str(args.hf_cache_dir) if args.hf_cache_dir is not None else None

    if args.query_feature is not None:
        return _load_query_feature(Path(args.query_feature), device=device)

    if device == "cpu" and not args.enable_feature_model_on_cpu:
        raise RuntimeError(
            "Model-backed query encoding on CPU is disabled. Use --query-feature, run with --device cuda, "
            "or add --enable-feature-model-on-cpu."
        )

    if feature_type in DINO_TYPES:
        if args.query_image is None:
            raise RuntimeError("DINO/DINOv2 score rendering requires --query-image or --query-feature.")
        net = DINOv2Network(
            DINOv2NetworkConfig(
                model_name=str(args.dino_model),
                cache_dir=cache_dir,
            ),
            device=device,
        )
        return net.encode_image(args.query_image).to(device)

    if not args.query_text:
        raise RuntimeError("Text score rendering requires --query-text or --query-feature.")

    if feature_type == "clip":
        net = OpenCLIPNetwork(OpenCLIPNetworkConfig(), device=device)
        return net.encode_text(args.query_text).to(device)

    net = SigLIPNetwork(
        SigLIPNetworkConfig(
            model_name=str(args.siglip_model),
            cache_dir=cache_dir,
        ),
        device=device,
    )
    return net.encode_text(args.query_text).to(device)


def _compute_query_scores(
    *,
    splats: SplatData,
    args: argparse.Namespace,
    device: str,
) -> torch.Tensor:
    features = splats.get_large("cpu")
    if features is None or features.numel() == 0:
        raise RuntimeError("Query-score rendering needs an aligned --feature-file/--language-feature/--dino-feature tensor.")

    query = _build_model_query(args, device=device)
    if query is None:
        raise RuntimeError("Could not build query embedding.")
    if query.shape[-1] != features.shape[-1]:
        raise RuntimeError(f"Query dim {query.shape[-1]} != feature dim {features.shape[-1]}.")

    chunk_size = max(1, int(args.score_chunk_size))
    scores: list[torch.Tensor] = []
    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        chunk = features[start:end].to(device, non_blocking=True)
        scores.append(CosineClassifier.score(chunk, query).detach().cpu())
        del chunk
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    raw = torch.cat(scores, dim=0).float()
    lo = raw.min()
    hi = raw.max()
    norm = ((raw - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
    print(
        "[render] query scores: "
        f"raw_min={float(lo):.4f}, raw_max={float(hi):.4f}, "
        f"selected@{args.threshold:.3f}={int((norm >= args.threshold).sum())}/{len(norm)}"
    )
    if args.score_output is not None:
        args.score_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.score_output, norm.numpy())
        print(f"[render] saved scores to {args.score_output}")
    return norm


def _build_render_colors(
    *,
    splats: SplatData,
    args: argparse.Namespace,
    device: str,
) -> tuple[torch.Tensor, str]:
    """Return CPU [N,3] colors and viewer_render_fn render_mode."""
    layer = _normalize_layer(args.render_layer)
    data_cpu = splats.get_data("cpu")

    if layer == "rgb":
        return _normalize_colors(data_cpu["colors"]), "rgb"

    if layer == "depth":
        return _normalize_colors(data_cpu["colors"]), "depth"

    if layer == "normal":
        normals = data_cpu["normals"].detach().float().cpu()
        return ((normals + 1.0) * 0.5).clamp(0.0, 1.0).contiguous(), "rgb"

    if layer == "feature":
        feature_preview = data_cpu.get("language_feature")
        if feature_preview is None or feature_preview.numel() == 0 or feature_preview.shape[-1] == 0:
            raise RuntimeError("Feature-map rendering needs --feature-file/--language-feature/--dino-feature.")
        return _normalize_colors(feature_preview), "rgb"

    if layer in {"query-score", "query-mask"}:
        scores = _compute_query_scores(splats=splats, args=args, device=device)
        if layer == "query-score":
            return _score_to_heatmap(scores), "rgb"
        colors = torch.full((len(scores), 3), 0.80, dtype=torch.float32)
        mask = scores >= float(args.threshold)
        colors[mask] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        return colors.contiguous(), "rgb"

    raise AssertionError(f"Unhandled layer: {layer}")


def _apply_feature_aliases(args: argparse.Namespace) -> None:
    feature_path = None
    for candidate in (args.feature_file, args.language_feature, args.dino_feature):
        if candidate is not None:
            feature_path = candidate
            break
    args.feature_file = feature_path
    # SplatData currently uses the historical name internally.
    args.language_feature = feature_path


def _default_output(args: argparse.Namespace) -> Path:
    layer = _normalize_layer(args.render_layer)
    if args.camera_state is not None:
        name = "render" if layer == "rgb" else layer.replace("-", "_")
        return Path("outputs") / f"{name}.png"
    name = "render" if layer == "rgb" else layer.replace("-", "_")
    return Path("outputs") / f"{name}.mp4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render RGB/feature/query maps from a Mini Viewer camera pose/path.")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--ply", type=Path)
    data.add_argument("--folder-npy", "--folder_npy", dest="folder_npy", type=Path)

    camera = parser.add_mutually_exclusive_group(required=True)
    camera.add_argument("--camera-path", "--camera_path", dest="camera_path", type=Path)
    camera.add_argument("--camera-state", "--camera_state", dest="camera_state", type=Path)

    parser.add_argument("--feature-file", "--feature", dest="feature_file", type=Path, default=None)
    parser.add_argument("--language-feature", "--language_feature", dest="language_feature", type=Path, default=None)
    parser.add_argument("--dino-feature", "--dino_feature", dest="dino_feature", type=Path, default=None)
    parser.add_argument(
        "--render-layer",
        "--render_layer",
        "--render-mode",
        "--render_mode",
        dest="render_layer",
        default="rgb",
        help="rgb, depth, normal, feature, query-score, or query-mask.",
    )
    parser.add_argument(
        "--feature-type",
        "--feature_type",
        dest="feature_type",
        default="siglip2",
        choices=["siglip2", "siglip", "clip", "dino", "dinov2", "dino2"],
    )
    parser.add_argument("--query-text", "--query_text", dest="query_text", default=None)
    parser.add_argument("--query-image", "--query_image", dest="query_image", type=Path, default=None)
    parser.add_argument("--query-feature", "--query_feature", dest="query_feature", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--score-output", "--score_output", dest="score_output", type=Path, default=None)
    parser.add_argument("--score-chunk-size", "--score_chunk_size", dest="score_chunk_size", type=int, default=200_000)

    parser.add_argument("--siglip-model", "--siglip_model", dest="siglip_model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--dino-model", "--dino_model", dest="dino_model", default="facebook/dinov2-base")
    parser.add_argument("--hf-cache-dir", "--hf_cache_dir", dest="hf_cache_dir", type=Path, default=None)
    parser.add_argument("--enable-feature-model-on-cpu", "--enable_feature_model_on_cpu", action="store_true")

    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frame-output-dir", "--frame_output_dir", dest="frame_output_dir", type=Path, default=None)
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
    args.render_layer = _normalize_layer(args.render_layer)
    args.output = args.output or _default_output(args)
    _apply_feature_aliases(args)

    camera_path, width, height, fps = _load_cameras(args)
    render_device = "cpu" if args.force_cpu_render else args.device
    render_backend = "torch" if args.force_cpu_render else args.backend
    score_device = args.device if args.device == "cuda" else "cpu"

    splats = SplatData(args, device=args.device)
    data = splats.get_data()
    cpu_data = splats.get_data("cpu")
    colors_cpu, render_mode = _build_render_colors(splats=splats, args=args, device=score_device)
    colors_render = colors_cpu if args.force_cpu_render else colors_cpu.to(render_device)

    render_data = cpu_data if args.force_cpu_render else data
    frames: list[np.ndarray] = []

    if args.frame_output_dir is not None:
        args.frame_output_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(camera_path):
        c2w = _as_4x4_matrix(item["camera_to_world"])
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
            colors=colors_render,
            sh_degree=args.sh_degree,
            device=render_device,
            backend=render_backend,
            render_mode=render_mode,
            max_cpu_splats=args.max_cpu_splats,
            fallback_to_cpu=args.cpu_render_fallback,
            cpu_fallback_splats=args.cpu_fallback_splats,
            cpu_means=cpu_data["means"],
            cpu_scales=cpu_data["scales"],
            cpu_opacities=cpu_data["opacities"],
            cpu_colors=colors_cpu,
        )
        image_np = image_to_uint8_numpy(image)
        frames.append(image_np)
        if args.frame_output_dir is not None:
            Image.fromarray(image_np).save(args.frame_output_dir / f"frame_{i:05d}.png")
        if (i + 1) % 30 == 0 or i + 1 == len(camera_path):
            print(f"[render] {i + 1}/{len(camera_path)} frames")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_suffix = args.output.suffix.lower()
    if len(frames) == 1 and output_suffix not in VIDEO_SUFFIXES:
        if output_suffix and output_suffix not in IMAGE_SUFFIXES:
            print(f"[render] Unknown image suffix '{output_suffix}', saving PNG-compatible bytes anyway.")
        Image.fromarray(frames[0]).save(args.output)
    else:
        imageio.mimsave(args.output, frames, fps=fps, macro_block_size=1)
    print(f"[render] Saved {args.output}")


if __name__ == "__main__":
    main()
