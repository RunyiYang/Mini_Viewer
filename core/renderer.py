"""Rendering adapters for Mini Viewer.

The renderer has three execution paths:

1. CUDA + gsplat for the normal fast path.
2. Torch point-splat rendering on the requested device.
3. Explicit CPU fallback rendering after a CUDA/gsplat failure.

The CPU path is intentionally approximate and downsampled. Its job is to keep
interactive rerendering alive after feature-map/color changes, CUDA OOM, or a
backend API mismatch rather than to replace the high-quality gsplat renderer.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Sequence

import numpy as np
import torch

RenderBackend = Literal["auto", "gsplat", "torch"]

_FALLBACK_MESSAGES_SEEN: set[str] = set()


def _log_once(key: str, message: str, *, enabled: bool = True) -> None:
    if not enabled:
        return
    if key in _FALLBACK_MESSAGES_SEEN:
        return
    _FALLBACK_MESSAGES_SEEN.add(key)
    print(message)


def _normalize_device(device: str | torch.device) -> str:
    device_str = str(device)
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device_str


def _as_tensor(value: torch.Tensor | np.ndarray, device: str | torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def _len_first_dim(value: torch.Tensor | np.ndarray) -> int:
    return int(value.shape[0])


def _index_first_dim(value: torch.Tensor | np.ndarray, indices_np: np.ndarray) -> torch.Tensor | np.ndarray:
    """Index the first dimension without moving tensors between devices."""
    if isinstance(value, torch.Tensor):
        idx = torch.as_tensor(indices_np, dtype=torch.long, device=value.device)
        return value.index_select(0, idx)
    return np.asarray(value)[indices_np]


def _downsample_first_dim(
    values: Sequence[torch.Tensor | np.ndarray],
    max_items: int | None,
) -> list[torch.Tensor | np.ndarray]:
    """Deterministically downsample splat-aligned tensors before device transfer."""
    if max_items is None or int(max_items) <= 0 or not values:
        return list(values)
    n = _len_first_dim(values[0])
    max_items = int(max_items)
    if n <= max_items:
        return list(values)
    indices_np = np.linspace(0, n - 1, max_items).astype(np.int64)
    return [_index_first_dim(value, indices_np) for value in values]


def _prepare_rgb(colors: torch.Tensor) -> torch.Tensor:
    """Convert common color/SH layouts to RGB-like [N, 3] values in [0, 1]."""
    if colors.ndim == 3:
        # For SH-style [N, C, 3], the first coefficient is usually DC/RGB-ish.
        colors = colors[:, 0, :]
    if colors.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 color channels, got {tuple(colors.shape)}")
    colors = colors[..., :3].float()
    if torch.nan_to_num(colors).max() > 1.5:
        colors = colors / 255.0
    return colors.clamp(0.0, 1.0)


def image_to_uint8_numpy(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return a contiguous HxWx3 uint8 NumPy image for viser/nerfview."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    else:
        image = np.asarray(image)

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"Rendered image must be HxWxC, got shape {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = image[..., :3]

    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    image = np.nan_to_num(image.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    if float(np.nanmax(image)) <= 1.5:
        image = image * 255.0
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


def _camera_intrinsics(width: int, height: int, fov_y: float) -> torch.Tensor:
    fov_y = float(fov_y)
    if fov_y <= 0.0 or not math.isfinite(fov_y):
        fov_y = math.radians(45.0)
    fy = 0.5 * height / math.tan(0.5 * fov_y)
    fx = fy
    return torch.tensor(
        [[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _camera_intrinsics_from_state(camera_state: Any, width: int, height: int, device: str) -> torch.Tensor:
    """Return gsplat intrinsics with support for the current nerfview API."""
    if hasattr(camera_state, "get_K"):
        try:
            return torch.as_tensor(
                camera_state.get_K((width, height)),
                dtype=torch.float32,
                device=device,
            )
        except Exception:
            pass
    fov = float(getattr(camera_state, "fov", math.radians(45.0)))
    return _camera_intrinsics(width, height, fov).to(device)


def _resolve_img_wh(camera_state: Any, img_wh_or_render_state: Any) -> tuple[int, int]:
    """Accept old and new nerfview render APIs."""
    if isinstance(img_wh_or_render_state, (tuple, list)) and len(img_wh_or_render_state) == 2:
        return int(img_wh_or_render_state[0]), int(img_wh_or_render_state[1])
    render_tab_state = img_wh_or_render_state
    if hasattr(render_tab_state, "preview_render"):
        if bool(getattr(render_tab_state, "preview_render")):
            return int(getattr(render_tab_state, "render_width")), int(getattr(render_tab_state, "render_height"))
        return int(getattr(render_tab_state, "viewer_width", 1280)), int(getattr(render_tab_state, "viewer_height", 720))
    aspect = float(getattr(camera_state, "aspect", 16.0 / 9.0))
    height = int(getattr(render_tab_state, "viewer_res", 720)) if render_tab_state is not None else 720
    return max(1, int(round(height * aspect))), height


def _view_matrix(camera_state: Any, device: str) -> torch.Tensor:
    c2w = getattr(camera_state, "c2w", None)
    if c2w is None:
        c2w = np.eye(4, dtype=np.float32)
    c2w = torch.as_tensor(c2w, dtype=torch.float32, device=device)
    if c2w.shape == (3, 4):
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
        c2w = torch.cat([c2w, bottom], dim=0)
    if c2w.shape != (4, 4):
        raise ValueError(f"camera_state.c2w must be 4x4 or 3x4, got {tuple(c2w.shape)}")
    return torch.linalg.inv(c2w)


def _torch_point_splat_rasterization(
    *,
    camera_state: Any,
    img_wh: tuple[int, int],
    means: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    device: str,
    render_mode: str,
    max_cpu_splats: int | None,
) -> torch.Tensor:
    """Approximate point-splat renderer used by CPU mode and fallback mode."""
    device = _normalize_device(device)
    width, height = int(img_wh[0]), int(img_wh[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {img_wh}")

    means, scales, opacities, colors = _downsample_first_dim(
        [means, scales, opacities, colors],
        max_cpu_splats,
    )

    means = _as_tensor(means, device).float()
    scales = _as_tensor(scales, device).float()
    opacities = _as_tensor(opacities, device).float().flatten().clamp(0.0, 1.0)
    rgb = _prepare_rgb(_as_tensor(colors, device))

    if means.numel() == 0:
        return torch.ones((height, width, 3), dtype=torch.float32, device=device)

    view = _view_matrix(camera_state, device=device)
    ones = torch.ones((len(means), 1), dtype=torch.float32, device=device)
    cam = (torch.cat([means, ones], dim=-1) @ view.T)[:, :3]

    # Viser/Nerf conventions can differ. Use whichever z-sign leaves more points
    # in front of the camera.
    z_plus = cam[:, 2]
    z_minus = -cam[:, 2]
    if (z_minus > 1e-4).sum() > (z_plus > 1e-4).sum():
        z = z_minus
        x = -cam[:, 0]
    else:
        z = z_plus
        x = cam[:, 0]
    y = cam[:, 1]

    visible = z > 1e-4
    if not torch.any(visible):
        return torch.ones((height, width, 3), dtype=torch.float32, device=device)

    x, y, z = x[visible], y[visible], z[visible]
    scales = scales[visible]
    opacities = opacities[visible]
    rgb = rgb[visible]

    fov = float(getattr(camera_state, "fov", math.radians(45.0)))
    K = _camera_intrinsics(width, height, fov).to(device)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * (x / z) + cx
    v = -fy * (y / z) + cy

    if scales.ndim == 1:
        world_radius = scales.abs()
    else:
        world_radius = scales.abs().mean(dim=-1)
    px_radius = (world_radius * fy / z).clamp(min=1.0, max=12.0)

    in_frame = (u >= -16) & (u < width + 16) & (v >= -16) & (v < height + 16)
    if not torch.any(in_frame):
        return torch.ones((height, width, 3), dtype=torch.float32, device=device)
    u, v, z, px_radius, opacities, rgb = [t[in_frame] for t in (u, v, z, px_radius, opacities, rgb)]

    # Far-to-near alpha compositing.
    order = torch.argsort(z, descending=True)
    u, v, z, px_radius, opacities, rgb = [t[order] for t in (u, v, z, px_radius, opacities, rgb)]

    canvas = torch.ones((height, width, 3), dtype=torch.float32, device=device)
    alpha_acc = torch.zeros((height, width, 1), dtype=torch.float32, device=device)
    depth = torch.zeros((height, width, 1), dtype=torch.float32, device=device)

    # This is deliberately simple. The fast/high-quality path is gsplat; this
    # fallback prioritizes not crashing the viewer after renderer swaps.
    for ui, vi, zi, ri, ai, ci in zip(u, v, z, px_radius, opacities, rgb):
        uu = int(round(float(ui)))
        vv = int(round(float(vi)))
        rr = max(1, int(round(float(ri))))
        x0, x1 = max(0, uu - rr), min(width, uu + rr + 1)
        y0, y1 = max(0, vv - rr), min(height, vv + rr + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, device=device),
            torch.arange(x0, x1, device=device),
            indexing="ij",
        )
        dist2 = (xx.float() - ui) ** 2 + (yy.float() - vi) ** 2
        sigma2 = max(float(ri) ** 2, 1.0)
        local_a = (ai * torch.exp(-0.5 * dist2 / sigma2)).clamp(0.0, 1.0)[..., None]
        old_a = alpha_acc[y0:y1, x0:x1]
        comp_a = local_a * (1.0 - old_a)
        canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1.0 - comp_a) + ci.view(1, 1, 3) * comp_a
        alpha_acc[y0:y1, x0:x1] = old_a + comp_a
        depth[y0:y1, x0:x1] = depth[y0:y1, x0:x1] + zi * comp_a

    if render_mode == "depth":
        valid = alpha_acc[..., 0] > 1e-6
        if torch.any(valid):
            d = torch.zeros_like(depth[..., 0])
            d[valid] = depth[..., 0][valid] / alpha_acc[..., 0][valid]
            vals = d[valid]
            lo, hi = torch.quantile(vals, 0.02), torch.quantile(vals, 0.98)
            d = ((d - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
            d[~valid] = 1.0
            return d[..., None].repeat(1, 1, 3)
    return canvas.clamp(0.0, 1.0)


def _try_gsplat_rasterization(
    *,
    camera_state: Any,
    img_wh: tuple[int, int],
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    sh_degree: int,
    device: str,
    render_mode: str,
) -> torch.Tensor:
    width, height = int(img_wh[0]), int(img_wh[1])
    device = _normalize_device(device)
    if not device.startswith("cuda"):
        raise RuntimeError("gsplat requires a CUDA device")

    means = _as_tensor(means, device).float()
    quats = _as_tensor(quats, device).float()
    scales = _as_tensor(scales, device).float()
    opacities = _as_tensor(opacities, device).float().flatten().clamp(0.0, 1.0)
    colors = _as_tensor(colors, device).float()

    viewmat = _view_matrix(camera_state, device=device)[None]
    K = _camera_intrinsics_from_state(camera_state, width, height, device)[None]
    backgrounds = torch.ones((1, 3), dtype=torch.float32, device=device)

    try:
        from gsplat.rendering import rasterization

        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            sh_degree=sh_degree if colors.ndim == 3 else None,
            render_mode="RGB+ED" if render_mode == "depth" else "RGB",
            backgrounds=backgrounds,
            # Work around gsplat 1.5.x packed-mode background shape assertion.
            packed=False,
        )
        image = render_colors[0]
        if render_mode == "depth" and image.shape[-1] >= 4:
            depth = image[..., 3]
            valid = render_alphas[0, ..., 0] > 1e-6
            if torch.any(valid):
                vals = depth[valid]
                lo, hi = torch.quantile(vals, 0.02), torch.quantile(vals, 0.98)
                depth = ((depth - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
                depth[~valid] = 1.0
                return depth[..., None].repeat(1, 1, 3)
        return image[..., :3].clamp(0.0, 1.0)
    except Exception as first_error:
        raise RuntimeError(f"gsplat rasterization failed: {first_error}") from first_error


def viewer_render_fn(
    camera_state: Any,
    img_wh_or_render_state: Any,
    *,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    sh_degree: int = 3,
    device: str = "cuda",
    backend: RenderBackend = "auto",
    render_mode: str = "rgb",
    fov: float | None = None,
    max_cpu_splats: int = 180_000,
    fallback_to_cpu: bool = True,
    cpu_fallback_splats: int | None = None,
    cpu_means: torch.Tensor | None = None,
    cpu_scales: torch.Tensor | None = None,
    cpu_opacities: torch.Tensor | None = None,
    cpu_colors: torch.Tensor | None = None,
    log_fallbacks: bool = True,
) -> np.ndarray:
    """Render one frame for nerfview/viser.

    Returns a contiguous [H, W, 3] uint8 NumPy image.

    ``fallback_to_cpu`` is the important runtime safety valve: when the CUDA
    gsplat path fails during a rerender, the same frame is retried with the
    approximate torch point renderer on real CPU tensors. This avoids killing
    nerfview's render thread when feature maps or query recoloring produce a
    shape/API combination that gsplat rejects.
    """
    width, height = _resolve_img_wh(camera_state, img_wh_or_render_state)
    render_device = _normalize_device(device)

    if fov is not None:
        try:
            camera_state.fov = float(fov)
        except Exception:
            pass

    backend = backend.lower().strip()  # type: ignore[assignment]
    if backend == "auto":
        backend = "gsplat" if render_device.startswith("cuda") else "torch"  # type: ignore[assignment]

    if backend == "gsplat" and render_device.startswith("cuda"):
        try:
            image = _try_gsplat_rasterization(
                camera_state=camera_state,
                img_wh=(width, height),
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                sh_degree=sh_degree,
                device=render_device,
                render_mode=render_mode,
            )
            return image_to_uint8_numpy(image)
        except Exception as exc:
            if fallback_to_cpu:
                fallback_splats = cpu_fallback_splats if cpu_fallback_splats is not None else max_cpu_splats
                _log_once(
                    "gsplat_to_cpu",
                    "[renderer] gsplat failed; rerendering with CPU fallback "
                    f"({fallback_splats:,} splats max): {exc}",
                    enabled=log_fallbacks,
                )
                image = _torch_point_splat_rasterization(
                    camera_state=camera_state,
                    img_wh=(width, height),
                    means=cpu_means if cpu_means is not None else means,
                    scales=cpu_scales if cpu_scales is not None else scales,
                    opacities=cpu_opacities if cpu_opacities is not None else opacities,
                    colors=cpu_colors if cpu_colors is not None else colors,
                    device="cpu",
                    render_mode=render_mode,
                    max_cpu_splats=fallback_splats,
                )
                return image_to_uint8_numpy(image)

            _log_once(
                "gsplat_to_same_device_torch",
                f"[renderer] gsplat failed; CPU fallback disabled, using torch renderer on {render_device}: {exc}",
                enabled=log_fallbacks,
            )

    image = _torch_point_splat_rasterization(
        camera_state=camera_state,
        img_wh=(width, height),
        means=means,
        scales=scales,
        opacities=opacities,
        colors=colors,
        device=render_device,
        render_mode=render_mode,
        max_cpu_splats=max_cpu_splats,
    )
    return image_to_uint8_numpy(image)
