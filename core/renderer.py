from typing import Tuple, Union

import nerfview
import torch
from gsplat.rendering import rasterization
from utils.geometry_fns import depth_to_normal, depth_to_rgb
import numpy as np


def _resolve_img_dims(img_wh: Union[Tuple[int, int], "nerfview.render_panel.RenderTabState"]) -> Tuple[int, int]:
    """Normalize nerfview image dimension metadata across versions."""
    if isinstance(img_wh, tuple):
        return int(img_wh[0]), int(img_wh[1])
    width = getattr(img_wh, "render_width", None) or getattr(img_wh, "viewer_width", None) or getattr(
        img_wh, "viewer_res", None
    )
    height = getattr(img_wh, "render_height", None) or getattr(img_wh, "viewer_height", None) or getattr(
        img_wh, "viewer_res", None
    )
    if width is None or height is None:
        raise ValueError("Unable to infer image dimensions from RenderTabState")
    return int(width), int(height)


@torch.no_grad()
def viewer_render_fn(camera_state: nerfview.CameraState, 
                     img_wh: Union[Tuple[int, int], "nerfview.render_panel.RenderTabState"],
                     means: torch.Tensor,
                     quats: torch.Tensor,
                     scales: torch.Tensor,
                     opacities: torch.Tensor,
                     colors: torch.Tensor,
                     sh_degree: int,
                     device: str,
                     backend: str = "gsplat",
                     render_mode="rgb",
                     fov=45.0 / 180 * 3.1415,
                     ):
    
    width, height = _resolve_img_dims(img_wh)
    c2w = camera_state.c2w
    camera_state.fov = fov
    K = camera_state.get_K((width, height))

    c2w = torch.from_numpy(c2w).float().to(device)
    K = torch.from_numpy(K).float().to(device)
    viewmat = c2w.inverse()

    # if args.backend == "gsplat":
    if backend == "gsplat":
        rasterization_fn = rasterization
    # elif args.backend == "gsplat_legacy":
    elif backend == "gsplat_legacy":
        from gsplat import rasterization_legacy_wrapper

        rasterization_fn = rasterization_legacy_wrapper
    # elif args.backend == "inria":
    elif backend == "inria":
        from gsplat import rasterization_inria_wrapper

        rasterization_fn = rasterization_inria_wrapper
    else:
        raise ValueError

    render_colors, render_alphas, meta = rasterization_fn(
        means,  # [N, 3]
        quats,  # [N, 4]
        scales,  # [N, 3]
        opacities,  # [N]
        colors,  # [N, 3]
        viewmat[None],  # [1, 4, 4]
        K[None],  # [1, 3, 3]
        width,
        height,
        sh_degree=sh_degree,
        backgrounds=None,
        render_mode="RGB+D",
        # this is to speedup large-scale rendering by skipping far-away Gaussians.
        radius_clip=3,
    )
    render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
    render_depths = render_colors[0, ..., 3].cpu().numpy()
    render_alphas = render_alphas[0, ..., 0].cpu().numpy()

    def _apply_white_background(image: np.ndarray, alphas: np.ndarray) -> np.ndarray:
        bg_mask = alphas <= 1e-5
        if not np.any(bg_mask):
            return image
        out = image.copy()
        out[bg_mask] = 1.0
        return out

    if render_mode == "rgb":
        return _apply_white_background(render_rgbs, render_alphas)
    elif render_mode == "depth":
        return depth_to_rgb(render_depths)
    elif render_mode == "normal":
        return depth_to_normal(render_depths)
    else:
        return render_rgbs
