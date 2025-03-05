from typing import Tuple

import nerfview
import torch
from gsplat.rendering import rasterization
from utils.geometry_fns import depth_to_normal, depth_to_rgb


@torch.no_grad()
def viewer_render_fn(camera_state: nerfview.CameraState, 
                     img_wh: Tuple[int, int],
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
    
    width, height = img_wh
    c2w = camera_state.c2w
    camera_state.fov = fov
    K = camera_state.get_K(img_wh)

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
        backgrounds = torch.tensor([1.0, 1.0, 1.0], device=device).reshape(1, 3),
        render_mode="RGB+D",
        # this is to speedup large-scale rendering by skipping far-away Gaussians.
        radius_clip=3,
    )
    render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
    render_depths = render_colors[0, ..., 3].cpu().numpy()
    if render_mode == "rgb":
        return render_rgbs
    elif render_mode == "depth":
        return depth_to_rgb(render_depths)
    elif render_mode == "normal":
        return depth_to_normal(render_depths)
    else:
        return render_rgbs