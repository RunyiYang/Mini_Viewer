"""A simple example to render a (large-scale) Gaussian Splats
Found in gsplat/examples/simple_viewer.py

Originally from nerfview
```
"""

import argparse
import math
import os
import time
from typing import Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import viser
from gsplat._helper import load_test_data
from gsplat.rendering import rasterization

from data_loader import load_data
from utils.ply_to_ckpt import generate_gsplat_compatible_data
from viewer import ViewerEditor
from IPython import embed
import functools
import cv2

def apply_bilateral_filter(depth_map, d=9, sigma_color=75, sigma_space=75):
    # Apply bilateral filter
    filtered_depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), d, sigma_color, sigma_space)
    return filtered_depth_map

def apply_histogram_equalization(depth_map):
    # Normalize depth map to 0-255 range
    depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Apply histogram equalization
    equalized_depth_map = cv2.equalizeHist(depth_map_norm)
    return equalized_depth_map

def apply_unsharp_mask(depth_map, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(depth_map, (0, 0), sigma)
    sharpened = cv2.addWeighted(depth_map, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def depth_map_to_rgb(depth_map):
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    
    # Normalize the depth map to [0, 1]
    normalized_depth_map = (depth_map - min_depth) / (max_depth - min_depth)
    
    # Scale to [0, 255] and convert to integers
    grayscale_map = (normalized_depth_map * 255).astype(np.uint8)
    
    # Create an RGB image by stacking the grayscale map into 3 channels
    rgb_image = np.stack([grayscale_map] * 3, axis=-1)
    return rgb_image

def depth_to_normal_map(depth_map):
    # Compute gradients in x and y directions
    grad_y, grad_x = np.gradient(depth_map)

    # Compute the normal for each pixel
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = np.ones_like(depth_map)

    # Normalize the normals
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm

    # Convert to RGB format: map [-1, 1] to [0, 255]
    normal_map = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
    normal_map[..., 0] = ((normal_x + 1) * 0.5 * 255).astype(np.uint8)  # Red channel
    normal_map[..., 1] = ((normal_y + 1) * 0.5 * 255).astype(np.uint8)  # Green channel
    normal_map[..., 2] = ((normal_z + 1) * 0.5 * 255).astype(np.uint8)  # Blue channel

    return normal_map

def smooth_depth_map(depth_map, ksize=5, sigma=2):
    # Apply Gaussian blur to smooth the depth map
    smoothed_depth_map = cv2.GaussianBlur(depth_map, (ksize, ksize), sigma)
    return smoothed_depth_map

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
                     mode: str = "rgb",
                     ):
    width, height = img_wh
    c2w = camera_state.c2w
    K = camera_state.get_K(img_wh)
    c2w = torch.from_numpy(c2w).float().to(device)
    K = torch.from_numpy(K).float().to(device)
    viewmat = c2w.inverse()
    # embed()
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
        render_mode="RGB+D",
        # this is to speedup large-scale rendering by skipping far-away Gaussians.
        radius_clip=3,
    )
    render_rgbs = render_colors[0, ..., 0:3].detach().cpu().numpy()
    render_depths = render_colors[0, ..., 3].detach().cpu().numpy()
    # render_bags = {
    #     "render_rgbs": render_rgbs,
    #     "render_depths": render_depths,
    #     "render_alphas":render_alphas,
    # }
    if mode == "rgb":
        return render_rgbs
    elif mode == "depth":
        # enhanced_depth_map = apply_bilateral_filter(render_depths)
        # enhanced_depth_map = apply_histogram_equalization(enhanced_depth_map)
        # enhanced_depth_map = smooth_depth_map(enhanced_depth_map)
        # enhanced_depth_map = apply_unsharp_mask(enhanced_depth_map)
        return depth_map_to_rgb(render_depths)
    elif mode == "normal":
        # enhanced_depth_map = apply_bilateral_filter(render_depths).astype(np.float32)
        # enhanced_depth_map = apply_histogram_equalization(enhanced_depth_map).astype(np.float32)
        # enhanced_depth_map = smooth_depth_map(enhanced_depth_map).astype(np.float32)
        # enhanced_depth_map = apply_unsharp_mask(enhanced_depth_map).astype(np.float32)
        return depth_to_normal_map(render_depths)

    
    else:
        print("Wrong Mode, rendering rgb")
        return render_rgbs

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, default=None, help="Instead of ckpt, provide ply from Inria and get the view")
    parser.add_argument("--port", type=int, default=8080, help="port for the viewer server")
    parser.add_argument(
        "--backend", type=str, default="gsplat", help="gsplat, gsplat_legacy, inria"
    )
    parser.add_argument("--language_feature", type=str, help="Whether to load language feature")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    torch.manual_seed(42)
    # device = "cuda"


    # register and open viewer
    splat_data = load_data(args)
    if args.language_feature:
        means, quats, scales, opacities, colors, sh_degree, language_feature = splat_data
    else:
        means, quats, scales, opacities, colors, sh_degree = splat_data

    viewer_render_fn_partial = functools.partial(viewer_render_fn, 
                                                 means=means, 
                                                 quats=quats, 
                                                 scales=scales, 
                                                 opacities=opacities, 
                                                 colors=colors, 
                                                 sh_degree=sh_degree, 
                                                 device=args.device, 
                                                 backend=args.backend,
                                                 mode="rgb",
                                                 )

    server = viser.ViserServer(port=args.port, verbose=False)

    _ = ViewerEditor(
        server=server,
        splat_args=args,
        splat_data=splat_data,
        render_fn=viewer_render_fn_partial,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    main()