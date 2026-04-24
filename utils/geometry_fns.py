"""Geometry visualization helpers."""

from __future__ import annotations

import torch


def depth_to_rgb(depth: torch.Tensor) -> torch.Tensor:
    """Normalize a depth map to a 3-channel grayscale RGB tensor."""
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    valid = torch.isfinite(depth)
    if not torch.any(valid):
        return torch.ones((*depth.shape, 3), device=depth.device, dtype=torch.float32)
    vals = depth[valid]
    lo, hi = torch.quantile(vals, 0.02), torch.quantile(vals, 0.98)
    norm = ((depth - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
    norm[~valid] = 1.0
    return norm[..., None].repeat(1, 1, 3)


def depth_to_normal(depth: torch.Tensor) -> torch.Tensor:
    """Approximate normals from a depth image using central differences."""
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    dzdx = torch.zeros_like(depth)
    dzdy = torch.zeros_like(depth)
    dzdx[:, 1:-1] = depth[:, 2:] - depth[:, :-2]
    dzdy[1:-1, :] = depth[2:, :] - depth[:-2, :]
    normal = torch.stack([-dzdx, -dzdy, torch.ones_like(depth)], dim=-1)
    normal = torch.nn.functional.normalize(normal, dim=-1, eps=1e-6)
    return (normal + 1.0) * 0.5
