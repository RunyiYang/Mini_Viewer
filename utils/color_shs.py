"""Small SH/RGB helpers used by Mini Viewer loaders and actions."""

from __future__ import annotations

import torch

C0 = 0.28209479177387814


def RGB2SH(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB values in [0, 1] to degree-0 SH coefficients."""
    return (rgb - 0.5) / C0


def SH2RGB(sh: torch.Tensor) -> torch.Tensor:
    """Convert degree-0 SH coefficients to RGB values in [0, 1]."""
    return (sh * C0 + 0.5).clamp(0.0, 1.0)
