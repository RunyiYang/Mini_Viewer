"""Spherical-harmonic color helpers."""

from __future__ import annotations

import torch

C0 = 0.28209479177387814


def SH2RGB(sh: torch.Tensor) -> torch.Tensor:
    """Convert 0th-order 3DGS SH coefficients to approximate RGB."""
    return (sh.float() * C0 + 0.5).clamp(0.0, 1.0)
