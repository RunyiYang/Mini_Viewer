"""Geometry helper placeholders kept for backwards-compatible imports."""

from __future__ import annotations

import torch


def normalize_vectors(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize vectors along the last dimension."""
    return values / values.norm(dim=-1, keepdim=True).clamp_min(eps)
