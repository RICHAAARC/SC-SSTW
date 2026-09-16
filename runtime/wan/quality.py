"""Shared descriptive RGB quality metrics; no acceptance thresholds."""
from __future__ import annotations
from math import log10
from typing import Any

def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("C2A write/read primitives require the optional torch runtime") from exc
    return torch


def rgb_quality_metrics(reference: Any, candidate: Any) -> dict[str, float | None]:
    """Fixed full-frame [0,1] RGB metrics; no candidate numerical gate is applied."""

    torch = _torch()
    if reference.shape != candidate.shape or reference.ndim != 4 or reference.shape[-1] != 3:
        raise ValueError("RGB comparison requires equal [T,H,W,3] clips")
    difference = candidate.to(torch.float64) - reference.to(torch.float64)
    mse = float(difference.square().mean().item())
    psnr = None if mse == 0.0 else 10.0 * log10(1.0 / mse)
    if reference.shape[0] < 2:
        temporal_ratio = None
    else:
        base = reference[1:].to(torch.float64) - reference[:-1].to(torch.float64)
        altered = candidate[1:].to(torch.float64) - candidate[:-1].to(torch.float64)
        baseline_energy = float(base.square().mean().item())
        temporal_ratio = None if baseline_energy == 0.0 else float(altered.square().mean().item() / baseline_energy)
    return {"rgb_mse": mse, "rgb_psnr_db": psnr, "temporal_difference_energy_ratio": temporal_ratio}
