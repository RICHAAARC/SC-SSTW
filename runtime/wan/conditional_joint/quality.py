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

import math

def residual_temporal_metrics(reference_rgb8, candidate_rgb8, *, source_start):
    """Condition-minus-P1 residual structure, separate from video self-difference ratio."""
    import torch
    if reference_rgb8.shape!=candidate_rgb8.shape or reference_rgb8.ndim!=4:
        raise ValueError("equal [T,H,W,3] clips required")
    frames=int(reference_rgb8.shape[0]); energy=roughness=boundary_energy=internal_energy=0.0
    previous=None; lag=prev_energy=next_energy=0.0; boundary_edges=0
    for index in range(frames):
        value=(candidate_rgb8[index].to(torch.float64)-reference_rgb8[index].to(torch.float64))/255.0
        energy+=float(value.square().sum())
        if previous is not None:
            edge=float((value-previous).square().sum());roughness+=edge
            if (source_start+index)%4==0:
                boundary_edges+=1;boundary_energy+=edge
            else:internal_energy+=edge
            lag+=float((previous*value).sum())
            prev_energy+=float(previous.square().sum());next_energy+=float(value.square().sum())
        previous=value
    return dict(definition="delta=condition-P1 in RGB[0,1]; unscaled noncyclic adjacent difference",
        source_start=int(source_start),frames=frames,edges=frames-1,boundary_edges=boundary_edges,
        internal_edges=frames-1-boundary_edges,residual_l2_squared=energy,
        residual_dt_squared=roughness,residual_dt_boundary_squared=boundary_energy,
        residual_dt_internal_squared=internal_energy,
        residual_normalized_roughness=roughness/energy if energy else None,
        residual_lag1_uncentered_cosine=lag/math.sqrt(prev_energy*next_energy) if prev_energy and next_energy else None,
        no_perceptual_flicker_interpretation=True)
