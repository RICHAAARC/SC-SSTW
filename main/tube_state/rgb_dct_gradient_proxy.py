"""Differentiable form of the frozen 30-group RGB-DCT receiver statistic."""
from __future__ import annotations

from typing import Any

from . import rgb_dct_group_consistency as receiver
from . import rgb_dct_presence as baseline


def _feature_tensor(rgb: Any) -> Any:
    """Return the exact 181 by 160 DCT feature geometry in float64."""
    import torch

    if not torch.is_tensor(rgb):
        raise TypeError("torch RGB tensor required")
    if tuple(rgb.shape) != (baseline.FRAMES, baseline.HEIGHT,
                            baseline.WIDTH, baseline.CHANNELS):
        raise ValueError("full [181,320,512,3] RGB required")
    if rgb.dtype not in (torch.float32, torch.float64):
        raise ValueError("float32 or float64 RGB required")
    if not bool(torch.isfinite(rgb).all()) or bool((rgb < 0).any()) or bool((rgb > 1).any()):
        raise ValueError("finite RGB in [0,1] required")
    weights = torch.as_tensor(baseline._Y_WEIGHTS, device=rgb.device,
                              dtype=torch.float64)
    kernel = torch.as_tensor(baseline._KERNEL, device=rgb.device,
                             dtype=torch.float64)
    rows = []
    # Frame-wise extraction keeps the full VAE graph but avoids a second full
    # float64 video allocation. Reduction operands and means remain float64.
    for frame in rgb.unbind(0):
        luminance = torch.einsum("hwc,c->hw", frame.to(torch.float64), weights)
        blocks = luminance.reshape(baseline.BLOCK_ROWS, baseline.BLOCK_SIZE,
                                   baseline.BLOCK_COLS, baseline.BLOCK_SIZE)
        blocks = blocks.permute(0, 2, 1, 3)
        rows.append(torch.einsum("abij,ij->ab", blocks, kernel).reshape(-1))
    features = torch.stack(rows)
    if tuple(features.shape) != (baseline.FRAMES, baseline.BLOCKS):
        raise RuntimeError("RGB-DCT feature geometry changed")
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError("nonfinite RGB-DCT features")
    return features


def group_scores_from_features(features: Any, key: bytes) -> tuple[Any, dict]:
    """Apply the frozen float64 receiver reduction to 181 by 160 features."""
    import torch

    spatial_np, _, temporal_np = baseline.key_codes(key)
    if (not torch.is_tensor(features)
            or tuple(features.shape) != (baseline.FRAMES, baseline.BLOCKS)
            or features.dtype != torch.float64
            or not bool(torch.isfinite(features).all())):
        raise ValueError("finite float64 [181,160] features required")
    spatial = torch.as_tensor(spatial_np, device=features.device, dtype=torch.float64)
    temporal = torch.as_tensor(temporal_np, device=features.device, dtype=torch.float64)
    denominator = torch.sqrt(features.square().mean(dtype=torch.float64)
                             + (1.0 / 255.0) ** 2)
    centered = features - features.mean(dim=0, dtype=torch.float64)
    values = []
    start = 0
    for length in receiver._LENGTHS:
        stop = start + length
        weighted = temporal[start:stop, None] * spatial[None, :] * centered[start:stop]
        values.append(weighted.mean(dtype=torch.float64) / denominator)
        start = stop
    q = torch.stack(values)
    score = (temporal[:, None] * spatial[None, :] * features).mean(
        dtype=torch.float64) / denominator
    reconstructed = sum(length * value / baseline.FRAMES
                        for length, value in zip(receiver._LENGTHS, q, strict=True))
    if tuple(q.shape) != (30,) or not bool(torch.isfinite(q).all()):
        raise FloatingPointError("nonfinite differentiable group scores")
    if not bool(torch.isfinite(score)) or not bool(torch.isfinite(reconstructed)):
        raise FloatingPointError("nonfinite differentiable receiver reduction")
    return q, dict(score=score, reconstructed_score=reconstructed,
                   denominator=denominator, features=features)


def group_scores(rgb: Any, key: bytes) -> tuple[Any, dict]:
    """Return differentiable q[30] plus the shared continuous score metadata."""
    return group_scores_from_features(_feature_tensor(rgb), key)


def loss_from_rgb(rgb: Any, key: bytes) -> tuple[Any, Any, dict]:
    """Fixed local objective: mean_g relu(-q_g)^2, minimized."""
    import torch

    q, metadata = group_scores(rgb, key)
    loss = torch.relu(-q).square().mean(dtype=torch.float64)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite RGB-DCT local objective")
    return loss, q, metadata
