"""Differentiable continuous proxy for the fixed RGB-DCT group receiver."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import rgb_dct_group_consistency as receiver
from . import rgb_dct_presence as baseline


GROUP_LENGTHS = (6,) * 29 + (7,)
LOSS_ID = "Video-WM/RGB-DCT-Terminal-Gradient-V1/L=mean(relu(-q_g)^2)"


def numpy_proxy(rgb: np.ndarray, key: bytes) -> dict:
    """Return the continuous q vector and loss using the frozen CPU receiver."""
    scored = receiver.score_rgb(rgb, key)
    if scored["status"] != "SCORED":
        raise ValueError(f"frozen receiver rejected continuous RGB: {scored['reason']}")
    q = np.asarray(scored["group_scores"], dtype=np.float64)
    loss = float(np.mean(np.maximum(-q, 0.0) ** 2, dtype=np.float64))
    if not math.isfinite(loss):
        raise FloatingPointError("nonfinite NumPy RGB-DCT proxy loss")
    return dict(q=q, loss=loss, score=float(scored["score"]))


def torch_proxy(rgb: Any, key: bytes) -> dict:
    """Compute all 30 q values and the fixed loss without detaching RGB.

    The tensor path follows the receiver's gamma RGB luminance, 160 nonoverlapping
    32x32 blocks, DCT difference, time centering, key signs and denominator.  It
    intentionally returns continuous values only; MP4 decisions remain on the
    NumPy/FFmpeg receiver.
    """
    import torch

    if not torch.is_tensor(rgb) or tuple(rgb.shape) != (
        baseline.FRAMES, baseline.HEIGHT, baseline.WIDTH, baseline.CHANNELS
    ):
        raise ValueError("fixed [181,320,512,3] RGB tensor required")
    if not rgb.is_floating_point() or not bool(torch.isfinite(rgb).all()):
        raise ValueError("finite floating RGB tensor required")
    if bool((rgb < 0).any()) or bool((rgb > 1).any()):
        raise ValueError("RGB tensor outside [0,1]")

    dtype, device = rgb.dtype, rgb.device
    weights = torch.as_tensor(baseline._Y_WEIGHTS, dtype=dtype, device=device)
    kernel = torch.as_tensor(baseline._KERNEL, dtype=dtype, device=device)
    spatial_np, _, temporal_np = baseline.key_codes(key)
    spatial = torch.as_tensor(spatial_np, dtype=dtype, device=device)
    temporal = torch.as_tensor(temporal_np, dtype=dtype, device=device)

    luminance = torch.einsum("thwc,c->thw", rgb, weights)
    blocks = luminance.reshape(
        baseline.FRAMES, baseline.BLOCK_ROWS, baseline.BLOCK_SIZE,
        baseline.BLOCK_COLS, baseline.BLOCK_SIZE,
    ).permute(0, 1, 3, 2, 4)
    features = torch.einsum("tabij,ij->tab", blocks, kernel).reshape(
        baseline.FRAMES, baseline.BLOCKS
    )
    denominator = torch.sqrt(features.square().mean() + rgb.new_tensor((1 / 255) ** 2))
    centered = features - features.mean(dim=0)
    values = []
    start = 0
    for length in GROUP_LENGTHS:
        stop = start + length
        values.append(
            (temporal[start:stop, None] * spatial[None, :]
             * centered[start:stop]).mean() / denominator
        )
        start = stop
    q = torch.stack(values)
    loss = torch.relu(-q).square().mean()
    score = (temporal[:, None] * spatial[None, :] * features).mean() / denominator
    reconstructed = sum(
        length * value / baseline.FRAMES
        for length, value in zip(GROUP_LENGTHS, q.unbind(), strict=True)
    )
    if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite differentiable RGB-DCT proxy")
    return dict(q=q, loss=loss, score=score, reconstructed=reconstructed)


def validate_against_numpy(rgb: Any, key: bytes, torch_result: dict,
                           *, rtol: float, atol: float) -> dict:
    """Validate one exact input against the frozen NumPy continuous receiver."""
    import torch

    if not (math.isfinite(rtol) and math.isfinite(atol) and rtol >= 0 and atol >= 0):
        raise ValueError("finite nonnegative proxy tolerances required")
    # Explicit float32 conversion is required for BF16-capable environments.
    rgb_np = rgb.detach().to(dtype=torch.float32, device="cpu").numpy()
    expected = numpy_proxy(rgb_np, key)
    actual_q = torch_result["q"].detach().to(dtype=torch.float64, device="cpu")
    expected_q = torch.from_numpy(expected["q"])
    difference = (actual_q - expected_q).abs()
    match = bool(torch.allclose(actual_q, expected_q, rtol=rtol, atol=atol))
    receipt = dict(
        matched=match, rtol=rtol, atol=atol,
        max_abs_q_error=float(difference.max()),
        torch_loss=float(torch_result["loss"].detach().double()),
        numpy_loss=expected["loss"],
        torch_score=float(torch_result["score"].detach().double()),
        numpy_score=expected["score"],
        q_torch=[float(value) for value in actual_q],
        q_numpy=[float(value) for value in expected_q],
    )
    if not match:
        raise RuntimeError(
            f"differentiable/NumPy RGB-DCT proxy mismatch: max_abs={receipt['max_abs_q_error']}"
        )
    return receipt
