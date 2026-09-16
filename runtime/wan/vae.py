"""Shared frozen Wan VAE and RGB8 adapters, preserving existing semantics."""
from __future__ import annotations
from typing import Any

def _clear_cache(vae: Any) -> None:
    clear = getattr(vae, "clear_cache", None) or getattr(vae, "_clear_cache", None)
    if callable(clear):
        clear()


def _scale_tensors(vae: Any, normalized_latent: Any) -> tuple[Any, Any]:
    import torch

    config = vae.config
    channels = int(normalized_latent.shape[1])
    mean = normalized_latent.new_tensor(config.latents_mean, dtype=torch.float32).reshape(1, -1, 1, 1, 1)
    std = normalized_latent.new_tensor(config.latents_std, dtype=torch.float32).reshape(1, -1, 1, 1, 1)
    if mean.shape[1] != channels or std.shape[1] != channels or not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
        raise ValueError("VAE normalized-latent scaling does not match supplied terminal latent")
    return mean, std


def decode_normalized_latent(vae: Any, normalized_latent: Any) -> Any:
    """Decode `z_norm * latents_std + latents_mean` to RGB [T,H,W,3] in [0,1]."""

    import torch

    if normalized_latent.ndim != 5 or normalized_latent.shape[0] != 1:
        raise ValueError("C2A decode requires one [1,C,T,H,W] terminal latent")
    mean, std = _scale_tensors(vae, normalized_latent)
    _clear_cache(vae)
    try:
        with torch.inference_mode():
            decoded = vae.decode((normalized_latent.to(torch.float32) * std + mean), return_dict=False)[0]
    finally:
        _clear_cache(vae)
    if decoded.ndim != 5 or decoded.shape[:2] != (1, 3):
        raise ValueError("frozen VAE did not return one [1,3,T,H,W] RGB video")
    return (decoded[0].permute(1, 2, 3, 0) / 2.0 + 0.5).clamp(0.0, 1.0)


def quantize_rgb8_no_codec(rgb: Any) -> Any:
    """Reuse the fixed RGB8 rounding before FFmpeg, without invoking FFmpeg.

    The returned CPU uint8 tensor is exactly the raster supplied to the C2A
    MP4 encoder before any YUV conversion or codec operation.
    """

    import numpy as np
    import torch

    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("RGB8 quantization input must be [T,H,W,3]")
    pixels = np.rint(rgb.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    return torch.from_numpy(pixels.copy())


def reencode_rgb24_readback(vae: Any, rgb: Any) -> Any:
    """Use posterior mode and restore the normalized coordinate system.

    The frame tensor follows the old Wan endpoint adapter: RGB [0,1] becomes
    [1,3,T,H,W] in [-1,1].  This first 2A adapter deliberately rejects an
    incompatible VAE call rather than silently swapping encoder semantics.
    """

    import torch

    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("C2A VAE encode requires RGB24 readback [T,H,W,3]")
    video = (rgb.permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0).to(
        device=next(vae.parameters()).device,
        dtype=next(vae.parameters()).dtype,
    )
    _clear_cache(vae)
    try:
        with torch.inference_mode():
            encoded = vae.encode(video)
            distribution = getattr(encoded, "latent_dist", None)
            if distribution is None or not callable(getattr(distribution, "mode", None)):
                raise ValueError("frozen VAE encode lacks deterministic latent_dist.mode()")
            raw = distribution.mode()
    finally:
        _clear_cache(vae)
    if raw.ndim != 5 or raw.shape[0] != 1:
        raise ValueError("frozen VAE encode did not return one [1,C,T,H,W] latent")
    mean, std = _scale_tensors(vae, raw)
    normalized = (raw.to(dtype=torch.float32) - mean) / std
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("NONFINITE_REENCODED_NORMALIZED_LATENT")
    return normalized
