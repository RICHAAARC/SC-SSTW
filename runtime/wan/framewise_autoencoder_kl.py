"""Pinned 2D AutoencoderKL adapter for post-Wan RGB framewise reconstruction."""
from __future__ import annotations

from typing import Any

MODEL_ID = "stabilityai/sd-vae-ft-mse"
MODEL_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
SCALED_LATENT_FACTOR = 0.18215


def load_frozen_framewise_vae(*, device: str) -> Any:
    """Load only the adopted 2D VAE at its immutable revision."""

    import torch
    from diffusers import AutoencoderKL

    dtype = torch.float32
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)
    verify_scaling_factor(vae)
    return vae


def _model_device_dtype(vae: Any) -> tuple[Any, Any]:
    parameter = next(vae.parameters())
    return parameter.device, parameter.dtype


def verify_scaling_factor(vae: Any) -> float:
    """Require the resolved config/default to match the adopted latent units."""

    value = float(getattr(vae.config, "scaling_factor", SCALED_LATENT_FACTOR))
    if value != SCALED_LATENT_FACTOR:
        raise ValueError(
            f"framewise VAE scaling_factor {value!r} does not match {SCALED_LATENT_FACTOR}"
        )
    return value


def scaling_receipt(vae: Any) -> dict[str, Any]:
    value = verify_scaling_factor(vae)
    return {
        "scaled_latent_factor": value,
        "expected_scaled_latent_factor": SCALED_LATENT_FACTOR,
        "resolved_from": (
            "model_config"
            if hasattr(vae.config, "scaling_factor")
            else "AutoencoderKL_adopted_default"
        ),
        "matches_fixed_protocol": True,
        "vae_compute_dtype": str(next(vae.parameters()).dtype),
        "method_coordinate_dtype": "torch.float32",
        "force_upcast": bool(getattr(vae.config, "force_upcast", False)),
    }


def encode_rgb_frames(
    vae: Any,
    rgb: Any,
    *,
    batch_frames: int,
) -> Any:
    """Deterministic posterior mode in explicit float32 scaled-latent units."""

    import torch

    if rgb.ndim != 4 or rgb.shape[-1] != 3 or int(rgb.shape[0]) < 1:
        raise ValueError("framewise VAE input must be [T,H,W,3]")
    if not isinstance(batch_frames, int) or batch_frames < 1:
        raise ValueError("batch_frames must be positive")
    if not bool(torch.isfinite(rgb).all()):
        raise ValueError("framewise VAE RGB contains nonfinite values")
    device, dtype = _model_device_dtype(vae)
    normalized = (rgb.permute(0, 3, 1, 2).to(device=device, dtype=dtype) * 2.0) - 1.0
    batches = []
    with torch.inference_mode():
        for start in range(0, int(normalized.shape[0]), batch_frames):
            encoded = vae.encode(normalized[start : start + batch_frames])
            distribution = getattr(encoded, "latent_dist", None)
            mode = getattr(distribution, "mode", None)
            if distribution is None or not callable(mode):
                raise ValueError("framewise VAE encode lacks deterministic latent_dist.mode()")
            batch = mode().float() * SCALED_LATENT_FACTOR
            if batch.ndim != 4:
                raise ValueError("framewise VAE must return [T,C,H,W] latent batches")
            batches.append(batch.detach().float().cpu())
    output = torch.cat(batches, dim=0).contiguous()
    if not bool(torch.isfinite(output).all()):
        raise ValueError("framewise VAE returned nonfinite latent")
    return output


def decode_rgb_frames(
    vae: Any,
    latent: Any,
    *,
    batch_frames: int,
) -> Any:
    """Decode explicit scaled latent to CPU float RGB [T,H,W,3] in [0,1]."""

    import torch

    if latent.ndim != 4 or int(latent.shape[0]) < 1:
        raise ValueError("framewise latent must be [T,C,H,W]")
    if latent.dtype != torch.float32:
        raise TypeError("framewise method latent must be float32")
    if not isinstance(batch_frames, int) or batch_frames < 1:
        raise ValueError("batch_frames must be positive")
    if not bool(torch.isfinite(latent).all()):
        raise ValueError("framewise latent contains nonfinite values")
    device, dtype = _model_device_dtype(vae)
    batches = []
    scale = verify_scaling_factor(vae)
    with torch.inference_mode():
        for start in range(0, int(latent.shape[0]), batch_frames):
            decoded = vae.decode(
                latent[start : start + batch_frames].to(device=device, dtype=dtype) / scale
            )
            sample = getattr(decoded, "sample", None)
            if sample is None and isinstance(decoded, tuple) and decoded:
                sample = decoded[0]
            if sample is None or sample.ndim != 4 or int(sample.shape[1]) != 3:
                raise ValueError("framewise VAE decode must return [T,3,H,W]")
            if not bool(torch.isfinite(sample).all()):
                raise ValueError("framewise VAE decode returned nonfinite sample before RGB clamp")
            batches.append(sample.detach().float().cpu())
    video = torch.cat(batches, dim=0)
    if not bool(torch.isfinite(video).all()):
        raise ValueError("framewise VAE decode returned nonfinite sample before RGB clamp")
    video = (video.permute(0, 2, 3, 1) / 2.0 + 0.5).clamp(0.0, 1.0).contiguous()
    return video


def reconstruction_receipt(
    *,
    encoded_frames: int,
    decoded_frames: int,
    batch_frames: int,
    scaling: dict[str, Any],
) -> dict[str, Any]:
    if encoded_frames != decoded_frames or encoded_frames < 1:
        raise ValueError("framewise VAE reconstruction must preserve frame count")
    return {
        "backend": "diffusers_autoencoder_kl_framewise",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_role": "2D reconstruction VAE only; not Stable Diffusion generation",
        "encode_mode": "deterministic_latent_dist_mode_times_0.18215",
        "decode_mode": "framewise_scaled_latent_divide_0.18215",
        "method_coordinate_dtype": "float32",
        "encoded_frames": encoded_frames,
        "decoded_frames": decoded_frames,
        "batch_frames": batch_frames,
        "scaling": scaling,
        "replaces_wan_native_decoder": False,
    }
