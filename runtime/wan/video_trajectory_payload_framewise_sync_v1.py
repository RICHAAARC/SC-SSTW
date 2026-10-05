"""Runtime bridge for the adopted post-Wan framewise sync candidate."""
from __future__ import annotations

from typing import Any

import numpy as np

from main.tube_state import video_trajectory_payload_framewise_sync_v1 as method
from runtime.wan import framewise_autoencoder_kl as framewise


def _rgb01(rgb8: Any) -> Any:
    import torch

    if rgb8.ndim != 4 or int(rgb8.shape[-1]) != 3 or rgb8.dtype != torch.uint8:
        raise ValueError("received video must be uint8 [T,H,W,3]")
    return rgb8.detach().cpu().float().div(255.0)


def encode_received_video(
    received_rgb8: Any,
    vae: Any,
    *,
    batch_frames: int,
    public: method.PublicProtocol = method.PUBLIC,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Receiver-side encode from only the received video and public VAE protocol."""

    latent = framewise.encode_rgb_frames(vae, _rgb01(received_rgb8), batch_frames=batch_frames)
    value = latent.numpy()
    method.validate_received_latent(value, public)
    return value, {
        "status": "COMPLETE",
        "received_frames": int(value.shape[0]),
        "latent_shape": list(value.shape),
        "dtype": str(value.dtype),
        "scaling": framewise.scaling_receipt(vae),
        "truth_inputs": False,
    }


def blind_receive(
    received_rgb8: Any,
    key: str,
    vae: Any,
    *,
    batch_frames: int,
    public: method.PublicProtocol = method.PUBLIC,
) -> dict[str, Any]:
    """Pure blind boundary: received video, key, and fixed public protocol only."""

    latent, encode_receipt = encode_received_video(
        received_rgb8, vae, batch_frames=batch_frames, public=public
    )
    return {
        "encode_receipt": encode_receipt,
        "sync_readout": method.score_received_latent(latent, key, public),
        "truth_inputs": False,
    }


def build_p1_p2(
    source_rgb8: Any,
    key: str,
    vae: Any,
    *,
    batch_frames: int,
    public: method.PublicProtocol = method.PUBLIC,
) -> tuple[Any, Any, dict[str, Any]]:
    """One writer encode, copied latents, then separate P1 and P2 decode calls."""

    import torch

    if int(source_rgb8.shape[0]) != public.source_frames:
        raise ValueError("writer source must be the full public source")
    encoded = framewise.encode_rgb_frames(vae, _rgb01(source_rgb8), batch_frames=batch_frames)
    source_latent = encoded.numpy()
    method.validate_source_latent(source_latent, public)
    p1_latent = encoded.clone()
    p2_numpy, writer_receipt = method.apply_projection_margin(source_latent.copy(), key, public)
    p2_latent = torch.from_numpy(p2_numpy.copy())
    p1_rgb = framewise.decode_rgb_frames(vae, p1_latent, batch_frames=batch_frames)
    p2_rgb = framewise.decode_rgb_frames(vae, p2_latent, batch_frames=batch_frames)
    reconstruction = framewise.reconstruction_receipt(
        encoded_frames=int(encoded.shape[0]),
        decoded_frames=int(p1_rgb.shape[0]),
        batch_frames=batch_frames,
        scaling=framewise.scaling_receipt(vae),
    )
    return p1_rgb, p2_rgb, {
        "status": "COMPLETE",
        "writer_encode_calls": 1,
        "writer_decode_calls": 2,
        "p1_uses_unmodified_copy": bool(torch.equal(p1_latent, encoded)),
        "p2_uses_same_encoding": True,
        "p0_uses_framewise_decoder": False,
        "writer_receipt": writer_receipt,
        "reconstruction": reconstruction,
    }
