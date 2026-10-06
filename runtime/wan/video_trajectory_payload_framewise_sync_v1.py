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


class M05Backend:
    """Fixed .5 writer and fresh framewise receiver, independent of experiments."""
    def __init__(self,config):
        import torch
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.vae=framewise.load_frozen_framewise_vae(device=self.device)
        self.batch_frames=config["batch_frames"]
    @staticmethod
    def read_source(spec):
        from runtime.wan.rgb8_source import read_rgb8_source
        return read_rgb8_source(spec["path"],expected_sha256=spec["sha256"],shape=tuple(spec["shape"]))
    def encode(self,rgb):return encode_received_video(rgb,self.vae,batch_frames=self.batch_frames)[0]
    @staticmethod
    def write(latent,key):return method.apply_projection_margin(latent,key,method.PUBLIC,target_margin=0.5)
    def decode(self,latent):
        import torch
        from runtime.wan.vae import quantize_rgb8_no_codec
        return quantize_rgb8_no_codec(framewise.decode_rgb_frames(self.vae,torch.from_numpy(latent),batch_frames=self.batch_frames))
    @staticmethod
    def transport(rgb,output,count,event):
        from pathlib import Path
        from runtime.wan import fixed_rgb_media as media
        output=Path(output);r=media.save_raster(rgb,output/"m05.rgb8");event("raster",r)
        return media.mp4_roundtrip(r["path"],r["sha256"],output/"m05.mp4",output/"received.rgb8",count=count,event=event)
    @staticmethod
    def score(latent,key):return method.score_received_latent(latent,key,method.PUBLIC)
    def close(self):
        import gc,torch
        self.vae=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
