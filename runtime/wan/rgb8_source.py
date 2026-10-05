"""Strict raw RGB8 source and post-MP4 crop helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_rgb8_source(
    path: str | Path,
    *,
    expected_sha256: str,
    shape: tuple[int, int, int, int],
) -> Any:
    """Read one fixed RGB8 source without download, padding, or truncation."""

    import numpy as np
    import torch

    if len(shape) != 4 or shape[-1] != 3 or any(int(value) < 1 for value in shape):
        raise ValueError("RGB8 shape must be positive [T,H,W,3]")
    source_path = Path(path)
    expected_bytes = int(np.prod(shape))
    if source_path.stat().st_size != expected_bytes:
        raise ValueError("RGB8 source byte count mismatch")
    actual_sha256 = file_sha256(source_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("RGB8 source SHA-256 mismatch")
    raw = source_path.read_bytes()
    value = np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
    return torch.from_numpy(value)


def crop_received_rgb(
    received_rgb8: Any,
    *,
    source_start: int,
    source_stop: int,
) -> tuple[Any, dict[str, Any]]:
    """Slice a public view only after full MP4 RGB24 readback."""

    import hashlib
    import torch

    if received_rgb8.ndim != 4 or received_rgb8.shape[-1] != 3:
        raise ValueError("received RGB8 must be [T,H,W,3]")
    if received_rgb8.dtype != torch.uint8:
        raise TypeError("received crop source must retain exact RGB8 bytes")
    frames = int(received_rgb8.shape[0])
    if not (0 < source_start < source_stop <= frames):
        raise ValueError("crop must have a nonzero valid source start")
    crop = received_rgb8[source_start:source_stop].contiguous().clone()
    raw = crop.cpu().numpy().tobytes()
    return crop, {
        "status": "SAVED",
        "source_start": source_start,
        "source_stop": source_stop,
        "frames": source_stop - source_start,
        "shape": list(crop.shape),
        "dtype": "uint8",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "derived_after_full_mp4_readback": True,
        "second_codec_save": False,
    }