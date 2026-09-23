"""Thin RGB24 MP4 readback entry for the pure RGB/DCT presence statistic."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from main.tube_state.rgb_dct_presence import score_rgb
from runtime.wan.io import read_mp4


def score_mp4(path: str | Path, key: bytes) -> dict:
    """Decode via the shared ffmpeg RGB24 reader, then score all decoded frames."""
    initial = score_rgb(None, key)
    if initial["reason"] == "INVALID_KEY":
        return initial
    try:
        frames = read_mp4(Path(path))
        if hasattr(frames, "detach"):
            frames = frames.detach().cpu().numpy()
        else:
            frames = np.asarray(frames)
    except Exception:
        initial["reason"] = "MP4_READ_FAILED"
        return initial
    return score_rgb(frames, key)
