"""Single FFmpeg RGB24 MP4 readback for the blind group receiver."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from main.tube_state.rgb_dct_group_consistency import score_rgb
from runtime.wan.io import read_mp4


def score_mp4(path: str | Path, key: bytes) -> dict:
    initial = score_rgb(None, key)
    if initial["reason"] == "INVALID_KEY":
        return initial
    try:
        frames = read_mp4(Path(path))
        frames = (frames.detach().cpu().numpy() if hasattr(frames, "detach")
                  else np.asarray(frames))
    except Exception:
        initial["reason"] = "MP4_READ_FAILED"
        return initial
    return score_rgb(frames, key)
