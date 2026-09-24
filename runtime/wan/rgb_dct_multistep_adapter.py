"""One FFmpeg RGB24 readback supplies blind C and paired-media diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from main.tube_state.rgb_dct_group_consistency import score_rgb
from runtime.wan.io import read_mp4


def score_mp4_once(path: str | Path, key: bytes) -> tuple[dict, np.ndarray | None]:
    """Return (blind receiver record, full decoded RGB); call read_mp4 once."""
    initial = score_rgb(None, key)
    if initial["reason"] == "INVALID_KEY":
        return initial, None
    try:
        frames = read_mp4(Path(path))
        frames = (frames.detach().cpu().numpy() if hasattr(frames, "detach")
                  else np.asarray(frames))
    except Exception:
        initial["reason"] = "MP4_READ_FAILED"
        return initial, None
    result = score_rgb(frames, key)
    return result, frames if result["status"] == "SCORED" else None
