"""Strict original-MP4 read and blind RGB-DCT score adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from main.tube_state.rgb_dct_group_consistency import score_rgb
from runtime.wan.io import read_mp4


def read_original_mp4(path: str | Path) -> np.ndarray:
    """Read once and let the original exception escape for raw persistence."""
    frames: Any = read_mp4(Path(path))
    frames = (frames.detach().float().cpu().numpy()
              if hasattr(frames, "detach") else np.asarray(frames))
    return frames


def score_original_rgb(rgb: np.ndarray, key: bytes) -> dict:
    """Blind receiver boundary: only decoded RGB and the frozen key enter."""
    return score_rgb(rgb, key)
