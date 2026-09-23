"""Pure fixed RGB carrier and finite normalized-latent lift geometry."""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .rgb_dct_presence import dct_difference_kernel, key_codes


AMPLITUDE = 0.1
RGB_SHAPE = (181, 320, 512, 3)
LATENT_SHAPE = (1, 16, 46, 40, 64)


def carrier_patch(key: bytes, frame: int, block: int) -> np.ndarray:
    """One 32x32 signed patch; add the same values to each RGB channel."""
    if not 0 <= frame < 181 or not 0 <= block < 160:
        raise ValueError("fixed frame/block index required")
    spatial, _, temporal = key_codes(key)
    return AMPLITUDE * temporal[frame] * spatial[block] * dct_difference_kernel()


def _full_spatial_pattern(key: bytes) -> tuple[np.ndarray, np.ndarray]:
    spatial, _, temporal = key_codes(key)
    blocks = spatial.reshape(10, 16, 1, 1) * dct_difference_kernel()
    return blocks.transpose(0, 2, 1, 3).reshape(320, 512), temporal


def apply_carrier(rgb: np.ndarray, key: bytes, polarity: int) -> tuple[np.ndarray, dict]:
    """Return clip(X0 +/- A,0,1) in the input float dtype and clipping counts."""
    if not isinstance(rgb, np.ndarray) or rgb.shape != RGB_SHAPE or rgb.dtype not in (
        np.dtype("float32"), np.dtype("float64")
    ):
        raise ValueError("full float32/float64 RGB required")
    if polarity not in (-1, 1):
        raise ValueError("polarity must be fixed +1 or -1")
    pattern, temporal = _full_spatial_pattern(key)
    output = np.empty(RGB_SHAPE, dtype=rgb.dtype)
    clipped_values = 0
    max_abs_carrier = 0.0
    for t in range(181):
        frame = np.asarray(rgb[t], dtype=np.float64)
        if not np.isfinite(frame).all() or np.any(frame < 0) or np.any(frame > 1):
            raise ValueError("invalid X0 RGB")
        carrier = polarity * AMPLITUDE * temporal[t] * pattern
        raw = frame + carrier[:, :, None]
        if not np.isfinite(raw).all():
            raise ValueError("nonfinite carrier result")
        clipped_values += int(np.count_nonzero((raw < 0) | (raw > 1)))
        max_abs_carrier = max(max_abs_carrier, float(np.max(np.abs(carrier))))
        output[t] = np.clip(raw, 0, 1)
    return output, dict(
        amplitude=AMPLITUDE, polarity=polarity,
        clipped_rgb_values=clipped_values,
        total_rgb_values=int(np.prod(RGB_SHAPE)),
        clipped_fraction=clipped_values / int(np.prod(RGB_SHAPE)),
        max_abs_carrier=max_abs_carrier,
        output_dtype=str(output.dtype),
    )


def lift_direction(raw: np.ndarray) -> tuple[np.ndarray, dict]:
    """Mask latent time endpoints 0 and 45; finite exact zero is a method negative."""
    if not isinstance(raw, np.ndarray) or raw.shape != LATENT_SHAPE or raw.dtype not in (
        np.dtype("float32"), np.dtype("float64")
    ) or not np.isfinite(raw).all():
        raise ValueError("finite normalized latent difference with fixed shape required")
    masked = raw.copy()
    masked[:, :, 0] = 0
    masked[:, :, 45] = 0
    raw64 = raw.astype(np.float64)
    masked64 = masked.astype(np.float64)
    raw_rms = math.sqrt(float(np.mean(raw64 * raw64)))
    support_rms = math.sqrt(float(np.mean(masked64[:, :, 1:45] ** 2)))
    metrics = dict(
        status="ZERO_LIFT_DIRECTION" if support_rms == 0 else "READY",
        raw_global_rms=raw_rms,
        raw_peak_abs=float(np.max(np.abs(raw64))),
        masked_global_rms=math.sqrt(float(np.mean(masked64 * masked64))),
        masked_support_rms=support_rms,
        masked_peak_abs=float(np.max(np.abs(masked64))),
        raw_sha256=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
        masked_sha256=hashlib.sha256(np.ascontiguousarray(masked).tobytes()).hexdigest(),
        support_time=[1, 45],
    )
    return masked, metrics


def classify_unit_response(response_support_rms: float) -> str:
    """A finite exact zero is a method negative; invalid values are engineering errors."""
    if not math.isfinite(response_support_rms) or response_support_rms < 0:
        raise ValueError("nonfinite/negative native unit response")
    return "ZERO_NATIVE_RESPONSE" if response_support_rms == 0 else "READY"
