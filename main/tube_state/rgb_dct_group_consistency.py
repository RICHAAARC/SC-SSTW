"""Blind, fixed RGB-DCT temporal group consistency receiver."""
from __future__ import annotations

import hashlib
import math

import numpy as np

from . import rgb_dct_presence as baseline


SPEC_ID = "Video-WM/RGB-DCT-Group-Consistency-H0-H1-V1"
SPEC_TEXT = "\n".join((
    SPEC_ID,
    f"Baseline RGB-DCT feature/key/continuous-score specification SHA-256: {baseline.SPEC_SHA256}.",
    "Input: one complete finite float32/float64 RGB [181,320,512,3] in [0,1].",
    "Use baseline luminance, 32x32 DCT(2,1)-DCT(1,2), 160 row-major blocks,",
    "baseline spatial signs s_b and centered unit-RMS temporal code c_t.",
    "D=sqrt(mean_(t,b)(d[t,b]^2)+(1/255)^2); m_b=mean_t d[t,b].",
    "Groups 0..28 contain six frames each; group 29 contains seven frames.",
    "q_g=mean_(t in g,b)(c_t*s_b*(d[t,b]-m_b))/D.",
    "C=sum_(g=0..29) 1(q_g>0); exact zero is nonpositive.",
    "For valid complete media, C>=24 is H1 and C<24 is H0.",
    "sum_g(n_g/181)*q_g must equal baseline S within absolute 1e-10.",
    "Invalid read, shape, finite/range, computation or consistency is INVALID, never H0.",
    "S and q_g are audit values only; no calibration, scan, fallback or writer input.",
))
SPEC_SHA256 = hashlib.sha256(SPEC_TEXT.encode("ascii")).hexdigest()
THRESHOLD = 24
_LENGTHS = (6,) * 29 + (7,)


def _base(key: bytes | object) -> dict:
    key_record = baseline.score_rgb(None, key)
    return dict(status="INVALID", reason=key_record["reason"],
                spec_id=SPEC_ID, spec_sha256=SPEC_SHA256,
                baseline_spec_sha256=baseline.SPEC_SHA256,
                key_id=key_record["key_id"], frames_used=0,
                blocks_per_frame=baseline.BLOCKS, score=None,
                group_scores=None, positive_groups=None, decision=None)


def decide(positive_groups: int) -> str:
    if isinstance(positive_groups, bool) or not isinstance(positive_groups, int) or not 0 <= positive_groups <= 30:
        raise ValueError("integer C in [0,30] required")
    return "H1" if positive_groups >= THRESHOLD else "H0"


def _score_features(features: np.ndarray, spatial: np.ndarray,
                    temporal: np.ndarray) -> dict:
    """Pure fixed reduction after the baseline feature extraction."""
    denominator = math.sqrt(float(np.mean(features * features, dtype=np.float64))
                            + (1 / 255) ** 2)
    weighted = temporal[:, None] * spatial[None, :] * features
    score = float(np.mean(weighted, dtype=np.float64)) / denominator
    centered = features - np.mean(features, axis=0, dtype=np.float64)
    group_scores = []
    start = 0
    for length in _LENGTHS:
        stop = start + length
        q = float(np.mean(temporal[start:stop, None] * spatial[None, :]
                          * centered[start:stop], dtype=np.float64)) / denominator
        group_scores.append(q)
        start = stop
    reconstructed = math.fsum(length * q / baseline.FRAMES
                               for length, q in zip(_LENGTHS, group_scores, strict=True))
    if not all(math.isfinite(x) for x in (denominator, score, reconstructed, *group_scores)):
        raise FloatingPointError("nonfinite group reduction")
    if abs(reconstructed - score) > 1e-10:
        raise ArithmeticError("group score inconsistent with baseline S")
    positive_groups = sum(q > 0 for q in group_scores)
    return dict(score=score, group_scores=group_scores,
                positive_groups=positive_groups,
                score_reconstructed=reconstructed)


def score_rgb(rgb: np.ndarray, key: bytes) -> dict:
    """Score only RGB and a fixed key; return one blind C decision or INVALID."""
    result = _base(key)
    if result["key_id"] is None:
        return result
    if not isinstance(rgb, np.ndarray):
        result["reason"] = "CPU_NUMPY_ARRAY_REQUIRED"
        return result
    if rgb.shape != (baseline.FRAMES, baseline.HEIGHT, baseline.WIDTH, baseline.CHANNELS):
        result["reason"] = "FULL_RGB_SHAPE_REQUIRED"
        return result
    if rgb.dtype not in (np.dtype("float32"), np.dtype("float64")):
        result["reason"] = "FLOAT32_OR_FLOAT64_REQUIRED"
        return result
    try:
        spatial, _, temporal = baseline.key_codes(key)
        features = np.empty((baseline.FRAMES, baseline.BLOCKS), dtype=np.float64)
        for t in range(baseline.FRAMES):
            frame = np.asarray(rgb[t], dtype=np.float64)
            if not np.isfinite(frame).all():
                result["reason"] = "NONFINITE_RGB"
                return result
            if np.any(frame < 0) or np.any(frame > 1):
                result["reason"] = "RGB_OUT_OF_RANGE"
                return result
            luminance = np.einsum("hwc,c->hw", frame, baseline._Y_WEIGHTS)
            blocks = luminance.reshape(baseline.BLOCK_ROWS, baseline.BLOCK_SIZE,
                                       baseline.BLOCK_COLS, baseline.BLOCK_SIZE)
            coefficients = np.einsum("abij,ij->ab", blocks.transpose(0, 2, 1, 3),
                                     baseline._KERNEL)
            if not np.isfinite(coefficients).all():
                result["reason"] = "NONFINITE_INTERMEDIATE"
                return result
            features[t] = coefficients.reshape(baseline.BLOCKS)
            result["frames_used"] = t + 1
        reduction = _score_features(features, spatial, temporal)
    except FloatingPointError:
        result["reason"] = "NONFINITE_INTERMEDIATE"
        return result
    except ArithmeticError:
        result["reason"] = "GROUP_SCORE_INCONSISTENT"
        return result
    except (OverflowError, ValueError):
        result["reason"] = "NONFINITE_INTERMEDIATE"
        return result
    result.update(status="SCORED", reason=None, **reduction)
    return result
