"""Uncalibrated, blind RGB/DCT temporal-balanced presence statistic on CPU."""
from __future__ import annotations

import hashlib
import math

import numpy as np


SPEC_ID = "Video-WM/RGB-DCT-Temporal-Balanced-V1"
SPEC_TEXT = "\n".join((
    "Video-WM/RGB-DCT-Temporal-Balanced-V1",
    "Input: full gamma-encoded RGB float32 or float64, [181,320,512,3], finite in [0,1].",
    "No transfer-function linearization, resize, crop, or frame selection.",
    "Y'=0.2126R+0.7152G+0.0722B computed in float64.",
    "Blocks: 32x32 nonoverlapping, 10 rows x 16 columns, row-major index b=16*row+column.",
    "DCT-II: D(u,v)=sum_ij Y[i,j]*alpha(u)*cos(pi*(2i+1)*u/64)*"
    "alpha(v)*cos(pi*(2j+1)*v/64), alpha(0)=1/sqrt(32),"
    "alpha(k>0)=sqrt(2/32), zero-index u vertical/v horizontal.",
    "d[t,b]=D_t,b(2,1)-D_t,b(1,2), computed in float64.",
    "Key: nonempty bytes; for each domain SHAKE256(domain||uint64_be(len(key))||key).",
    "Space domain ASCII Video-WM/RGB-DCT-Temporal-Balanced-V1/space followed by byte 00;"
    " output 160*8 bytes.",
    "Time domain ASCII Video-WM/RGB-DCT-Temporal-Balanced-V1/time followed by byte 00;"
    " output 30*8 bytes.",
    "Parse each 8-byte chunk as big-endian unsigned integer; sort (hash,index) ascending.",
    "Spatial first 80 signs +1, last 80 -1; temporal first 15 group signs +1, last 15 -1.",
    "Groups 0..28 each cover frames 6g..6g+5; group 29 covers frames 174..180.",
    "r_t=group_sign[frame_group(t)]; mu=mean_181(r);"
    " c_t=(r_t-mu)/sqrt(mean_181((r-mu)^2)).",
    "S=mean_{181*160}(c_t*s_b*d[t,b]) /"
    " sqrt(mean_{181*160}(d[t,b]^2)+(1/255)^2).",
    "The denominator is independent of key, arm, writer, and truth; all-zero d scores zero.",
    "Any invalid input or nonfinite intermediate is INVALID. Score is continuous and uncalibrated;"
    " there is no decision or threshold.",
))
SPEC_SHA256 = hashlib.sha256(SPEC_TEXT.encode("ascii")).hexdigest()
FRAMES, HEIGHT, WIDTH, CHANNELS = 181, 320, 512, 3
BLOCK_SIZE, BLOCK_ROWS, BLOCK_COLS = 32, 10, 16
BLOCKS = BLOCK_ROWS * BLOCK_COLS
_Y_WEIGHTS = np.array((0.2126, 0.7152, 0.0722), dtype=np.float64)
_SPACE_DOMAIN = b"Video-WM/RGB-DCT-Temporal-Balanced-V1/space\x00"
_TIME_DOMAIN = b"Video-WM/RGB-DCT-Temporal-Balanced-V1/time\x00"
_GROUPS = np.repeat(np.arange(30, dtype=np.intp), [6] * 29 + [7])


def dct_difference_kernel() -> np.ndarray:
    """The 32x32 pixel-space kernel for DCT(2,1)-DCT(1,2)."""
    x = np.arange(BLOCK_SIZE, dtype=np.float64)
    alpha = math.sqrt(2.0 / BLOCK_SIZE)
    basis1 = alpha * np.cos(math.pi * (2 * x + 1) / (2 * BLOCK_SIZE))
    basis2 = alpha * np.cos(math.pi * (2 * x + 1) * 2 / (2 * BLOCK_SIZE))
    return np.outer(basis2, basis1) - np.outer(basis1, basis2)


_KERNEL = dct_difference_kernel()


def _shake_bytes(domain: bytes, key: bytes, count: int) -> bytes:
    return hashlib.shake_256(domain + len(key).to_bytes(8, "big") + key).digest(8 * count)


def _balanced_signs(domain: bytes, key: bytes, count: int) -> np.ndarray:
    hashes = np.frombuffer(_shake_bytes(domain, key, count), dtype=">u8")
    order = np.lexsort((np.arange(count, dtype=np.intp), hashes))
    signs = np.full(count, -1, dtype=np.int8)
    signs[order[:count // 2]] = 1
    return signs


def key_codes(key: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return spatial signs, group signs, and 181 centered unit-RMS weights."""
    if not isinstance(key, bytes) or not key or len(key) >= 1 << 64:
        raise ValueError("nonempty bytes key required")
    spatial = _balanced_signs(_SPACE_DOMAIN, key, BLOCKS)
    group = _balanced_signs(_TIME_DOMAIN, key, 30)
    raw = group[_GROUPS].astype(np.float64)
    centered = raw - np.mean(raw, dtype=np.float64)
    rms = math.sqrt(float(np.mean(centered * centered, dtype=np.float64)))
    temporal = centered / rms
    if not np.isfinite(temporal).all() or not math.isfinite(rms) or rms <= 0:
        raise ArithmeticError("nonfinite temporal code")
    return spatial, group, temporal


def _base_record(key: bytes | object) -> dict:
    valid_key = isinstance(key, bytes) and bool(key) and len(key) < 1 << 64
    return dict(
        status="INVALID", reason=None, score=None, decision=None,
        calibration_status="UNCALIBRATED", spec_id=SPEC_ID,
        spec_sha256=SPEC_SHA256,
        key_id=(hashlib.sha256(b"Video-WM/RGB-DCT-Temporal-Balanced-V1/key\x00" + key)
                .hexdigest()[:16] if valid_key else None),
        spatial_code_sha256=None, temporal_code_sha256=None,
        frames_used=0, blocks_per_frame=BLOCKS, feature_count=0,
    )


def score_rgb(rgb: np.ndarray, key: bytes) -> dict:
    """Score exactly 181 complete RGB frames; return a continuous score or INVALID.

    Input storage may be float32 or float64. Luminance, DCT and reductions use
    float64. No writer state, arm, labels, or threshold enter this interface.
    """
    result = _base_record(key)
    if result["key_id"] is None:
        result["reason"] = "INVALID_KEY"
        return result
    if not isinstance(rgb, np.ndarray):
        result["reason"] = "CPU_NUMPY_ARRAY_REQUIRED"
        return result
    if rgb.shape != (FRAMES, HEIGHT, WIDTH, CHANNELS):
        result["reason"] = "FULL_RGB_SHAPE_REQUIRED"
        return result
    if rgb.dtype not in (np.dtype("float32"), np.dtype("float64")):
        result["reason"] = "FLOAT32_OR_FLOAT64_REQUIRED"
        return result

    try:
        spatial, group, temporal = key_codes(key)
        result["spatial_code_sha256"] = hashlib.sha256(spatial.tobytes()).hexdigest()
        result["temporal_code_sha256"] = hashlib.sha256(group.tobytes()).hexdigest()
        features = np.empty((FRAMES, BLOCKS), dtype=np.float64)
        for t in range(FRAMES):
            frame = np.asarray(rgb[t], dtype=np.float64)
            if not np.isfinite(frame).all():
                result["reason"] = "NONFINITE_RGB"
                return result
            if np.any(frame < 0.0) or np.any(frame > 1.0):
                result["reason"] = "RGB_OUT_OF_RANGE"
                return result
            luminance = np.einsum("hwc,c->hw", frame, _Y_WEIGHTS)
            if not np.isfinite(luminance).all():
                result["reason"] = "NONFINITE_INTERMEDIATE"
                return result
            blocks = luminance.reshape(BLOCK_ROWS, BLOCK_SIZE, BLOCK_COLS, BLOCK_SIZE)
            blocks = blocks.transpose(0, 2, 1, 3)
            coefficients = np.einsum("abij,ij->ab", blocks, _KERNEL)
            if not np.isfinite(coefficients).all():
                result["reason"] = "NONFINITE_INTERMEDIATE"
                return result
            features[t] = coefficients.reshape(BLOCKS)
            result["frames_used"] = t + 1
            result["feature_count"] = (t + 1) * BLOCKS
        weighted = temporal[:, None] * spatial[None, :] * features
        numerator = float(np.mean(weighted, dtype=np.float64))
        energy = float(np.mean(features * features, dtype=np.float64))
        denominator = math.sqrt(energy + (1.0 / 255.0) ** 2)
        value = numerator / denominator
        if not all(map(math.isfinite, (numerator, energy, denominator, value))):
            result["reason"] = "NONFINITE_INTERMEDIATE"
            return result
    except (ArithmeticError, OverflowError, ValueError, FloatingPointError):
        result["reason"] = "NONFINITE_INTERMEDIATE"
        return result
    result.update(status="SCORED", reason=None, score=value)
    return result
