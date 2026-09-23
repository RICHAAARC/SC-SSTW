"""Uncalibrated CPU reduction of VideoSeal v1.0 *raw per-frame* detect logits.

Official Python interface: ``model.detect(video, is_video=True)["preds"]``;
the TorchScript wrapper returns the tensor directly. Both raw outputs have
shape ``[F, 1+K, H, W]``; channel zero is the detection mask/logit. See
https://github.com/facebookresearch/videoseal/blob/main/docs/torchscript.md and
https://github.com/facebookresearch/videoseal/blob/main/inference_streaming.py.
Video-level ``extract_message`` output ``[1,K]`` is already aggregated and is
not accepted here. This module imports no model, checkpoint, media or GPU code.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

FRAME_COUNT = 181
MESSAGE_BITS = 256
CHANNELS = MESSAGE_BITS + 1
KEY_DOMAIN = b"Video-WM/VideoSeal-diagnostic-v1\x00"
KEY_ID_DOMAIN = b"SC-SSTW/VideoSeal-v1.0/diagnostic-key-id/v1\x00"
SCORE_FORMULA = "mean_{t,h,w}(logit[t,1:257,h,w]) dot (2*SHAKE256_key_bits-1) / 256"


def key_bits(key: bytes) -> np.ndarray:
    """Map a nonempty key via SHAKE256(domain + key), 256 bits MSB first."""
    if type(key) is not bytes or not key:
        raise ValueError("nonempty key bytes required")
    digest = hashlib.shake_256(KEY_DOMAIN + key).digest(MESSAGE_BITS // 8)
    return np.unpackbits(np.frombuffer(digest, dtype=np.uint8), bitorder="big").copy()


def key_identifier(key: bytes) -> str:
    if type(key) is not bytes or not key:
        raise ValueError("nonempty key bytes required")
    return hashlib.sha256(KEY_ID_DOMAIN + len(key).to_bytes(8, "big") + key).hexdigest()[:16]


def reduce_raw_detect_logits(logits: np.ndarray, key: bytes) -> dict:
    """Score only [181,257,H,W] CPU logits; no truth, OFF or writer inputs.

    Score is the dimension-normalized signed dot product: first average each
    message logit uniformly over every frame and spatial point, then divide
    its dot product with key signs by 256. This is not cosine normalization,
    calibration, a detection decision, or evidence of VideoSeal compatibility.
    """
    base = {
        "status": "INVALID", "score": None, "detection_logit_mean": None,
        "key_id": None, "score_formula": SCORE_FORMULA,
        "frame_count_required": FRAME_COUNT, "message_bits": MESSAGE_BITS,
        "source_api": "VideoSeal v1.0 detect(video,is_video=True)['preds'] raw per-frame",
        "spec_sha256": None, "checkpoint_sha256": None,
        "binding_status": "UNBOUND_DEVELOPMENT_DIAGNOSTIC",
        "calibration_status": "UNCALIBRATED", "decision": None,
    }
    try:
        bits = key_bits(key)
        base["key_id"] = key_identifier(key)
    except (TypeError, ValueError, OverflowError) as exc:
        return dict(base, reason="INVALID_KEY", detail=str(exc))
    if not isinstance(logits, np.ndarray):
        return dict(base, reason="CPU_NUMPY_ARRAY_REQUIRED")
    base["input_shape"] = list(logits.shape)
    if logits.ndim != 4 or logits.shape[0] != FRAME_COUNT or logits.shape[1] != CHANNELS or logits.shape[2] <= 0 or logits.shape[3] <= 0:
        return dict(base, reason="RAW_PER_FRAME_SHAPE_REQUIRED")
    if logits.dtype.kind != "f":
        return dict(base, reason="FLOAT_LOGITS_REQUIRED")
    if not bool(np.isfinite(logits).all()):
        return dict(base, reason="NONFINITE_LOGITS")
    message_mean = logits[:, 1:, :, :].mean(axis=(0, 2, 3), dtype=np.float64)
    detect_mean = float(logits[:, 0, :, :].mean(dtype=np.float64))
    signs = bits.astype(np.float64) * 2.0 - 1.0
    score = float(np.dot(message_mean, signs) / MESSAGE_BITS)
    if not math.isfinite(score) or not math.isfinite(detect_mean):
        return dict(base, reason="NONFINITE_REDUCTION")
    return dict(base, status="SCORED", score=score,
        detection_logit_mean=detect_mean,
        aggregation="uniform_all_181_frames_and_all_spatial_positions",
        detection_channel_in_score=False,
        meaning="uncalibrated continuous diagnostic only; no H0/H1 threshold or PASS")
