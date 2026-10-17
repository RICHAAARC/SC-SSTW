"""One fixed native terminal state writer and truth-free clean-grid reader."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math

import numpy as np
import torch

LATENT_SHAPE = (1, 16, 46, 40, 64)
VIDEO_SHAPE = (181, 320, 512, 3)
CHANNELS = tuple(range(4, 16))
ROIS = ((8, 16, 12, 20), (8, 16, 44, 52), (28, 36, 12, 20), (28, 36, 44, 52))
STATES = tuple(range(22))
NOMINAL_L2 = 0.0199
L2_CAP = 0.02
ARMS = ("PAYLOAD_ONLY", "PAYLOAD_STATE")
LAYERS = ("terminal", "float_rgb", "mp4")
KEYS = {"CORRECT": "watermark", "WRONG": "watermark-wrong"}


class ConstructionError(ValueError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def validate_latent(value):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != LATENT_SHAPE:
        raise ValueError("expected normalized native latent [1,16,46,40,64]")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("normalized latent must be finite floating point")


def support_mapping():
    return [dict(state=s, latent_half_open=[1 + 2*s, 3 + 2*s],
                 nominal_video_half_open=[8*s + 1, 8*s + 9]) for s in STATES]


@lru_cache(maxsize=8)
def _codebook(key: str):
    """PRNG key derivation only; this digest is never a source integrity check."""
    if not isinstance(key, str) or not key:
        raise ValueError("nonempty receiver key required")
    books = []
    for roi in range(4):
        seed = int.from_bytes(hashlib.sha256(
            f"staged-native-state-v1|{key}|{roi}".encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        rows = rng.permutation(np.arange(1, 512))[:22]
        walsh = np.array([[1 - 2*((int(row) & col).bit_count() % 2)
                           for col in range(512)] for row in rows], dtype=np.float64)
        code = np.tile(walsh, (1, 3))[:, rng.permutation(1536)]
        code *= rng.choice([-1, 1], size=(22, 1))
        books.append(code / math.sqrt(1536))
    return torch.from_numpy(np.stack(books))


def codebook(key):
    return _codebook(key).clone()


def write_state(native, key):
    """Writer knows s; only add the single declared state code, never payload."""
    validate_latent(native)
    base = native.detach().to(device="cpu", dtype=torch.float32)
    validate_latent(base)
    out = base.clone()
    codes = _codebook(key)
    for s in STATES:
        for r, (y0, y1, x0, x1) in enumerate(ROIS):
            delta = (codes[r, s] * (NOMINAL_L2 / math.sqrt(88))).float().reshape(12, 2, 8, 8)
            out[0, 4:16, 1+2*s:3+2*s, y0:y1, x0:x1] += delta
    actual = out.double() - base.double()
    mask = torch.zeros(LATENT_SHAPE, dtype=torch.bool)
    for y0, y1, x0, x1 in ROIS:
        mask[0, 4:16, 1:45, y0:y1, x0:x1] = True
    l2 = float(torch.linalg.vector_norm(actual))
    receipt = dict(status="CONSTRUCTED", nominal_l2=NOMINAL_L2, l2_cap=L2_CAP,
                   actual_fp32_l2=l2, support_rms=float(actual[mask].square().mean().sqrt()),
                   global_rms=float(actual.square().mean().sqrt()),
                   max_abs=float(actual.abs().max()), support_coordinates=int(mask.sum()),
                   outside_support_max=float(actual[~mask].abs().max()),
                   payload_channel_max=float(actual[:, :4].abs().max()),
                   changed_coordinates=int(torch.count_nonzero(actual)),
                   coordinate_system="normalized terminal latent; FP32 addition",
                   nominal_code_l2_per_tube=NOMINAL_L2/math.sqrt(88))
    if l2 > L2_CAP or not math.isfinite(l2):
        raise ConstructionError("actual FP32 delta exceeds 0.02 cap", receipt)
    if l2 == 0:
        raise ConstructionError("all fixed additions lost to FP32 rounding", receipt)
    return out, receipt


def read_state(received, key):
    """No source, writer target, message, attack metadata, or truth parameters."""
    validate_latent(received)
    z = received.detach().to(device="cpu", dtype=torch.float64)
    codes = _codebook(key)
    roi_scores = torch.empty((22, 4, 22), dtype=torch.float64)
    for position in STATES:
        for r, (y0, y1, x0, x1) in enumerate(ROIS):
            block = z[0, 4:16, 1+2*position:3+2*position, y0:y1, x0:x1].reshape(-1)
            roi_scores[position, r] = codes[r] @ (block - block.mean())
    scores = roi_scores.mean(dim=1)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("nonfinite state scores")
    decisions = []
    for position, row in enumerate(scores):
        top = float(row.max())
        winners = torch.nonzero(row == top).flatten().tolist()
        ordered = row.sort(descending=True).values
        decisions.append(dict(position=position, status="READ" if len(winners)==1 else "UNRESOLVED",
                              label=winners[0] if len(winners)==1 else None,
                              top_score=top, runner_up=float(ordered[1]),
                              top_gap=float(ordered[0]-ordered[1]), tied_candidates=winners))
    return dict(status="READ", truth_used=False, candidate_states=list(STATES),
                pooled_scores=scores.tolist(), roi_scores=roi_scores.tolist(), decisions=decisions,
                geometry="full clean 181-frame public grid; no unknown-phase recovery",
                rejection="exact maximum tie only; no calibrated presence threshold")


def evaluate_state(blind, targets):
    """Evaluation after raw persistence. Never feeds back into read_state."""
    if len(targets) != 22 or any(s not in STATES for s in targets):
        raise ValueError("22 evaluation state labels required")
    rows = []
    if blind.get("status") != "READ":
        return dict(status=blind.get("status", "NOT_ENTERED"), denominator=22,
                    rows=[dict(position=p, status=blind.get("status", "NOT_ENTERED")) for p in STATES])
    for p, target in enumerate(targets):
        scores = blind["pooled_scores"][p]
        wrong = max(scores[j] for j in STATES if j != target)
        dec = blind["decisions"][p]
        label = dec["label"]
        rows.append(dict(position=p, target=target, label=label, status=dec["status"],
                         correct_score=scores[target], max_wrong_score=wrong,
                         true_minus_max_wrong=scores[target]-wrong,
                         true_rank=1+sum(v > scores[target] for v in scores),
                         exact=label == target,
                         absolute_position_error=None if label is None else abs(label-target)))
    return dict(status="EVALUATED", denominator=22, rows=rows,
                exact_count=sum(row["exact"] for row in rows),
                unresolved_count=sum(row["label"] is None for row in rows))


def evaluate_payload(blind, bits):
    if len(bits) != 32 or any(b not in (0, 1) for b in bits):
        raise ValueError("32 evaluation bits required")
    if blind.get("status") != "READ":
        return dict(status=blind.get("status", "NOT_ENTERED"), denominator=32,
                    rows=[dict(bit=b, status=blind.get("status", "NOT_ENTERED")) for b in range(32)])
    rows = [dict(bit=i, decoded=b, target=bits[i], correct=b==bits[i])
            for i, b in enumerate(blind["decoded_bits"])]
    errors = sum(not row["correct"] for row in rows)
    return dict(status="EVALUATED", denominator=32, rows=rows, bit_errors=errors, ber=errors/32)
