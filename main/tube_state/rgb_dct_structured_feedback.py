"""Pure, fixed three-direction RGB-DCT terminal feedback decisions."""
from __future__ import annotations

import itertools
import math

import numpy as np


R_STAR = 0.042943312697648145
PROBE_RADIUS = R_STAR / 8
POINT_CAP = R_STAR
MARGIN = 1e-4
RIDGE = 1e-4
MIN_IMPROVEMENT = 1e-8
PEAK_CAP = 0.25
BINS = ((1, 15), (15, 30), (30, 45))
SCALES = (1.0, 0.5, 0.25)


def split_basis(raw: np.ndarray) -> tuple[list[np.ndarray], list[dict]]:
    """Partition one deterministic VAE lift, retaining exact zero bins."""
    if raw.shape != (1, 16, 46, 40, 64) or raw.dtype != np.float32:
        raise ValueError("fixed FP32 latent lift required")
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite lift")
    directions, records = [], []
    for lo, hi in BINS:
        direction = np.zeros_like(raw)
        window = raw[:, :, lo:hi].astype(np.float64)
        rms = math.sqrt(float(np.mean(window * window)))
        support_rms = rms * math.sqrt((hi - lo) / 44)
        if support_rms:
            direction[:, :, lo:hi] = raw[:, :, lo:hi] / support_rms
        directions.append(direction)
        records.append(dict(time=[lo, hi], raw_bin_rms=rms,
                            raw_support_rms=support_rms,
                            status="READY" if rms else "ZERO_BASIS"))
    return directions, records


def weighted_hinge(q: np.ndarray, positive_reference: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    reference = np.asarray(positive_reference, dtype=np.float64)
    if (q.shape != (30,) or reference.shape != (30,)
            or not np.isfinite(q).all() or not np.isfinite(reference).all()):
        raise ValueError("finite 30-group q required")
    weights = np.where(reference > 0, 4.0, 1.0)
    return float(np.mean(weights * np.square(np.maximum(MARGIN - q, 0))))


def select_coefficients(q: np.ndarray, jacobian: np.ndarray, remaining: float,
                        available: tuple[bool, bool, bool] = (True, True, True)) -> dict:
    """Bounded 125-prediction deterministic lattice; no model calls."""
    q = np.asarray(q, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if q.shape != (30,) or jacobian.shape != (30, 3):
        raise ValueError("fixed q/J shape required")
    if not np.isfinite(q).all() or not np.isfinite(jacobian).all():
        raise ValueError("finite q/J required")
    if not math.isfinite(remaining) or not 0 <= remaining <= R_STAR:
        raise ValueError("finite remaining radius in [0,R*] required")
    radius = min(POINT_CAP, max(0.0, remaining))
    best = None
    tried = 0
    for fractions in itertools.product((0.0, 0.25, 0.5, 0.75, 1.0), repeat=3):
        tried += 1
        coeff = np.asarray(fractions, dtype=np.float64) * radius
        if any(coeff[j] != 0 and not available[j] for j in range(3)):
            continue
        if float(np.sum(coeff)) > radius + 1e-14:
            continue
        predicted = q + jacobian @ coeff
        objective = weighted_hinge(predicted, q) + RIDGE * float(coeff @ coeff)
        if not math.isfinite(objective):
            raise FloatingPointError("nonfinite predicted objective")
        choice = (objective, tuple(float(x) for x in coeff))
        if best is None or choice < best[0]:
            best = (choice, predicted)
    assert best is not None
    return dict(coefficients=list(best[0][1]), predicted_q=best[1].tolist(),
                predicted_objective=best[0][0], predictions_tried=tried,
                radius=radius)


def judge_candidate(base_q: np.ndarray, candidate_q: np.ndarray,
                    immediate_rms: float, immediate_peak: float,
                    spent: float) -> dict:
    base = np.asarray(base_q, dtype=np.float64)
    candidate = np.asarray(candidate_q, dtype=np.float64)
    if (base.shape != (30,) or candidate.shape != (30,)
            or not np.isfinite(base).all() or not np.isfinite(candidate).all()):
        raise ValueError("finite 30-group q required")
    gained = np.flatnonzero((base <= 0) & (candidate > 0)).tolist()
    lost = np.flatnonzero((base > 0) & (candidate <= 0)).tolist()
    before = weighted_hinge(base, base)
    after = weighted_hinge(candidate, base)
    reason = "ACCEPT"
    if not all(map(math.isfinite, (immediate_rms, immediate_peak, spent))):
        raise FloatingPointError("nonfinite native response")
    if min(immediate_rms, immediate_peak, spent) < 0:
        raise ValueError("negative native response ledger")
    if immediate_rms > POINT_CAP or spent + immediate_rms > R_STAR:
        reason = "NATIVE_RMS_BUDGET"
    elif immediate_peak > PEAK_CAP:
        reason = "NATIVE_PEAK_CAP"
    elif lost:
        reason = "POSITIVE_GROUP_LOSS"
    elif before - after < MIN_IMPROVEMENT:
        reason = "NO_TRUE_HINGE_IMPROVEMENT"
    return dict(accepted=reason == "ACCEPT", reason=reason,
                gained_groups=gained, lost_groups=lost,
                baseline_hinge=before, candidate_hinge=after,
                true_hinge_improvement=before - after)
