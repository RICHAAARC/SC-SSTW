"""Adopted R-MEAN-2H: two fixed half steps within one BASE-relative ball."""
from __future__ import annotations

import math

from main.tube_state.local_joint_readout_m0_v1 import tensor_readout, UndefinedActiveTie

VIEWS = ("BASE", "ONE_MEAN", "ONE_COMMON", "MID_MEAN", "TWO_MEAN")
FRAMES = (1, 44, 88, 112, 116, 120, 132, 176)
HALF = .5
CAP = 1.


def mean_objective(read):
    """One VJP of this scalar equals the mean of the five unscaled gradients."""
    import torch
    if read["tied"]:
        raise UndefinedActiveTie("ACTIVE_MIN_MAX_TIE: refreshed MEAN is undefined")
    values = read["objectives"]
    if tuple(values.shape) != (5,) or not bool(torch.isfinite(values).all()):
        raise FloatingPointError("five finite objectives required")
    return values.mean()


def midpoint(saved_mean):
    """Use the saved first direction verbatim; no new first-point VJP."""
    return HALF * saved_mean


def refresh_delta(midpoint_delta, masked_gradient):
    """Project the cumulative delta from original BASE, never fill unused radius."""
    import torch
    g = masked_gradient.detach().cpu().double()
    mid = midpoint_delta.detach().cpu()
    if g.shape != mid.shape or not bool(torch.isfinite(g).all()) or not bool(torch.isfinite(mid).all()):
        raise FloatingPointError("finite equal-shape midpoint and masked gradient required")
    norm = float(g.norm())
    report = dict(status="ZERO_MEAN" if norm == 0 else "DEFINED", half_step=HALF,
                  total_cap=CAP, gradient_norm=norm, midpoint_l2=float(mid.double().norm()),
                  budget_reference="original BASE", no_radius_refill=True,
                  individual_objective_prediction=None,
                  ceiling="One average-objective VJP does not certify five positive component slopes.")
    if norm == 0:
        return None, report
    if not math.isfinite(norm):
        raise FloatingPointError("nonfinite gradient norm")
    unit = (g/norm).to(dtype=mid.dtype)
    candidate = mid + HALF*unit
    before = float(candidate.double().norm())
    scale = min(1., CAP/before) if before else 1.
    delta = candidate*scale
    increment = delta-mid
    report.update(cumulative_pre_cap_l2=before, cap_scale=scale,
                  cumulative_post_cap_l2=float(delta.double().norm()),
                  actual_second_increment_l2=float(increment.double().norm()),
                  predicted_average_change=float((g*increment.double()).sum()))
    return delta, report
