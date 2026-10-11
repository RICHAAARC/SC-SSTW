"""Adopted all-32-bit positive-margin deficit; two fresh BASE-relative steps."""
from __future__ import annotations

import math

from main.tube_state.local_joint_readout_m0_v1 import tensor_readout, UndefinedActiveTie

VIEWS = ("BASE", "TWO_MEAN_SAVED", "ALLBIT_HALF", "ALLBIT_TWO")
FRAMES = (1, 44, 88, 112, 116, 120, 132, 176)
MU, GAMMA = .02, .01
PAYLOAD_WEIGHT, STATE_WEIGHT = .8, .2
HALF, CAP = .5, 1.
PARAMETERS = dict(mu=MU, gamma=GAMMA, payload_weight=PAYLOAD_WEIGHT,
                  state_weight=STATE_WEIGHT, half_step=HALF, cap=CAP)


def huber_deficit(x, target):
    """C1 at both zero and target; linear deficit for nonpositive margins."""
    import torch
    return torch.where(x >= target, torch.zeros_like(x),
                       torch.where(x > 0, (target-x).square()/(2*target), target/2-x))


def loss_summary(margins, gap, other_offsets):
    """Plain-number report, also valid when an active state tie blocks a VJP."""
    m = [float(x) for x in margins]
    if len(m) != 32 or not all(math.isfinite(x) for x in m+[gap]):
        raise FloatingPointError("32 finite margins and finite state gap required")
    def h(x,a):
        return 0. if x >= a else ((a-x)**2/(2*a) if x > 0 else a/2-x)
    def dh(x,a):
        return 0. if x >= a else (-(1-x/a) if x > 0 else -1.)
    payload = sum(h(x,MU) for x in m)/32
    state = h(gap,GAMMA)
    return dict(loss=PAYLOAD_WEIGHT*payload+STATE_WEIGHT*state,
        payload_loss=payload, state_loss=state, parameters=PARAMETERS,
        payload_coefficients=[PAYLOAD_WEIGHT*dh(x,MU)/32 for x in m],
        state_coefficient=STATE_WEIGHT*dh(gap,GAMMA),
        payload_active=[x < MU for x in m], state_active=gap < GAMMA,
        state_wrong_offsets=list(other_offsets), state_max_tied=len(other_offsets)>1,
        active_state_tie=gap < GAMMA and len(other_offsets)>1,
        positive_bits=sum(x>0 for x in m), minimum_margin=min(m), mean_margin=sum(m)/32,
        nonpositive_deficits=[max(0.,-x) for x in m],
        target_deficits=[max(0.,MU-x) for x in m], state_target_deficit=max(0.,GAMMA-gap))


def read_summary(read):
    return loss_summary(read["margins"].detach().cpu().reshape(-1).tolist(),
                        float(read["objectives"][0].detach()), read["active"]["other_offsets"])


def scalar_loss(read):
    report = read_summary(read)
    if report["active_state_tie"]:
        raise UndefinedActiveTie("ACTIVE_STATE_MAX_TIE: G<gamma requires a unique wrong-offset maximum")
    # The old payload-min tied flag is descriptive only; all 32 C1 terms participate.
    return (PAYLOAD_WEIGHT*huber_deficit(read["margins"], MU).mean()
            + STATE_WEIGHT*huber_deficit(read["objectives"][0], GAMMA))


def descent_delta(previous_delta, masked_gradient):
    """One fixed half step; cumulative projection centered on original BASE."""
    import torch
    g = masked_gradient.detach().cpu().double()
    previous = previous_delta.detach().cpu()
    if g.shape != previous.shape or not bool(torch.isfinite(g).all()) or not bool(torch.isfinite(previous).all()):
        raise FloatingPointError("finite equal-shape delta and masked gradient required")
    norm = float(g.norm())
    if not math.isfinite(norm):
        raise FloatingPointError("nonfinite masked gradient norm")
    report = dict(status="ZERO_GRADIENT" if norm == 0 else "DEFINED", half_step=HALF,
        total_cap=CAP, gradient_norm=norm, previous_delta_l2=float(previous.double().norm()),
        budget_reference="original BASE", no_radius_refill=True,
        component_predictions=None, prediction_meaning="scalar loss change, not per-bit slopes")
    if norm == 0:
        return None, report
    candidate = previous-HALF*(g/norm).to(previous.dtype)
    before = float(candidate.double().norm())
    scale = min(1.,CAP/before) if before else 1.
    delta = candidate*scale
    increment = delta.double()-previous.double()
    report.update(cumulative_pre_cap_l2=before, cap_scale=scale,
        cumulative_post_cap_l2=float(delta.double().norm()), increment_l2=float(increment.norm()),
        predicted_loss_change=float((g*increment).sum()), cumulative_zero=before == 0.)
    return delta, report
