"""Adopted .10 x 2 continuation from an actual saved terminal, in the original ball."""
from __future__ import annotations

import math
from main.tube_state import local_joint_all_bit_v1 as original
from main.tube_state.local_joint_all_bit_v1 import tensor_readout, scalar_loss, UndefinedActiveTie

VIEWS = ("BASE", "START_SAVED", "CONT_MID", "CONT_FINAL")
FRAMES = original.FRAMES
MU, GAMMA = original.MU, original.GAMMA
STEP, CAP = .10, 1.
PARAMETERS = dict(mu=MU,gamma=GAMMA,payload_weight=.8,state_weight=.2,step_size=STEP,cap=CAP)
STEP_PAIRS = (("START_SAVED","CONT_MID"),("CONT_MID","CONT_FINAL"))
TRACKED_BITS = (4,12,18,20,27,30)
PRIOR_POSITIVE_BITS = tuple(i for i in range(32) if i not in TRACKED_BITS)


def loss_summary(margins,gap,other_offsets):
    return {**original.loss_summary(margins,gap,other_offsets),"parameters":PARAMETERS}


def read_summary(read):
    return loss_summary(read["margins"].detach().cpu().reshape(-1).tolist(),
                        float(read["objectives"][0].detach()),read["active"]["other_offsets"])


def project_terminal(base,candidate):
    """FP64 radius projection and only-if-outside BASE-directed FP32 rounding."""
    import torch
    base=base.detach().cpu()
    candidate=candidate.detach().cpu().double()
    if base.dtype!=torch.float32 or base.shape!=candidate.shape or not bool(torch.isfinite(base).all()) or not bool(torch.isfinite(candidate).all()):
        raise ValueError("finite FP32 BASE and equal-shape finite displacement required")
    norm=float(candidate.norm())
    if not math.isfinite(norm):
        raise FloatingPointError("nonfinite candidate norm")
    scale=min(1.,CAP/norm) if norm else 1.
    projected=candidate*scale
    ulps=0
    while float(projected.norm())>CAP:
        scale=math.nextafter(scale,0.)
        projected=candidate*scale
        ulps+=1
    nearest=(base.double()+projected).float()
    if not bool(torch.isfinite(nearest).all()):
        raise FloatingPointError("FP32 terminal overflow")
    actual=nearest.double()-base.double()
    repair=float(actual.norm())>CAP
    overshot=(((projected>0)&(actual>projected))|((projected<0)&(actual<projected))) if repair else torch.zeros_like(projected,dtype=torch.bool)
    terminal=torch.where(overshot,torch.nextafter(nearest,base),nearest)
    displacement=terminal.double()-base.double()
    if not bool(torch.isfinite(displacement).all()) or float(displacement.norm())>CAP:
        raise FloatingPointError("saved FP32 terminal exceeds original BASE radius after inward rounding")
    return terminal,projected,dict(cumulative_pre_cap_l2=norm,cap_scale=scale,
        scale_fp64_inward_ulps=ulps,projected_l2=float(projected.norm()),
        nearest_fp32_actual_l2=float(actual.norm()),inward_repair_needed=repair,
        inward_fp32_components=int(overshot.sum()),rounding_l2=float((displacement-projected).norm()),
        cumulative_post_cap_l2=float(displacement.norm()))


def descent_terminal(base,current,masked_gradient):
    import torch
    base,current=base.detach().cpu(),current.detach().cpu()
    g=masked_gradient.detach().cpu().double()
    if base.dtype!=torch.float32 or current.dtype!=torch.float32 or base.shape!=current.shape or g.shape!=current.shape or not all(bool(torch.isfinite(x).all()) for x in (base,current,g)):
        raise ValueError("finite FP32 BASE/current and equal-shape gradient required")
    previous=current.double()-base.double()
    previous_norm,norm=float(previous.norm()),float(g.norm())
    if previous_norm>CAP or not math.isfinite(norm):
        raise FloatingPointError("actual input outside original BASE ball or nonfinite gradient norm")
    report=dict(status="ZERO_GRADIENT" if norm==0 else "DEFINED",step_size=STEP,total_cap=CAP,
        gradient_norm=norm,previous_delta_l2=previous_norm,budget_reference="actual terminal minus original BASE",
        no_radius_refill=True,component_predictions=None,prediction_meaning="one scalar loss slope, not per-bit slopes")
    if norm==0:
        return None,report
    candidate=previous-STEP*g/norm
    terminal,projected,rounding=project_terminal(base,candidate)
    increment=terminal.double()-current.double()
    report.update(rounding,increment_l2=float(increment.norm()),actual_terminal_increment_l2=float(increment.norm()),
        predicted_loss_change=float((g*increment).sum()),
        predicted_loss_change_actual_terminal_increment=float((g*increment).sum()),
        predicted_preprojection_loss_change=float((g*(candidate-previous)).sum()),
        predicted_projected_loss_change=float((g*(projected-previous)).sum()),
        cumulative_zero=bool(torch.count_nonzero(terminal.double()-base.double())==0))
    return terminal,report
