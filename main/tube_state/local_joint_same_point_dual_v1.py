"""Adopted same-S, .10-step MEAN/ALLBIT comparison; no projection or selection."""
from __future__ import annotations

import math
from main.tube_state import local_joint_all_bit_v1 as allbit
from main.tube_state.local_joint_gradient_sum_v1 import NONPOSITIVE_ROSTER, summarize as old_summary
from main.tube_state.local_joint_readout_m0_v1 import tensor_readout, UndefinedActiveTie

ARMS = ("NEXT_MEAN", "NEXT_ALLBIT")
REFERENCES = ("BASE", "SUM_STEP", "HISTORICAL_26")
VIEWS = REFERENCES + ARMS
HARD_BITS = (4, 12, 18, 20, 27, 30)
STEP, CAP = .10, 1.0
CEILING = (
    "Two fixed recipes at one seen saved SUM_STEP, each with one fresh scalar VJP and one .10 step. "
    "Both payload aggregation and state weights differ; differences are not isolated payload causality. "
    "No individual-bit or separate new A/G gradient guarantees. Historical 26/32 remains a saved reference. "
    "No scan, selection, projection, rollback, extra step, codec, blind recovery, FPR, independent source, "
    "generalization, native trajectory causality or combined A/B scientific PASS."
)


def summarize(values):
    row = old_summary(values)
    row.pop("thresholds_are_descriptive_only", None)
    row["no_result_threshold_gate"] = True
    m, c = row["margins"], row["correlations"]
    loss = allbit.loss_summary(m, row["state_gap"], row["actual_max_wrong_offsets"])
    # The historical loss is reused, but its old half_step is not this task's update.
    loss["parameters"] = dict(mu=.02, gamma=.01, payload_weight=.8, state_weight=.2)
    row.update(allbit_loss=loss, minimum_margin=min(m),
        deepest_negative_indices=[i for i, x in enumerate(m) if x == min(m) and x <= 0],
        hard_bits=[dict(index=i, margin=m[i], nonpositive_deficit=max(0., -m[i])) for i in HARD_BITS],
        mean_scalar=row["fixed_mean_A"] + row["state_gap"],
        threshold_roles=dict(mean="descriptive", allbit="adopted loss parameters; not result PASS"))
    return row


def read_report(name, read):
    values = dict(margins=read["margins"].detach().cpu().flatten().tolist(),
                  correlations=read["correlations"].detach().cpu().flatten().tolist())
    summary = summarize(values)
    return dict(name=name, values=values, summary=summary,
        objective="maximize A16+trueG" if name == ARMS[0] else "minimize historical all-bit loss",
        scalar=summary["mean_scalar"] if name == ARMS[0] else summary["allbit_loss"]["loss"],
        separate_component_derivatives=False, derivative_point="SUM_STEP",
        saved_expected_wrong_branch=5, actual_wrong_branches=summary["actual_max_wrong_offsets"])


def scalar(name, read):
    if name == "NEXT_ALLBIT":
        return allbit.scalar_loss(read)
    if name != "NEXT_MEAN":
        raise ValueError("unknown fixed arm")
    correlations = read["correlations"].flatten()
    maximum = correlations[1:].max()
    if int((correlations[1:] == maximum).sum()) != 1:
        raise UndefinedActiveTie("MEAN trueG requires a unique actual wrong-state maximum")
    return read["margins"].flatten()[list(NONPOSITIVE_ROSTER)].mean() + correlations[0] - maximum


def construct(name, base, start, gradient, support):
    import torch
    if name not in ARMS:
        raise ValueError("unknown fixed arm")
    base, start, g = base.detach().cpu(), start.detach().cpu(), gradient.detach().cpu().double()
    if base.dtype != torch.float32 or start.dtype != torch.float32 or not (base.shape == start.shape == g.shape):
        raise ValueError("finite matching FP32 original BASE/S and gradient required")
    if not all(bool(x.isfinite().all()) for x in (base, start, g)):
        raise FloatingPointError("nonfinite BASE/S/gradient")
    support.validate_for(tuple(base.shape))
    def masked(x):
        value = torch.zeros_like(x)
        for y0,y1,x0,x1 in support.rois:
            value[:,:,support.time_range[0]:support.time_range[1],y0:y1,x0:x1] = x[:,:,support.time_range[0]:support.time_range[1],y0:y1,x0:x1]
        return value
    previous = start.double() - base.double()
    if float(previous.norm()) > CAP or torch.count_nonzero(previous-masked(previous)):
        raise ValueError("S violates original BASE radius or original support")
    if torch.count_nonzero(g-masked(g)):
        raise ValueError("gradient must use the original support mask")
    norm = float(g.norm())
    if not math.isfinite(norm):
        raise FloatingPointError("nonfinite gradient norm")
    sign = 1 if name == "NEXT_MEAN" else -1
    report = dict(status="ZERO_GRADIENT" if norm == 0 else "DEFINED", arm=name, step=.10,
        direction_sign=sign, masked_gradient_norm=norm, per_component_normalization=False,
        original_BASE_start_radius=float(previous.norm()), radius_cap=1., projection=False,
        derivative_point="SUM_STEP", component_predictions=None)
    if norm == 0:
        return None, report
    terminal = (start.double()+sign*STEP*g/norm).to(start.dtype)
    delta, increment = terminal.double()-base.double(), terminal.double()-start.double()
    if float(delta.norm()) > CAP or torch.count_nonzero(delta-masked(delta)):
        raise ValueError("actual candidate violates original BASE radius or support; no projection")
    report.update(original_BASE_radius=float(delta.norm()), actual_increment_l2=float(increment.norm()),
                  predicted_scalar_change=float((g*increment).sum()))
    return terminal, report
