"""One protected tangent common direction at the saved START_FINAL.

The six objectives retain their raw margin units. Four homogeneous protection
constraints are numerical first-order predictions, never endpoint guarantees.
"""
from __future__ import annotations

import math

from main.tube_state.local_joint_readout_m0_v1 import tensor_readout
from main.tube_state.local_joint_six_bit_final_v1 import values55, validate_values
from main.tube_state.local_joint_tangent_common_v1 import candidate as straight_candidate

BITS = (4, 12, 18, 20, 27, 30)
PROTECTIONS = ("bit29", "barrier25", "state_G", "quality_Q")
VIEWS = ("BASE", "START_FINAL", "TANGENT", "PROTECTED")
MU, GAMMA, STEP = .02, .01, .10
# Numerical resolution only; these are not scientific success thresholds.
SOLVER_TOL = 1e-9
RESIDUAL_TOL = 1e-8
CEILING = ("One saved START_FINAL, six reused margin gradients and four fresh protection VJPs; "
           "at most one .10 tangent step in the original BASE radius 1. First-order protection "
           "does not guarantee any finite-step margin, state or quality outcome. Fixed four views "
           "and all missing/failed slots remain. No scan, second step, projection, relaxed protection, "
           "cap increase, codec, blind decoding, FPR, trajectory/CFG/native causality or scientific PASS.")


class UndefinedProtection(ValueError):
    """A scalar derivative has no adopted differentiable definition at this point."""


def barrier_parameters(reference):
    validate_values(reference)
    margins = reference["margins"]
    positive = [i for i, value in enumerate(margins) if value > 0]
    if len(positive) != 26 or set(positive) != set(range(32))-set(BITS):
        raise ValueError("START must retain the adopted six negative and 26 positive bits")
    if not all(margins[i] < 0 for i in BITS):
        raise ValueError("all six adopted failure bits must be negative at START")
    indices = [i for i in positive if i != 29]
    gaps = [margins[i]-MU for i in indices]
    if margins[29] <= MU or any(g <= 0 for g in gaps):
        raise UndefinedProtection("START positive margins must exceed mu for the adopted protections")
    return dict(indices=indices, start_gaps=gaps,
                coefficients=[-1/(25*g) for g in gaps], mu=MU)


def protection_scalar(name, read, rgb, base_rgb, parameters):
    """Each call is differentiated once on an independent decoder graph."""
    import torch
    if name == "bit29":
        return read["margins"].reshape(-1)[29]
    if name == "barrier25":
        coefficients = torch.tensor(parameters["coefficients"], dtype=read["margins"].dtype,
                                    device=read["margins"].device)
        # Fixed START weights: exactly dB at START, not a newly tuned loss.
        return (read["margins"].reshape(-1)[parameters["indices"]]*coefficients).sum()
    if name == "state_G":
        if len(read["active"]["other_offsets"]) != 1:
            raise UndefinedProtection("G has a tied maximum wrong offset; no extra VJP")
        return read["objectives"][0]
    if name == "quality_Q":
        if rgb.shape != base_rgb.shape or not bool(torch.isfinite(base_rgb).all()):
            raise ValueError("Q requires finite saved BASE RGB with the same full shape")
        # FP64 residual arithmetic matches the existing full-frame quality report.
        # Transfer only one reference frame at a time; never clip/quantize residuals.
        return torch.stack([(frame.double()-reference.to(frame.device).double()).square().sum()
                            for frame, reference in zip(rgb,base_rgb)]).sum()/rgb.numel()
    raise ValueError("unknown fixed protection scalar")


def barrier_value(margins, parameters):
    bad = [i for i in parameters["indices"] if margins[i] <= MU]
    if bad:
        return dict(status="OUTSIDE_LOG_DOMAIN", value=None, indices_at_or_below_mu=bad)
    value = -sum(math.log((margins[i]-MU)/gap)
                 for i,gap in zip(parameters["indices"],parameters["start_gaps"]))/25
    return dict(status="DEFINED", value=value, indices_at_or_below_mu=[])


def solve_direction(gradients, base, start):
    """Ten masked vectors -> at most ten dimensional SOCP, solved once.

Row normalization is used only to construct a stable span and scale protection
constraints. All six target rows share one scale, retaining raw max-min units.
"""
    import numpy as np
    import torch
    report = dict(status="NUMERICALLY_UNRESOLVED", zero_direction_feasible=True,
                  solver="CLARABEL", solver_tolerance=SOLVER_TOL, residual_tolerance=RESIDUAL_TOL,
                  target_bits=list(BITS), protection_order=list(PROTECTIONS),
                  constraint_senses=[">=0", "<=0", ">=0", "<=0"],
                  no_per_bit_objective_normalization=True, ceiling=CEILING)
    if len(gradients) != 10:
        raise ValueError("six saved and four new gradients required")
    raw = torch.stack([g.detach().cpu().double().reshape(-1) for g in gradients])
    delta = (start.detach().cpu().double()-base.detach().cpu().double()).reshape(-1)
    if not bool(torch.isfinite(raw).all()) or not 0 < float(delta.norm()) <= 1:
        raise ValueError("finite masked gradients and nonzero START radius <=1 required")
    u = delta/delta.norm()
    tangent = raw-(raw@u)[:,None]*u
    norms = tangent.norm(dim=1)
    # Twice-reorthogonalized MGS on scaled rows avoids losing a small Q gradient
    # merely because its physical units are much smaller than margin units.
    basis = []
    for row,norm in zip(tangent,norms):
        if norm == 0:
            continue
        residual = row/norm
        for _ in range(2):
            for b in basis:
                residual = residual-(residual@b)*b
        if float(residual.norm()) > 1e-12:
            basis.append(residual/residual.norm())
    report.update(tangent_norms=norms.tolist(), span_dimension=len(basis),
                  projected_radial_residuals=(tangent@u).tolist())
    if not basis or float(norms[:6].max()) == 0:
        report["status"] = "NO_CLEAR_POSITIVE_DIRECTION"
        report["reason"] = "zero objective tangent span; zero is feasible"
        return None,report
    basis = torch.stack(basis)
    rows = tangent@basis.T
    reconstruction_error = (tangent-rows@basis).norm(dim=1)
    report["span_relative_residuals"] = [0. if n == 0 else float(e/n) for e,n in zip(reconstruction_error,norms)]
    objective_scale = float(norms[:6].max())
    a = rows[:6].numpy()/objective_scale
    signs = torch.tensor([1.,-1.,1.,-1.],dtype=torch.float64)
    h = rows[6:]*signs[:,None]
    protection_norms = norms[6:]
    h = h/torch.where(protection_norms>0,protection_norms,torch.ones_like(protection_norms))[:,None]
    try:
        import cvxpy as cp
        x,t = cp.Variable(len(basis)),cp.Variable()
        constraints = [a@x >= t, h.numpy()@x >= 0, cp.norm(x,2) <= 1]
        problem = cp.Problem(cp.Maximize(t),constraints)
        problem.solve(solver="CLARABEL",tol_gap_abs=SOLVER_TOL,tol_gap_rel=SOLVER_TOL,
                      tol_feas=SOLVER_TOL,max_iter=200)
        report.update(solver_status=problem.status, objective_scale=objective_scale,
                      objective=None if problem.value is None or not math.isfinite(problem.value)
                      else float(problem.value*objective_scale))
        if problem.status != "optimal" or x.value is None or not np.isfinite(x.value).all():
            report["reason"] = "solver failed or did not resolve an optimal solution; not method infeasibility"
            return None,report
        v = torch.tensor(x.value,dtype=torch.float64)@basis
        if float(v.norm()) <= RESIDUAL_TOL:
            report.update(status="NO_CLEAR_POSITIVE_DIRECTION",reason="optimal numerical direction is near zero")
            return None,report
        d = v/v.norm()
        slopes = raw@d
        signed = signs*slopes[6:]
        scaled_protection = signed/torch.where(protection_norms>0,protection_norms,torch.ones_like(protection_norms))
        relative_gain = float(slopes[:6].min())/objective_scale
        residual = max(0.,-float(scaled_protection.min()),abs(float(u@d)),
                       abs(float(d.norm())-1),max(report["span_relative_residuals"]))
        report.update(slopes=slopes.tolist(), scaled_signed_protection_slopes=scaled_protection.tolist(),
                      min_target_slope=float(slopes[:6].min()), relative_min_target_slope=relative_gain,
                      numerical_gain_resolution=RESIDUAL_TOL*objective_scale,
                      unit_norm=float(d.norm()), radial_dot=float(delta@d), residual=residual,
                      solver_scaled_objective=float(problem.value))
        if residual > RESIDUAL_TOL:
            report["reason"] = "normalized direction failed numerical constraints; no relaxation or candidate"
            return None,report
        if relative_gain <= RESIDUAL_TOL:
            report.update(status="NO_CLEAR_POSITIVE_DIRECTION",reason="positive common gain is not numerically clear")
            return None,report
        report["status"] = "POSITIVE_COMMON_DIRECTION"
        return d.reshape(start.shape),report
    except Exception as exc:
        report["reason"] = repr(exc)
        return None,report


def candidate(base, start, direction, gradients):
    terminal,report = straight_candidate(base,start,direction,STEP)
    actual = terminal.double()-start.double()
    report.update(direction_semantics="protected max-min unit tangent at START_FINAL",ceiling=CEILING,
                  predictions={name:dict(slope=float((g.double()*direction).sum()),
                      planned_change=float((g.double()*(STEP*direction)).sum()),
                      actual_increment_change=float((g.double()*actual).sum()))
                      for name,g in zip(tuple(f"bit_{i:02d}" for i in BITS)+PROTECTIONS,gradients)})
    return terminal,report
