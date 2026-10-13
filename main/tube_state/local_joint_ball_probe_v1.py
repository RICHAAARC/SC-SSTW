"""One guarded full-BASE-ball endpoint probe at the saved START_FINAL.

The CPU SOCP maximizes the worst predicted endpoint margin of the six adopted
failure bits.  It is deliberately not the old common-gain direction and not a
tangent step.  Four saved START derivatives constrain first-order changes only.
"""
from __future__ import annotations

import math

from main.tube_state.local_joint_protected_common_v1 import (
    BITS,
    GAMMA,
    MU,
    PROTECTIONS,
    barrier_parameters,
    barrier_value,
    validate_values,
)

VIEWS = ("BASE", "START_FINAL", "PROTECTED", "BALL_PROBE")
CAP = 1.0
# Pure numerical interior margins.  They do not relax or redefine the radius or
# protection constraints.  They protect those constraints against FP32 storage.
FP32_RADIUS_MARGIN = 2e-5
FP32_GUARD_MARGIN = 2e-5
SOLVER_TOL = 1e-9
RESIDUAL_TOL = 1e-8
CEILING = (
    "One saved START_FINAL and ten reused masked gradients define one guarded full-BASE-ball "
    "endpoint that maximizes the worst predicted margin of bits 4/12/18/20/27/30. "
    "One ordinary VAE decode only; no new VJP, encode, DiT, native/scheduler step, codec, "
    "step scan, second candidate, endpoint projection, cap increase or gradient refresh. "
    "First-order feasibility does not guarantee finite-step protection, 32-bit recovery, "
    "all wrong-offset protection, per-bit protection of the other25, local-peak quality, "
    "blind decoding, codec survival, FPR, generalization or scientific PASS."
)


def _basis(raw):
    """Stable orthonormal basis of the ten gradient rows."""
    import torch

    norms = raw.norm(dim=1)
    basis = []
    for row, norm in zip(raw, norms):
        if float(norm) == 0:
            continue
        residual = row / norm
        for _ in range(2):
            for vector in basis:
                residual = residual - (residual @ vector) * vector
        if float(residual.norm()) > 1e-12:
            basis.append(residual / residual.norm())
    if not basis:
        return None, norms
    return torch.stack(basis), norms


def linear_predictions(gradients, start_values, start, terminal):
    """Evaluate the saved START linear model on an actual stored terminal."""
    import torch

    validate_values(start_values)
    raw = torch.stack([g.detach().cpu().double().reshape(-1) for g in gradients])
    increment = terminal.detach().cpu().double().reshape(-1) - start.detach().cpu().double().reshape(-1)
    changes = raw @ increment
    margins = torch.tensor([start_values["margins"][i] for i in BITS], dtype=torch.float64) + changes[:6]
    signs = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    guard_norms = raw[6:].norm(dim=1)
    normalized_signed = signs * changes[6:] / guard_norms
    names = tuple(f"bit_{bit:02d}" for bit in BITS) + PROTECTIONS
    return dict(
        changes={name: float(value) for name, value in zip(names, changes)},
        target_margins={f"bit_{bit:02d}": float(value) for bit, value in zip(BITS, margins)},
        minimum_target_margin=float(margins.min()),
        normalized_signed_guard_changes={name: float(value) for name, value in zip(PROTECTIONS, normalized_signed)},
        max_normalized_guard_violation=max(0.0, -float(normalized_signed.min())),
    )


def solve_candidate(gradients, base, start, start_values):
    """Solve once in the ambient masked domain and return one FP32 terminal."""
    import cvxpy as cp
    import numpy as np
    import torch

    report = dict(
        status="NUMERICALLY_UNRESOLVED",
        solver="CLARABEL",
        solver_tolerance=SOLVER_TOL,
        residual_tolerance=RESIDUAL_TOL,
        target_bits=list(BITS),
        protection_order=list(PROTECTIONS),
        constraint_senses=[">=0", "<=0", ">=0", "<=0"],
        objective="maximize minimum predicted endpoint margin",
        domain="ambient adopted mask and original BASE ball; no constraint on the START displacement span",
        coordinate_reduction=(
            "lossless for this linear program: an optimal BASE-relative endpoint may be chosen "
            "in the ten-row span because an orthogonal endpoint component changes no saved "
            "linear objective or guard and only consumes radius"
        ),
        no_per_bit_objective_normalization=True,
        endpoint_projection=False,
        radius_cap=CAP,
        solver_radius_limit=CAP-FP32_RADIUS_MARGIN,
        normalized_guard_numerical_margin=FP32_GUARD_MARGIN,
        ceiling=CEILING,
    )
    if len(gradients) != 10:
        raise ValueError("six target and four protection gradients required")
    validate_values(start_values)
    raw = torch.stack([g.detach().cpu().double().reshape(-1) for g in gradients])
    base64 = base.detach().cpu().double().reshape(-1)
    start64 = start.detach().cpu().double().reshape(-1)
    delta = start64-base64
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(delta).all()):
        raise ValueError("finite saved gradients and BASE/START required")
    if not 0 < float(delta.norm()) <= CAP:
        raise ValueError("START must be inside the original BASE radius 1")
    basis, norms = _basis(raw)
    if basis is None or bool((norms == 0).any()):
        report["reason"] = "all ten constraint rows must have nonzero finite norm"
        return None, report
    rows = raw @ basis.T
    reconstruction = (raw-rows@basis).norm(dim=1)
    relative_reconstruction = reconstruction/norms
    objective_scale = float(norms[:6].max())
    start_margins = torch.tensor([start_values["margins"][i] for i in BITS], dtype=torch.float64)
    offsets = start_margins-raw[:6]@delta
    signs = torch.tensor([1.0,-1.0,1.0,-1.0],dtype=torch.float64)
    guard_start_dots = raw[6:]@delta
    try:
        x = cp.Variable(basis.shape[0])
        t = cp.Variable()
        a = rows[:6].numpy()/objective_scale
        b = offsets.numpy()/objective_scale
        h = (signs[:,None]*rows[6:]/norms[6:,None]).numpy()
        guard_rhs = (signs*guard_start_dots/norms[6:]).numpy()+FP32_GUARD_MARGIN
        constraints = [b+a@x >= t, h@x >= guard_rhs,
                       cp.norm(x,2) <= CAP-FP32_RADIUS_MARGIN]
        problem = cp.Problem(cp.Maximize(t),constraints)
        problem.solve(solver="CLARABEL",tol_gap_abs=SOLVER_TOL,tol_gap_rel=SOLVER_TOL,
                      tol_feas=SOLVER_TOL,max_iter=500)
        report.update(solver_status=problem.status,
                      solver_scaled_objective=None if problem.value is None else float(problem.value),
                      objective_scale=objective_scale,
                      span_dimension=int(basis.shape[0]),
                      gradient_norms=norms.tolist(),
                      span_relative_residuals=relative_reconstruction.tolist())
        if problem.status != "optimal" or x.value is None or not np.isfinite(x.value).all():
            report["reason"] = "solver did not return a finite optimal endpoint; not nonlinear infeasibility"
            return None,report
        endpoint = torch.tensor(x.value,dtype=torch.float64)@basis
        increment = endpoint-delta
        exact_margins = start_margins+raw[:6]@increment
        exact_guard_changes = raw[6:]@increment
        exact_signed = signs*exact_guard_changes/norms[6:]
        terminal = (base64+endpoint).reshape(base.shape).float()
        actual_endpoint = terminal.double().reshape(-1)-base64
        actual_increment = terminal.double().reshape(-1)-start64
        actual = linear_predictions(gradients,start_values,start,terminal)
        rounding = actual_endpoint-endpoint
        active_targets = [bit for bit,value in zip(BITS,exact_margins)
                          if float(value-exact_margins.min()) <= 1e-7]
        active_guards = [name for name,value in zip(PROTECTIONS,exact_signed)
                         if float(value-FP32_GUARD_MARGIN) <= 1e-7]
        residual = max(
            max(0.0,float(endpoint.norm())-(CAP-FP32_RADIUS_MARGIN)),
            max(0.0,FP32_GUARD_MARGIN-float(exact_signed.min())),
            max(relative_reconstruction.tolist()),
        )
        report.update(
            status="FIXED_BALL_CANDIDATE",
            predicted_minimum_margin=float(exact_margins.min()),
            predicted_target_margins={f"bit_{bit:02d}":float(value) for bit,value in zip(BITS,exact_margins)},
            predicted_guard_changes={name:float(value) for name,value in zip(PROTECTIONS,exact_guard_changes)},
            normalized_signed_guard_changes={name:float(value) for name,value in zip(PROTECTIONS,exact_signed)},
            active_target_constraints=active_targets,
            active_guard_constraints=active_guards,
            exact_endpoint_BASE_radius=float(endpoint.norm()),
            exact_distance_from_START=float(increment.norm()),
            exact_START_radial_component=float((increment@delta)/delta.norm()),
            delta_orthogonal_to_gradient_span_norm=float((delta-(delta@basis.T)@basis).norm()),
            solver_residual=residual,
            fp32=dict(
                BASE_radius=float(actual_endpoint.norm()),
                distance_from_START=float(actual_increment.norm()),
                rounding_l2=float(rounding.norm()),
                predictions=actual,
            ),
        )
        if residual > RESIDUAL_TOL:
            report.update(status="NUMERICALLY_UNRESOLVED",reason="solver endpoint failed numerical residual checks")
            return None,report
        if report["fp32"]["BASE_radius"] > CAP:
            report.update(status="NUMERICALLY_UNRESOLVED",reason="FP32 candidate exceeds the original BASE radius 1")
            return None,report
        if actual["max_normalized_guard_violation"] > RESIDUAL_TOL:
            report.update(status="NUMERICALLY_UNRESOLVED",reason="FP32 candidate violates a saved linear protection constraint")
            return None,report
        return terminal,report
    except Exception as exc:
        report["reason"] = repr(exc)
        return None,report
