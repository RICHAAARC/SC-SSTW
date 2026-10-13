"""Fixed FINAL six-margin derivative diagnostics. No terminal updates."""
from __future__ import annotations

import math

from main.tube_state.local_joint_readout_m0_v1 import simplex_minimum, tensor_readout
from main.tube_state import local_joint_all_bit_v1 as loss_method

BITS = (4, 12, 18, 20, 27, 30)
VIEWS = tuple(f"bit_{i:02d}" for i in BITS)
LOSS_PARAMETERS = dict(mu=.02, gamma=.01, payload_weight=.8, state_weight=.2)
CEILING = ("Six repeated derivative reads at the same saved CONT_FINAL, not six independent samples. "
           "Numerical local first-order diagnostics of stored masked gradients only; no update, "
           "finite-budget reachability or route infeasibility claim. The derivatives of the other "
           "26 bits, state and quality are unmeasured. No COMMON refresh, M1/M2 or scientific PASS.")


def selected_margin(read, index):
    if index not in BITS:
        raise ValueError("index outside adopted six bits")
    return read["margins"].reshape(-1)[index]


def values55(read):
    return dict(correlations=read["correlations"].detach().cpu().tolist(),
                state_gap=float(read["objectives"][0].detach()),
                margins=read["margins"].detach().cpu().reshape(-1).tolist(),
                active=read["active"])


def validate_values(values):
    if len(values["correlations"]) != 22 or len(values["margins"]) != 32:
        raise ValueError("22 correlations and 32 margins required")
    if not all(math.isfinite(x) for x in values["correlations"]+[values["state_gap"]]+values["margins"]):
        raise ValueError("nonfinite original readout")


def loss_condition(values):
    validate_values(values)
    summary = loss_method.loss_summary(values["margins"], values["state_gap"],
                                       values.get("active", {}).get("other_offsets", []))
    summary["parameters"] = LOSS_PARAMETERS
    active = [i for i, enabled in enumerate(summary["payload_active"]) if enabled]
    negative = all(values["margins"][i] < 0 for i in BITS)
    coefficients = all(summary["payload_coefficients"][i] == -.025 for i in BITS)
    valid = active == list(BITS) and negative and not summary["state_active"] and coefficients
    return dict(valid=valid, payload_active_indices=active, six_strictly_negative=negative,
                coefficients_are_minus_point025=coefficients, loss_summary=summary,
                fragment_and_wrong_offset_branches_are_not_loss_active_set=True)


def readout_report(read, reference, index):
    values = values55(read)
    validate_values(values); validate_values(reference)
    differences = {name: [a-b for a,b in zip(values[name], reference[name])]
                   for name in ("correlations", "margins")}
    differences["state_gap"] = values["state_gap"]-reference["state_gap"]
    return dict(bit_index=index, fragment=index//8, fragment_bit=index%8,
                differentiated_margin=values["margins"][index], derivative_sign="positive_margin",
                values=values, loss_condition=loss_condition(values),
                differences_from_saved_FINAL=differences,
                max_abs_difference_55=max(abs(x) for x in differences["correlations"]+
                                         [differences["state_gap"]]+differences["margins"]),
                diagnostic_only=True, metrics=55, windows=88, chips=1408,
                payload_denominators=[24,24,20,20])


def gamma(n):
    """Standard FP64 rounding accumulation bound, no score threshold."""
    eps = 2.0**-52
    return n*eps/(1-n*eps)


def gram_report(flat):
    import torch
    gram = flat @ flat.T
    error = gamma(flat.shape[1]+2) * (flat.abs() @ flat.abs().T)
    norms = flat.norm(dim=1)
    cosine = [[None if norms[i] == 0 or norms[j] == 0 else
               float(gram[i,j]/(norms[i]*norms[j])) for j in range(6)] for i in range(6)]
    return gram, error, dict(gram=gram.tolist(), gram_abs_roundoff_bound=error.tolist(),
                            norms=norms.tolist(), cosine=cosine)


def direction_report(flat, direction, delta, *, vector_error=0.):
    """Unit-direction slopes and exact straight-line unit-ball geometry."""
    import torch
    norm = float(direction.norm())
    if norm == 0:
        return None, dict(status="UNDEFINED_ZERO_VECTOR", norm=0., slopes=None,
                          slope_lower_bounds=None, ball_distance=None)
    d = direction/norm
    slopes = flat@d
    errors = gamma(flat.shape[1]+8)*(flat.abs()@d.abs())
    errors += flat.norm(dim=1)*(vector_error/norm + gamma(flat.shape[1]+8))
    radius = float(delta.norm())
    radial = float(delta@d)
    remaining = max(0., 1-radius*radius)
    distance = remaining/(math.sqrt(radial*radial+remaining)+radial) if radial > 0 else math.sqrt(radial*radial+remaining)-radial
    return d, dict(status="DEFINED", norm=norm, slopes=slopes.tolist(),
                   slope_abs_error_bound=errors.tolist(), slope_lower_bounds=(slopes-errors).tolist(),
                   ball_distance=distance, delta_dot_unit_direction=radial,
                   unit_norm_residual=abs(float(d.norm())-1),
                   geometry_ceiling="Straight-line distance only, not margin gain or steps to cross zero.")


def common_report(flat, delta, *, input_vector_error=None):
    import torch
    gram, gram_error, report = gram_report(flat)
    solution = simplex_minimum(gram.numpy())
    report["simplex"] = solution
    report["status"] = "NUMERICALLY_UNRESOLVED"
    report["ceiling"] = "Unresolved is neither an infeasibility certificate nor a reason to discard this row."
    if solution["weights"] is None:
        report.update(direction=None, primal_residual=None, dual_residual=None, kkt_residual=None)
        return None, report
    alpha = torch.tensor(solution["weights"], dtype=torch.float64)
    v = alpha@flat
    ga = gram@alpha
    q = float(alpha@ga)
    dual_residual = max(0., q-float(ga.min()))
    primal_residual = max(abs(float(alpha.sum())-1), float((-alpha).clamp_min(0).max()))
    complementarity = float((alpha*(ga-q)).abs().max())
    report.update(primal_objective=q, dual_lower_bound=2*float(ga.min())-q,
                  duality_gap=2*dual_residual, primal_residual=primal_residual,
                  dual_residual=dual_residual, complementarity_residual=complementarity,
                  kkt_residual=max(primal_residual,dual_residual,complementarity))
    # Conservative distance uncertainty around the numerical minimum-norm solve.
    # Even a nonzero vector is unresolved when it cannot be distinguished from zero.
    # Covers both objective/min-coordinate Gram error and the six-term QP
    # matvecs, in addition to the solver's reported scale-relative tolerance.
    gap_uncertainty = 8*(solution["numerical_tolerance"]+float(gram_error.max())+
                         gamma(64)*float(gram.abs().max()))
    solve_error = math.sqrt(2*(dual_residual+gap_uncertainty))
    combine_error = gamma(8)*float((alpha.abs()@flat.abs()).norm())
    if input_vector_error is not None:
        combine_error += float(alpha.abs()@input_vector_error)
    vector_error = solve_error+combine_error
    unit, direction = direction_report(flat,v,delta,vector_error=vector_error)
    if unit is not None and input_vector_error is not None:
        errors = torch.tensor(direction["slope_abs_error_bound"],dtype=torch.float64)+input_vector_error
        lower = torch.tensor(direction["slopes"],dtype=torch.float64)-errors
        direction.update(slope_abs_error_bound=errors.tolist(),slope_lower_bounds=lower.tolist())
    report.update(direction=direction, solution_vector_error_bound=vector_error,
                  gram_scale_error=gap_uncertainty)
    if unit is not None and all(x > 0 for x in direction["slope_lower_bounds"]):
        report["status"] = "NUMERICALLY_SUPPORTED_STRICT_COMMON_DIRECTION"
    return unit, report


def analyze(gradients, base, final, readouts):
    import torch
    if len(gradients) != 6 or len(readouts) != 6:
        raise ValueError("all six fixed rows required for joint diagnosis")
    flat = torch.stack([g.detach().cpu().double().reshape(-1) for g in gradients])
    delta = (final.detach().cpu().double()-base.detach().cpu().double()).reshape(-1)
    if not bool(torch.isfinite(flat).all()) or not bool(torch.isfinite(delta).all()):
        raise ValueError("finite gradients and actual displacement required")
    radius = float(delta.norm())
    if radius > 1:
        raise ValueError("actual FINAL lies outside original BASE radius 1")
    _, _, raw = gram_report(flat)
    aggregate, aggregate_report = direction_report(flat,flat.sum(0),delta)
    common, common_result = common_report(flat,delta)
    conditions = [loss_condition(read["values"]) for read in readouts]
    formula_valid = all(row["valid"] for row in conditions)
    report = dict(bits=list(BITS), raw=raw, aggregate=aggregate_report, common=common_result,
                  loss_gradient_formula_valid_on_all_six_reads=formula_valid,
                  loss_gradient_formula="-.025 * sum(masked_positive_margin_gradients)" if formula_valid else None,
                  loss_gradient_norm=float((-.025*flat.sum(0)).norm()) if formula_valid else None,
                  loss_conditions=conditions, radius=radius, outward_radial_distance=1-radius,
                  pure_tangent_distance=math.sqrt(max(0.,1-radius*radius)), ceiling=CEILING,
                  numerical_error_scope="FP64 arithmetic on stored gradients, including conservative simplex and projection uncertainty. No bound on model/autodiff FP32 derivative error.",
                  error_assumptions="Finite normal FP64 arithmetic; gamma_n dot-product bound, no underflow/overflow. Bounds are diagnostics, not formal integrity proofs.")
    directions = {"aggregate":aggregate,"common":common}
    if radius == 0:
        report["tangent"] = dict(status="UNDEFINED_RADIAL_AXIS_AT_BASE")
        directions["tangent_common"] = None
    else:
        u = delta/radius
        radial = flat@u
        tangent = flat-radial[:,None]*u
        radial_error = gamma(flat.shape[1]+8)*(flat.abs()@u.abs())
        projection_error = radial_error + gamma(flat.shape[1]+12)*(flat.norm(dim=1)+radial.abs())
        tangent_direction, tangent_report = common_report(tangent,delta,input_vector_error=projection_error)
        tangent_report.update(radial_slopes=radial.tolist(),radial_slope_abs_error_bound=radial_error.tolist(),
                              tangential_norms=tangent.norm(dim=1).tolist(),
                              radial_axis_norm_residual=abs(float(u.norm())-1),
                              projected_gradient_radial_residuals=(tangent@u).tolist(),
                              projection_vector_error_bounds=projection_error.tolist(),
                              unit_direction_radial_residual=None if tangent_direction is None else float(u@tangent_direction),
                              definition="t_i = h_i - (h_i dot u)u; simplex uses raw t_i, never per-bit normalization")
        report["tangent"] = tangent_report
        directions["tangent_common"] = tangent_direction
    return report, {k:None if v is None else v.reshape(final.shape) for k,v in directions.items()}
