"""One adopted output-aware P48 correction; no model calls or truth-aware receiver."""
from __future__ import annotations
import math
from main.tube_state.local_joint_all_bit_v1 import huber_deficit
from main.tube_state.local_joint_readout_m0_v1 import tensor_readout
from main.tube_state.local_joint_p0_input_vjp_v1 import finite_tensor, statistics

VIEWS = ("ZERO", "ANCHOR", "CANDIDATE")
OBJECTIVES = ("S", "P", "W", "Q")
ENERGIES = ("latent_out_l2_squared", "rgb_out_mse", "rgb_mse", "temporal_mse",
            "worst_frame_mse", "worst_transition_mse", "pixel_peak_squared", "temporal_peak_squared")
CAP = .02
CEILING = ("One saved source/fork, one fixed nonzero anchor and one joint correction. "
    "BF16 path pullbacks are approximate, not derivatives of the actual discrete finite map. "
    "Eight half-energy goals are adopted finite-construction targets, not perceptual or scientific PASS. "
    "No scan, retry, P49 fallback, codec, new source or generalization claim.")


class UndefinedConstruction(ValueError):
    pass


def support_mask(shape, support, *, device="cpu"):
    import torch
    support.validate_for(tuple(shape))
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    for y0, y1, x0, x1 in support.rois:
        mask[:, :, support.time_range[0]:support.time_range[1], y0:y1, x0:x1] = True
    return mask


def restrict(value, support):
    return value * support_mask(value.shape, support, device=value.device)


def objectives(read):
    # amin/amax over all equal active entries supply the adopted equal tie rule.
    m, c = read["margins"].flatten(), read["correlations"]
    return dict(S=huber_deficit(c[0]-c[1:].amax(), .01),
                P=huber_deficit(m, .02).mean(), W=huber_deficit(m.amin(), .02))


def quality_data(z, rgb, zero_z, zero_rgb, support, protocol):
    """CPU FP64 exact formulas on saved float RGB; no clipping of residuals."""
    import torch
    z, zero_z, rgb, zero_rgb = (v.detach().cpu().double() for v in (z, zero_z, rgb, zero_rgb))
    if z.shape != zero_z.shape or rgb.shape != zero_rgb.shape or tuple(rgb.shape) != tuple(protocol.video_shape):
        raise ValueError("quality inputs must share fixed latent/RGB coordinates")
    for v in (z, zero_z, rgb, zero_rgb): finite_tensor(v)
    lm = support_mask(z.shape, support)
    rm = torch.zeros(rgb.shape[1:3], dtype=torch.bool)
    for y0, y1, x0, x1 in protocol.rois: rm[y0:y1, x0:x1] = True
    if bool(rm.all()) or len(rgb) < 2: raise ValueError("quality requires RGB outside support and temporal pairs")
    dz = z-zero_z
    r = rgb-zero_rgb
    dt = r[1:]-r[:-1]
    frame = r.square().mean((1,2,3))
    transition = dt.square().mean((1,2,3))
    values = torch.stack((dz[~lm].square().sum(), r[:,~rm,:].square().mean(), r.square().mean(),
                          dt.square().mean(), frame.amax(), transition.amax(), r.square().amax(), dt.square().amax()))
    if not bool(values.isfinite().all()) or bool((values < 0).any()): raise FloatingPointError("invalid quality energies")
    return values, (dz, r, dt, lm, rm, frame, transition)


def quality_values(z, rgb, zero_z, zero_rgb, support, protocol):
    return quality_data(z, rgb, zero_z, zero_rgb, support, protocol)[0]


def quality_cotangents(z, rgb, zero_z, zero_rgb, denominators, support, protocol):
    """Exact F_Q chain rule including equal max ties; no extra model/VJP call.

    Construct the full RGB seed on CPU, then one decoder VJP supplies its latent
    contribution. Add the latent-out term in the FP32 input coordinate before
    DiT/native transport. Neither output contribution is locally masked away.
    """
    import torch
    den = denominators.detach().cpu().double()
    finite_tensor(den, (8,), "same-run anchor quality denominators")
    if bool((den == 0).any()): raise UndefinedConstruction("ZERO_QUALITY_DENOMINATOR")
    if bool((den < 0).any()): raise ValueError("negative quality denominator")
    values, (dz,r,dt,lm,rm,frames,transitions) = quality_data(z,rgb,zero_z,zero_rgb,support,protocol)
    w = 1/(8*den)
    gz = 2*dz*(~lm)*w[0]
    gr = 2*r*(w[2]/r.numel())
    gr[:,~rm,:] += 2*r[:,~rm,:]*(w[1]/r[:,~rm,:].numel())
    gd = 2*dt*(w[3]/dt.numel())
    ties = {}
    active = frames == frames.amax()
    ties["frames"] = active.nonzero().flatten().tolist()
    gr += 2*r*(active.to(r.dtype)*w[4]/(r[0].numel()*int(active.sum())))[:,None,None,None]
    active = transitions == transitions.amax()
    ties["transitions"] = active.nonzero().flatten().tolist()
    gd += 2*dt*(active.to(dt.dtype)*w[5]/(dt[0].numel()*int(active.sum())))[:,None,None,None]
    active = r.square() == values[6]
    ties["pixels"] = dict(count=int(active.sum()), coordinates_file="pixel_peak_ties.pt")
    gr += 2*r*active*(w[6]/int(active.sum()))
    active = dt.square() == values[7]
    ties["temporal_pixels"] = dict(count=int(active.sum()), coordinates_file="temporal_peak_ties.pt")
    gd += 2*dt*active*(w[7]/int(active.sum()))
    gr[1:] += gd
    gr[:-1] -= gd
    return gz, gr, dict(value=float((values/den).mean()), energies=values.tolist(),
                        full_rgb_cotangent=statistics(gr), direct_latent_cotangent=statistics(gz),
                        ties=ties, tie_rule="equal active subgradient", output_support_masked=False)


def solve_correction(anchor, gradients, support):
    """One lexicographic SOCP in <=5 dimensions, then verify actual FP32 control."""
    import numpy as np
    import torch
    import cvxpy as cp
    report = dict(status="NUMERICALLY_UNRESOLVED", solver="CLARABEL", solver_tolerance=1e-10,
                  arithmetic_resolution=1e-8, physical_candidates=0, automatic_retry=False)
    a = anchor.detach().cpu().double().reshape(-1)
    if len(gradients) != 4: raise ValueError("four objective gradients required")
    finite_tensor(anchor, name="anchor", dtype=torch.float32)
    if float(a.norm()) > CAP or bool((anchor-restrict(anchor,support)).count_nonzero()):
        raise ValueError("actual anchor exceeds fixed control support/radius")
    rows = torch.stack([g.detach().cpu().double().reshape(-1) for g in gradients])
    if rows.shape[1] != a.numel() or not bool(rows.isfinite().all()):
        report.update(status="UNDEFINED", reason="nonfinite/malformed control gradient"); return None, report
    if any(bool((g-restrict(g,support)).count_nonzero()) for g in gradients): raise ValueError("control gradient outside original support")
    norms = rows.norm(dim=1)
    report["gradient_norms"] = norms.tolist()
    if bool((norms == 0).any()):
        report.update(status="UNDEFINED", reason="zero objective control gradient; no objective dropped"); return None, report
    normalized = rows/norms[:,None]
    vectors = torch.cat((normalized, (a/CAP)[None]))
    basis = []
    for v in vectors:
        v = v.clone()
        for _ in range(2):
            for b in basis: v -= (v@b)*b
        if float(v.norm()) > 1e-12: basis.append(v/v.norm())
    basis = torch.stack(basis)
    h = (normalized@basis.T).numpy()
    ac = (a@basis.T).numpy()/CAP
    residual = float((vectors-(vectors@basis.T)@basis).norm(dim=1).max())
    report.update(span_dimension=len(basis), span_residual=residual)
    if residual > 1e-8: report["reason"]="unresolved span arithmetic"; return None, report
    u,t = cp.Variable(len(basis)),cp.Variable()
    constraints = [h@u <= -t, cp.norm(u,2) <= 1, cp.norm(ac+u,2) <= 1]
    kwargs = dict(solver="CLARABEL",tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=200)
    primary = cp.Problem(cp.Maximize(t),constraints)
    primary.solve(**kwargs)
    report.update(primary_status=primary.status, primary_t=None if t.value is None else float(t.value))
    if primary.status != "optimal" or u.value is None or not np.isfinite(u.value).all():
        report["reason"]="primary solver did not resolve optimum; not a proof of infeasibility"; return None,report
    def certificate(extra=None):
        uv, tv = np.asarray(u.value), float(t.value)
        lam = np.asarray(constraints[0].dual_value)
        mu, nu = (float(c.dual_value) for c in constraints[1:])
        norm_u, norm_c = np.linalg.norm(uv), np.linalg.norm(ac+uv)
        stationarity = h.T@lam
        # At a zero cone argument its nonzero norm subgradient is not identified.
        identifiable = (norm_u > 1e-14 or abs(mu) < 1e-8) and (norm_c > 1e-14 or abs(nu) < 1e-8)
        if norm_u > 1e-14: stationarity += mu*uv/norm_u
        if norm_c > 1e-14: stationarity += nu*(ac+uv)/norm_c
        primal = max(0.,float((h@uv+tv).max()),float(norm_u-1),float(norm_c-1))
        complementarity = max(float(np.abs(lam*(h@uv+tv)).max()),abs(mu*(norm_u-1)),abs(nu*(norm_c-1)))
        dual_min = min(float(lam.min()),mu,nu)
        if extra is None:
            t_stationarity = abs(float(lam.sum())-1)
        else:
            eta = float(extra.dual_value)
            stationarity += 2*uv
            t_stationarity = abs(float(lam.sum())-eta)
            primal = max(primal, target-tv)
            complementarity = max(complementarity,abs(eta*(target-tv)))
            dual_min = min(dual_min,eta)
        return dict(solver_t=tv,realized_t=float(-(h@uv).max()),delta_scaled_l2=float(norm_u),
            control_scaled_l2=float(norm_c),primal_residual=primal,dual_nonnegative_residual=max(0.,-dual_min),
            stationarity_residual=float(max(np.abs(stationarity).max(),t_stationarity)) if identifiable else None,
            complementarity_residual=float(complementarity),dual_norm_subgradient_identified=bool(identifiable))
    report["primary_certificate"] = certificate()
    cert = report["primary_certificate"]
    report["certificate_tolerance"] = 1e-6
    if cert["primal_residual"] > 1e-8 or cert["dual_nonnegative_residual"] > 1e-8 or cert["complementarity_residual"] > 1e-6 or cert["stationarity_residual"] is None or cert["stationarity_residual"] > 1e-6:
        report["reason"]="primary numerical certificate unresolved"; return None,report
    t_star = cert["realized_t"]
    # Numerical resolution is explicitly recorded; no physical threshold/scan.
    if t_star <= 1e-8:
        report.update(status="NO_RESOLVED_COMMON_DESCENT",reason="positive max-min descent not numerically resolved"); return None,report
    target = t_star-1e-10
    report["secondary_target"] = target
    lex = t >= target
    secondary = cp.Problem(cp.Minimize(cp.sum_squares(u)), constraints+[lex])
    secondary.solve(**kwargs)
    report["secondary_status"] = secondary.status
    if secondary.status != "optimal" or u.value is None or not np.isfinite(u.value).all():
        report["reason"]="minimum-norm tie problem unresolved"; return None,report
    report["secondary_certificate"] = certificate(lex)
    cert = report["secondary_certificate"]
    if cert["primal_residual"] > 1e-8 or cert["dual_nonnegative_residual"] > 1e-8 or cert["complementarity_residual"] > 1e-6 or cert["stationarity_residual"] is None or cert["stationarity_residual"] > 1e-6:
        report["reason"]="secondary numerical certificate unresolved"; return None,report
    proposed_delta = CAP*(torch.tensor(u.value,dtype=torch.float64)@basis)
    candidate = (a+proposed_delta).reshape(anchor.shape).float()
    actual_delta = candidate.double().reshape(-1)-a
    planned_norm, actual_norm = float(proposed_delta.norm()),float(actual_delta.norm())
    actual_control_norm = float(candidate.double().norm())
    slopes = normalized@actual_delta
    report.update(planned_delta_l2=planned_norm, actual_delta_l2=actual_norm, actual_control_l2=actual_control_norm,
                  rounding_l2=float((actual_delta-proposed_delta).norm()),
                  normalized_slopes=slopes.tolist(), raw_predicted_changes=(rows@actual_delta).tolist(),
                  minimum_predicted_descent=float(-slopes.max()), tie_optimum_resolution=1e-10,
                  candidate_semantics="single FP32 rounded theta_A+delta; no corrective rescale")
    if actual_norm > CAP or actual_control_norm > CAP or bool((candidate-restrict(candidate,support)).count_nonzero()):
        report["reason"]="actual FP32 candidate violates a hard control bound; no rescale or replacement"; return None,report
    if float(-slopes.max()) <= CAP*1e-8:
        report["reason"]="actual FP32 positive common descent unresolved"; return None,report
    if max(float((h@u.value+t.value).max()),float(np.linalg.norm(u.value)-1),float(np.linalg.norm(ac+u.value)-1)) > 1e-8:
        report["reason"]="solver primal residual unresolved"; return None,report
    report.update(status="CANDIDATE_DEFINED",physical_candidates=1)
    return candidate,report


def finite_comparison(zero, anchor, candidate, anchor_energy, candidate_energy):
    """Separate adopted construction goals; never scientific PASS."""
    if candidate is None:
        return dict(status="MISSING_CANDIDATE",scientific_pass=False,ceiling=CEILING)
    mz, ma, mc = (v["margins"] for v in (zero,anchor,candidate))
    def loss(v):
        def h(x,a): return 0. if x>=a else (a-x)**2/(2*a) if x>0 else a/2-x
        gap=v["correlations"][0]-max(v["correlations"][1:])
        return dict(S=h(gap,.01),P=sum(h(x,.02) for x in v["margins"])/32,W=h(min(v["margins"]),.02),gap=gap)
    lz,la,lc=(loss(v) for v in (zero,anchor,candidate))
    quality = [dict(name=n,anchor=float(a),candidate=float(c),ratio=float(c/a),half_energy_met=bool(c<=.5*a))
               for n,a,c in zip(ENERGIES,anchor_energy,candidate_energy)]
    goals=dict(state_improves_vs_ZERO=lc["S"]<lz["S"],payload_improves_vs_ZERO=lc["P"]<lz["P"],
        weakest_margin_nondecrease_vs_ZERO=min(mc)>=min(mz),ZERO_positive_bits_preserved=all(b>0 for a,b in zip(mz,mc) if a>0),
        all_eight_half_energy_goals=all(r["half_energy_met"] for r in quality))
    return dict(status="FINITE_CANDIDATE_REPORTED", losses=dict(ZERO=lz,ANCHOR=la,CANDIDATE=lc),goals=goals,
        all_adopted_construction_goals_met=all(goals.values()),quality=quality,
        candidate_improvement_vs_ANCHOR={k:lc[k]<la[k] for k in ("S","P","W")},
        margins=dict(ZERO=mz,ANCHOR=ma,CANDIDATE=mc),correlations={n:v["correlations"] for n,v in zip(VIEWS,(zero,anchor,candidate))},
        scientific_pass=False,ceiling=CEILING)
