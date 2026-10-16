"""Fixed-target secondary reformulation; no primary solve or numerical retry."""
from __future__ import annotations
import copy
import math
from main.tube_state import local_joint_p48_joint_control_v1 as original
from main.tube_state.local_joint_p0_input_vjp_v1 import finite_tensor


def coordinates(anchor, gradients, support):
    """Same normalization and two-pass span construction as the adopted primary."""
    import torch
    finite_tensor(anchor,name='anchor',dtype=torch.float32)
    a=anchor.detach().cpu().double().reshape(-1)
    if float(a.norm())>original.CAP or bool((anchor-original.restrict(anchor,support)).count_nonzero()):
        raise ValueError('actual anchor exceeds original support/radius')
    if len(gradients)!=4: raise ValueError('four saved objective gradients required')
    for g in gradients:
        finite_tensor(g,anchor.shape,'saved control cotangent',torch.float32)
        if bool((g-original.restrict(g,support)).count_nonzero()): raise ValueError('control gradient outside support')
    rows=torch.stack([g.detach().cpu().double().reshape(-1) for g in gradients])
    norms=rows.norm(dim=1)
    if bool((norms==0).any()): raise original.UndefinedConstruction('zero objective control gradient; none dropped')
    normalized=rows/norms[:,None]
    vectors=torch.cat((normalized,(a/original.CAP)[None]))
    basis=[]
    for v in vectors:
        v=v.clone()
        for _ in range(2):
            for b in basis:v-=(v@b)*b
        if float(v.norm())>1e-12:basis.append(v/v.norm())
    basis=torch.stack(basis)
    residual=float((vectors-(vectors@basis.T)@basis).norm(dim=1).max())
    if residual>1e-8:raise ValueError('unresolved span arithmetic')
    return dict(a=a,rows=rows,normalized=normalized,norms=norms,basis=basis,
        h=(normalized@basis.T).numpy(),ac=(a@basis.T).numpy()/original.CAP,span_residual=residual)


def certificate(c, target, primal, dual):
    import numpy as np
    u=np.asarray(primal,dtype=float);lam=np.asarray(dual['objective'],dtype=float)
    mu,nu=float(dual['delta_ball']),float(dual['control_ball'])
    if u.shape!=c['ac'].shape or lam.shape!=(4,) or not all(np.isfinite(x).all() for x in (u,lam,np.array([mu,nu]))):
        raise ValueError('invalid secondary primal/dual coordinates')
    norm_u,norm_c=np.linalg.norm(u),np.linalg.norm(c['ac']+u)
    identifiable=(norm_u>1e-14 or abs(mu)<1e-8) and (norm_c>1e-14 or abs(nu)<1e-8)
    stationarity=2*u+c['h'].T@lam
    if norm_u>1e-14:stationarity+=mu*u/norm_u
    if norm_c>1e-14:stationarity+=nu*(c['ac']+u)/norm_c
    slack=c['h']@u+target
    return dict(realized_t=float(-(c['h']@u).max()),delta_scaled_l2=float(norm_u),control_scaled_l2=float(norm_c),
        primal_residual=max(0.,float(slack.max()),float(norm_u-1),float(norm_c-1)),
        dual_nonnegative_residual=max(0.,-float(lam.min()),-mu,-nu),
        stationarity_residual=float(np.abs(stationarity).max()) if identifiable else None,
        complementarity_residual=max(float(np.abs(lam*slack).max()),abs(mu*(norm_u-1)),abs(nu*(norm_c-1))),
        dual_norm_subgradient_identified=bool(identifiable))


def checked_candidate(anchor, gradients, saved_solution, receipt, support):
    """Re-read saved inputs and validate the one recorded solve; never re-solve."""
    import numpy as np
    import torch
    report=copy.deepcopy(receipt)
    c=coordinates(anchor,gradients,support)
    target=float(saved_solution['secondary_target'])
    if not math.isfinite(target) or target<=0:raise ValueError('original positive fixed secondary target required')
    if report['secondary_target']!=target:raise ValueError('secondary witness uses a different mathematical target')
    report['reloaded_gradient_norms']=c['norms'].tolist()
    if report.get('secondary_primal') is not None and report.get('secondary_dual') is not None:
        report['secondary_certificate']=certificate(c,target,report['secondary_primal'],report['secondary_dual'])
    if report.get('secondary_status')!='optimal':
        report.update(status='NUMERICALLY_UNRESOLVED',physical_candidates=0)
        report.setdefault('reason','fixed-target secondary solver did not return optimal')
        return None,report
    if report.get('secondary_primal') is None or report.get('secondary_dual') is None:
        report.update(status='NUMERICALLY_UNRESOLVED',physical_candidates=0,reason='secondary primal/dual unavailable')
        return None,report
    cert=report['secondary_certificate']
    if cert['primal_residual']>1e-8 or cert['dual_nonnegative_residual']>1e-8 or cert['complementarity_residual']>1e-6 or cert['stationarity_residual'] is None or cert['stationarity_residual']>1e-6:
        report.update(status='NUMERICALLY_UNRESOLVED',physical_candidates=0,reason='fixed-target secondary certificate unresolved')
        return None,report
    proposed=original.CAP*(torch.tensor(report['secondary_primal'],dtype=torch.float64)@c['basis'])
    candidate=(c['a']+proposed).reshape(anchor.shape).float()
    delta=candidate.double().reshape(-1)-c['a'];slopes=c['normalized']@delta
    report.update(planned_delta_l2=float(proposed.norm()),actual_delta_l2=float(delta.norm()),actual_control_l2=float(candidate.double().norm()),
        rounding_l2=float((delta-proposed).norm()),normalized_slopes=slopes.tolist(),raw_predicted_changes=(c['rows']@delta).tolist(),
        minimum_predicted_descent=float(-slopes.max()),candidate_semantics='single FP32 rounded theta_A+delta; no corrective rescale')
    if report['actual_delta_l2']>original.CAP or report['actual_control_l2']>original.CAP or bool((candidate-original.restrict(candidate,support)).count_nonzero()):
        report.update(status='NUMERICALLY_UNRESOLVED',physical_candidates=0,reason='actual FP32 candidate violates a hard control bound; no rescale or replacement')
        return None,report
    if report['minimum_predicted_descent']<=original.CAP*1e-8:
        report.update(status='NUMERICALLY_UNRESOLVED',physical_candidates=0,reason='actual FP32 positive common descent unresolved')
        return None,report
    report.update(status='CANDIDATE_DEFINED',physical_candidates=1)
    report.pop('reason',None)
    return candidate,report


def solve_saved_secondary(anchor, gradients, saved_solution, support):
    """Exactly one CLARABEL call at the original saved target, without auxiliary t."""
    import cvxpy as cp
    import numpy as np
    c=coordinates(anchor,gradients,support)
    target=float(saved_solution['secondary_target'])
    if not math.isfinite(target) or target<=0:raise ValueError('original positive fixed secondary target required')
    report=dict(status='NUMERICALLY_UNRESOLVED',solver='CLARABEL',solver_tolerance=1e-10,max_iter=200,
        arithmetic_resolution=1e-8,certificate_tolerance=1e-6,secondary_target=target,physical_candidates=0,
        automatic_retry=False,primary_solve_count=0,secondary_solve_count=1,
        formulation='min ||u||^2; h@u <= -original_target; ||u||<=1; ||ac+u||<=1',
        gradient_norms=c['norms'].tolist(),span_dimension=len(c['basis']),span_residual=c['span_residual'],
        original_primary_primal_saved=False,original_primary_dual_saved=False)
    u=cp.Variable(len(c['basis']))
    constraints=[c['h']@u<=-target,cp.norm(u,2)<=1,cp.norm(c['ac']+u,2)<=1]
    problem=cp.Problem(cp.Minimize(cp.sum_squares(u)),constraints)
    try:
        problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=200)
    except Exception as exc:
        report.update(secondary_status='solver_exception',reason='fixed-target secondary solver exception',solver_error=repr(exc))
    else:
        report['secondary_status']=problem.status
    report['secondary_primal']=None if u.value is None else np.asarray(u.value).tolist()
    report['secondary_dual']=None if any(x.dual_value is None for x in constraints) else dict(objective=np.asarray(constraints[0].dual_value).tolist(),delta_ball=float(constraints[1].dual_value),control_ball=float(constraints[2].dual_value))
    s=problem.solver_stats
    report['solver_stats']=None if s is None else dict(num_iters=s.num_iters,solve_time=s.solve_time,setup_time=s.setup_time)
    report['secondary_objective']=None if problem.value is None or not math.isfinite(float(problem.value)) else float(problem.value)
    if report['secondary_primal'] is not None and report['secondary_dual'] is not None:
        report['secondary_certificate']=certificate(c,target,report['secondary_primal'],report['secondary_dual'])
    return checked_candidate(anchor,gradients,saved_solution,report,support)
