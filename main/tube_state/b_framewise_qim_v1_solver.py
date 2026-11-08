"""Matrix-free lexicographic solver; finite KKT tolerance is not a proof of feasibility."""
from dataclasses import dataclass
import math
import time
import torch
from main.tube_state import b_framewise_qim_v1 as protocol


@dataclass(frozen=True)
class Limits:
    iterations: int = 4000
    check_every: int = 20
    seconds: float = 7200
    memory_bytes: int = 16*1024**3
    rgb_tolerance: float = 1e-7
    q_tolerance: float = 1e-3
    kkt_tolerance: float = 1e-7


def project_budget(value, mask, rms=protocol.RMS, linf=protocol.LINF):
    """Exact box/ball intersection projection to FP64 bisection precision."""
    v = torch.where(mask, value, 0).flatten(1)
    radii = mask.flatten(1).sum(1).double().sqrt()*rms
    clipped = v.clamp(-linf, linf)
    if bool((clipped.norm(dim=1) <= radii).all()): return clipped.reshape_as(value)
    lo = torch.ones_like(radii)
    hi = torch.maximum(lo, v.norm(dim=1)/radii.clamp_min(1e-300))
    for _ in range(64):
        mid = (lo+hi)/2
        inside = (v/mid[:, None]).clamp(-linf, linf).norm(dim=1) <= radii
        hi = torch.where(inside, mid, hi); lo = torch.where(inside, lo, mid)
    return (v/hi[:, None]).clamp(-linf, linf).reshape_as(value)


def budget_metrics(r, mask, rms=protocol.RMS, linf=protocol.LINF):
    counts = mask.flatten(1).sum(1)
    norm = torch.where(mask, r, 0).flatten(1).norm(dim=1)
    values = norm/counts.clamp_min(1).sqrt()
    outside = torch.where(mask, 0, r).abs().max()
    maximum = r.abs().max()
    return dict(rms=values.tolist(), linf=float(maximum), outside=float(outside),
                excess=max(0., float((values-rms).max()), float(maximum)-linf, float(outside)),
                support_rgb_components=counts.tolist())


def solve(operator, d, target_mask, budget_mask, *, limits=Limits(),
          measure=protocol.project, transpose=protocol.project_transpose,
          measurement_bound=math.sqrt(sum(x*x for x in protocol.LUMA)),
          rms=protocol.RMS, linf=protocol.LINF, progress=None):
    """A from coefficients to packed RGB, all inputs CPU FP64. No fallback/retry."""
    if d.dtype != torch.float64 or d.device.type != 'cpu' or not bool(torch.isfinite(d).all()):
        raise ValueError('finite CPU FP64 target displacement required')
    if target_mask.shape != d.shape or budget_mask.shape != operator.output_shape:
        raise ValueError('target/support dimensions do not match operator')
    if any(x <= 0 for x in (limits.iterations, limits.check_every, limits.seconds, limits.memory_bytes)):
        raise ValueError('positive solver limits required')
    required = 8*(16*math.prod(operator.output_shape)+8*math.prod(operator.input_shape)+5*d.numel())
    result = dict(status='NUMERICALLY_UNRESOLVED', estimated_bytes=required, iterations=0,
                  metrics=None, first_objective=None, second_objective=None)
    if required > limits.memory_bytes: return dict(result, status='RESOURCE_UNRESOLVED', reason='solver live-array bound')
    if operator.bound == 0 and bool((d[target_mask] != 0).any()):
        return dict(result, status='CERTIFIED_INFEASIBLE_ZERO_RESPONSE', reason='A=0, nonzero equality RHS')
    deadline = time.monotonic()+limits.seconds
    a = torch.zeros(operator.input_shape, dtype=torch.float64)
    r = torch.zeros(operator.output_shape, dtype=torch.float64)
    y, z = torch.zeros_like(r), torch.zeros_like(d)
    abar, rbar = a.clone(), r.clone()
    step = .99/math.sqrt(operator.bound**2+1+measurement_bound**2)
    try:
        for iteration in range(1, limits.iterations+1):
            if time.monotonic() > deadline:
                result.update(status='RESOURCE_UNRESOLVED', reason='solver deadline'); break
            y += step*(operator.forward(abar)-rbar)
            z += step*torch.where(target_mask, measure(rbar)-d, 0)
            anew = a-step*operator.adjoint(y)
            rnew = project_budget((r+step*y-step*transpose(z))/(1+step), budget_mask, rms, linf)
            abar, rbar = 2*anew-a, 2*rnew-r
            a, r = anew, rnew
            result['iterations'] = iteration
            if iteration % limits.check_every and iteration != limits.iterations: continue
            actual = operator.forward(a)
            at_y = operator.adjoint(y)
            # prox_f(r+y-P^T z) = projection_B((r+y-P^T z)/2).
            fixed = r-project_budget((r+y-transpose(z))/2, budget_mask, rms, linf)
            metrics = dict(split=float((actual-r).abs().max()),
                           target=float(torch.where(target_mask, measure(actual)-d, 0).abs().max())*255,
                           a_stationarity=float(at_y.norm())/max(1., float(y.norm())),
                           r_stationarity=float(fixed.norm())/max(1., float(r.norm())),
                           budget=budget_metrics(actual, budget_mask, rms, linf))
            result.update(metrics=metrics, first_objective=float(actual.square().sum()), second_objective=float(a.square().sum()))
            if not all(math.isfinite(x) for x in (metrics['split'], metrics['target'], metrics['a_stationarity'], metrics['r_stationarity'], result['first_objective'], result['second_objective'])):
                result.update(reason='nonfinite iterate'); break
            if progress: progress(dict(result))
            if (metrics['split'] <= limits.rgb_tolerance and metrics['target'] <= limits.q_tolerance
                    and metrics['budget']['excess'] <= limits.rgb_tolerance
                    and max(metrics['a_stationarity'], metrics['r_stationarity']) <= limits.kkt_tolerance):
                result.update(status='SOLVED_APPROXIMATE_KKT', reason='fixed numeric tolerances'); break
        else: result['reason'] = 'iteration cap'
    except (MemoryError, RuntimeError) as error:
        result.update(status='RESOURCE_UNRESOLVED' if isinstance(error, MemoryError) or 'memory' in str(error).lower() else 'NUMERICALLY_UNRESOLVED',
                      reason=f'{type(error).__name__}: {error}')
    result['coefficients'] = a
    return result
