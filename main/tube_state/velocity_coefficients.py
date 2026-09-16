"""Torch form of the existing normalized 1760-block carrier and coefficient norm."""
from __future__ import annotations
import math
import torch
from .projection_margin import SHAPE, SUPPORT_COUNT

ACTIVE = (44, 45, 46)
COEFFICIENT_SHAPE = (3, SUPPORT_COUNT)
ROUNDING_RESERVE = 1e-3
PROBE_FRACTION = 0.1


def blocks(z):
    if tuple(z.shape) != SHAPE:
        raise ValueError(f'normalized carrier shape must be {SHAPE}')
    x = z[0, :, 1:45].permute(1,0,2,3)
    return x.reshape(11,4,16,10,4,16,4).permute(0,3,5,1,2,4,6).reshape(1760,1024)


def scatter(a, directions):
    """No VAE scale: coefficients multiply original unit directions directly."""
    x = (a[:,None]*directions).reshape(11,10,16,4,16,4,4)
    support = x.permute(0,3,4,1,5,2,6).reshape(44,16,40,64).permute(1,0,2,3)[None]
    padding = support.new_zeros((1,16,1,40,64))
    return torch.cat((padding,support,padding),dim=2)


def projections(z, directions):
    # Match the receiver's float64 dot accumulation; retain the real input graph.
    return (blocks(z).double()*directions.double()).sum(dim=1)


def terminal_loss(z, directions, codes):
    p = projections(z,directions)
    return torch.relu(1.-codes*p).square().mean()


def direction_and_amplitude(gradient, directions, sigmas, responses, radius):
    """One symmetric radius from the coefficient Euclidean direction and budgets.

    h_i is the actual UniPC scalar derivative of step output w.r.t. current
    velocity at fixed history/sample; it is NOT dF/da or a propagation estimate.
    """
    g = gradient.detach().double()
    if tuple(g.shape) != COEFFICIENT_SHAPE or not bool(torch.isfinite(g).all()):
        raise ValueError('invalid/nonfinite coefficient gradient')
    norm = float(torch.linalg.vector_norm(g))
    if norm == 0.:
        raise ValueError('zero coefficient gradient; no direction')
    # Store the exact FP32 direction that the sampler will actually use.
    q = (-g/norm).float()
    q_norm = float(torch.linalg.vector_norm(q.double()))
    if abs(q_norm-1.) > 1e-6:
        raise ValueError('coefficient direction is not unit Euclidean norm')
    norm2 = directions.detach().double().square().sum(dim=1)
    s = ((q.double().square()*norm2[None]).sum(dim=1)/(1760*1024)).sqrt()
    ku = s*torch.as_tensor(sigmas,dtype=torch.float64,device=s.device).abs()
    kd = s*torch.as_tensor(responses,dtype=torch.float64,device=s.device).abs()
    candidates = []
    for name,k in (('U',ku),('D',kd)):
        for index,value in zip(ACTIVE,k.tolist()):
            if value>0: candidates.append((f'{name}/step{index}',radius/(3*value)))
        total = float(k.sum())
        if total>0: candidates.append((f'{name}/sum',radius/total))
    if not candidates or not math.isfinite(radius) or radius<=0:
        raise ValueError('no positive finite OFF-derived budget radius')
    limiting, maximum = min(candidates,key=lambda pair:pair[1])
    epsilon = PROBE_FRACTION*(1.-ROUNDING_RESERVE)*maximum
    if not math.isfinite(epsilon) or epsilon<=0:
        raise ValueError('invalid derived epsilon')
    return q, {'gradient_l2':norm,'q_l2':q_norm,'epsilon':epsilon,
        'epsilon_cap':maximum,'rho':PROBE_FRACTION,'rounding_reserve_fraction':ROUNDING_RESERVE,
        'limiting_constraint':limiting,'all_upper_bounds':dict(candidates),
        'scatter_rms_per_unit_epsilon':s.tolist(),'U_rms_per_unit_epsilon':ku.tolist(),
        'D_rms_per_unit_epsilon':kd.tolist(),'sigmas':list(sigmas),'responses':list(responses),
        'AD_directional_derivative':float((g*q.double()).sum()),
        'direction_rule':'q=-g/||g||_2 in 5280 coefficient coordinates; stored in FP32',
        'amplitude_rule':'epsilon_probe=0.1*(1-0.001)*epsilon_cap; cap=min(per-step R/(3*k), cumulative R/sum(k)) for k=|sigma|s and |h|s',
        'meaning':'rho=0.1 is a fixed engineering probe convention, not a guarantee of locality or optimality'}
