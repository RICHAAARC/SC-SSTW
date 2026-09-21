"""Frozen terminal-only paired objective diagnostic; no receiver or model dependency."""
import math
import torch
from . import projection_margin as carrier
from .velocity_coefficients import blocks,projections

OBJECTIVES=('hinge','tanh')

def loss(z,directions,codes,message,objective):
    p=projections(z,directions)
    if objective=='hinge':return .5*torch.relu(1-codes[message]*p).square().mean()
    if objective=='tanh':return -(torch.tanh(p)*(codes[message]-codes[1-message])).mean()
    raise ValueError('unknown frozen objective')

def rms(z):return float(z[:,:,1:45].double().square().mean().sqrt())

def metrics(z,directions,codes,message):
    with torch.no_grad():
        p=projections(z,directions);hard=(p.clamp(-1,1)[None]*codes).mean(1);soft=(p.tanh()[None]*codes).mean(1)
        gap=float(hard[message]-hard[1-message])
        return dict(hinge_by_message=[float(.5*torch.relu(1-c*p).square().mean()) for c in codes],
                    hard_clipped_message_scores=hard.tolist(),hard_gap=gap,nominal_rank='CORRECT' if gap>0 else ('WRONG' if gap<0 else 'TIED'),
                    tanh_message_scores=soft.tolist(),tanh_gap=float(soft[message]-soft[1-message]))

def gradient_shares(z,g,directions,codes,message):
    p=projections(z,directions).detach();energy=blocks(g).double().square().sum(1);total=float(energy.sum())
    critical=codes[0]!=codes[1];interior=p.abs()<1;saturated=p.abs()>1;boundary=p.abs()==1
    masks=dict(critical=critical,noncritical=~critical,interior=interior,saturated=saturated,boundary=boundary,
               critical_interior=critical&interior,critical_saturated=critical&saturated,critical_boundary=critical&boundary,
               critical_target_correct_saturated=critical&saturated&(codes[message]*p>0),
               critical_target_wrong_saturated=critical&saturated&(codes[message]*p<0))
    return dict(total_squared_l2=total,groups={name:dict(block_count=int(mask.sum()),squared_l2=float(energy[mask].sum()),
                    squared_l2_fraction=None if total==0 else float(energy[mask].sum())/total) for name,mask in masks.items()},
                meaning='squared gradient L2 shares; overlapping partitions, not all rows summing to one')

def update(z,directions,codes,message,objective,budget):
    if tuple(z.shape)!=carrier.SHAPE or z.dtype!=torch.float32 or not torch.isfinite(z).all():raise ValueError('finite original float32 terminal required')
    if not math.isfinite(budget) or budget<=0:raise ValueError('zero/nonfinite/negative conditional budget')
    with torch.enable_grad():
        leaf=z.detach().double().clone().requires_grad_(True);value=loss(leaf,directions,codes,message,objective)
        g,=torch.autograd.grad(value,leaf)
    g=g.detach();norm=rms(g)
    if not math.isfinite(norm) or norm<=0 or not torch.isfinite(g).all():raise ValueError('zero/nonfinite gradient; no fallback')
    if torch.count_nonzero(g[:,:,0])+torch.count_nonzero(g[:,:,45]):raise ValueError('gradient outside fixed support')
    u=(-budget*g/norm).float();after=z.detach()+u;actual=after-z.detach()
    if not torch.isfinite(after).all():raise ValueError('nonfinite updated latent')
    if not torch.equal(after[:,:,0],z[:,:,0]) or not torch.equal(after[:,:,45],z[:,:,45]):raise ValueError('changed support boundary')
    before=metrics(z,directions,codes,message);end=metrics(after,directions,codes,message)
    row=dict(status='COMPLETE',objective=objective,temperature=1 if objective=='tanh' else None,requested_budget=budget,scale_multiplier=1,
             raw_gradient_support_rms=norm,gradient_shares=gradient_shares(z,g,directions,codes,message),
             requested_control_support_rms=rms(u),actual_control_support_rms=rms(actual),actual_control_global_rms=float(actual.double().square().mean().sqrt()),
             actual_control_peak=float(actual.abs().max()),budget_relative_error=abs(rms(actual)-budget)/budget,
             actual_rounded_control_zero=rms(actual)==0,
             support_boundary_unchanged=True,before=before,after=end,hard_gap_gain=end['hard_gap']-before['hard_gap'],
             tanh_gap_gain=end['tanh_gap']-before['tanh_gap'],own_objective_decrease=float(value.detach())-float(loss(after,directions,codes,message,objective).detach()),
             updated_media_evidence=None)
    return row,actual
