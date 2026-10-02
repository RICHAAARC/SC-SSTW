"""Writer-only full-family margin objective; the blind receiver is unchanged."""
from __future__ import annotations
import numpy as np
from main.tube_state import video_overlap_zero_mean_state as state

R=44

def objective_spec(key):
    f=state.family(key,R,np.ones((R,4),bool))
    true_class=int(f['membership'][f['valid_indices'].index(0)])
    wrong=[i for i in range(len(f['classes'])) if i!=true_class]
    means=f['means']; truth=means[true_class]
    distances=np.mean((means[wrong]-truth)**2,axis=1)
    if len(wrong)!=173 or np.any(distances<=0):
        raise ValueError('fixed full-support R44 distinct-class family changed')
    return dict(means=means,true_class=true_class,wrong_classes=wrong,
                global_margin=float(distances.min()/2),local_margins=distances/2,
                valid_catalog_indices=f['valid_indices'],classes=f['classes'],
                support='all 44 rows and all 1408 projection coordinates',
                margin_semantics='half of public ideal-template pair separation; writer only')


def project(normalized,key):
    import torch
    if tuple(normalized.shape)!=(1,16,46,40,64):raise ValueError('full normalized video required')
    U=torch.as_tensor(state.bases(key),device=normalized.device,dtype=torch.float64)
    return torch.stack([(normalized[0,4,1:45,h0:h1,w0:w1].double().reshape(R,16)@U[i]).reshape(R,4,2)
                        for i,(h0,h1,w0,w1) in enumerate(state.PUBLIC.blocks)],dim=1)


def margin_loss(projection,key,*,objective="composite"):
    """All competitors, including starts/repeats/skips; known truth is writer-only.

    Each local pair term cancels identically at equal-template coordinates, so
    it measures the differing temporal positions without dropping any receiver
    observation. All terms use the original M=1408 denominator.
    """
    import torch
    if objective not in ('composite','worst_only'):raise ValueError('unknown margin objective')
    spec=objective_spec(key);q=projection.double().reshape(-1)
    if q.numel()!=1408 or not bool(q.isfinite().all()):raise ValueError('finite R44 projection required')
    means=torch.as_tensor(spec['means'],device=q.device,dtype=q.dtype)
    true=means[spec['true_class']];wrong=means[spec['wrong_classes']]
    # Algebraically J_wrong-J_true, with shared observation energy cancelled.
    margins=((wrong-true)*(wrong+true-2*q)).mean(dim=1)
    target=torch.as_tensor(spec['local_margins'],device=q.device,dtype=q.dtype)
    global_term=torch.relu(q.new_tensor(spec['global_margin'])-margins.min())
    local_terms=torch.relu(target-margins)
    loss=global_term+local_terms.mean() if objective=='composite' else global_term
    return loss,dict(delta=float(margins.min().detach()),loss=float(loss.detach()),
                     global_term=float(global_term.detach()),local_mean=float(local_terms.mean().detach()),
                     global_margin=spec['global_margin'],local_margins=spec['local_margins'].tolist(),
                     margins=margins.detach().cpu().tolist(),wrong_classes=spec['wrong_classes'],
                     distinct_wrong_classes=len(spec['wrong_classes']),local_active=int((local_terms>0).sum()))


def constrained_step(gradient,key,eta=174.,cap=1.):
    """One descent step in the original 1392-dimensional pilot write space."""
    import torch
    if tuple(gradient.shape)!=(1,16,46,40,64) or not bool(gradient.isfinite().all()):
        raise ValueError('full finite terminal gradient required')
    if eta<=0 or cap<=0:raise ValueError('positive step/cap required')
    U=torch.as_tensor(state.bases(key),device=gradient.device,dtype=torch.float64)
    active=torch.as_tensor(state.composite_signs(key)!=0,device=gradient.device)
    supported=torch.zeros_like(gradient)
    for i,(h0,h1,w0,w1) in enumerate(state.PUBLIC.blocks):
        values=gradient[0,4,1:,h0:h1,w0:w1].double().reshape(45,16)@U[i]
        values=values*active[:,i].reshape(45,8)
        supported[0,4,1:,h0:h1,w0:w1]=(values@U[i].T).reshape(45,4,4).to(gradient.dtype)
    raw=-float(eta)*supported;norm=float(raw.double().norm())
    if norm==0:raise ValueError('zero supported gradient; no alternate direction or retry')
    scale=min(1.,float(cap)/norm);step=raw*scale
    return step,dict(eta=float(eta),cap=float(cap),raw_l2=norm,scale=scale,actual_l2=float(step.double().norm()),
                     gradient_dot_step=float((gradient.double()*step.double()).sum()),
                     support='original channel4/four blocks/regular1..45/1392 active coefficients',
                     no_receiver_change=True,no_budget_scan=True)
