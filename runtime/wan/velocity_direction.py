"""Differentiable Wan/UniPC tail with pure-Transformer checkpoint replay."""
from __future__ import annotations
import copy
import torch
from torch.utils.checkpoint import checkpoint
from main.tube_state.velocity_coefficients import ACTIVE, scatter
from .flow_step import measures


def detached_tree(value):
    if torch.is_tensor(value): return value.detach().clone()
    if isinstance(value,list): return [detached_tree(x) for x in value]
    if isinstance(value,tuple): return tuple(detached_tree(x) for x in value)
    if isinstance(value,dict): return type(value)({k:detached_tree(v) for k,v in value.items()})
    return copy.deepcopy(value)


def detached_scheduler(scheduler):
    """Snapshot for a new independent run/monitor, never a replacement live graph."""
    result=copy.copy(scheduler)
    result.__dict__={k:detached_tree(v) for k,v in scheduler.__dict__.items()}
    return result


def require_affine_flow(scheduler):
    if (type(scheduler).__name__ != 'UniPCMultistepScheduler' or
        scheduler.config.prediction_type != 'flow_prediction' or scheduler.config.thresholding or
        not scheduler.predict_x0 or scheduler.solver_p is not None):
        raise ValueError('direction experiment requires native non-thresholded UniPC flow_prediction')


@torch.no_grad()
def response_coefficients(snapshot,count):
    """Exact affine scalar response from the real scheduler, not model probing.

    UniPC is coordinatewise affine in sample/history/velocity. Null their values
    but preserve cursor, order, corrector history presence and the real schedule.
    step(v=1)-step(v=0) therefore gives its current-velocity coefficient h_i.
    """
    require_affine_flow(snapshot)
    scheduler=detached_scheduler(snapshot)
    scheduler.sigmas=scheduler.sigmas.cpu()
    scheduler.timesteps=scheduler.timesteps.cpu()
    scheduler.timestep_list=[t.cpu() if torch.is_tensor(t) else t for t in scheduler.timestep_list]
    zero=torch.zeros((1,1),dtype=torch.float64)
    scheduler.model_outputs=[None if x is None else zero.clone() for x in scheduler.model_outputs]
    scheduler.last_sample=None if scheduler.last_sample is None else zero.clone()
    rows=[]
    for index in ACTIVE:
        if scheduler.step_index != index: raise ValueError('response cursor differs from active index')
        timestep=scheduler.timesteps[index]
        base,perturbed=detached_scheduler(scheduler),detached_scheduler(scheduler)
        count('response_probe_step',False)
        y0=base.step(zero,timestep,zero,return_dict=False)[0]
        count('response_probe_step',True)
        count('response_probe_step',False)
        y1=perturbed.step(torch.ones_like(zero),timestep,zero,return_dict=False)[0]
        count('response_probe_step',True)
        h=float((y1-y0).item())
        if not torch.isfinite(y1-y0).all(): raise FloatingPointError('nonfinite UniPC response')
        rows.append({'index':index,'sigma':float(scheduler.sigmas[index]),'h':h,
                     'order':scheduler.this_order,'meaning':'fixed-history current-velocity response only'})
        scheduler=base
    return rows


def transformer_output(transformer,hidden,timestep,embedding,count,use_checkpoint):
    invocation=0
    def pure_forward(x,t,e):
        nonlocal invocation
        kind='transformer' if invocation==0 else 'transformer_replay'
        invocation+=1
        count(kind,False)
        try:
            out=transformer(hidden_states=x,timestep=t,encoder_hidden_states=e,
                            attention_kwargs=None,return_dict=False)[0]
        except Exception as exc:
            # Successful partial replay: non-reentrant checkpoint stops as soon
            # as all tensors needed by autograd have been recomputed.
            if kind=='transformer_replay' and type(exc).__name__=='_StopRecomputationError':
                count(kind,True)
            raise
        count(kind,True)
        return out
    if use_checkpoint and torch.is_grad_enabled() and hidden.requires_grad:
        return checkpoint(pure_forward,hidden,timestep,embedding,use_reentrant=False,preserve_rng_state=True)
    return pure_forward(hidden,timestep,embedding)


def tail(pipe,snapshot,prefix,prompt,negative,input_dtype,guidance_scale,
         coefficients,directions,radius,count,record_step,*,use_checkpoint=True,responses=None):
    """F(a): real six-step trajectory; live model_outputs/last_sample keep graph."""
    scheduler=detached_scheduler(snapshot)
    require_affine_flow(scheduler)
    sample=prefix.detach().clone()
    totals={'U':0.,'D':0.}
    for index in range(44,50):
        timestep=scheduler.timesteps[index]
        hidden=sample.to(input_dtype)
        time=timestep.expand(sample.shape[0])
        conditional=transformer_output(pipe.transformer,hidden,time,prompt,count,use_checkpoint)
        unconditional=transformer_output(pipe.transformer,hidden,time,negative,count,use_checkpoint)
        raw_cfg=unconditional+guidance_scale*(conditional-unconditional)
        velocity=raw_cfg.float()
        if not bool(torch.isfinite(velocity).all()): raise FloatingPointError('nonfinite CFG velocity')
        if index in ACTIVE:
            delta=scatter(coefficients[index-44],directions)
            controlled_velocity=velocity+delta
            # Monitoring is isolated and detached; only this live graph reaches
            # the real step. There is no clipping, projection, or rescaling.
            with torch.no_grad():
                baseline=detached_scheduler(scheduler)
                controlled=detached_scheduler(scheduler)
                count('shadow_step',False)
                without=baseline.step(velocity.detach(),timestep,sample.detach(),return_dict=False)[0]
                count('shadow_step',True)
                count('shadow_step',False)
                shadow=controlled.step(controlled_velocity.detach(),timestep,sample.detach(),return_dict=False)[0]
                count('shadow_step',True)
                sigma=float(scheduler.sigmas[scheduler.step_index])
            count('scheduler_step',False)
            updated=scheduler.step(controlled_velocity,timestep,sample,return_dict=False)[0]
            count('scheduler_step',True)
            with torch.no_grad():
                effective=controlled_velocity.detach()-velocity.detach()
                U=-sigma*effective
                D=updated.detach()-without
                um,dm=measures(U),measures(D)
                totals['U']+=um['support_rms'];totals['D']+=dm['support_rms']
                requested=delta.detach()
                support_requested=requested[:,:,1:45]
                support_effective=effective[:,:,1:45]
                nonzero=int((support_requested!=0).sum())
                swallowed=int(((support_requested!=0)&(support_effective==0)).sum())
                row={'index':index,'sigma':sigma,'U':um,'D':dm,
                     'requested_delta_velocity':measures(requested),'effective_delta_velocity':measures(effective),
                     'requested_U':measures(-sigma*requested),
                     'addition_rounding_error':measures(effective-requested),
                     'nonzero_requested_coordinates':nonzero,'swallowed_coordinates':swallowed,
                     'swallowed_fraction':swallowed/nonzero if nonzero else None,
                     'sum_rms':dict(totals),'radius':radius,
                     'step_within_budget':None if radius is None else um['support_rms']<=radius/3 and dm['support_rms']<=radius/3,
                     'sum_within_budget':None if radius is None else totals['U']<=radius and totals['D']<=radius,
                     'live_shadow_error':measures(updated.detach()-shadow),
                     'cfg_dtype':str(raw_cfg.dtype),'scheduler_dtype':str(velocity.dtype),
                     'U_meaning':'frozen-sample/velocity predicted endpoint budget quantity, not true endpoint change'}
                if responses is not None:
                    h=responses[index-44]['h']
                    residual=measures(D-h*effective)
                    tolerance=2e-6*max(1.,measures(sample.detach())['support_rms'])
                    row.update(response_h=h,response_residual=residual,response_roundoff_tolerance=tolerance,
                               response_valid=residual['support_rms']<=tolerance)
                record_step(row,{'U':U.cpu(),'D':D.cpu(),'requested_delta_velocity':requested.cpu(),
                                 'effective_delta_velocity':effective.cpu()})
            del baseline,controlled,without,shadow,U,D,effective,requested,support_requested,support_effective
            sample=updated
        else:
            count('scheduler_step',False)
            sample=scheduler.step(velocity,timestep,sample,return_dict=False)[0]
            count('scheduler_step',True)
        if scheduler.step_index!=index+1: raise RuntimeError('live UniPC cursor changed unexpectedly')
        if not bool(torch.isfinite(sample).all()): raise FloatingPointError('nonfinite live terminal trajectory')
    return sample  # Do not detach: loss -> all six real solver/model steps -> a.
