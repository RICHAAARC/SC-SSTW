"""One shared-state step30 input Jacobian. Only that model step has a graph."""
import copy
import time
from contextlib import contextmanager
import torch
from torch.utils.checkpoint import checkpoint
from main.tube_state import grow_temporal_difference as method
from main.tube_state.grow_control_budget import matching_scale
from .grow_control_transfer import cfg, require_scheduler, rms, finite, check_delta

STEP=30
ARMS=('OFF','LOCAL_A','LOCAL_B','JAC_A','JAC_B')

def enable_input_checkpointing(transformer, stats):
    """Use the public diffusers hook; no train mode or parameter gradients."""
    transformer.eval()
    for p in transformer.parameters():p.requires_grad_(False)
    @contextmanager
    def context(key):
        stats[key]=stats.get(key,0)+1
        yield
    def checkpoint_function(module,*args):
        stats['checkpoint_calls']=stats.get('checkpoint_calls',0)+1
        return checkpoint(module,*args,use_reentrant=False,
            context_fn=lambda:(context('checkpoint_forward_contexts'),context('checkpoint_recompute_contexts')))
    transformer.enable_gradient_checkpointing(gradient_checkpointing_func=checkpoint_function)
    if not transformer.is_gradient_checkpointing:raise RuntimeError('official input checkpoint did not enable')


def differentiable_cfg(pipe,z,prompt,negative,dtype,guidance,count):
    hidden=z.to(dtype);t=pipe.scheduler.timesteps[STEP].expand(z.shape[0]);values=[]
    for embedding in (prompt,negative):
        count('transformer',False)
        values.append(pipe.transformer(hidden_states=hidden,timestep=t,encoder_hidden_states=embedding,attention_kwargs=None,return_dict=False)[0])
        count('transformer',True)
    return (values[1]+guidance*(values[0]-values[1])).float()


def direction(pipe,z,v,prompt,negative,dtype,guidance,book,message,kind,count,stats):
    """Exact autograd of this CFG implementation, never a detached-v shortcut for JAC."""
    sigma=float(pipe.scheduler.sigmas[STEP]);cuda=z.is_cuda
    if cuda:torch.cuda.synchronize(z.device);torch.cuda.reset_peak_memory_stats(z.device)
    baseline=torch.cuda.memory_allocated(z.device) if cuda else None
    baseline_reserved=torch.cuda.memory_reserved(z.device) if cuda else None
    start=time.perf_counter();before=dict(stats);stats['last_direction_diagnostics']={}
    try:
        with torch.enable_grad():
            leaf=z.detach().requires_grad_(True)
            prediction=differentiable_cfg(pipe,leaf,prompt,negative,dtype,guidance,count) if kind=='JAC' else v.detach()
            consistency=check_delta(prediction.detach(),v,z,v)
            stats['last_direction_diagnostics']={'forward_consistency':consistency}
            if not consistency['pass']:raise RuntimeError('SAME_FIELD_FORWARD_MISMATCH')
            clean=leaf-sigma*prediction
            loss=method.loss(clean,book,message)
            call='input_vjp' if kind=='JAC' else 'local_gradient'
            count(call,False);gradient=torch.autograd.grad(loss,leaf)[0];count(call,True)
            info=dict(forward_consistency=consistency,loss=float(loss.detach()),gradient_rms=rms(gradient),
                forward_velocity_maxabs=float((prediction.detach()-v).abs().max()),
                prediction_has_graph=prediction.requires_grad,gradient_has_graph=gradient.requires_grad)
            answer=gradient.detach()
        if not finite((answer,loss,prediction)):raise FloatingPointError('nonfinite loss, prediction or direction')
        stats['last_direction_diagnostics']=info
        return answer,info
    finally:
        if cuda:torch.cuda.synchronize(z.device)
        stats['last_direction_resources']=dict(seconds=time.perf_counter()-start,
            baseline_allocated_bytes=baseline,baseline_reserved_bytes=baseline_reserved,
            after_allocated_bytes=torch.cuda.memory_allocated(z.device) if cuda else None,
            after_reserved_bytes=torch.cuda.memory_reserved(z.device) if cuda else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(z.device) if cuda else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(z.device) if cuda else None,
            checkpoint_delta={k:stats.get(k,0)-before.get(k,0) for k in ('checkpoint_calls','checkpoint_forward_contexts','checkpoint_recompute_contexts')},
            after_memory_meaning='measured before direction function exits; not graph-free release measurement',
            checkpoint_counter_meaning='block context entries, not full Transformer calls or successful recomputation count')


@torch.no_grad()
def prefix(pipe,initial,prompt,negative,dtype,guidance,count,record):
    require_scheduler(pipe.scheduler);z=initial.detach().float()
    for index in range(STEP):
        t=pipe.scheduler.timesteps[index];v=cfg(pipe,z,prompt,negative,dtype,guidance,index,count)
        count('scheduler_step',False);z=pipe.scheduler.step(v,t,z,return_dict=False)[0];count('scheduler_step',True)
        if not finite(z):raise FloatingPointError('nonfinite shared prefix')
        record(dict(index=index,controlled=False))
    return z.detach()


@torch.no_grad()
def one_step(pipe,z,v,book,message,gradient,count,off_next=None,target_energy=None):
    sigma=float(pipe.scheduler.sigmas[STEP]);t=pipe.scheduler.timesteps[STEP]
    if not sigma>0:raise ValueError('positive step30 sigma required')
    row=dict(index=STEP,sigma=sigma,controlled=message is not None)
    u=torch.zeros_like(z) if gradient is None else -method.ETA*gradient
    controlled=v-u/sigma
    if target_energy is not None:
        shadow=copy.deepcopy(pipe.scheduler)
        count('unit_response_probe_step',False);unit=shadow.step(controlled,t,z.clone(),return_dict=False)[0];count('unit_response_probe_step',True)
        match=matching_scale(target_energy,rms(unit-off_next)**2);row['energy_matching']=match
        if match['scale'] is None:raise RuntimeError('unmatchable single-step native response: '+str(match))
        u=u*match['scale'];controlled=v-u/sigma
    if not finite((u,controlled)):raise FloatingPointError('nonfinite matched update')
    count('scheduler_step',False);next_z=pipe.scheduler.step(controlled,t,z,return_dict=False)[0];count('scheduler_step',True)
    if not finite(next_z):raise FloatingPointError('nonfinite native response')
    row.update(control_rms=rms(u),delta_velocity_rms=rms(controlled-v),
        control_induced_delta_rms=rms(next_z-off_next) if off_next is not None else 0.,
        loss_before=float(method.loss(z-sigma*v,book,message)) if message is not None else None,
        loss_after_local=float(method.loss(z-sigma*controlled,book,message)) if message is not None else None)
    row['actual_response_energy']=row['control_induced_delta_rms']**2
    if target_energy is not None:row['energy_matching'].update(actual_energy=row['actual_response_energy'],
        absolute_error=abs(row['actual_response_energy']-target_energy),
        relative_error=abs(row['actual_response_energy']-target_energy)/target_energy if target_energy else None)
    return next_z.detach(),row


@torch.no_grad()
def tail(pipe,z,prompt,negative,dtype,guidance,count,record):
    for index in range(STEP+1,50):
        t=pipe.scheduler.timesteps[index];v=cfg(pipe,z,prompt,negative,dtype,guidance,index,count)
        count('scheduler_step',False);z=pipe.scheduler.step(v,t,z,return_dict=False)[0];count('scheduler_step',True)
        if not finite(z):raise FloatingPointError('nonfinite tail')
        record(dict(index=index,controlled=False))
    return z.detach()
