"""Fixed late-window local control; shadows measure response, never update live history."""
import copy
import math
import torch
from main.tube_state import grow_temporal_difference as method
from .grow_control_transfer import cfg,require_scheduler,rms,finite,check_delta

CONTROL_INDICES=tuple(range(30,50))


@torch.no_grad()
def pulse(z,v,sigma,book,message,count):
    clean=z-sigma*v
    with torch.enable_grad():
        leaf=clean.detach().requires_grad_(True);loss=method.loss(leaf,book,message)
        count('local_gradient',False);gradient=torch.autograd.grad(loss,leaf)[0];count('local_gradient',True)
    u=-method.ETA*gradient.detach();controlled=v-u/sigma;after=z-sigma*controlled
    if not finite((u,controlled,after,loss)):raise FloatingPointError('nonfinite local difference control')
    return u,controlled,after,dict(controlled=True,loss_before=float(loss.detach()),loss_after=float(method.loss(after,book,message)),
        loss_ratio=float(method.loss(after,book,message)/loss.detach()) if float(loss.detach()) else None,
        algebra=check_delta(after-clean,u,z,clean,after),u_rms=rms(u))


def observation(z,book):
    row=method.read(z,book)['aggregate']
    means=z.new_tensor(row['coefficient_mean_diagnostic'],dtype=torch.float64)
    row['candidate_direction_scores']=[float((means*means.new_tensor(p)).mean()) for p in book['payloads']]
    row['candidate_losses']=[float(method.loss(z,book,m)) for m in (0,1)]
    return row


@torch.no_grad()
def generate(pipe,initial,prompt,negative,dtype,guidance,book,message,count,record,artifact,*,control_indices):
    if tuple(control_indices)!=CONTROL_INDICES:raise ValueError('runtime requires fixed 30..49 indices')
    require_scheduler(pipe.scheduler)
    if tuple(initial.shape)!=method.SHAPE:raise ValueError('fixed latent shape required')
    if not all(math.isfinite(float(pipe.scheduler.sigmas[i])) and float(pipe.scheduler.sigmas[i])>0 for i in control_indices):
        raise ValueError('all controlled sigmas must be finite and positive; no skipping')
    z=initial.detach().float()
    for index,timestep in enumerate(pipe.scheduler.timesteps):
        sigma=float(pipe.scheduler.sigmas[index]);v=cfg(pipe,z,prompt,negative,dtype,guidance,index,count)
        clean=z-sigma*v;active=message is not None and index in control_indices
        row=dict(index=index,sigma=sigma,next_sigma=float(pipe.scheduler.sigmas[index+1]),controlled=active)
        controlled=v;after=clean
        if active:
            shadow=copy.deepcopy(pipe.scheduler)
            u,controlled,after,local=pulse(z,v,sigma,book,message,count)
            row.update(local,loss_after_local=local['loss_after'],effective_clean_control_rms=rms(sigma*(controlled-v)),
                delta_velocity_rms=rms(controlled-v),control_rms=rms(u),
                clean_before=observation(clean,book),clean_after=observation(after,book))
            count('response_probe_step',False)
            off=shadow.step(v,timestep,z.clone(),return_dict=False)[0]
            count('response_probe_step',True)
            del shadow,u
        if index==49:
            for name,value in [('before_state',z),('predicted_clean_before',clean),('predicted_clean_after',after)]:
                artifact('last49_'+name,value)
            row.update(last_before=observation(z,book),last_clean_before=observation(clean,book),last_clean_after=observation(after,book))
        count('scheduler_step',False)
        next_z=pipe.scheduler.step(controlled,timestep,z,return_dict=False)[0]
        count('scheduler_step',True)
        row['whole_transition_rms']=rms(next_z-z)
        if active:
            row['control_induced_delta_rms']=rms(next_z-off)
            row['control_induced_delta_readout']=observation(next_z-off,book)
            row['actual_after']=observation(next_z,book)
            del off
        if index==49:
            artifact('last49_actual_after',next_z)
            row.update(last_after=observation(next_z,book),terminal_vs_last_controlled_clean_rms=rms(next_z-after),
                terminal_vs_last_controlled_clean_maxabs=float((next_z-after).abs().max()),
                terminal_lower_order=pipe.scheduler.this_order)
        if pipe.scheduler.step_index!=index+1 or not finite(next_z):raise RuntimeError('invalid native step')
        record(row);z=next_z
    return z.detach()
