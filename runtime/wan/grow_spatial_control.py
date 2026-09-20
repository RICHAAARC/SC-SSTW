"""Fixed late-window local control; shadows measure response, never update live history."""
import copy
import math
import torch
from main.tube_state import grow_frequency as method
from main.tube_state.grow_control_budget import matching_scale
from .grow_control_transfer import cfg,require_scheduler,rms,finite,check_delta

CONTROL_INDICES=tuple(range(30,50))


@torch.no_grad()
def pulse(z,v,sigma,book,message,count,loss_fn=None):
    objective=method.loss if loss_fn is None else loss_fn
    clean=z-sigma*v
    with torch.enable_grad():
        leaf=clean.detach().requires_grad_(True);loss=objective(leaf,book,message)
        count('local_gradient',False);gradient=torch.autograd.grad(loss,leaf)[0];count('local_gradient',True)
    u=-method.ETA*gradient.detach();controlled=v-u/sigma;after=z-sigma*controlled
    if not finite((u,controlled,after,loss)):raise FloatingPointError('nonfinite local spatial control')
    return u,controlled,after,dict(controlled=True,loss_before=float(loss.detach()),loss_after=float(objective(after,book,message)),
        loss_ratio=float(objective(after,book,message)/loss.detach()) if float(loss.detach()) else None,
        algebra=check_delta(after-clean,u,z,clean,after),u_rms=rms(u))


def observation(z,book,loss_fn=None):
    row=method.read(z,book)['aggregate']
    means=z.new_tensor(row['coefficient_mean_diagnostic'],dtype=torch.float64)
    row['candidate_direction_scores']=[float((means*means.new_tensor(p)).mean()) for p in book['payloads']]
    row['candidate_losses']=[float(method.loss(z,book,m)) for m in (0,1)]
    if loss_fn is not None:row['writer_objective_candidate_losses']=[float(loss_fn(z,book,m)) for m in (0,1)]
    return row


@torch.no_grad()
def generate(pipe,initial,prompt,negative,dtype,guidance,book,message,count,record,artifact,*,control_indices,loss_fn=None,target_energy=None):
    if loss_fn is not None:raise ValueError('paired protocol fixes original MSE objective')
    if target_energy is not None and tuple(control_indices)!=(49,):raise ValueError('energy matching only allowed for last-only arm')
    if tuple(control_indices) not in (CONTROL_INDICES,(49,)):raise ValueError('fixed multi or last-only window required')
    if tuple(control_indices)==(49,) and message is not None and target_energy is None:raise ValueError('paired energy target required')
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
            u,controlled,after,local=pulse(z,v,sigma,book,message,count,loss_fn=loss_fn)
            row.update(local,loss_after_local=local['loss_after'],effective_clean_control_rms=rms(sigma*(controlled-v)),
                delta_velocity_rms=rms(controlled-v),control_rms=rms(u),
                clean_before=observation(clean,book,loss_fn=loss_fn),clean_after=observation(after,book,loss_fn=loss_fn))
            count('response_probe_step',False)
            off=shadow.step(v,timestep,z.clone(),return_dict=False)[0]
            count('response_probe_step',True)
            if target_energy is not None:
                unit_shadow=copy.deepcopy(pipe.scheduler)
                count('unit_response_probe_step',False)
                unit_next=unit_shadow.step(controlled,timestep,z.clone(),return_dict=False)[0]
                count('unit_response_probe_step',True)
                match=matching_scale(target_energy,rms(unit_next-off)**2)
                if match['scale'] is None:
                    row.update(unit_control_local=local,energy_matching=match,step_status='NOT_EXECUTED_UNMATCHABLE');record(row)
                    raise RuntimeError(str(match))
                controlled=v+(controlled-v)*match['scale'];u=u*match['scale'];after=z-sigma*controlled
                row.update(unit_control_local=local,energy_matching=match,effective_clean_control_rms=rms(sigma*(controlled-v)),
                    delta_velocity_rms=rms(controlled-v),control_rms=rms(u),clean_after=observation(after,book),
                    loss_after_local=float(method.loss(after,book,message)),loss_after=float(method.loss(after,book,message)),
                    loss_ratio=float(method.loss(after,book,message))/local['loss_before'] if local['loss_before'] else None,u_rms=rms(u),
                    algebra=check_delta(after-clean,u,z,clean,after))
                if not finite((controlled,after,u)):raise FloatingPointError('nonfinite matched control')
                del unit_shadow,unit_next
            del shadow,u
        if index==49:
            for name,value in [('before_state',z),('predicted_clean_before',clean),('predicted_clean_after',after)]:
                artifact('last49_'+name,value)
            row.update(last_before=observation(z,book,loss_fn=loss_fn),last_clean_before=observation(clean,book,loss_fn=loss_fn),last_clean_after=observation(after,book,loss_fn=loss_fn))
        count('scheduler_step',False)
        next_z=pipe.scheduler.step(controlled,timestep,z,return_dict=False)[0]
        count('scheduler_step',True)
        row['whole_transition_rms']=rms(next_z-z)
        if active:
            row['control_induced_delta_rms']=rms(next_z-off)
            if target_energy is not None:
                actual=row['control_induced_delta_rms']**2
                row['energy_matching'].update(actual_energy=actual,absolute_error=abs(actual-target_energy),relative_error=abs(actual-target_energy)/target_energy if target_energy else None)
            row['control_induced_delta_readout']=observation(next_z-off,book,loss_fn=loss_fn)
            row['actual_after']=observation(next_z,book,loss_fn=loss_fn)
            del off
        if index==49:
            artifact('last49_actual_after',next_z)
            row.update(last_after=observation(next_z,book,loss_fn=loss_fn),terminal_vs_last_controlled_clean_rms=rms(next_z-after),
                terminal_vs_last_controlled_clean_maxabs=float((next_z-after).abs().max()),
                terminal_lower_order=pipe.scheduler.this_order)
        if pipe.scheduler.step_index!=index+1 or not finite(next_z):raise RuntimeError('invalid native step')
        record(row);z=next_z
    return z.detach()
