"""Fixed full-margin early steps and optional final compensation; native histories stay live."""
import copy
import numpy as np
import torch
from main.tube_state import projection_margin as carrier,flow_control,terminal_guidance as method
from .flow_generation import continue_steps
from .flow_step import measures
from .tube_terminal_guidance import final_step,fingerprint
EARLY=(44,45,46)


def projection_diagnostics(z,book):
    records=[flow_control.projection_record(z.detach().cpu().numpy(),book,m) for m in (0,1)]
    scores=[float(np.clip(v['signed_projection'],-1,1).mean()) for v in records]
    return {'messages':{str(m):{'loss':v['loss'],'minimum_signed_projection':min(v['signed_projection']),
        'mean_signed_projection':float(np.mean(v['signed_projection'])),'nominal_matched_score':scores[m]} for m,v in enumerate(records)},
        'nominal_score_A_minus_B':scores[0]-scores[1],
        'meaning':'fixed clean-latent projection diagnostic, not blind MP4 receiver evidence'}


@torch.no_grad()
def shared_prefix(pipe,initial,prompt,negative,dtype,guidance,count):
    s=pipe.scheduler
    if type(s).__name__!='UniPCMultistepScheduler' or s.config.prediction_type!='flow_prediction' or not s.predict_x0 or s.config.thresholding or s.solver_p is not None:
        raise ValueError('native nonthresholded UniPC flow predict_x0 required')
    if len(s.timesteps)!=50 or not s.config.lower_order_final or float(s.sigmas[-1])!=0.:raise ValueError('fixed native50 zero-sigma final required')
    precision={};z=continue_steps(pipe,s,initial,prompt,negative,dtype,guidance,0,44,count,precision=precision)
    return z.detach(),copy.deepcopy(s),precision


@torch.no_grad()
def velocity(pipe,z,scheduler,prompt,negative,dtype,guidance,index,count):
    t=scheduler.timesteps[index].expand(z.shape[0]);values=[]
    for embedding in (prompt,negative):
        count('transformer',False);values.append(pipe.transformer(hidden_states=z.to(dtype),timestep=t,encoder_hidden_states=embedding,attention_kwargs=None,return_dict=False)[0]);count('transformer',True)
    v=(values[1]+guidance*(values[0]-values[1])).float()
    if not torch.isfinite(v).all():raise FloatingPointError('nonfinite CFG')
    return v


@torch.no_grad()
def advance(scheduler,z,v,book,message,index,count):
    """Only 44..48. A true native intermediate step, never a terminal overwrite."""
    if index not in range(44,49) or scheduler.step_index!=index:raise ValueError('invalid intermediate cursor')
    sigma=float(scheduler.sigmas[index]);clean=z-sigma*v;active=message is not None and index in EARLY
    row=dict(index=index,controlled=active,sigma=sigma,predicted_clean_before=projection_diagnostics(clean,book))
    controlled=v;off=None
    if active:
        if not sigma>0:raise ValueError('positive controlled sigma required')
        u,projected,evidence=method.correction(clean.cpu().numpy(),book,message);u=torch.from_numpy(u).to(z.device)
        controlled=v-u/sigma
        if not torch.isfinite(u).all() or not torch.isfinite(controlled).all():raise FloatingPointError('nonfinite full projection control')
        count('shadow_step',False);off=copy.deepcopy(scheduler).step(v,scheduler.timesteps[index],z.clone(),return_dict=False)[0];count('shadow_step',True)
        row.update(u=measures(u),delta_velocity=measures(controlled-v),requested_minimum_margin=evidence['minimum_signed_projection_after'],
            predicted_clean_after=projection_diagnostics(z-sigma*controlled,book))
    count('scheduler_step',False);after=scheduler.step(controlled,scheduler.timesteps[index],z,return_dict=False)[0];count('scheduler_step',True)
    if not torch.isfinite(after).all() or scheduler.step_index!=index+1:raise RuntimeError('invalid native intermediate step')
    if active:row['actual_D']=measures(after-off)
    row['native_state_after']=projection_diagnostics(after,book)
    return after.detach(),row


@torch.no_grad()
def final_fork(snapshot,z,v,book,message,count,uncontrolled=None):
    """Full projection through the same native49 interface as the successful LAST baseline."""
    sigma=float(snapshot.sigmas[49]);clean=z-sigma*v
    if uncontrolled is None:uncontrolled,_=final_step(snapshot,z,v,count)
    if message is None:return uncontrolled,dict(index=49,controlled=False,predicted_clean_before=projection_diagnostics(clean,book))
    u,projected,evidence=method.correction(clean.cpu().numpy(),book,message);u=torch.from_numpy(u).to(z.device)
    after,info=final_step(snapshot,z,v,count,u=u)
    row=dict(index=49,controlled=True,sigma=sigma,u=measures(u),delta_velocity=measures((v-u/sigma)-v),actual_D=measures(after-uncontrolled),
        predicted_clean_before=projection_diagnostics(clean,book),predicted_clean_after=projection_diagnostics(z-sigma*(v-u/sigma),book),
        requested_minimum_margin=evidence['minimum_signed_projection_after'],native=info,
        terminal_vs_projected_clean=method.numerical_equivalence(after.cpu().numpy(),projected))
    return after.detach(),row


def cumulative(rows):
    active=[v for v in rows if v.get('controlled')]
    return {'controlled_steps':len(active),'actual_D':{space:{'sum_rms':sum(v['actual_D'][space+'_rms'] for v in active),'peak_rms':max([v['actual_D'][space+'_rms'] for v in active],default=0.),'sum_rms_squared':sum(v['actual_D'][space+'_rms']**2 for v in active)} for space in ('support','global')},
        'u':{space:{'sum_rms':sum(v['u'][space+'_rms'] for v in active),'peak_rms':max([v['u'][space+'_rms'] for v in active],default=0.)} for space in ('support','global')},
        'meaning':'per-path same-history local responses; sums are not net terminal displacement or equal-budget comparison'}
