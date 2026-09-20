"""Single native write at fixed times; future OFF LAST scale is a diagnostic oracle."""
import copy
import math
import torch
from main.tube_state import terminal_guidance as method
from .tube_multistep import shared_prefix,velocity,projection_diagnostics,fingerprint
from .flow_step import measures
TIMES=(44,46,49)

@torch.no_grad()
def native_step(s,z,v,index,count,kind='scheduler_step'):
    if s.step_index!=index:raise ValueError('native cursor mismatch')
    count(kind,False);out=s.step(v,s.timesteps[index],z.clone(),return_dict=False)[0];count(kind,True)
    if s.step_index!=index+1 or not torch.isfinite(out).all():raise FloatingPointError('nonfinite native state or cursor')
    return out.detach()

@torch.no_grad()
def reference(pipe,initial,prompt,negative,dtype,guidance,count):
    z,s,precision=shared_prefix(pipe,initial,prompt,negative,dtype,guidance,count)
    nodes={};snapshots={}
    for t in range(44,50):
        v=velocity(pipe,z,s,prompt,negative,dtype,guidance,t,count)
        if t in TIMES:snapshots[t]=copy.deepcopy(s)
        nodes[t]={'z':z.cpu().clone(),'v':v.cpu().clone(),'clean':(z-float(s.sigmas[t])*v).cpu().clone()}
        z=native_step(s,z,v,t,count)
    nodes[50]={'z':z.cpu().clone()}
    return nodes,snapshots,precision

@torch.no_grad()
def matched_update(snapshot,z,v,off_next,book,message,index,count,target_rms=None):
    before=fingerprint(vars(snapshot));s=copy.deepcopy(snapshot);sigma=float(s.sigmas[index])
    if not sigma>0:raise ValueError('positive sigma required')
    clean=z-sigma*v;raw,_,_=method.correction(clean.cpu().numpy(),book,message)
    unit=torch.from_numpy(raw).to(z.device);unit_velocity=v-unit/sigma
    if not torch.isfinite(unit_velocity).all():raise FloatingPointError('nonfinite unit control')
    unit_D=None;scale=1.
    if target_rms is not None:
        if not math.isfinite(target_rms) or target_rms<=0:raise ValueError('nonpositive/nonfinite LAST reference budget')
        probe=native_step(copy.deepcopy(s),z,unit_velocity,index,count,'unit_response_probe_step')
        unit_D=measures(probe-off_next);r=unit_D['support_rms']
        if not math.isfinite(r) or r<=0:raise ValueError('unit native response cannot match positive budget')
        scale=target_rms/r
    u=scale*unit;controlled=v-u/sigma
    if not torch.isfinite(controlled).all():raise FloatingPointError('nonfinite matched velocity')
    after=native_step(s,z,controlled,index,count);D=after-off_next;actual=measures(D)
    if fingerprint(vars(snapshot))!=before:raise RuntimeError('shared history changed')
    row=dict(index=index,sigma=sigma,scale=scale,target_D_support_rms=target_rms,unit_D=unit_D,
        actual_D=actual,u=measures(u),delta_velocity=measures(controlled-v),
        matching_relative_error=None if target_rms is None else abs(actual['support_rms']-target_rms)/target_rms,
        matching_signed_error=None if target_rms is None else actual['support_rms']-target_rms,
        before_clean=projection_diagnostics(clean,book),controlled_clean=projection_diagnostics(z-sigma*controlled,book),
        input_fingerprint=fingerprint(z),history_fingerprint=before,
        budget_meaning='per case/message future OFF LAST full-control native D RMS; diagnostic oracle, not online rule')
    return after,s,row,{'applied_u':u.cpu(),'actual_D':D.cpu()}

@torch.no_grad()
def continue_single(pipe,z,s,nodes,prompt,negative,dtype,guidance,index,book,count):
    stages={}
    for t in range(index+1,50):
        v=velocity(pipe,z,s,prompt,negative,dtype,guidance,t,count)
        if t in (index+1,index+2):
            clean=z-float(s.sigmas[t])*v;off=nodes[t]
            stages['immediate_next' if t==index+1 else 'one_free_step']={
                'node':t,'state_response':measures(z-off['z'].to(z.device)),
                'clean_response':measures(clean-off['clean'].to(z.device)),
                'clean':projection_diagnostics(clean,book),'OFF_clean':projection_diagnostics(off['clean'],book)}
        z=native_step(s,z,v,t,count)
    if index==49:stages={'status':'NOT_APPLICABLE_FINAL_WRITE'}
    return z,stages
