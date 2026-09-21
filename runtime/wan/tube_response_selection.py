"""No-gradient native T46 direction probing and independent formal replay."""
import copy,math
import torch
from .tube_retention import reference,native_step,continue_single
from .tube_multistep import fingerprint,projection_diagnostics
from .flow_step import measures
INDEX=46

@torch.no_grad()
def prepare_direction(snapshot,z,v,off_next,raw,target,count):
    before=fingerprint(vars(snapshot));sigma=float(snapshot.sigmas[INDEX])
    if snapshot.step_index!=INDEX or not math.isfinite(sigma) or sigma<=0:raise ValueError('invalid T46 history/sigma')
    if not math.isfinite(target) or target<0:raise ValueError('invalid LAST oracle target')
    raw=raw.detach().to(device=z.device,dtype=torch.float32)
    if not torch.isfinite(raw).all():raise ValueError('nonfinite direction')
    norm=measures(raw)['support_rms'];zero=norm==0
    if not math.isfinite(norm):raise ValueError('nonfinite direction norm')
    q=torch.zeros_like(raw) if zero else raw/norm
    probe=native_step(copy.deepcopy(snapshot),z,v-q/sigma,INDEX,count,'unit_response_probe_step')
    unit_D=measures(probe-off_next);r=unit_D['support_rms']
    if zero:epsilon=0.
    else:
        if not math.isfinite(r) or r<=0 or target<=0:raise ValueError('cannot match positive native budget')
        epsilon=target/r
    if not math.isfinite(epsilon):raise ValueError('nonfinite epsilon')
    if fingerprint(vars(snapshot))!=before:raise RuntimeError('unit probe changed original history')
    row=dict(status='ZERO_DIRECTION' if zero else 'READY',raw_direction=measures(raw),unit_direction=measures(q),
             unit_D=unit_D,epsilon=epsilon,target_D_support_rms=target,index=INDEX,sigma=sigma,
             input_fingerprint=fingerprint(z),history_fingerprint=before,
             epsilon_meaning='clean-control coordinate; both signs share one epsilon from actual unit native response',
             budget_meaning='same-case/message future LAST native RMS oracle; not deployable unified parameter')
    return q,epsilon,row

@torch.no_grad()
def replay(pipe,snapshot,z,v,off_next,q,epsilon,sign,target,nodes,prompt,negative,dtype,guidance,book,count):
    before=fingerprint(vars(snapshot));sigma=float(snapshot.sigmas[INDEX])
    if snapshot.step_index!=INDEX or not sigma>0 or sign not in (-1,0,1) or not math.isfinite(epsilon) or epsilon<0:raise ValueError('invalid replay control')
    u=(sign*epsilon)*q;controlled=v-u/sigma
    if not torch.isfinite(controlled).all():raise FloatingPointError('nonfinite velocity control')
    s=copy.deepcopy(snapshot);after=native_step(s,z,controlled,INDEX,count);D=after-off_next;actual=measures(D)
    terminal,stages=continue_single(pipe,after,s,nodes,prompt,negative,dtype,guidance,INDEX,book,count)
    if not torch.isfinite(terminal).all():raise FloatingPointError('nonfinite replay terminal')
    if fingerprint(vars(snapshot))!=before:raise RuntimeError('replay changed original history')
    expected=0. if sign==0 or epsilon==0 else target
    row=dict(index=INDEX,sigma=sigma,sign=sign,epsilon=epsilon,applied_clean_scale=sign*epsilon,target_D_support_rms=target,
             actual_D=actual,u=measures(u),delta_velocity=measures(controlled-v),
             matching_signed_error=actual['support_rms']-target,
             matching_relative_error=None if target==0 else abs(actual['support_rms']-target)/target,
             oracle_mismatch_meaning='difference from future LAST target, not failure for zero/skip control',
             expected_signed_arm_D_support_rms=expected,expected_arm_absolute_error=abs(actual['support_rms']-expected),
             input_fingerprint=fingerprint(z),history_fingerprint=before,terminal_fingerprint=fingerprint(terminal),
             tail_graph_retained=False,terminal_reused_from_probe=False)
    return terminal,row,stages,dict(applied_u=u.cpu(),actual_D=D.cpu())
