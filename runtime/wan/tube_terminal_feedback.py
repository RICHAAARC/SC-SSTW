"""One detached terminal-feedback update at fixed46; identity transport is an approximation."""
import copy,math
import torch
from .tube_retention import reference,native_step,continue_single
from .tube_multistep import velocity,projection_diagnostics,fingerprint
from .flow_step import measures
INDEX=46

@torch.no_grad()
def preview(pipe,snapshot,z,v,prompt,negative,dtype,guidance,count):
    before=fingerprint(vars(snapshot));s=copy.deepcopy(snapshot)
    terminal=native_step(s,z,v,INDEX,count)
    for index in range(INDEX+1,50):
        current=velocity(pipe,terminal,s,prompt,negative,dtype,guidance,index,count)
        terminal=native_step(s,terminal,current,index,count)
    if fingerprint(vars(snapshot))!=before:raise RuntimeError('preview changed original history')
    return terminal.detach(),dict(input_fingerprint=fingerprint(z),history_fingerprint=before,terminal_fingerprint=fingerprint(terminal),tail_graph_retained=False)

@torch.no_grad()
def matched_direction(snapshot,z,v,off_next,direction,terminal_gradient,target,count):
    before=fingerprint(vars(snapshot));sigma=float(snapshot.sigmas[INDEX])
    if snapshot.step_index!=INDEX or not sigma>0 or not math.isfinite(target) or target<=0:raise ValueError('fixed cursor and positive oracle budget required')
    raw=direction.detach().to(device=z.device,dtype=torch.float32);norm=measures(raw)['support_rms']
    if not torch.isfinite(raw).all() or not math.isfinite(norm) or norm<=0:raise ValueError('zero/nonfinite direction')
    unit=raw/norm
    probe=native_step(copy.deepcopy(snapshot),z,v-unit/sigma,INDEX,count,'unit_response_probe_step')
    probe_r=measures(probe-off_next)['support_rms']
    if not math.isfinite(probe_r) or probe_r<=0:raise ValueError('zero/nonfinite unit native response')
    scale=target/probe_r;u=scale*unit;controlled=v-u/sigma
    if not torch.isfinite(controlled).all():raise FloatingPointError('nonfinite controlled velocity')
    s=copy.deepcopy(snapshot);after=native_step(s,z,controlled,INDEX,count);D=after-off_next
    actual=measures(D);g=terminal_gradient.detach().to(z.device).double()
    if not torch.isfinite(D).all():raise FloatingPointError('nonfinite actual native response')
    if fingerprint(vars(snapshot))!=before:raise RuntimeError('formal update changed original history')
    row=dict(index=INDEX,sigma=sigma,raw_direction=measures(raw),unit_direction=measures(unit),unit_D_support_rms=probe_r,
        applied_unit_scale=scale,effective_raw_scale=scale/norm,target_D_support_rms=target,actual_D=actual,u=measures(u),delta_velocity=measures(controlled-v),
        matching_relative_error=abs(actual['support_rms']-target)/target,matching_signed_error=actual['support_rms']-target,
        predicted_terminal_loss_decrease_identity_tail=float(-(g*D.double()).sum()),
        clean_control_dot_negative_gradient_aux=float(-(g*u.double()).sum()),
        prediction_meaning='first-order -g_terminal dot actual_D assumes identity next-state-to-terminal Jacobian; not measured improvement',
        clean_control_meaning='u enters velocity as -u/sigma; u is not native D or terminal displacement',
        input_fingerprint=fingerprint(z),history_fingerprint=before,
        budget_meaning='same case/message future LAST native response RMS; diagnostic oracle, not deployable online rule')
    return after,s,row,dict(raw_direction=raw.cpu(),unit_direction=unit.cpu(),applied_u=u.cpu(),actual_D=D.cpu())
