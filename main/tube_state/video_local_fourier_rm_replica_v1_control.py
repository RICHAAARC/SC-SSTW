"""Unchanged OLD8 control plus one norm-matched two-spatial-copy candidate."""
from __future__ import annotations
import math
from main.tube_state import video_local_fourier_rm_control as old
from main.tube_state import video_local_fourier_rm_replica_v1_state as state
ARMS=('OFF','PAYLOAD_MULTI','STATE_OLD_MULTI','STATE_REPLICA_MULTI')
message_bits=old.message_bits
payload_read=old.payload_read

def control_enabled(arm,index):
    if arm not in ARMS or not 0<=index<50:raise ValueError('fixed arm/index required')
    return arm!='OFF' and index>=25

def build_targets(latent,key,bits,arm):
    import torch
    if arm not in ARMS:raise ValueError('fixed arm required')
    targets=old.build_targets(latent,key,bits)
    targets.update(blocks=old.PUBLIC.blocks,groups=state.GROUPS,carrier=old.PUBLIC.method_version if arm=='STATE_OLD_MULTI' else state.VERSION)
    if arm=='STATE_REPLICA_MULTI':targets['pilot_target']=targets['pilot_target']/math.sqrt(2.)
    return targets

project_tensor=old.project_tensor

def project_groups(value,basis):
    import torch
    if tuple(value.shape)!=(1,16,46,40,64):raise ValueError('full writer latent geometry required')
    with torch.no_grad():return torch.stack([torch.stack([(value.detach().float()[0,4,1:,h0:h1,w0:w1].reshape(45,64)@basis[b]).reshape(45,4,8) for b,(h0,h1,w0,w1) in enumerate(group)],dim=1) for group in state.GROUPS])

def group_physical_norms(value):
    return [math.sqrt(sum(float(value[0,4,1:,h0:h1,w0:w1].double().square().sum()) for h0,h1,w0,w1 in group)) for group in state.GROUPS]

def pilot_delta(clean,targets,arm,matched_norm=None):
    import torch
    if arm=='STATE_OLD_MULTI':return old.pilot_delta(clean,targets)
    if arm!='STATE_REPLICA_MULTI':raise ValueError('pilot arm required')
    if matched_norm is None or not math.isfinite(float(matched_norm)) or float(matched_norm)<0:raise ValueError('same-run OLD physical pilot norm required')
    matched_norm=float(matched_norm)
    with torch.enable_grad():
        leaf=clean.detach().float().clone().requires_grad_(True)
        features=torch.stack([torch.stack([(leaf[0,4,1:,h0:h1,w0:w1].reshape(45,64)@targets['basis'][b]).reshape(45,4,8) for b,(h0,h1,w0,w1) in enumerate(group)],dim=1) for group in state.GROUPS])
        active=targets['pilot_active'].unsqueeze(0).expand(2,-1,-1,-1,-1)
        target=targets['pilot_target'].unsqueeze(0).expand_as(features)
        residual=features[active]-target[active];loss=residual.square().mean();grad=torch.autograd.grad(loss,leaf)[0];raw=(-1392.*grad).detach();norm=float(raw.double().square().sum().sqrt())
        if not math.isfinite(norm) or not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite replica pilot update')
        if matched_norm==0:scale=0.;delta=torch.zeros_like(raw)
        elif norm==0:raise ValueError('zero replica raw direction cannot match positive OLD pilot norm')
        else:scale=matched_norm/norm;delta=raw*scale
    actual=float(delta.double().square().sum().sqrt())
    if not math.isfinite(actual) or not bool(torch.isfinite(delta).all()):raise FloatingPointError('nonfinite norm-matched update')
    return delta,dict(loss=float(loss.detach()),active_coefficients=11136,boundary_coefficients=384,eta=1392.,target_l2=1.,group_alpha=state.GROUP_ALPHA,raw_delta_l2=norm,norm_match_scale=scale,matched_actual_norm_l2=matched_norm,actual_delta_l2=actual,norm_match_residual=actual-matched_norm,raw_group_l2=group_physical_norms(raw),actual_group_l2=group_physical_norms(delta),normalization='joint raw direction matched to same-run OLD full physical pilot norm; scale may exceed1',budget_source='writer-only same-run STATE_OLD_MULTI step norm; no receiver input',cap=None,scale_is_shrink_only=False)

def delta_closure(delta,projected,targets):
    import torch
    with torch.no_grad():
        total=float(delta.double().square().sum());patches=0.;complement=0.
        for g,group in enumerate(state.GROUPS):
            for b,(h0,h1,w0,w1) in enumerate(group):
                patch=delta[0,4,1:,h0:h1,w0:w1].double().reshape(45,64);reconstructed=projected[g,:,b].double().reshape(45,32)@targets['basis'][b].double().T
                patches+=float(patch.square().sum());complement+=float((patch-reconstructed).square().sum())
        active=targets['pilot_active'].unsqueeze(0).expand_as(projected);a2=float(projected[active].double().square().sum());b2=float(projected[~active].double().square().sum())
    return dict(actual_spatial_l2=math.sqrt(total),projection_l2=math.sqrt(a2+b2),active_projection_energy=a2,boundary_projection_energy=b2,spatial_U_complement_l2=math.sqrt(complement),outside_ROI_energy=total-patches,parseval_error=total-(a2+b2+complement),active_coefficients=11136,boundary_coefficients=384,actual_group_l2=group_physical_norms(delta),note='two physical groups; FP32 basis/projection residuals retained; groups are not independent votes')

def guided_velocity(z,conditional,unconditional,sigma,targets,arm,index,count=None,*,diagnostics=None,matched_norm=None):
    import torch
    if not math.isfinite(float(sigma)) or sigma<=0:raise ValueError('positive native sigma required')
    z,c,u=z.float(),conditional.float(),unconditional.float();enabled=control_enabled(arm,index);detail=dict(enabled=enabled,state_control_cfg='float32',arm=arm,carrier=targets['carrier'])
    if enabled:
        clean=z-float(sigma)*c
        if count:count('payload_gradient',False)
        dp,pinfo=old.payload.local_delta(clean,targets['payload_target'],targets['payload_mask'])
        if count:count('payload_gradient',True)
        ds=torch.zeros_like(dp);sinfo=None
        if arm.startswith('STATE_'):
            if count:count('pilot_gradient',False)
            ds,sinfo=pilot_delta(clean,targets,arm,matched_norm)
            if count:count('pilot_gradient',True)
        delta=dp+ds;p2=float(dp.double().square().sum());s2=float(ds.double().square().sum());cross=float((dp.double()*ds.double()).sum());d2=float(delta.double().square().sum())
        detail.update(payload=pinfo,pilot=sinfo,payload_delta_l2=math.sqrt(p2),pilot_delta_l2=math.sqrt(s2),merged_delta_l2=math.sqrt(d2),merged_delta_rms=math.sqrt(d2/delta.numel()),cross_inner_product=cross,norm_decomposition_error=d2-(p2+s2+2*cross),cfg_clean_delta_l2=5*math.sqrt(d2),cross_arm_total_budget_matched=False)
        c=c-delta/float(sigma)
    if diagnostics is not None:
        try:
            applied=ds if enabled and arm.startswith('STATE_') else torch.zeros_like(z)
            groups=project_groups(applied,targets['basis']);diagnostics.update(pilot_delta=groups[0].detach(),pilot_delta_groups=groups.detach(),closure=delta_closure(applied,groups,targets))
        except Exception as exc:diagnostics['error']=f'{type(exc).__name__}: {exc}'
    v=u+5.*(c-u)
    if not bool(torch.isfinite(v).all()):raise FloatingPointError('nonfinite CFG velocity')
    return v.detach(),detail
