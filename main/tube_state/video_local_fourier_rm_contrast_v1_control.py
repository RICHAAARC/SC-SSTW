"""Native MULTI pilot direction in one public masked union space; old arms frozen."""
from __future__ import annotations
import math
import numpy as np
from main.tube_state import video_local_fourier_rm_lowband_v1_control as original
from main.tube_state import video_local_fourier_rm_lowband_v1_state as low
from main.tube_state import video_local_fourier_rm_contrast_v1_space as space
VERSION=space.VERSION
ARMS=('OFF','PAYLOAD_MULTI','STATE_LOWBAND_MULTI','STATE_CONTRAST_MULTI')
message_bits=original.message_bits
payload_read=original.payload_read
project_tensor=original.project_tensor
delta_closure=original.delta_closure


def control_enabled(arm,index):
    if arm not in ARMS or not 0<=index<50:raise ValueError('fixed arm/index required')
    return arm!='OFF' and index>=25


def build_targets(latent,key,bits,arm):
    import torch
    if arm not in ARMS:raise ValueError('fixed arm required')
    targets=original.build_targets(latent,key,bits,'STATE_LOWBAND_MULTI' if arm=='STATE_CONTRAST_MULTI' else arm)
    targets['writer_key']=key
    if arm=='STATE_CONTRAST_MULTI':
        public=space.build_space(key);targets.update(contrast_basis=torch.tensor(public['basis'],device=latent.device,dtype=torch.float64),contrast_mask=torch.tensor(public['mask'],device=latent.device,dtype=torch.bool),contrast_recipe=public['receipt']['recipe_sha256'])
    return targets


def restrict_raw(raw,targets):
    """Only first44 local coefficients change; raw physical row45 remains exact pre-cap."""
    import torch
    coefficients=project_tensor(raw,targets['basis'],targets['blocks']);x=(coefficients[:44].double()*targets['contrast_mask']).reshape(-1);u=targets['contrast_basis']
    restricted=(u@(u.T@x)).reshape(44,4,4,8);restricted=restricted.masked_fill(~targets['contrast_mask'],0.)
    out=raw.clone()
    for b,(h0,h1,w0,w1) in enumerate(targets['blocks']):
        out[0,4,1:45,h0:h1,w0:w1]=(restricted[:,b].float().reshape(44,32)@targets['basis'][b].T).reshape(44,h1-h0,w1-w0)
    return out,coefficients,restricted


def pilot_delta(clean,targets,*,diagnostics=None):
    import torch
    with torch.enable_grad():
        leaf=clean.detach().float().clone().requires_grad_(True)
        features=torch.stack([(leaf[0,4,1:,h0:h1,w0:w1].reshape(45,(h1-h0)*(w1-w0))@targets['basis'][b]).reshape(45,4,8) for b,(h0,h1,w0,w1) in enumerate(targets['blocks'])],dim=1)
        residual=features[targets['pilot_active']]-targets['pilot_target'][targets['pilot_active']];loss=residual.square().mean();raw=(-696.*torch.autograd.grad(loss,leaf)[0]).detach()
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite contrast raw direction')
    restricted,raw_q,precap_q=restrict_raw(raw,targets);before=float(raw.double().norm());norm=float(restricted.double().norm())
    if not math.isfinite(norm) or not bool(torch.isfinite(restricted).all()):raise FloatingPointError('nonfinite restricted contrast direction')
    scale=min(1.,1./norm) if norm>0 else 1.;delta=restricted*scale
    if diagnostics is not None:
        diagnostics.update(direction_raw=raw_q[:44].detach(),direction_precap=precap_q.detach(),direction_actual=project_tensor(delta,targets['basis'],targets['blocks'])[:44],direction_source='raw actual gradient projection; precap FP64 union projection; actual FP32 applied pilot projection')
    return delta,dict(loss=float(loss.detach()),active_coefficients=5568,eta=696.,target_l2=1.,raw_delta_l2=before,restricted_raw_delta_l2=norm,raw_R44_l2=float(raw_q[:44].double().norm()),restricted_R44_l2=float(precap_q.double().norm()),row45_raw_spatial_l2=float(raw[0,4,45].double().norm()),row45_precap_spatial_l2=float(restricted[0,4,45].double().norm()),row45_actual_spatial_l2=float(delta[0,4,45].double().norm()),row45_precap_exact=bool(torch.equal(raw[0,4,45],restricted[0,4,45])),cap=1.,cap_scale=scale,actual_delta_l2=float(delta.double().norm()),recipe_sha256=targets['contrast_recipe'],row45_note='raw pre-cap unchanged; full pilot cap can change actual row45 update')


def guided_velocity(z,conditional,unconditional,sigma,targets,arm,index,count=None,*,diagnostics=None):
    import torch
    if arm not in ARMS:raise ValueError('fixed arm required')
    if arm!='STATE_CONTRAST_MULTI':
        value,detail=original.guided_velocity(z,conditional,unconditional,sigma,targets,arm,index,count,diagnostics=diagnostics)
        if diagnostics is not None and control_enabled(arm,index) and arm=='STATE_LOWBAND_MULTI' and 'error' not in diagnostics:
            actual=diagnostics['pilot_delta'][:44];raw=actual/detail['pilot']['cap_scale']
            diagnostics.update(direction_raw=raw.detach(),direction_precap=raw.detach(),direction_actual=actual.detach(),direction_source='LOW control unchanged: raw/precap algebraically reconstructed from saved actual projection divided by original cap_scale; FP32 rounding, not separately saved raw physical gradient')
        return value,detail
    if not math.isfinite(float(sigma)) or sigma<=0:raise ValueError('positive native sigma required')
    z,c,u=z.float(),conditional.float(),unconditional.float();enabled=control_enabled(arm,index);detail=dict(enabled=enabled,state_control_cfg='float32',arm=arm,carrier=targets['carrier'])
    if enabled:
        clean=z-float(sigma)*c
        if count:count('payload_gradient',False)
        dp,pinfo=original.old.payload.local_delta(clean,targets['payload_target'],targets['payload_mask'])
        if count:count('payload_gradient',True)
        if count:count('pilot_gradient',False)
        ds,sinfo=pilot_delta(clean,targets,diagnostics=diagnostics)
        if count:count('pilot_gradient',True)
        delta=dp+ds;p2=float(dp.double().square().sum());s2=float(ds.double().square().sum());cross=float((dp.double()*ds.double()).sum());d2=float(delta.double().square().sum())
        detail.update(payload=pinfo,pilot=sinfo,payload_delta_l2=math.sqrt(p2),pilot_delta_l2=math.sqrt(s2),merged_delta_l2=math.sqrt(d2),merged_delta_rms=math.sqrt(d2/delta.numel()),cross_inner_product=cross,norm_decomposition_error=d2-(p2+s2+2*cross),cfg_clean_delta_l2=5*math.sqrt(d2),cross_arm_total_budget_matched=False);c=c-delta/float(sigma)
    if diagnostics is not None:
        try:
            ds_applied=ds if enabled else torch.zeros_like(z);q=project_tensor(ds_applied,targets['basis'],targets['blocks']);diagnostics.update(pilot_delta=q.detach(),closure=delta_closure(ds_applied,q,targets))
        except Exception as exc:diagnostics.update(error=f'{type(exc).__name__}: {exc}')
    velocity=u+5.*(c-u)
    if not bool(torch.isfinite(velocity).all()):raise FloatingPointError('nonfinite CFG velocity')
    return velocity.detach(),detail


def direction_diagnostic(capture,key):
    """Diagnostic only; complete public candidate effects for all three directions."""
    f=space.build_space(key);effects={name:space.relative_effects(capture['direction_'+name].detach().double().cpu().numpy(),key) for name in ('raw','precap','actual')}
    return dict(status='COMPLETE',recipe_sha256=f['receipt']['recipe_sha256'],valid_catalog_indices=f['valid_catalog_indices'],anchor_catalog_index=f['valid_catalog_indices'][0],effects=effects,source=capture['direction_source'],effect_semantics='change in C_i-C_anchor when q is incremented by pilot; all candidates, no winner or truth',candidate_effect_values=1044,cfg_clean_pilot_multiplier=5.,effects_do_not_include_payload_or_native_transition=True,truth_inputs=False,accepted_payload=False,state_path_accepted=False)
