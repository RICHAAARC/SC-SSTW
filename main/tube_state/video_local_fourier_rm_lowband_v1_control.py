"""Four real trajectory arms: unchanged old state and one fixed lowband carrier."""
from __future__ import annotations
import math
from main.tube_state import video_local_fourier_rm_control as old
from main.tube_state import video_local_fourier_rm_lowband_v1_state as low
ARMS=('OFF','PAYLOAD_MULTI','STATE_OLD_MULTI','STATE_LOWBAND_MULTI')
message_bits=old.message_bits
payload_read=old.payload_read


def carrier_for_writer(arm):
    if arm not in ARMS:raise ValueError('fixed arm required')
    return old.state if arm=='STATE_OLD_MULTI' else low

def control_enabled(arm,index):
    if arm not in ARMS or not 0<=index<50:raise ValueError('fixed arm/index required')
    return arm!='OFF' and index>=25

def build_targets(latent,key,bits,arm):
    import torch
    carrier=carrier_for_writer(arm);targets=old.build_targets(latent,key,bits)
    targets.update(basis=torch.tensor(carrier.bases(key),device=latent.device,dtype=torch.float32),blocks=carrier.PUBLIC.blocks,carrier=carrier.PUBLIC.method_version)
    return targets

def project_tensor(value,basis,blocks):
    import torch
    if tuple(value.shape)!=(1,16,46,40,64):raise ValueError('full writer latent geometry required')
    with torch.no_grad():
        return torch.stack([(value.detach().float()[0,4,1:,h0:h1,w0:w1].reshape(45,(h1-h0)*(w1-w0))@basis[b]).reshape(45,4,8) for b,(h0,h1,w0,w1) in enumerate(blocks)],dim=1)

def pilot_delta(clean,targets,arm):
    import torch
    if arm=='STATE_OLD_MULTI':return old.pilot_delta(clean,targets)
    if arm!='STATE_LOWBAND_MULTI':raise ValueError('pilot arm required')
    with torch.enable_grad():
        leaf=clean.detach().float().clone().requires_grad_(True)
        features=torch.stack([(leaf[0,4,1:,h0:h1,w0:w1].reshape(45,(h1-h0)*(w1-w0))@targets['basis'][b]).reshape(45,4,8) for b,(h0,h1,w0,w1) in enumerate(targets['blocks'])],dim=1)
        residual=features[targets['pilot_active']]-targets['pilot_target'][targets['pilot_active']]
        loss=residual.square().mean();grad=torch.autograd.grad(loss,leaf)[0];raw=(-696.*grad).detach();norm=raw.double().square().sum().sqrt()
        if not bool(torch.isfinite(norm)) or not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite lowband pilot update')
        scale=min(1.,1./float(norm)) if float(norm)>0 else 1.;delta=raw*scale
    return delta,dict(loss=float(loss.detach()),active_coefficients=5568,eta=696.,target_l2=1.,raw_delta_l2=float(norm),cap=1.,cap_scale=scale,actual_delta_l2=float(delta.double().square().sum().sqrt()))

def delta_closure(delta,projected,targets):
    import torch
    with torch.no_grad():
        total=float(delta.double().square().sum());patches=0.;complement=0.
        for b,(h0,h1,w0,w1) in enumerate(targets['blocks']):
            patch=delta[0,4,1:,h0:h1,w0:w1].double().reshape(45,(h1-h0)*(w1-w0));reconstructed=projected[:,b].double().reshape(45,32)@targets['basis'][b].double().T
            patches+=float(patch.square().sum());complement+=float((patch-reconstructed).square().sum())
        active=targets['pilot_active'];active2=float(projected[active].double().square().sum());boundary2=float(projected[~active].double().square().sum())
    return dict(actual_spatial_l2=math.sqrt(total),projection_l2=math.sqrt(active2+boundary2),active_projection_energy=active2,boundary_projection_energy=boundary2,spatial_U_complement_l2=math.sqrt(complement),outside_ROI_energy=total-patches,parseval_error=total-(active2+boundary2+complement),active_coefficients=5568,boundary_coefficients=192,note='actual FP32 projection closure; diagnostic only')

def guided_velocity(z,conditional,unconditional,sigma,targets,arm,index,count=None,*,diagnostics=None):
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
            ds,sinfo=pilot_delta(clean,targets,arm)
            if count:count('pilot_gradient',True)
        delta=dp+ds;p2=float(dp.double().square().sum());s2=float(ds.double().square().sum());cross=float((dp.double()*ds.double()).sum());d2=float(delta.double().square().sum())
        detail.update(payload=pinfo,pilot=sinfo,payload_delta_l2=math.sqrt(p2),pilot_delta_l2=math.sqrt(s2),merged_delta_l2=math.sqrt(d2),merged_delta_rms=math.sqrt(d2/delta.numel()),cross_inner_product=cross,norm_decomposition_error=d2-(p2+s2+2*cross),cfg_clean_delta_l2=5*math.sqrt(d2),cross_arm_total_budget_matched=False)
        c=c-delta/float(sigma)
    if diagnostics is not None:
        try:
            ds_applied=ds if enabled and arm.startswith('STATE_') else torch.zeros_like(z);projected=project_tensor(ds_applied,targets['basis'],targets['blocks'])
            diagnostics.update(pilot_delta=projected.detach(),closure=delta_closure(ds_applied,projected,targets))
        except Exception as exc:diagnostics.update(error=f'{type(exc).__name__}: {exc}')
    velocity=u+5.*(c-u)
    if not bool(torch.isfinite(velocity).all()):raise FloatingPointError('nonfinite CFG velocity')
    return velocity.detach(),detail
