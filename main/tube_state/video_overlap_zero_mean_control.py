"""Spatial-zero-mean Wan control; unchanged state code, loss, cap and payload."""
from __future__ import annotations
import math
import numpy as np
from main.tube_state import video_overlap_zero_mean_state as state
from main.tube_state import grow_video_reference as payload
PUBLIC=state.PUBLIC
ARMS=('OFF','PAYLOAD_MULTI','OVERLAP_MULTI')
VIEWS=('FULL_RESAVED181','CROP4_129','CROP5_129','CROP6_129','CROP7_129')
message_bits=payload.message_bits


def phase_spec(length,g):
    if length not in (181,129,125,133) or g not in range(4):raise ValueError('unsupported public frame geometry')
    k=(length-g-1)//4;R=(length-4)//4
    if R not in PUBLIC.observed_lengths:raise ValueError('common regular support outside frozen family')
    return dict(g=g,left=g,right=g+1+4*k,regular_count=k,R=R,first_latent_excluded=True,
        nominal_grid='newly supplied RGB frames only; not VAE receptive field')


def control_enabled(arm,index):
    if arm not in ARMS or not 0<=index<50:raise ValueError('fixed arm/index required')
    return arm!='OFF' and index>=25


def build_targets(latent,key,bits):
    import torch
    target,mask=payload.build_target(latent,key,bits)
    basis=torch.tensor(state.bases(key),device=latent.device,dtype=torch.float32)
    signs=torch.tensor(state.composite_signs(key),device=latent.device,dtype=torch.float32)
    active=signs!=0
    if int(active.sum())!=1392:raise ValueError('frozen active pilot coefficient count')
    return dict(payload_target=target,payload_mask=mask,basis=basis,pilot_target=signs*PUBLIC.alpha,pilot_active=active)


def pilot_delta(clean,targets):
    import torch
    with torch.enable_grad():
        leaf=clean.detach().float().clone().requires_grad_(True);features=[]
        for i,(h0,h1,w0,w1) in enumerate(PUBLIC.blocks):
            patch=leaf[0,4,1:,h0:h1,w0:w1].reshape(45,16)
            features.append((patch@targets['basis'][i]).reshape(45,4,2))
        features=torch.stack(features,dim=1)
        residual=features[targets['pilot_active']]-targets['pilot_target'][targets['pilot_active']]
        loss=residual.square().mean();grad=torch.autograd.grad(loss,leaf)[0]
        raw=(-174.*grad).detach();norm=raw.double().square().sum().sqrt()
        if not bool(torch.isfinite(norm)) or not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite overlap pilot update')
        scale=min(1.,1./float(norm)) if float(norm)>0 else 1.
        delta=raw*scale
    return delta,dict(loss=float(loss.detach()),active_coefficients=1392,eta=174.,target_l2=1.,
      raw_delta_l2=float(norm),cap=1.,cap_scale=scale,actual_delta_l2=float(delta.double().square().sum().sqrt()))


def guided_velocity(z,conditional,unconditional,sigma,targets,arm,index,count=None,*,diagnostics=None):
    import torch
    if not math.isfinite(float(sigma)) or sigma<=0:raise ValueError('positive native sigma required')
    z,c,u=z.float(),conditional.float(),unconditional.float();enabled=control_enabled(arm,index)
    detail=dict(enabled=enabled,state_control_cfg='float32',arm=arm)
    if enabled:
        clean=z-float(sigma)*c
        if count:count('payload_gradient',False)
        dp,pinfo=payload.local_delta(clean,targets['payload_target'],targets['payload_mask'])
        if count:count('payload_gradient',True)
        ds=torch.zeros_like(dp);sinfo=None
        if arm=='OVERLAP_MULTI':
            if count:count('pilot_gradient',False)
            ds,sinfo=pilot_delta(clean,targets)
            if count:count('pilot_gradient',True)
        delta=dp+ds
        p2=float(dp.double().square().sum());s2=float(ds.double().square().sum())
        cross=float((dp.double()*ds.double()).sum());d2=float(delta.double().square().sum())
        detail.update(payload=pinfo,pilot=sinfo,payload_delta_l2=math.sqrt(p2),pilot_delta_l2=math.sqrt(s2),
          merged_delta_l2=math.sqrt(d2),merged_delta_rms=math.sqrt(d2/delta.numel()),cross_inner_product=cross,
          norm_decomposition_error=d2-(p2+s2+2*cross),cfg_clean_delta_l2=5*math.sqrt(d2),cross_arm_total_budget_matched=False)
        c=c-delta/float(sigma)
    if diagnostics is not None:
        # Side channel only. Actual applied delta, never an analytic surrogate.
        try:
            ds_applied=ds if enabled and arm=='OVERLAP_MULTI' else torch.zeros_like(z)
            projected=project_tensor(ds_applied,targets['basis'])
            diagnostics.update(pilot_delta=projected.detach(),closure=delta_closure(ds_applied,projected,targets))
        except Exception as exc:
            diagnostics.update(error=f'{type(exc).__name__}: {exc}')
    v=u+5.*(c-u)
    if not bool(torch.isfinite(v).all()):raise FloatingPointError('nonfinite CFG velocity')
    return v.detach(),detail



def project_tensor(value,basis):
    """Full 45 x 4 x 4 x 2, including inactive boundary slots; no temporal centering."""
    import torch
    if tuple(value.shape)!=(1,16,46,40,64):raise ValueError('full writer latent geometry required')
    with torch.no_grad():
        return torch.stack([(value.detach().float()[0,4,1:,h0:h1,w0:w1].reshape(45,16)@basis[i]).reshape(45,4,2)
                            for i,(h0,h1,w0,w1) in enumerate(PUBLIC.blocks)],dim=1)


def delta_closure(delta,projected,targets):
    """Actual tensor energy audit; U-complement is not called host or score residual."""
    import torch
    with torch.no_grad():
        total=float(delta.double().square().sum())
        patches=0.;complement=0.
        for i,(h0,h1,w0,w1) in enumerate(PUBLIC.blocks):
            patch=delta[0,4,1:,h0:h1,w0:w1].double().reshape(45,16)
            reconstructed=projected[:,i].double().reshape(45,8)@targets['basis'][i].double().T
            patches+=float(patch.square().sum());complement+=float((patch-reconstructed).square().sum())
        active=targets['pilot_active'];active2=float(projected[active].double().square().sum())
        boundary2=float(projected[~active].double().square().sum())
    return dict(actual_spatial_l2=math.sqrt(total),projection_l2=math.sqrt(active2+boundary2),
                active_projection_energy=active2,boundary_projection_energy=boundary2,
                spatial_U_complement_l2=math.sqrt(complement),outside_ROI_energy=total-patches,
                parseval_error=total-(active2+boundary2+complement),
                active_coefficients=1392,boundary_coefficients=48,
                note='FP32 projection closure; signed rounding residuals retained')

def payload_read(normalized,key,R):
    """Repeated payload diagnostic only; never used to rank state paths."""
    import torch
    from collections import Counter
    if normalized.ndim!=5 or tuple(normalized.shape[:2])!=(1,16) or tuple(normalized.shape[-2:])!=(40,64) or normalized.shape[2]<R+1:raise ValueError('received geometry mismatch')
    values=torch.fft.fft2(normalized.float(),dim=(-2,-1),norm='ortho').real
    coords=payload.coordinates(key);h=[a for a,b in coords];w=[b for a,b in coords];bits=[];votes=[]
    for ch in range(4):
        raw=(values[0,ch,1:R+1,h,w]>0).to(torch.int64).cpu().numpy()
        for bit in range(8):
            row=raw[:,bit::8].reshape(-1).tolist();bits.append(Counter(row).most_common(1)[0][0])
            votes.append(dict(ones=sum(row),zeros=len(row)-sum(row),count=len(row)))
    return dict(status='READ',decoded_bits=bits,votes=votes,truth_used=False,R=R,role='repeated payload diagnostic; not a path selector')


def combine_phases(phases):
    """Global uncalibrated minimum over four independently re-encoded phases."""
    result=dict(status='SEARCH_INCOMPLETE',score_status='UNCALIBRATED_DIAGNOSTIC',state_path_accepted=False,
       accepted_payload=False,canonical=None,top=[],unique_model_hypothesis=False)
    if set(phases)!=set(range(4)) or any(p['summary']['status'] not in ('COMPLETE','NO_ENERGY') for p in phases.values()):return result
    minimum=min(p['summary']['min_cost'] for p in phases.values());top=[]
    for g,p in phases.items():
        for index,cost in zip(p['valid_catalog_indices'],p['path_costs']):
            if cost-minimum<=PUBLIC.tie_atol:top.append(dict(g=g,catalog_index=index,cost=cost))
    R=phases[0]['received_length'];rows=state.catalog(R)
    canonical=min(top,key=lambda v:(tuple(rows[v['catalog_index']]['taus']),v['g'],rows[v['catalog_index']]['event_type'],rows[v['catalog_index']]['event_i'] or 0))
    energy=phases[canonical['g']]['summary']['observed_projection_energy']
    result.update(status='NO_ENERGY' if energy==0 else 'COMPLETE',min_cost=minimum,top=top,
        canonical=None if energy==0 else {**canonical,'path':rows[canonical['catalog_index']]},unique_model_hypothesis=energy>0 and len(top)==1,
        all_top_require_event=all(rows[t['catalog_index']]['event_type']!='ZERO_EDIT' for t in top),
        interpretation='finite phase/path model only; no real synchronization acceptance or calibrated threshold')
    return result
