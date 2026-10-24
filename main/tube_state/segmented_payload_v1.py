"""Four actual temporal bytes, fixed net 32 bits and original physical support."""
from __future__ import annotations
from collections import Counter
import math
from main.tube_state import grow_video_reference as original
from main.tube_state.given_time_mapping_v1 import plan_given_time_mapping


def source_segment(t):
    if type(t) is not int or not 0<=t<=45:raise ValueError('source Wan time outside 0..45')
    return 0 if t==0 else 3 if t==45 else (t-1)//11


def build_target(latent,key,bits):
    """All 46 layers keep the original mask/alpha; effective layers 1..44."""
    import torch
    original._validate(latent,original.PUBLIC)
    if len(bits)!=32 or any(type(b) is not int or b not in (0,1) for b in bits):
        raise ValueError('exactly 32 binary writer bits required')
    target=torch.zeros_like(latent);mask=torch.zeros_like(latent,dtype=torch.bool)
    coords=original.coordinates(key);h=[p[0] for p in coords];w=[p[1] for p in coords]
    for t in range(46):
        g=source_segment(t)
        values=latent.new_tensor([(2*bits[8*g+j%8]-1)*original.ALPHA for j in range(len(coords))])
        for ch in original.PUBLIC.channels:
            target[0,ch,t,h,w]=values;mask[0,ch,t,h,w]=True
    return target,mask


def segment_plan(estimated_path):
    """Group only complete cells under a supplied map; perform no synchronization."""
    estimated_path=list(estimated_path)
    operation=plan_given_time_mapping(estimated_path)
    support=set(estimated_path)
    segments=[]
    for g in range(4):
        cells=[]
        for r in range(1,operation['wan_support']+1):
            source_t=operation['anchor']//4+r
            if 1<=source_t<=44 and source_segment(source_t)==g:
                sources=list(range(4*source_t-3,4*source_t+1))
                if all(s in support for s in sources):
                    cells.append(dict(receiver_wan_time=r,source_wan_time=source_t,source_frames=sources))
        segments.append(dict(segment=g,cells=cells,status='SUPPORTED' if cells else 'MISSING',
                             observed_cells=len(cells),planned_bits=8))
    # Preserve nominal partial/absent support separately; vote eligibility above
    # is unchanged. These are frame-map facts, not claims about a VAE receptive field.
    cell_support=[]
    for source_t in range(1,45):
        frames=list(range(4*source_t-3,4*source_t+1))
        present=[frame for frame in frames if frame in support]
        receiver_t=source_t-operation['anchor']//4
        in_span=1<=receiver_t<=operation['wan_support']
        cell_support.append(dict(source_wan_time=source_t,segment=source_segment(source_t),
            source_frames=frames,present_frames=present,missing_frames=[f for f in frames if f not in support],
            support_status='FULL' if len(present)==4 else 'PARTIAL' if present else 'MISSING',
            in_read_span=in_span,eligible_for_votes=in_span and len(present)==4))
    return dict(operation=operation,segments=segments,cell_support=cell_support,estimated_path=list(estimated_path),
                mapping_origin='CALLER_SUPPLIED',conditional_on_given_mapping=True,
                blind_sync_performed=False,message_inputs=False,
                support_rule='Only four source positions actually represented by received indices in estimated path; synthetic fill is context, not votes.')


def read_segments(normalized,key,estimated_path):
    """Read a latent already encoded after the supplied map was applied to RGB."""
    import torch
    import numpy as np
    plan=segment_plan(estimated_path);op=plan['operation']
    expected=(1,16,(op['output_frames']-1)//4+1,40,64)
    if tuple(normalized.shape)!=expected or not normalized.is_floating_point() or not bool(torch.isfinite(normalized).all()):
        raise ValueError('finite normalized Wan latent with selected output geometry required')
    if op['status']!='SUPPORTED':raise ValueError('selected span has no Wan support')
    spectrum=torch.fft.fft2(normalized.float(),dim=(-2,-1),norm='ortho').real
    coords=original.coordinates(key);h=[p[0] for p in coords];w=[p[1] for p in coords]
    values=torch.stack([spectrum[0,ch,1:op['wan_support']+1,h,w] for ch in range(4)])
    votes=(values>0).to(torch.int8).cpu().numpy()
    zeros=(values==0).to(torch.int8).cpu().numpy()
    return decode_signed_votes(plan,(2*votes-1).astype(np.int8),zeros)


def decode_signed_votes(plan,signed_votes,zeros):
    """One canonical vote reducer, also usable to read saved evidence without a VAE."""
    import numpy as np
    shape=(4,plan['operation']['wan_support'],240)
    if signed_votes.shape!=shape or signed_votes.dtype!=np.int8 or not np.isin(signed_votes,[-1,1]).all():
        raise ValueError('signed votes must be int8 +/-1 with public support geometry')
    if zeros.shape!=shape or zeros.dtype!=np.int8 or not np.isin(zeros,[0,1]).all():
        raise ValueError('zero coefficients must be int8 0/1 with public support geometry')
    if np.any((zeros==1)&(signed_votes!= -1)):raise ValueError('zero FFT coefficients must receive strict >0 vote zero')
    votes=(signed_votes>0).astype(np.int8)
    segments=[];bits=[None]*32
    for item in plan['segments']:
        row={**item,'bits':[]};times=[c['receiver_wan_time']-1 for c in item['cells']]
        for slot in range(8):
            sequence=votes[:,times,slot::8].reshape(-1).tolist()
            if sequence:
                decoded=int(Counter(sequence).most_common(1)[0][0]);ones=sum(sequence);count=len(sequence)
                bit=dict(bit_index=8*item['segment']+slot,decoded=decoded,ones=ones,zeros=count-ones,
                         count=count,tie=ones*2==count,zero_coefficients=int(zeros[:,times,slot::8].sum()))
                bits[bit['bit_index']]=decoded
            else:
                bit=dict(bit_index=8*item['segment']+slot,decoded=None,count=0,status='MISSING')
            row['bits'].append(bit)
        segments.append(row)
    return dict(status='READ',decoded_bits=bits,segments=segments,plan=plan,
                net_payload_bits=32,decoded_bit_count=sum(b is not None for b in bits),mapping_origin='CALLER_SUPPLIED',conditional_on_given_mapping=True,blind_sync_performed=False,
                vote_order='channel, eligible receiver time, coordinate slot::8; Counter encounter ties; strict >0',
                signed_votes=signed_votes,zero_coefficient_mask=zeros)


def guided_velocity(z,conditional,unconditional,sigma,target,mask,index,count):
    """Exact PAYLOAD_MULTI active arithmetic; no pilot or old state dependency."""
    import torch
    if type(index) is not int or not 0<=index<50 or not math.isfinite(float(sigma)) or sigma<=0:raise ValueError("fixed native step/sigma")
    z,c,u=z.float(),conditional.float(),unconditional.float()
    enabled=index>=25;detail=dict(enabled=enabled,arm="PAYLOAD_MULTI",state_control_cfg="float32")
    if enabled:
        count("payload_gradient",False)
        dp,pinfo=original.local_delta(z-float(sigma)*c,target,mask)
        count("payload_gradient",True)
        delta=dp+torch.zeros_like(dp) # retain successful payload-only branch arithmetic
        c=c-delta/float(sigma)
        detail.update(payload=pinfo,pilot=None,payload_delta_l2=float(dp.double().square().sum().sqrt()))
    velocity=u+5.0*(c-u)
    if not bool(torch.isfinite(velocity).all()):raise FloatingPointError("nonfinite CFG")
    return velocity.detach(),detail
