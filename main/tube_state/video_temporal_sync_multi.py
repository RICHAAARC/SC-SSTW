"""MULTI control and public window observations; no clock estimator or runtime I/O."""
from __future__ import annotations
import math
import hashlib
from main.tube_state import video_temporal_sync_bridge as bridge

PUBLIC=bridge.PUBLIC
ARMS=('OFF','PAYLOAD_MULTI','PILOT_MULTI')
VIEWS=('FULL_RESAVED181','CROP4_129','CROP5_129','CROP6_129','CROP7_129')
WINDOW_SCHEMA='video-temporal-sync-multi-window-v1'
message_bits=bridge.message_bits
layout_receipt=bridge.layout_receipt
candidates=bridge.candidates
phases=bridge.phases
phase_slice=bridge.phase_slice
score_candidate=bridge.score_candidate
decide=bridge.decide
phase_features=bridge.phase_features  # Separate terminal diagnostic only.
guided_velocity=bridge.guided_velocity


def build_target(latent,key,bits,arm,public=PUBLIC):
    mapped={'OFF':'OFF','PAYLOAD_MULTI':'PAYLOAD_LAST','PILOT_MULTI':'PILOT_LAST'}
    if arm not in mapped:raise ValueError('fixed MULTI arm required')
    return bridge.build_target(latent,key,bits,mapped[arm],public)


def control_enabled(arm,index):
    if arm not in ARMS or not 0<=index<50:raise ValueError('fixed arm and native50 index required')
    return arm!='OFF' and 25<=index<=49


def window_row_count(length,g):
    left,right=phase_slice(length,g)
    return (right-left-1)//4


def aggregate_ordered_windows(rows):
    """Aggregate an explicit public observation order, including repeats/skips.

    No path estimation or source-clock interpretation occurs here. A global
    hard-vote tie uses the first coefficient bit of the first selected window,
    not that window's majority. Negative and exact-zero coefficients vote0.
    """
    if not rows:raise ValueError('at least one observed window required')
    for row in rows:
        if len(row['payload_stats'])!=32:raise ValueError('32 bit statistics required')
    decoded=[];votes=[];soft=[]
    for bit in range(32):
        stats=[row['payload_stats'][bit] for row in rows]
        for s in stats:
            counts=[s[k] for k in ('positive_count','negative_count','zero_count','support_count')]
            if any(type(x) is not int or x<0 for x in counts) or counts[3]!=30 or sum(counts[:3])!=30:
                raise ValueError('invalid per-bit coefficient counts')
            if s['first_bit'] not in (0,1) or not math.isfinite(s['sum']) or not math.isfinite(s['sumsq']) or s['sumsq']<0:
                raise ValueError('invalid soft statistic or first bit')
        ones=sum(s['positive_count'] for s in stats);negative=sum(s['negative_count'] for s in stats)
        zero=sum(s['zero_count'] for s in stats);count=sum(s['support_count'] for s in stats)
        tie=ones*2==count
        decoded.append(stats[0]['first_bit'] if tie else int(ones*2>count))
        votes.append(dict(ones=ones,zeros=negative+zero,count=count,tie=tie))
        total=math.fsum(s['sum'] for s in stats);squares=math.fsum(s['sumsq'] for s in stats)
        if not math.isfinite(total) or not math.isfinite(squares):raise FloatingPointError('nonfinite aggregate')
        soft.append(dict(sum=total,sumsq=squares,positive_count=ones,negative_count=negative,zero_count=zero,
            support_count=count,first_bit=stats[0]['first_bit']))
    return dict(decoded_bits=decoded,votes=votes,payload_stats=soft,ordered_window_count=len(rows),truth_used=False)


def phase_observations(normalized,key,length,g,public=PUBLIC):
    """One FFT supplies unchanged primary support and all public regular windows."""
    import torch
    if public!=PUBLIC:raise ValueError('fixed public protocol required')
    left,right=phase_slice(length,g);all_R=window_row_count(length,g);R=candidates(length)[0]['R']
    if tuple(normalized.shape)!=(1,16,all_R+1,40,64):raise ValueError('observed phase shape mismatch')
    if not bool(torch.isfinite(normalized).all()):raise FloatingPointError('nonfinite observation')
    spectrum=torch.fft.fft2(normalized.float(),norm='ortho').real
    payload_coords=bridge.payload_coordinates(key);h=[x[0] for x in payload_coords];w=[x[1] for x in payload_coords]
    # [regular time,32 bits,30 coefficients]. Preserve key-coordinate order.
    channel_values=[spectrum[0,ch,1:,h,w].reshape(all_R,30,8).transpose(1,2) for ch in public.channels]
    values=torch.cat(channel_values,dim=1).detach().cpu().double()
    sums=values.sum(-1);squares=values.square().sum(-1)
    positives=(values>0).sum(-1);negatives=(values<0).sum(-1);zeros=(values==0).sum(-1)
    first=(values[:,:,0]>0).to(torch.int64)
    coords,pn=bridge.pilot_layout(key);ph=[x[0] for x in coords];pw=[x[1] for x in coords]
    pilot=spectrum[0,4,1:,ph,pw].detach().cpu().double().tolist()
    windows=[]
    for i in range(all_R):
        stats=[dict(sum=float(sums[i,k]),sumsq=float(squares[i,k]),positive_count=int(positives[i,k]),
            negative_count=int(negatives[i,k]),zero_count=int(zeros[i,k]),support_count=30,first_bit=int(first[i,k])) for k in range(32)]
        j=i+1
        windows.append(dict(phase_g=g,observed_regular_j=j,nominal_stride4=4,
            nominal_received_new_frame_interval=[g+4*j-3,g+4*j+1],pilot_fft=pilot[i],payload_stats=stats))
    primary=aggregate_ordered_windows(windows[:R])
    primary=dict(status='READ',decoded_bits=primary['decoded_bits'],votes=primary['votes'],pilot_fft=pilot[:R],R=R,
        truth_used=False,excluded_first_latent=True,normalized_shape=list(normalized.shape))
    artifact=dict(schema=WINDOW_SCHEMA,phase_g=g,received_frame_count=length,used_received_frame_interval=[left,right],
        public_key_sha256=hashlib.sha256(key.encode('utf-8')).hexdigest(),
        public_layout_sha256=bridge.digest(dict(payload_coordinates=payload_coords,pilot_coordinates=coords,pilot_pn=pn)),
        nominal_stride4=4,grid_meaning='nominal newly supplied RGB frames, not VAE receptive field',
        first_latent_excluded=True,primary_regular_count=R,window_count=all_R,payload_bit_count=32,pilot_coefficient_count=64,
        transform='fft2_ortho_real_float32',statistic_reduction='float64',time_dependent_payload='disabled_unverified',
        windows=windows)
    return primary,artifact
