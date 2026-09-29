"""Public pilot synchronization and local FFT-real Flow control; no runtime imports."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math

ARMS=("OFF","PAYLOAD_LAST","PILOT_LAST")
VIEWS=("FULL_RESAVED181","CROP5_129")
LFSR_BITS="111100010011010"


@dataclass(frozen=True)
class PublicProtocol:
    height:int=320
    width:int=512
    full_frames:int=181
    crop_frames:int=129
    channels:tuple=(0,1,2,3)
    payload_bits:int=32
    pilot_channel:int=4
    pilot_count:int=64
    threshold:float=0.5
    tie_atol:float=1e-12
    alpha:float=0.5


PUBLIC=PublicProtocol()


def digest(value):
    return hashlib.sha256(json.dumps(value,separators=(",",":"),sort_keys=True).encode()).hexdigest()


def message_bits(value):
    return [int(bit) for byte in value.encode("utf-8") for bit in f"{byte:08b}"]


def band():return [(h,w) for h in range(8,20) for w in range(12,32)]


def payload_coordinates(key):
    import numpy as np
    coords=band();np.random.default_rng(sum(map(ord,key))).shuffle(coords)
    return coords


def pilot_hash(domain,key,h,w):
    if not isinstance(key,str) or "\0" in key:raise ValueError("public key must be a NUL-free string")
    # Exact bytes: UTF-8 domain NUL key NUL decimal h NUL decimal w; no final NUL.
    return hashlib.sha256(f"{domain}\0{key}\0{h}\0{w}".encode("utf-8")).digest()


def pilot_layout(key):
    coords=sorted(band(),key=lambda x:(pilot_hash("VTSB1/pilot/coords",key,*x),*x))[:64]
    pn=[2*(pilot_hash("VTSB1/pilot/sign",key,*x)[0]&1)-1 for x in coords]
    return coords,pn


def layout_receipt(key,wrong_key):
    def one(k):
        coords,pn=pilot_layout(k);payload=payload_coordinates(k)
        return dict(pilot_support_sha256=digest(sorted(coords)),pilot_pn_sha256=digest(sorted([h,w,s] for (h,w),s in zip(coords,pn))),
            pilot_coordinate_order_sha256=digest(coords),payload_assignment_sha256=digest(sorted([h,w,i%8] for i,(h,w) in enumerate(payload))),
            payload_seed=sum(map(ord,k)),pilot_support=coords,pilot_pn=pn)
    right,wrong=one(key),one(wrong_key)
    for field in ("pilot_support_sha256","pilot_pn_sha256","payload_assignment_sha256"):
        if right[field]==wrong[field]:raise ValueError("correct/wrong key layout collision: "+field)
    return dict(correct=right,wrong=wrong,domain_encoding="UTF8(domain NUL key NUL decimal_h NUL decimal_w); no trailing NUL",
        pn_rule="2*(SHA256(sign-domain bytes)[0] & 1)-1",lfsr_bits=LFSR_BITS)


def chips():
    bits=[1,1,1,1]
    for n in range(11):bits.append(bits[n+1]^bits[n])
    assert ''.join(map(str,bits))==LFSR_BITS
    return [2*x-1 for x in bits]


def temporal_sign(t):
    if not 0<=t<=45:raise ValueError("finite temporal pilot has no cyclic extension")
    return chips()[0 if t==0 else (t-1)//3]


def candidates(length):
    if length==181:return [dict(b=0,g=0,a=0,R=45)]
    if length==129:return [dict(b=b,g=(-b)%4,a=(b+(-b)%4)//4,R=31) for b in range(53)]
    raise ValueError("public receiver accepts only 181 or 129 frames")


def phases(length):return sorted({row['g'] for row in candidates(length)})


def phase_slice(length,g):
    if g not in phases(length):raise ValueError("phase outside public candidate set")
    kept=1+4*((length-g-1)//4)
    return g,g+kept


def build_target(latent,key,bits,arm,public=PUBLIC):
    import torch
    if public!=PUBLIC or tuple(latent.shape)!=(1,16,46,40,64):raise ValueError("fixed writer geometry required")
    if arm not in ARMS or len(bits)!=32 or any(type(b) is not int or b not in (0,1) for b in bits):raise ValueError("fixed arm and32 bits required")
    target=torch.zeros_like(latent);mask=torch.zeros_like(latent,dtype=torch.bool)
    for i,ch in enumerate(public.channels):
        for j,(h,w) in enumerate(payload_coordinates(key)):
            target[0,ch,:,h,w]=(2*bits[i*8+j%8]-1)*public.alpha;mask[0,ch,:,h,w]=True
    if arm=="PILOT_LAST":
        coords,pn=pilot_layout(key)
        signs=torch.tensor([temporal_sign(t) for t in range(46)],dtype=latent.dtype,device=latent.device)
        for (h,w),sign in zip(coords,pn):
            target[0,4,:,h,w]=sign*public.alpha*signs;mask[0,4,:,h,w]=True
    return target,mask


def local_delta(clean,target,mask):
    import torch
    import torch.nn.functional as F
    n=int(mask.sum());eta=n/8
    if n not in (44160,47104):raise ValueError("fixed payload or payload+pilot mask required")
    with torch.enable_grad():
        leaf=clean.detach().clone().requires_grad_(True)
        loss=F.mse_loss(torch.fft.fft2(leaf,norm="ortho").real[mask],target[mask])
        grad=torch.autograd.grad(loss,leaf)[0];delta=(-eta*grad).detach()
    if not bool(torch.isfinite(delta).all()) or not math.isfinite(float(loss)):raise FloatingPointError("nonfinite local update")
    payload=float(delta[:,:4].double().square().sum());pilot=float(delta[:,4:5].double().square().sum())
    return delta,dict(mask_count=n,eta=eta,loss=float(loss.detach()),payload_delta_squared_l2=payload,
        pilot_delta_squared_l2=pilot,total_delta_squared_l2=float(delta.double().square().sum()),
        channel_cross_inner_product=0.0,total_budget_matched=False)


def guided_velocity(z,c,u,sigma,target,mask,enabled):
    if not math.isfinite(float(sigma)) or sigma<=0:raise ValueError("positive native sigma required")
    z,c,u=z.float(),c.float(),u.float();row=dict(enabled=enabled,control_dtype="float32",cfg_dtype="float32")
    if enabled:
        delta,detail=local_delta(z-float(sigma)*c,target,mask);c=c-delta/float(sigma);row.update(detail)
    import torch
    velocity=(u+5*(c-u)).detach()
    if not bool(torch.isfinite(velocity).all()):raise FloatingPointError('nonfinite FP32 CFG')
    return velocity,row


def phase_features(normalized,key,R,public=PUBLIC):
    """Only observed normalized latent and public key/regular support enter."""
    import torch
    if public!=PUBLIC or normalized.ndim!=5 or tuple(normalized.shape[:2])!=(1,16) or tuple(normalized.shape[-2:])!=(40,64) or normalized.shape[2]<R+1:
        raise ValueError("observed phase latent geometry mismatch")
    if not bool(torch.isfinite(normalized).all()):raise FloatingPointError("nonfinite observation")
    spectrum=torch.fft.fft2(normalized.float(),norm="ortho").real
    coords=payload_coordinates(key);h=[x[0] for x in coords];w=[x[1] for x in coords]
    decoded=[];votes=[]
    for ch in public.channels:
        raw=(spectrum[0,ch,1:R+1,h,w]>0).to(torch.int64).cpu().numpy()
        for bit in range(8):
            values=raw[:,bit::8].reshape(-1).tolist();ones=sum(values)
            decoded.append(Counter(values).most_common(1)[0][0])
            votes.append(dict(ones=ones,zeros=len(values)-ones,count=len(values),tie=2*ones==len(values)))
    coords,_=pilot_layout(key);h=[x[0] for x in coords];w=[x[1] for x in coords]
    pilot=spectrum[0,4,1:R+1,h,w].detach().cpu().double().tolist()
    return dict(status="READ",decoded_bits=decoded,votes=votes,pilot_fft=pilot,R=R,truth_used=False,
        excluded_first_latent=True,normalized_shape=list(normalized.shape))


def score_candidate(feature,key,candidate):
    import numpy as np
    try:
        R=candidate['R'];X=np.asarray(feature['pilot_fft'],dtype=np.float64)
        if X.shape!=(R,64) or not np.isfinite(X).all():raise ValueError("nonfinite or wrong-size pilot observation")
        _,pn=pilot_layout(key)
        P=np.asarray([temporal_sign(j+candidate['a']) for j in range(1,R+1)])[:,None]*np.asarray(pn)[None,:]
        energy=float(np.sum(X*X,dtype=np.float64));dot=float(np.sum(X*P,dtype=np.float64))
        if not math.isfinite(energy) or not math.isfinite(dot):raise FloatingPointError("nonfinite pilot reduction")
        normalizer=X.size*energy
        if not math.isfinite(normalizer):raise FloatingPointError('nonfinite cosine normalization')
        score=0.0 if energy==0 else dot/math.sqrt(normalizer)
        if not math.isfinite(score):raise FloatingPointError("nonfinite cosine")
        return dict(**candidate,status="NO_ENERGY" if energy==0 else "SCORED",score=score,energy=energy,n=X.size)
    except Exception as exc:return dict(**candidate,status="FAILED",score=None,error=f"{type(exc).__name__}: {exc}")


def decide(rows,length,public=PUBLIC):
    expected=candidates(length)
    if len(rows)!=len(expected) or any(any(row.get(k)!=value for k,value in ref.items()) for row,ref in zip(rows,expected)):
        raise ValueError("candidate denominator/order mismatch")
    valid=[x for x in rows if x['status'] in ("SCORED","NO_ENERGY") and x.get('score') is not None and math.isfinite(x['score'])]
    base=dict(threshold=public.threshold,tie_atol=public.tie_atol,trivial_full=length==181,selected_b=None,selected_g=None,
        winner_is_not_confidence=True,truth_used=False)
    if len(valid)!=len(expected):return dict(**base,status="SEARCH_INCOMPLETE",ties=[],top_gap=None,max_score=None)
    ordered=sorted(valid,key=lambda x:(-x['score'],x['b']));best=ordered[0]['score']
    ties=[x['b'] for x in valid if best-x['score']<=public.tie_atol]
    status="NO_PILOT" if best<public.threshold else ("AMBIGUOUS" if len(ties)>1 else "LOCATED")
    if status=="LOCATED":base.update(selected_b=ordered[0]['b'],selected_g=ordered[0]['g'])
    return dict(**base,status=status,ties=ties,top_gap=best-ordered[1]['score'] if len(ordered)>1 else None,max_score=best,
        no_energy_candidates=[x['b'] for x in valid if x['status']=='NO_ENERGY'])
