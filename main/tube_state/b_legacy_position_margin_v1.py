"""Position-key signatures and nearest halfspace corrections; score-only clean probe."""
from collections import Counter
import math
import numpy as np
import torch
from . import local_segmented_terminal_sync_v1 as geometry
from . import b_segmented_payload_v1 as payload
from . import grow_video_reference as grow
from .legacy_reference_v1.codebook import _build_integer_seed, _build_normalized_direction, _build_binary_code

ARMS=("P0","PM")
KEY_IDS=("key00","key01")
KEYS=("watermark","watermark-wrong")
REFERENCE="B-legacy-position-margin-v1"
TIMES=tuple(range(1,45))
ROIS=geometry.ROIS
GAMMA=1.0  # single concrete recommendation, not a claim of adopted physical strength
DIMENSION=768


def codebook(key):
    if not isinstance(key,str) or not key:raise ValueError("nonempty key required")
    directions=[];signs=[]
    for r in range(4):
        ds=[];bs=[]
        for t in TIMES:
            index=(t-1)*4+r
            seed=_build_integer_seed(key+"|direction",REFERENCE,index)
            ds.append(_build_normalized_direction(seed,DIMENSION,"l2"))
            bs.append(_build_binary_code(key+"|symbol",REFERENCE,index))
        directions.append(ds);signs.append(bs)
    d=torch.tensor(directions,dtype=torch.float64).reshape(4,44,12,8,8)
    b=torch.tensor(signs,dtype=torch.float64)
    n=d.flatten(2).square().sum(2)
    if not bool(torch.isfinite(d).all() and (n>0).all()):raise ValueError("invalid direction")
    return dict(directions=d,signs=b,norm_squared=n,public_reference=REFERENCE,
                index_rule="(position-1)*4+ROI; position1..44",generator="legacy uniform/l2/round8")


def ranks(roi_q):
    q=torch.as_tensor(roi_q,dtype=torch.float64)
    if q.shape!=(44,4,44) or not bool(torch.isfinite(q).all()):raise ValueError("finite44x4x44 evidence required")
    pooled=q.mean(1);gaps=torch.empty_like(pooled);rows=[]
    for i,row in enumerate(pooled):
        best=float(row.max());ties=[j+1 for j in range(44) if float(row[j])==best]
        for k in range(44):gaps[i,k]=row[k]-torch.cat((row[:k],row[k+1:])).max()
        rows.append(dict(observed_time=i+1,provisional_best=ties[0] if len(ties)==1 else None,
                         exact_max_ties=ties,reliable_identity=None,phase_status="UNCALIBRATED"))
    return dict(status="SCORE_OBSERVED",evidence_level="DIRECTION_EVIDENCE_ONLY",roi_q=q.tolist(),pooled_q=pooled.tolist(),
                competitor_margins=gaps.tolist(),rows=rows,identity_status="UNCALIBRATED",
                reliable_placement=None,truth_inputs=False,message_inputs=False,
                scope="complete clean full-history canonical grid only; no unknown phase/path claim")


def observe(latent,book):
    geometry.validate_latent(latent)
    z=latent.detach().cpu().double();roi=torch.empty((44,4,44),dtype=torch.float64)
    for r,(y0,y1,x0,x1) in enumerate(ROIS):
        patches=z[0,4:16,1:45,y0:y1,x0:x1].permute(1,0,2,3).reshape(44,-1)
        roi[:,r]=(patches@book["directions"][r].reshape(44,-1).T)*book["signs"][r]
    return ranks(roi)


def unavailable(status):
    return dict(status=status,evidence_level="DIRECTION_EVIDENCE_ONLY",identity_status="UNCALIBRATED",reliable_placement=None,
                roi_q=[[[None]*44 for r in range(4)] for t in TIMES],
                pooled_q=[[None]*44 for t in TIMES],competitor_margins=[[None]*44 for t in TIMES],
                rows=[dict(observed_time=t,status=status,provisional_best=None,exact_max_ties=None,
                           reliable_identity=None,phase_status="UNCALIBRATED") for t in TIMES],
                truth_inputs=False,message_inputs=False)


def construct(latent,book,gamma=GAMMA,*,native_observation=None):
    geometry.validate_latent(latent)
    if latent.dtype!=torch.float32:raise ValueError("one final FP32 terminal required")
    if not math.isfinite(gamma) or gamma<=0:raise ValueError("finite positive gamma required")
    observed=native_observation if native_observation is not None else observe(latent,book)
    if observed["status"]!="SCORE_OBSERVED":raise ValueError("native scores unavailable")
    q=torch.tensor([[observed["roi_q"][t][r][t] for t in range(44)] for r in range(4)],dtype=torch.float64)
    short=(gamma-q).clamp_min(0);coef=short/book["norm_squared"]
    before=latent.detach().cpu();out=before.clone();rows=[]
    for r,(y0,y1,x0,x1) in enumerate(ROIS):
        original=before[0,4:16,1:45,y0:y1,x0:x1].permute(1,0,2,3).double()
        delta=coef[r,:,None,None,None]*book["signs"][r,:,None,None,None]*book["directions"][r]
        changed=(original+delta).float()  # one cast, no compensation/reprojection
        out[0,4:16,1:45,y0:y1,x0:x1]=changed.permute(1,0,2,3)
        for t in range(44):
            actual=changed[t].double()-original[t]
            rows.append(dict(position=t+1,roi=r,q_before=float(q[r,t]),active=bool(short[r,t]>0),
                norm_squared=float(book["norm_squared"][r,t]),reference_sign=int(book["signs"][r,t]),
                channels=[4,16],ROI=[y0,y1,x0,x1],flatten_order="channel,y,x",
                q_after_float64=float(q[r,t]+coef[r,t]*book["norm_squared"][r,t]),coefficient=float(coef[r,t]),
                ideal_squared_energy=float(delta[t].square().sum()),actual_FP32_squared_energy=float(actual.square().sum())))
    geometry.validate_latent(out)
    return out,dict(status="CONSTRUCTED",gamma=gamma,units=rows,unit_denominator=176,
        ideal_squared_energy=sum(x["ideal_squared_energy"] for x in rows),
        actual_FP32_squared_energy=sum(x["actual_FP32_squared_energy"] for x in rows),
        zero_host_squared_energy=float((gamma*gamma/book["norm_squared"]).sum()),
        nearest_step="max(0,gamma-q)/actual_norm_squared * sign * rounded_direction",
        FP32_correction_attempts=0,old22_constraint=False,payload_rewritten=False)


def payload_spectrum(latent):
    """Exactly one FFT per encoded arm; both keys consume this same spectrum."""
    geometry.validate_latent(latent)
    return torch.fft.fft2(latent[0,:4,1:45].float(),dim=(-2,-1),norm="ortho").real


def payload_from_spectrum(spectrum,key):
    if tuple(spectrum.shape)!=(4,44,40,64) or not bool(torch.isfinite(spectrum).all()):
        raise ValueError("finite shared spectrum [4,44,40,64] required")
    coords=grow.coordinates(key);h=[p[0] for p in coords];w=[p[1] for p in coords]
    values=spectrum[:,:,h,w]
    signed=(2*(values>0).to(torch.int8)-1).cpu().numpy().astype(np.int8)
    zeros=(values==0).to(torch.int8).cpu().numpy().astype(np.int8)
    return payload_from_votes(signed,zeros)


def payload_from_votes(signed,zeros):
    result=payload.decode_signed_votes(payload.segment_plan(list(range(181))),signed,zeros)
    bits=(signed>0).astype(np.int8);local=[]
    for t in range(44):
        row=[]
        for j in range(8):
            seq=bits[:,t,j::8].reshape(-1).tolist();ones=sum(seq)
            row.append(dict(bit_index=8*(t//11)+j,decoded=int(Counter(seq).most_common(1)[0][0]),
                            count=len(seq),ones=ones,tie=ones*2==len(seq),zero_coefficients=int(zeros[:,t,j::8].sum())))
        local.append(dict(observed_time=t+1,canonical_segment=t//11,bits=row))
    result.update(local_cells=local,scope="known clean canonical grouping; payload side effect, no sync benefit",
                  reliable_sync_placement=False)
    result["signed_votes"]=result["signed_votes"].tolist()
    result["zero_coefficient_mask"]=result["zero_coefficient_mask"].tolist()
    return result


def unavailable_payload(status):
    return dict(status=status,decoded_bits=[None]*32,
        local_cells=[dict(observed_time=t,bits=[dict(bit_index=8*((t-1)//11)+j,decoded=None,status=status) for j in range(8)]) for t in TIMES],
        signed_votes=None,zero_coefficient_mask=None,planned_signed_vote_slots=42240,
        scope="known clean canonical grouping; side effect only",reliable_sync_placement=False)
