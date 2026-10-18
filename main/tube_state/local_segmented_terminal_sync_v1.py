"""B terminal state only: no message or payload directions in writer/observer."""
import hashlib
import hmac
import math
import torch
from main.tube_state.local_joint_positive_projection_margin_v1 import (
    LATENT_SHAPE, VIDEO_SHAPE, ROIS, STATES, GAMMA_S, DOMAIN,
    validate_latent, project_state, delta_statistics,
)
ARMS=("P","S")
LAYERS=("terminal","mp4")
KEYS={"CORRECT":"watermark","WRONG":"watermark-wrong"}

def public_codebook(key):
    if not isinstance(key,str) or not key:raise ValueError("nonempty key required")
    def digest(text):return hmac.new(key.encode(),text.encode("ascii"),hashlib.sha256).digest()
    basis=[(c,a,b) for c in range(4,16) for a in range(8) for b in range(8) if (a,b)!=(0,0)]
    directions=torch.zeros((4,22,12,2,8,8),dtype=torch.float64);selected=[]
    for r in range(4):
        ids=sorted(range(756),key=lambda i:(digest(f"{DOMAIN}|roi={r}|basis={i}|perm"),i))[:22]
        selected.append(ids)
        for j,i in enumerate(ids):
            c,a,b=basis[i];polarity=1 if digest(f"{DOMAIN}|roi={r}|basis={i}|sign")[0]&1==0 else -1
            for y in range(8):
                for x in range(8):
                    v=polarity*(1-2*(((a&y).bit_count()+(b&x).bit_count())%2))/math.sqrt(128)
                    directions[r,j,c-4,0,y,x]=v;directions[r,j,c-4,1,y,x]=-v
    return dict(key=key,domain=DOMAIN,directions=directions,selected_basis=selected)

def observe_state(received,key,public):
    validate_latent(received)
    if public["key"]!=key or public["domain"]!=DOMAIN:raise ValueError("state codebook/key mismatch")
    if tuple(public["directions"].shape)!=(4,22,12,2,8,8):raise ValueError("state-only22 codebook required")
    z=received.detach().cpu().double();scores=torch.empty((22,4,22),dtype=torch.float64)
    for k in STATES:
        for r,(y0,y1,x0,x1) in enumerate(ROIS):
            scores[k,r]=public["directions"][r].reshape(22,-1)@z[0,4:16,1+2*k:3+2*k,y0:y1,x0:x1].reshape(-1)
    pooled=scores.mean(1)
    if not bool(torch.isfinite(pooled).all()):raise ValueError("nonfinite state scores")
    rows=[]
    for k,row in enumerate(pooled):
        winners=torch.nonzero(row==row.max()).flatten().tolist()
        rows.append(dict(position=k,label=winners[0] if len(winners)==1 else None,tied_candidates=winners,
                         status="READ" if len(winners)==1 else "UNRESOLVED"))
    return dict(status="READ",truth_used=False,message_used=False,roi_scores=scores.tolist(),
                pooled_scores=pooled.tolist(),states=rows,geometry="clean full public181 grid; not an edit path")

def write_state_only(source,host_state,public):
    validate_latent(source)
    if source.dtype!=torch.float32:raise ValueError("saved source must be FP32")
    host=torch.tensor(host_state.get("pooled_scores",[]),dtype=torch.float64)
    if host_state.get("status")!="READ" or host.shape!=(22,22) or not bool(torch.isfinite(host).all()):
        raise ValueError("finite source state table required")
    if public["directions"].shape!=(4,22,12,2,8,8):raise ValueError("state-only directions required")
    out=source.detach().cpu().double().clone();records=[];energy=0.
    for k in STATES:
        updated,active=project_state(host[k],k);delta=updated-host[k]
        energy+=4*float(delta.square().sum())
        records.append(dict(position=k,active_competitors=active,state_score_correction=delta.tolist()))
        for r,(y0,y1,x0,x1) in enumerate(ROIS):
            out[0,4:16,1+2*k:3+2*k,y0:y1,x0:x1]+=(delta@public["directions"][r].reshape(22,-1)).reshape(12,2,8,8)
    candidate=out.float();validate_latent(candidate)
    return candidate,dict(status="CONSTRUCTED",rows=records,gamma_s=GAMMA_S,
        continuous_distance_squared=energy,fp32_materializations=1,norm_cap=None,
        message_inputs=False,payload_directions=0,payload_corrections=0,
        interpretation="state-only on saved segmented host; real efficacy not assumed")

def unavailable_state(status):
    return dict(status=status,truth_used=False,message_used=False,
        roi_scores=None,pooled_scores=None,states=[dict(position=k,label=None,status=status,tied_candidates=[]) for k in STATES])

def unavailable_payload(status):
    return dict(status=status,truth_inputs=False,net_payload_bits=32,decoded_bits=[None]*32,
        segments=[dict(segment=g,status=status,cells=[],planned_bits=8,
                       bits=[dict(bit_index=8*g+b,decoded=None,count=0,status=status) for b in range(8)]) for g in range(4)])

