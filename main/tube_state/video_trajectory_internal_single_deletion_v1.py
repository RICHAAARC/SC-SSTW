"""Pure fixed single-jump search and input-map construction; no runtime or truth selection."""
import hashlib,math
import numpy as np
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as prior
sync=prior.sync
PUBLIC=prior.PUBLIC
N=177
MODES=("RAW","GLOBAL_ALIGN","PATH_ALIGN","TRUTH_PATH")
def hypotheses():
    return [dict(family="H0",b=b,k=None) for b in range(5)]+[
        dict(family="H1",b=b,k=k) for b in range(4) for k in range(1,N)]
def path_map(h):
    if h["family"]=="H0" and type(h["b"]) is int and h["b"] in range(5) and h["k"] is None:
        return [r+h["b"] for r in range(N)]
    if h["family"]=="H1" and type(h["b"]) is int and h["b"] in range(4) and type(h["k"]) is int and h["k"] in range(1,N):
        return [r+h["b"]+int(r>=h["k"]) for r in range(N)]
    raise ValueError("outside fixed709 grammar")
def correction(h):
    path_map(h);p=h["b"]%4;entries=[(r,False) for r in range(N)]
    if h["family"]=="H1":entries.insert(h["k"],(h["k"]-1,True))
    entries=[(entries[0][0],True)]*p+entries
    used=entries[:N];mapping=[r for r,_ in used]
    return dict(hypothesis={k:h[k] for k in ("family","b","k")},nominal_source_map=list(range(h["b"]-p,h["b"]-p+N)),received_index_map=mapping,synthetic_output_indices=[i for i,(_,s) in enumerate(used) if s],
        dropped_received_indices=sorted(set(range(N))-set(mapping)),phase=p,
        inserted_gap_output_index=p+h["k"] if h["family"]=="H1" and p+h["k"]<N else None,
        operation="prepend+truncate" if h["family"]=="H0" else "insert previous received frame, prepend, truncate")
def identity():
    return dict(hypothesis=None,nominal_source_map=None,received_index_map=list(range(N)),synthetic_output_indices=[],dropped_received_indices=[],phase=0,inserted_gap_output_index=None,operation="identity")
def argmax(rows):
    if not rows or any(r["status"]!="SCORED" or r["score"] is None or not math.isfinite(r["score"]) for r in rows):
        return dict(status="UNRESOLVED",reason="INCOMPLETE_OR_NONFINITE",path=None,top_paths=[],best_score=None,gap=None,sync_accepted=False)
    best=max(r["score"] for r in rows);top=[{k:r[k] for k in ("family","b","k")} for r in rows if abs(r["score"]-best)<=PUBLIC.tie_atol]
    values=sorted((r["score"] for r in rows),reverse=True)
    return dict(status="ESTIMATED" if len(top)==1 else "UNRESOLVED",reason=None if len(top)==1 else "TIED_TOP",
        path=top[0] if len(top)==1 else None,top_paths=top,best_score=best,gap=values[0]-values[1] if len(values)>1 else None,sync_accepted=False)
def reduce_cache(cache,key):
    if len(cache)!=885 or [(x["r"],x["d"]) for x in cache]!=[(r,d) for r in range(N) for d in range(5)]:
        raise ValueError("fixed frame-displacement cache required")
    values={(x["r"],x["d"]):x for x in cache}
    def aggregate(indices):
        num=rho=0.0
        for r,d in indices:
            x=values[r,d]
            if x["status"]!="SCORED" or any(x.get(k) is None or not math.isfinite(x[k]) for k in ("numerator","rho")) or x["rho"]<=0:
                return dict(status="FAILED",numerator=None,rho=None,score=None)
            num+=x["numerator"];rho+=x["rho"]
        return dict(status="SCORED",numerator=num,rho=rho,score=num/rho)
    rows=[dict(**h,**aggregate([(r,s-r) for r,s in enumerate(path_map(h))])) for h in hypotheses()]
    local=[dict(bin=j,received_start=4*j,received_stop=min(4*j+4,N),d=d,
                **aggregate([(r,d) for r in range(4*j,min(4*j+4,N))])) for j in range(45) for d in range(5)]
    local_top=[]
    for j in range(45):
        group=local[j*5:j*5+5]
        finite=all(x["status"]=="SCORED" and math.isfinite(x["score"]) for x in group)
        best=max(x["score"] for x in group) if finite else None
        top=[x["d"] for x in group if abs(x["score"]-best)<=PUBLIC.tie_atol] if finite else []
        local_top.append(dict(bin=j,status="ESTIMATED" if len(top)==1 else "UNRESOLVED",
            top_displacements=top,canonical_displacement=top[0] if len(top)==1 else None,best_score=best))
    h0=argmax(rows[:5]);h1=argmax(rows[5:]);joint=argmax(rows)
    return dict(status="COMPLETE" if joint["reason"]!="INCOMPLETE_OR_NONFINITE" else "RETAINED_FAILURES",truth_inputs=False,
        method_version=PUBLIC.method_version,diagnostic_version="internal-single-deletion-v1",key_id=sync.key_identifier(key),
        frame_cache=cache,candidate_rows=rows,local_grid=local,local_top=local_top,
        estimates=dict(H0=h0,joint=joint),
        family_comparison=dict(best_H0=h0["best_score"],best_H1=h1["best_score"],
            H1_minus_H0=h1["best_score"]-h0["best_score"] if h0["best_score"] is not None and h1["best_score"] is not None else None),
        counts=dict(frame_displacement_cells=885,path_candidates=709,local_grid_cells=225),
        interpretation="No threshold or complexity penalty. H1 winning is not detection; local bins are not Wan receptive fields.")
def score(received,key):
    z=sync.validate_received_latent(received,PUBLIC)
    if len(z)!=N:raise ValueError("only177")
    # Signed full-direction age slices; original terminal source180 keeps unit rho.
    weights=np.empty((181,4,40,64),np.float64)
    for t in range(46):
        start=4*t;stop=min(start+4,181);sign=sync.sync_sign(key,t,PUBLIC)
        for y,x in sync.spatial_patch_coordinates(40,64,PUBLIC):
            weights[start:stop,:,y:y+4,x:x+4]=sign*sync.patch_direction(key,t,y,x,PUBLIC).astype(np.float64)
    nums=[];rhos=[];value=z.astype(np.float64)
    for d in range(5):
        w=weights[d:d+N];nums.append(np.sum(value*w,axis=(1,2,3)));rhos.append(np.sum(w*w,axis=(1,2,3)))
    cache=[dict(r=r,d=d,source_index=r+d,source_tubelet=(r+d)//4,source_age=(r+d)%4,status="SCORED",
        numerator=float(nums[d][r]),rho=float(rhos[d][r])) for r in range(N) for d in range(5)]
    return reduce_cache(cache,key)
def missing(key,error):
    result=reduce_cache([dict(r=r,d=d,source_index=r+d,source_tubelet=(r+d)//4,source_age=(r+d)%4,
        status="FAILED",error=str(error),numerator=None,rho=None) for r in range(N) for d in range(5)],key)
    result["error"]=str(error);return result
def estimates(readout,key):
    try:
        if readout["key_id"]!=sync.key_identifier(key) or readout["truth_inputs"] is not False:raise ValueError("blind/key mismatch")
        rebuilt=reduce_cache(readout["frame_cache"],key)
        if readout["status"]!=rebuilt["status"] or readout["candidate_rows"]!=rebuilt["candidate_rows"] or readout["local_grid"]!=rebuilt["local_grid"] or readout["counts"]!=rebuilt["counts"]:raise ValueError("incomplete/inconsistent support")
        return rebuilt["estimates"]
    except (KeyError,TypeError,ValueError) as exc:
        e=dict(status="UNRESOLVED",reason=str(exc),path=None,top_paths=[],best_score=None,gap=None,sync_accepted=False)
        return dict(H0=e.copy(),joint=e.copy())
def add_slot(plan,oid,key,mode,operation,spec,*,error=None,oracle=False):
    lid=oid+"/"+key+"/"+mode
    if operation is None:
        plan["logical_slots"][lid]=dict(status="UNRESOLVED",physical_read=None,error=error,oracle=oracle);return
    mapping=operation["received_index_map"]
    map_id=hashlib.sha256(bytes(mapping)).hexdigest()
    gid=oid+"/map_"+map_id;rid=gid+"/"+key
    plan["encodes"].setdefault(gid,dict(observation_id=oid,frames=N,received_sha256=spec["sha256"],received_index_map=mapping))
    read=plan["reads"].setdefault(rid,dict(encode_id=gid,key_label=key,frames=N,R=44,logical_slots=[]))
    alias=read["logical_slots"][0] if read["logical_slots"] else None
    read["logical_slots"].append(lid)
    plan["logical_slots"][lid]=dict(status="PLANNED",physical_read=rid,alias_of=alias,operation=operation,oracle=oracle)
def plan_blind(inputs,est):
    plan=dict(encodes={},reads={},logical_slots={},truth_inputs=False)
    for oid,spec in inputs.items():
        for key in ("K0","K1"):
            e=est[oid+"/"+key]
            for mode,which in (("RAW",None),("GLOBAL_ALIGN","H0"),("PATH_ALIGN","joint")):
                op=identity() if which is None else correction(e[which]["path"]) if e[which]["status"]=="ESTIMATED" else None
                add_slot(plan,oid,key,mode,op,spec,error=e[which]["reason"] if which else None)
    return plan
def path_evaluation(estimate,truth):
    if estimate["status"]!="ESTIMATED":return dict(status="UNRESOLVED",reason=estimate["reason"],map_denominator=N)
    h=estimate["path"];a=path_map(h);b=path_map(truth);delta=[x-y for x,y in zip(a,b)]
    kerr=h["k"]-truth["k"] if h["family"]==truth["family"]=="H1" else None
    return dict(status="EVALUATED_POSTSEAL",estimate=h,true_family=truth["family"],true_b=truth["b"],true_k=truth["k"],
        family_correct=h["family"]==truth["family"],b_correct=h["b"]==truth["b"],
        map_correct=sum(x==0 for x in delta),map_denominator=N,map_signed_errors=delta,
        map_mean_absolute_error=sum(map(abs,delta))/N,map_max_absolute_error=max(map(abs,delta)),
        k_exact=h["k"]==truth["k"] if truth["family"]=="H1" and h["family"]=="H1" else False if truth["family"]=="H1" else None,
        k_signed_error=kerr,k_absolute_error=abs(kerr) if kerr is not None else None,
        false_jump_on_this_input=truth["family"]=="H0" and h["family"]=="H1",sync_accepted=False)
