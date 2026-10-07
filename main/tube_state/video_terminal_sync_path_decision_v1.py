"""Adopted empirical action gate. Original scores/winners remain unchanged."""
import math
from functools import lru_cache
from main.tube_state import video_trajectory_internal_single_deletion_v1 as deletion
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as prior
EPS=prior.PUBLIC.tie_atol
FAMILIES=(177,89)

def operation(frames,path):
    if frames==177:return deletion.correction(path)["received_index_map"]
    if frames==89 and path["family"]=="H0" and path["k"] is None and type(path["b"]) is int and path["b"] in range(93):
        return prior.phase_map(89,path["b"]%4)
    raise ValueError("fixed public path required")

@lru_cache(None)
def grammar(frames):
    hs=deletion.hypotheses() if frames==177 else [dict(family="H0",b=b,k=None) for b in range(93)] if frames==89 else []
    if not hs:raise ValueError("unsupported protocol")
    return tuple((h,tuple(deletion.path_map(h) if frames==177 else range(h["b"],h["b"]+89)),tuple(operation(frames,h))) for h in hs)

def failed(frames,error):
    return dict(status="FAILED",frames=frames,reason=str(error),M=None,m=None,unique=False,path=None,contrasts=[],same_action_paths=[],truth_inputs=False)

def analyze(readout,frames,key):
    """Inputs are blind original readout/public N/key only; no condition or labels."""
    try:
        hs=grammar(frames)
        if readout["truth_inputs"] is not False or readout["key_id"]!=prior.sync.key_identifier(key):raise ValueError("blind/key identity")
        if frames==177:
            est=deletion.estimates(readout,key)["joint"]
            if readout["status"]!="COMPLETE" or est["reason"] not in (None,"TIED_TOP"):raise ValueError("incomplete original709")
            q={(x["r"],x["d"]):x["numerator"]/x["rho"] for x in readout["frame_cache"]}
            scores=[x["score"] for x in readout["candidate_rows"]]
        else:
            est=prior.estimate(readout,89,key)
            if est.get("reason") not in (None,"TIED_TOP"):raise ValueError(est["reason"])
            groups={b:[] for b in range(93)}
            for row in readout["local_rows"]:groups[row["source_offset"]].append(row)
            Q={}
            for b,rows in groups.items():
                supports=prior.sync.tubelet_support_rows(89,b,prior.PUBLIC)
                if len(rows)!=len(supports):raise ValueError("local support missing")
                value=num=rho=0.0
                for row,support in zip(rows,supports):
                    if any(row.get(k)!=v for k,v in support.items()):raise ValueError("local geometry mismatch")
                    if row["status"]!="SCORED" or any(not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) for k in ("q","rho","signed_projection")) or row["rho"]<=0:raise ValueError("invalid local projection")
                    if row["q"]!=row["signed_projection"]/row["rho"]:raise ValueError("original local q mismatch")
                    value+=row["observed_frames"]*row["q"]
                    num+=row["signed_projection"];rho+=row["rho"]
                original=readout["candidate_rows"][b]
                if (original["numerator"],original["denominator"],original["score"])!=(num,rho,num/rho):raise ValueError("original S mismatch")
                Q[b]=value/89
            scores=[x["score"] for x in readout["candidate_rows"]]
        if len(scores)!=len(hs) or any(not math.isfinite(x) for x in scores):raise ValueError("full finite roster required")
        M=max(scores);tops=[i for i,x in enumerate(scores) if abs(x-M)<=EPS]
        result=dict(status="SCORED",frames=frames,M=M,m=None,unique=len(tops)==1,path=None,top_paths=[hs[i][0] for i in tops],
                    contrasts=[],same_action_paths=[],truth_inputs=False,original_sync_accepted=False,candidate_count=len(hs))
        if len(tops)!=1:return result
        idx=tops[0];h,source_map,op=hs[idx];result.update(path=h,received_index_map=list(op))
        for j,(g,gmap,gop) in enumerate(hs):
            if gop==op:
                result["same_action_paths"].append(g);continue
            if frames==89:E=Q[h["b"]]-Q[g["b"]];count=89
            else:
                diff=[r for r in range(177) if source_map[r]!=gmap[r]];count=len(diff);total=0.0
                for r in diff:total+=q[r,source_map[r]-r]-q[r,gmap[r]-r]
                E=total/count
            if not math.isfinite(E):raise ValueError("nonfinite contrast")
            result["contrasts"].append(dict(path=g,E=E,different_frames=count))
        result["m"]=min(x["E"] for x in result["contrasts"])
        return result
    except (KeyError,TypeError,ValueError,ZeroDivisionError,IndexError,OverflowError) as exc:return failed(frames,exc)

def calibrate(analyses,null_rosters):
    result={}
    for n,expected in ((177,8),(89,6)):
        ids=null_rosters.get(str(n),[])
        rows=[analyses.get(sid,failed(n,"missing")) for sid in ids]
        eligible=[x for x in rows if x.get("unique") and x.get("m") is not None and math.isfinite(x["m"])]
        good=len(ids)==expected and len(set(ids))==expected and all(x.get("frames")==n and x.get("status")=="SCORED" and x.get("M") is not None and math.isfinite(x["M"]) for x in rows) and bool(eligible)
        result[str(n)]=dict(status="AVAILABLE" if good else "UNAVAILABLE",null_ids=ids,null_denominator=expected,
            eligible_ids=[sid for sid,x in zip(ids,rows) if x in eligible],tau_M=max(x["M"] for x in rows) if good else None,
            tau_m=max(0.0,max(x["m"] for x in eligible)) if good else None,
            reason=None if good else "REQUIRED_NULL_MISSING_INVALID_OR_NO_UNIQUE_NULL")
    return dict(status="AVAILABLE" if all(x["status"]=="AVAILABLE" for x in result.values()) else "UNAVAILABLE",families=result,
                scope="Empirical development envelopes; no population FPR/coverage",truth_used="sealed-score null roles only")

def decide(analysis,thresholds):
    n=analysis["frames"];t=thresholds.get("families",{}).get(str(n),{})
    state="UNCERTAIN";reason="CALIBRATION_UNAVAILABLE"
    if n not in FAMILIES:reason="UNSUPPORTED_PROTOCOL"
    elif thresholds.get("status")=="AVAILABLE" and t.get("status")=="AVAILABLE":
        if analysis["status"]!="SCORED":reason="INCOMPLETE_OR_NONFINITE"
        elif not analysis["unique"]:reason="TIED_TOP"
        elif analysis["M"]<=t["tau_M"]+EPS:state="REJECT";reason="NO_TERMINAL_EVIDENCE"
        elif analysis["m"]<=t["tau_m"]+EPS:reason="ACTION_AMBIGUOUS"
        else:state="ACCEPT_ACTION";reason=None
    blocked=[x for x in analysis.get("contrasts",[]) if t.get("tau_m") is not None and x["E"]<=t["tau_m"]+EPS]
    return dict(state=state,reason=reason,frames=n,path=analysis.get("path"),M=analysis.get("M"),m=analysis.get("m"),
                original_point_path_unique=analysis.get("unique",False),same_action_paths=analysis.get("same_action_paths",[]),
                unexcluded_competitors=blocked,received_index_map=analysis.get("received_index_map") if state=="ACCEPT_ACTION" else None,truth_inputs=False)

def cpu_plan(decisions):
    plan={};seen={}
    for sid,row in decisions.items():
        n=row["frames"];op=row.get("received_index_map");oid=sid.rsplit("/",1)[0]
        alias=seen.get((oid,tuple(op))) if op is not None else None
        if op is not None:seen.setdefault((oid,tuple(op)),sid)
        plan[sid]=dict(RAW=dict(status="PLANNED",received_index_map=list(range(n))),
            GATED=dict(status="PLANNED" if op is not None else "NOT_AUTHORIZED",received_index_map=op,reason=row.get("reason"),alias_of=alias,
                operation_cost=operation_cost(n,row.get("path"),op)))
    return dict(operations=plan,receiver_wan_encodes=0,payload_reads=0,truth_inputs=False)

def operation_cost(frames,path,mapping):
    if mapping is None:return None
    synthetic=deletion.correction(path)["synthetic_output_indices"] if frames==177 else list(range(path["b"]%4))
    return dict(synthetic_output_indices=synthetic,dropped_received_indices=sorted(set(range(frames))-set(mapping)),
        repeated_received_indices=sorted({r for r in mapping if mapping.count(r)>1}),
        duplicate_occurrences=frames-len(set(mapping)),synthetic_count=len(synthetic),meaning="Input index edits only; no missing image recovery or quality metric")
