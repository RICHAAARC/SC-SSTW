"""Fixed conditional joint protocol; pure methods, no runtime or experiment imports."""
from __future__ import annotations
import hashlib,json,math
from main.tube_state import grow_video_reference as payload
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as align
from main.tube_state import video_trajectory_internal_single_deletion_v1 as deletion
KEY_LABELS=("K0","K1")
INPUT_FRAMES={"input_00":181,"input_01":177,"input_02":89,"input_03":177,"input_04":177,"input_05":89,"input_06":89,"input_07":177}
VIEWS={**{f"global_{i:02d}":dict(input_id=f"input_{i:02d}",protocol="GLOBAL") for i in range(7)},"path_00":dict(input_id="input_03",protocol="SINGLE_JUMP"),"path_01":dict(input_id="input_07",protocol="SINGLE_JUMP")}
FIXED=dict(logical_observations=9,physical_inputs=8,sync_reads=18,sync_candidates=3426,global_local_rows=14276,frame_cache_cells=3540,local_grid_cells=900,framewise_frames=1156,framewise_batches=151,blind_payload_reads=40,oracle_payload_reads=4,logical_payload_reads=44,logical_votes=1605120,logical_time_bit_rows=53504,logical_final_bits=1408,quality_rows=6)
UPPER=dict(blind_wan_encodes=28,blind_key_reads=36,total_wan_encodes=30,total_key_reads=40)

def guided_velocity(z,conditional,unconditional,sigma,target,mask,index,count):
    """Exact PAYLOAD_MULTI active arithmetic; no pilot or old state dependency."""
    import torch
    if type(index) is not int or not 0<=index<50 or not math.isfinite(float(sigma)) or sigma<=0:raise ValueError("fixed native step/sigma")
    z,c,u=z.float(),conditional.float(),unconditional.float()
    enabled=index>=25;detail=dict(enabled=enabled,arm="PAYLOAD_MULTI",state_control_cfg="float32")
    if enabled:
        count("payload_gradient",False)
        dp,pinfo=payload.local_delta(z-float(sigma)*c,target,mask)
        count("payload_gradient",True)
        delta=dp+torch.zeros_like(dp) # retain successful payload-only branch arithmetic
        c=c-delta/float(sigma)
        detail.update(payload=pinfo,pilot=None,payload_delta_l2=float(dp.double().square().sum().sqrt()))
    velocity=u+5.0*(c-u)
    if not bool(torch.isfinite(velocity).all()):raise FloatingPointError("nonfinite CFG")
    return velocity.detach(),detail

def logical_slots():
    rows={}
    for vid,v in VIEWS.items():
        modes=("BASELINE","EST_ALIGN") if v["protocol"]=="GLOBAL" else ("RAW","GLOBAL_ALIGN","PATH_ALIGN","TRUTH_PATH")
        n=INPUT_FRAMES[v["input_id"]];R=align.support(n)
        for key in KEY_LABELS:
            for mode in modes:
                lid=vid+"/"+key+"/"+mode
                rows[lid]=dict(view_id=vid,input_id=v["input_id"],protocol=v["protocol"],frames=n,R=R,key_label=key,mode=mode,oracle=mode=="TRUTH_PATH",status="PENDING",physical_read=None,planned_votes=32*30*R,planned_time_bits=32*R,planned_final_bits=32)
    return rows

def phase_operation(n,p):
    indices=align.phase_map(n,p)
    return dict(received_index_map=indices,phase=p,synthetic_output_indices=list(range(p)),dropped_received_indices=sorted(set(range(n))-set(indices)),inserted_gap_output_index=None,operation="prepend+truncate")

def operation_cost(op):
    a=op["received_index_map"]
    return dict(output_frames=len(a),synthetic_outputs=len(op["synthetic_output_indices"]),repeated_occurrences=len(a)-len(set(a)),dropped_received=len(op["dropped_received_indices"]),inserted_gap_output_index=op.get("inserted_gap_output_index"))

def estimates(raw,protocol,n,key):
    if protocol=="GLOBAL":return {"global":align.estimate(raw,n,key)}
    e=deletion.estimates(raw,key)
    return dict(global_=e["H0"],joint=e["joint"])

def blind_operation(row,estimate):
    if row["mode"] in ("RAW","BASELINE"):return phase_operation(row["frames"],0)
    e=estimate.get("global" if row["protocol"]=="GLOBAL" else "global_" if row["mode"]=="GLOBAL_ALIGN" else "joint",{})
    if e.get("status")!="ESTIMATED":return None
    if row["protocol"]=="GLOBAL":return phase_operation(row["frames"],e["phase"])
    return deletion.correction(e["path"])

def empty_plan():return dict(encodes={},reads={},logical_slots={},truth_inputs=False)

def add_slot(plan,lid,row,op,input_spec,key_value,*,error=None):
    if op is None:
        plan["logical_slots"][lid]=dict(status="UNRESOLVED",physical_read=None,error=error or "NO_UNIQUE_ESTIMATE",oracle=row["oracle"]);return
    mapping=op["received_index_map"]
    token=json.dumps([input_spec.get("sha256"),row["input_id"],row["frames"],mapping],separators=(",",":"))
    gid=hashlib.sha256(token.encode()).hexdigest();kid=align.sync.key_identifier(key_value);rid=gid+"/"+kid
    plan["encodes"].setdefault(gid,dict(input_id=row["input_id"],frames=row["frames"],received_sha256=input_spec.get("sha256"),received_index_map=mapping))
    read=plan["reads"].setdefault(rid,dict(encode_id=gid,key_label=row["key_label"],key_id=kid,frames=row["frames"],logical_slots=[]))
    alias=read["logical_slots"][0] if read["logical_slots"] else None
    if lid not in read["logical_slots"]:read["logical_slots"].append(lid)
    plan["logical_slots"][lid]=dict(status="PLANNED",physical_read=rid,alias_of=alias,operation=op,operation_cost=operation_cost(op),oracle=row["oracle"])

def missing_final(frames,error):
    x=align.missing_detail(frames,error)
    x["bit_rows"]=[dict(bit_index=b,payload_channel=b//8,status="FAILED",error=str(error)) for b in range(32)]
    return x
