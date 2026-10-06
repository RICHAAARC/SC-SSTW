"""Public-only Stage2 estimation, phase plan and detailed payload evidence."""
from dataclasses import replace
import math
import numpy as np
from main.tube_state import video_trajectory_payload_framewise_sync_v1 as sync
from main.tube_state.video_trajectory_receiver_origin_v1 import _counter, join_truth
PUBLIC=replace(sync.PUBLIC,received_frame_counts=(181,177,89))

def support(frames):
    if frames not in (181,177,89):raise ValueError("fixed received length required")
    return min(44,(frames-1)//4)

def phase_map(frames,phase):
    support(frames)
    if type(phase) is not int or phase not in range(4):raise ValueError("phase0..3 required")
    return [0]*phase+list(range(frames-phase))

def estimate(readout,frames,key):
    """No score threshold: require the unchanged complete unique argmax."""
    offsets=list(sync.candidate_offsets(frames,PUBLIC))
    expected_locals=sum(len(sync.tubelet_support_rows(frames,o,PUBLIC)) for o in offsets)
    try:
        rows=readout["candidate_rows"]
        if readout["status"]!="COMPLETE" or readout["truth_inputs"] is not False:
            raise ValueError("sync not complete/blind")
        if readout["key_id"]!=sync.key_identifier(key) or readout["candidate_offsets"]!=offsets:
            raise ValueError("key/public roster mismatch")
        if [x["source_offset"] for x in rows]!=offsets:raise ValueError("missing/duplicate candidate")
        if readout["counts"]!=dict(candidate_scores=len(offsets),candidate_tubelet_rows=expected_locals,failed_candidates=0):
            raise ValueError("incomplete candidate/local counts")
        if len(readout["local_rows"])!=expected_locals or any(x["status"]!="SCORED" for x in readout["local_rows"]):
            raise ValueError("incomplete local support")
        if any(x["status"]!="SCORED" or x["score"] is None or not math.isfinite(x["score"]) for x in rows):
            raise ValueError("nonfinite/failed score")
        best=max(x["score"] for x in rows)
        top=[x["source_offset"] for x in rows if abs(x["score"]-best)<=PUBLIC.tie_atol]
        summary=readout["summary"]
        canonical=top[0] if len(top)==1 else None
        if summary["top_offsets"]!=top or summary["canonical_offset"]!=canonical or summary["unique"]!=(len(top)==1):
            raise ValueError("inconsistent canonical summary")
        ordered=sorted((x["score"] for x in rows),reverse=True)
        return dict(status="ESTIMATED" if canonical is not None else "UNRESOLVED",
            reason=None if canonical is not None else "TIED_TOP",offset=canonical,
            phase=canonical%4 if canonical is not None else None,top_offsets=top,
            best_score=best,gap=ordered[0]-ordered[1] if len(ordered)>1 else None,
            sync_accepted=False,full_singleton_geometry_only=len(offsets)==1,truth_inputs=False)
    except (KeyError,ValueError,TypeError) as exc:
        return dict(status="UNRESOLVED",reason=str(exc),offset=None,phase=None,
                    sync_accepted=False,truth_inputs=False)

def missing_sync(frames,key,error):
    offsets=list(sync.candidate_offsets(frames,PUBLIC));locals_=[];candidates=[]
    for offset in offsets:
        rows=sync.tubelet_support_rows(frames,offset,PUBLIC)
        locals_.extend(dict(source_offset=offset,**x,status="FAILED",error=str(error),q=None) for x in rows)
        candidates.append(dict(source_offset=offset,status="FAILED",score=None,tubelet_rows=len(rows)))
    return dict(status="RETAINED_FAILURES",error=str(error),method_version=PUBLIC.method_version,
        key_id=sync.key_identifier(key),received_shape=[frames,4,40,64],candidate_offsets=offsets,
        candidate_rows=candidates,local_rows=locals_,
        summary=dict(top_offsets=[],canonical_offset=None,unique=False,best_score=None,sync_accepted=False),
        counts=dict(candidate_scores=len(offsets),candidate_tubelet_rows=len(locals_),failed_candidates=len(offsets)),truth_inputs=False)

def physical_plan(inputs,estimates):
    """Freeze after sync seal. Identity/phase and key alone select cache entries."""
    encodes={};reads={};logical={}
    for oid,spec in inputs.items():
        n=spec["frames"]
        for mode in ("BASELINE","EST_ALIGN"):
            for key in ("K0","K1"):
                sid=oid+"/"+key+"/"+mode
                est=estimates[oid+"/"+key]
                p=0 if mode=="BASELINE" else est.get("phase") if est["status"]=="ESTIMATED" else None
                if p is None:
                    logical[sid]=dict(status="UNRESOLVED",physical_read=None,error=est.get("reason","no estimate"),
                        frames=n,R=support(n),planned_votes=32*30*support(n),planned_time_bit_rows=32*support(n),planned_final_bits=32)
                    continue
                gid=oid+"/phase"+str(p);rid=gid+"/"+key
                encodes.setdefault(gid,dict(observation_id=oid,phase=p,frames=n,received_sha256=spec["sha256"],
                    received_index_map=phase_map(n,p),synthetic_output_indices=list(range(p)),
                    dropped_received_indices=list(range(n-p,n)) if p else []))
                row=reads.setdefault(rid,dict(encode_id=gid,key_label=key,frames=n,R=support(n),logical_slots=[]))
                alias=row["logical_slots"][0] if row["logical_slots"] else None
                row["logical_slots"].append(sid)
                logical[sid]=dict(status="PLANNED",physical_read=rid,alias_of=alias,frames=n,R=support(n),
                    planned_votes=32*30*support(n),planned_time_bit_rows=32*support(n),planned_final_bits=32)
    return dict(encodes=encodes,reads=reads,logical_slots=logical,truth_inputs=False,
        planned_calls=dict(wan_receiver_encode=len(encodes),payload_read=len(reads)),
        planned_physical_votes=sum(32*30*x["R"] for x in reads.values()),
        planned_physical_time_bit_rows=sum(32*x["R"] for x in reads.values()),
        planned_physical_final_bits=32*len(reads),alias_interpretation="References to one physical read; never independent evidence")

def detailed_votes(signs,coords,zero_mask,frames):
    R=support(frames);votes=np.asarray(signs);zero=np.asarray(zero_mask)
    if votes.shape!=(4,R,240) or not np.isin(votes,[-1,1]).all():raise ValueError("fixed polarity tensor required")
    if zero.shape!=votes.shape or not np.isin(zero,[0,1]).all() or np.any(votes[zero.astype(bool)]!=-1):
        raise ValueError("strict >0 / exact zero mask mismatch")
    coords=[[int(h),int(w)] for h,w in coords]
    if len(coords)!=240 or len({tuple(c) for c in coords})!=240:raise ValueError("original frequency coordinates required")
    bits=[];times=[]
    for ch in range(4):
        for slot in range(8):
            seq=votes[ch,:,slot::8]
            bits.append(dict(bit_index=ch*8+slot,payload_channel=ch,slot=slot,**_counter(seq.reshape(-1))))
            for t in range(R):
                times.append(dict(receiver_latent_index=t+1,nominal_stride_coordinate=4*(t+1),
                    bit_index=ch*8+slot,payload_channel=ch,slot=slot,**_counter(seq[t])))
    return dict(status="READ",R=R,received_frames=frames,shape=[4,R,240],coordinate_order=coords,
        receiver_latent_indices=list(range(1,R+1)),signed_votes=votes.astype(np.int8).tolist(),
        zero_coefficient_mask=zero.astype(np.int8).tolist(),bit_rows=bits,time_bit_rows=times,
        detailed_vote_count=32*30*R,time_bit_count=32*R,truth_inputs=False,
        signed_vote_definition="2*int(FFT.real>0)-1; exact zero is -1, not erasure",
        flatten_order="signs[ch,:,bit::8].reshape(-1), C/time-major order",
        time_semantics="Receiver latent indices, not independent four-frame blocks or matched source receptive fields")

def missing_detail(frames,error):
    R=support(frames)
    return dict(status="FAILED",R=R,received_frames=frames,error=str(error),shape=[4,R,240],signed_votes=None,
        planned_detailed_votes=32*30*R,actual_detailed_votes=0,truth_inputs=False,
        time_bit_rows=[dict(receiver_latent_index=t,bit_index=b,payload_channel=b//8,status="FAILED",error=str(error))
                       for b in range(32) for t in range(1,R+1)])

def summarize(joined):
    if joined["status"]!="EVALUATED_TRUTH":return dict(status="FAILED",error=joined.get("error"))
    bits=joined["bit_rows"];times=joined["time_bit_rows"];R=len(times)//32
    def stats(rows):
        return dict(count=len(rows),errors=sum(x["bit_error"] for x in rows),ties=sum(x["tie"] for x in rows),
            negative=sum(x["signed_normalized_margin"]<0 for x in rows),zero=sum(x["signed_normalized_margin"]==0 for x in rows),
            positive=sum(x["signed_normalized_margin"]>0 for x in rows),
            mean_signed_margin=sum(x["signed_normalized_margin"] for x in rows)/len(rows))
    return dict(status="DESCRIPTIVE_POSTSEAL",final=stats(bits),
        by_time_channel=[dict(receiver_latent_index=t,payload_channel=c,
            **stats([x for x in times if x["receiver_latent_index"]==t and x["payload_channel"]==c]))
            for t in range(1,R+1) for c in range(4)])
