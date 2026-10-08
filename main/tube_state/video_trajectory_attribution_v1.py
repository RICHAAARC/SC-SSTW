"""Pure trajectory attribution decisions from sealed RGB-derived evidence.

No model, experiment, source role, expected path, or posthoc truth dependency.
"""
from __future__ import annotations
import hashlib
import json
import math
from collections import Counter
from main.tube_state import grow_video_reference as payload
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as align
from main.tube_state import video_trajectory_internal_single_deletion_v1 as deletion

SOURCE_IDS=("DEV","C1","C2")
VIDEO_IDS=("OFF","A_P1","A_M05","B_M05")
KEY_LABELS=("K0","K1")
CLAIM_MESSAGE="OKOK"
CLAIM_BITS=tuple(payload.message_bits(CLAIM_MESSAGE))
VIEW_MAPS={
    "FULL181":tuple(range(181)),
    "CROP177_P1":tuple(range(1,178)),
    "CROP177_P2":tuple(range(2,179)),
    "CROP177_P3":tuple(range(3,180)),
    "SHORT89_S37":tuple(range(37,126)),
    "SHORT89_S38":tuple(range(38,127)),
    "SHORT89_S39":tuple(range(39,128)),
    "DELETE177_B2K88":tuple(range(2,90))+tuple(range(91,180)),
}
RULE_FORMULA_VERSION="trajectory-attribution-v1-action-folded-development-maxima"

def fixed_denominator(sources=3):
    if type(sources) is not int or sources<1:raise ValueError("positive source count")
    per_video_R=44+3*44+3*22+44
    return dict(sources=sources,native_trajectories=3*sources,received_conditions=4*sources,
        physical_observations=32*sources,logical_queries=64*sources,
        sync_candidates=24928*sources,framewise_frames=4624*sources,
        framewise_batches=604*sources,max_payload_reads=64*sources,
        logical_payload_votes=32*30*per_video_R*8*sources,
        logical_time_bit_rows=32*per_video_R*8*sources,
        logical_final_bits=32*64*sources)

def query_roster(source_ids=SOURCE_IDS):
    return [dict(query_id="/".join((s,v,w,k)),source_id=s,video_id=v,view_id=w,
        key_label=k,frames=len(mapping),claim_message=CLAIM_MESSAGE,claim_bits=list(CLAIM_BITS))
        for s in source_ids for v in VIDEO_IDS for w,mapping in VIEW_MAPS.items() for k in KEY_LABELS]

def _action(mapping):
    raw=json.dumps(list(mapping),separators=(",",":")).encode()
    return hashlib.sha256(raw).hexdigest()

def _candidate_rows(readout,frames,key):
    if frames==177:
        if readout.get("truth_inputs") is not False or readout.get("key_id")!=align.sync.key_identifier(key):
            raise ValueError("blind/key mismatch")
        rebuilt=deletion.reduce_cache(readout["frame_cache"],key)
        if (readout.get("status")!=rebuilt["status"] or readout.get("candidate_rows")!=rebuilt["candidate_rows"]
            or readout.get("local_grid")!=rebuilt["local_grid"] or readout.get("counts")!=rebuilt["counts"]):
            raise ValueError("incomplete/inconsistent J709 support")
        if rebuilt["status"]!="COMPLETE":raise ValueError("J709 not complete")
        rows=[]
        for index,row in enumerate(rebuilt["candidate_rows"]):
            if row["status"]!="SCORED" or not math.isfinite(row["score"]):raise ValueError("nonfinite J709 row")
            path={k:row[k] for k in ("family","b","k")}
            op=deletion.correction(path)
            rows.append(dict(candidate_index=index,path=path,score=float(row["score"]),
                received_index_map=op["received_index_map"],operation=op,action_id=_action(op["received_index_map"])))
        if len(rows)!=709:raise ValueError("J709 count")
        return rows,"J709"
    if frames not in (89,181):raise ValueError("public N must be 181/177/89")
    estimate=align.estimate(readout,frames,key)
    if estimate["status"] not in ("ESTIMATED","UNRESOLVED") or (
        estimate["status"]=="UNRESOLVED" and estimate.get("reason")!="TIED_TOP"):
        raise ValueError("G search not complete")
    rows=[]
    for index,row in enumerate(readout["candidate_rows"]):
        if row["status"]!="SCORED" or not math.isfinite(row["score"]):raise ValueError("nonfinite G row")
        offset=row["source_offset"];phase=offset%4
        mapping=align.phase_map(frames,phase)
        operation=dict(received_index_map=mapping,phase=phase,
            synthetic_output_indices=list(range(phase)),
            dropped_received_indices=sorted(set(range(frames))-set(mapping)),
            inserted_gap_output_index=None,operation="prepend+truncate")
        rows.append(dict(candidate_index=index,path=dict(family="G",offset=offset,phase=phase),
            score=float(row["score"]),received_index_map=mapping,operation=operation,action_id=_action(mapping)))
    expected=1 if frames==181 else 93
    if len(rows)!=expected:raise ValueError("G candidate count")
    return rows,"SINGLETON" if frames==181 else "G93"

def summarize_sync(readout,frames,key):
    try:
        rows,search=_candidate_rows(readout,frames,key)
        M=max(x["score"] for x in rows);atol=align.PUBLIC.tie_atol
        top=[x for x in rows if abs(x["score"]-M)<=atol]
        action_order=[];actions={}
        for row in top:
            if row["action_id"] not in actions:
                action_order.append(row["action_id"]);actions[row["action_id"]]=row
        margins=[]
        for action_id in action_order:
            representative=actions[action_id]
            other=[x["score"] for x in rows if x["action_id"]!=action_id]
            margins.append(None if not other else representative["score"]-max(other))
        m=None if not margins or any(x is None for x in margins) else min(margins)
        chosen=actions[action_order[0]] if len(action_order)==1 else None
        return dict(status="COMPLETE",search=search,frames=frames,M=M,m=m,
            top_paths=[x["path"] for x in top],top_action_ids=action_order,
            top_action_count=len(action_order),unique_action=len(action_order)==1,
            representative_path=chosen["path"] if chosen else top[0]["path"],
            chosen_action=chosen["operation"] if chosen else None,
            candidate_count=len(rows),truth_inputs=False)
    except (KeyError,TypeError,ValueError) as exc:
        return dict(status="TECHNICAL_INCOMPLETE",frames=frames,error=str(exc),M=None,m=None,
            top_paths=[],top_action_ids=[],top_action_count=0,unique_action=False,
            representative_path=None,chosen_action=None,truth_inputs=False)

def identity_evidence(detail,claim_bits=CLAIM_BITS):
    try:
        if detail["status"]!="READ" or detail.get("truth_inputs") is not False:raise ValueError("blind payload unavailable")
        rows=detail["bit_rows"]
        if len(rows)!=32 or tuple(claim_bits)!=CLAIM_BITS:raise ValueError("fixed public OKOK claim required")
        margins=[];decoded=[]
        for index,row in enumerate(rows):
            if row["bit_index"]!=index:raise ValueError("bit roster")
            values=(row["ones"],row["zeros"],row["count"],row["decoded"])
            if any(type(x) is not int for x in values):raise ValueError("integer vote evidence required")
            ones,zeros,count,value=values
            if count<=0 or ones<0 or zeros<0 or ones+zeros!=count:raise ValueError("vote counts")
            if value not in (0,1):raise ValueError("decoded bit")
            reconstructed=Counter(([1]*ones)+([0]*zeros)).most_common(1)[0][0]
            if reconstructed!=value and ones!=zeros:raise ValueError("decoded/count mismatch")
            decoded.append(value)
            margins.append((2*claim_bits[index]-1)*(ones-zeros)/count)
        exact=decoded==list(claim_bits)
        return dict(status="COMPLETE",attempted=True,I=min(margins),exact32=exact,decoded_bits=decoded,
            signed_claim_margins=margins,claim_message=CLAIM_MESSAGE,truth_inputs=False)
    except (KeyError,TypeError,ValueError) as exc:
        return dict(status="TECHNICAL_INCOMPLETE",error=str(exc),I=None,exact32=None,
            claim_message=CLAIM_MESSAGE,truth_inputs=False)

def _required(rows,predicate,label):
    selected=[r for r in rows if predicate(r)]
    if not selected:raise ValueError("missing "+label)
    if any(r["sync"]["status"]!="COMPLETE" for r in selected):raise ValueError("technical sync failure in "+label)
    return selected

def freeze_rules(development_rows,source_roster_sha256,config_sha256):
    """Freeze numeric maxima once from the complete preregistered DEV roster."""
    try:
        expected={x["query_id"] for x in query_roster(("DEV",))}
        actual={x["query_id"] for x in development_rows}
        if actual!=expected or len(development_rows)!=64:raise ValueError("complete DEV64 roster required")
        if any(r["identity"].get("attempted") is True and r["identity"].get("status")!="COMPLETE" for r in development_rows):
            raise ValueError("attempted DEV payload failure")
        thresholds={}
        for N in (181,177,89):
            same=[r for r in development_rows if r["frames"]==N]
            sync_null=_required(same,lambda r:r["video_id"] in ("OFF","A_P1") or r["key_label"]=="K1","sync-null")
            M_values=[r["sync"]["M"] for r in sync_null]
            if any(not math.isfinite(x) for x in M_values):raise ValueError("nonfinite M")
            m_values=[r["sync"]["m"] for r in sync_null if r["sync"]["m"] is not None]
            if N!=181 and (not m_values or any(not math.isfinite(x) for x in m_values)):raise ValueError("missing/nonfinite m")
            identity_classes=(("OFF","K0"),("A_M05","K1"),("B_M05","K0"))
            identity=[]
            for video,key in identity_classes:
                group=_required(same,lambda r,v=video,k=key:r["video_id"]==v and r["key_label"]==k,video+"/"+key)
                if any(r["identity"].get("attempted") is True and r["identity"]["status"]!="COMPLETE" for r in group):
                    raise ValueError("technical identity failure")
                eligible=[r for r in group if r["sync"].get("unique_action") and r["identity"]["status"]=="COMPLETE"]
                if not eligible:raise ValueError("no eligible identity-null row for "+video+"/"+key)
                identity.extend(eligible)
            I_values=[r["identity"]["I"] for r in identity]
            if any(not math.isfinite(x) for x in I_values):raise ValueError("nonfinite I")
            thresholds[str(N)]=dict(tau_M=max(M_values),tau_m=None if N==181 else max(0.0,max(m_values)),
                tau_I=max(0.0,max(I_values)),M_rows=len(M_values),m_rows=len(m_values),I_rows=len(I_values))
            positives=_required(same,lambda r:r["video_id"]=="A_M05" and r["key_label"]=="K0","positive")
            t=thresholds[str(N)]
            if any(not r["sync"].get("unique_action") or r["identity"]["status"]!="COMPLETE" or not r["identity"]["exact32"] or
                   not (r["sync"]["M"]>t["tau_M"]) or
                   (N!=181 and not (r["sync"]["m"]>t["tau_m"])) or
                   not (r["identity"]["I"]>t["tau_I"]) for r in positives):
                raise ValueError("development positive not strictly separable")
        frozen=dict(status="FROZEN",formula_version=RULE_FORMULA_VERSION,thresholds=thresholds,
            source_roster_sha256=source_roster_sha256,config_sha256=config_sha256,
            claim_message=CLAIM_MESSAGE,claim_bits=list(CLAIM_BITS))
        frozen["rule_sha256"]=hashlib.sha256(json.dumps(frozen,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        return frozen
    except (KeyError,TypeError,ValueError) as exc:
        return dict(status="NOT_FREEZABLE",formula_version=RULE_FORMULA_VERSION,error=str(exc),
            thresholds={},source_roster_sha256=source_roster_sha256,config_sha256=config_sha256,
            claim_message=CLAIM_MESSAGE,claim_bits=list(CLAIM_BITS))

def decide(sync,identity,frames,rules):
    base=dict(frames=frames,rule_sha256=rules.get("rule_sha256") if isinstance(rules,dict) else None,claim_message=CLAIM_MESSAGE)
    try:
        if not isinstance(rules,dict) or rules.get("status")!="FROZEN":
            return dict(base,decision="UNCERTAIN",reason="UNCERTAIN_RULE_NOT_FREEZABLE")
        if frames not in (181,177,89) or str(frames) not in rules.get("thresholds",{}):
            return dict(base,decision="UNCERTAIN",reason="UNCERTAIN_UNSUPPORTED_PROTOCOL")
        if sync.get("status")!="COMPLETE" or not math.isfinite(sync.get("M")):
            return dict(base,decision="UNCERTAIN",reason="UNCERTAIN_TECHNICAL_SYNC")
        threshold=rules["thresholds"][str(frames)]
        if not math.isfinite(threshold["tau_M"]) or not math.isfinite(threshold["tau_I"]):
            return dict(base,decision="UNCERTAIN",reason="UNCERTAIN_RULE_NOT_FREEZABLE")
        if sync["M"]<=threshold["tau_M"]:return dict(base,decision="REJECT",reason="REJECT_LOW_SYNC")
        if frames!=181:
            if not math.isfinite(sync.get("m")) or not math.isfinite(threshold.get("tau_m")):
                return dict(base,decision="UNCERTAIN",reason="UNCERTAIN_TECHNICAL_SYNC")
        if not sync.get("unique_action") or (frames!=181 and sync["m"]<=threshold["tau_m"]):
            return dict(base,decision="UNCERTAIN",reason="UNCERTAIN_SYNC")
        if identity.get("status")!="COMPLETE" or not math.isfinite(identity.get("I")):
            return dict(base,decision="UNCERTAIN",reason="UNCERTAIN_TECHNICAL_PAYLOAD")
        I=identity["I"]
        if I<0:return dict(base,decision="REJECT",reason="REJECT_IDENTITY")
        if I<=threshold["tau_I"]:return dict(base,decision="UNCERTAIN",reason="UNCERTAIN_IDENTITY_WEAK")
        if identity.get("exact32") is True:return dict(base,decision="ACCEPT",reason="ACCEPT")
        return dict(base,decision="UNCERTAIN",reason="UNCERTAIN_INTERNAL_INCONSISTENCY")
    except (KeyError,TypeError,ValueError):
        return dict(base,decision="UNCERTAIN",reason="UNCERTAIN_TECHNICAL_EVIDENCE")

def true_action_for_view(view_id):
    if view_id == "FULL181":return align.phase_map(181,0)
    if view_id.startswith("CROP177_P"):
        b=int(view_id[-1]);return deletion.correction(dict(family="H0",b=b,k=None))["received_index_map"]
    if view_id.startswith("SHORT89_S"):
        start=int(view_id.rsplit("S",1)[1]);return align.phase_map(89,start%4)
    if view_id == "DELETE177_B2K88":return deletion.correction(dict(family="H1",b=2,k=88))["received_index_map"]
    raise ValueError("fixed view id")

def posthoc_false_claim(decision,role):
    required=role.get("required_for_false_claim",True)
    false=decision["decision"]=="ACCEPT" and role.get("expected_accept") is not True
    return dict(false_claim=false,required=required,technical_complete=decision["reason"] not in
        ("UNCERTAIN_RULE_NOT_FREEZABLE","UNCERTAIN_TECHNICAL_SYNC","UNCERTAIN_TECHNICAL_PAYLOAD"))

def aggregate_false_claim(rows):
    required=[x for x in rows if x["posthoc"]["required"]]
    if any(x["posthoc"]["false_claim"] for x in required):return True
    if any(not x["posthoc"]["technical_complete"] for x in required):return "UNRESOLVED"
    return False
