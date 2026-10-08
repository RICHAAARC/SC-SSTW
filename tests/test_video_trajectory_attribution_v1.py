import ast
import json
import pytest
from pathlib import Path
from main.tube_state import video_trajectory_attribution_v1 as method
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as align

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.quick

def fake_global(frames,key,scores):
    row=align.missing_sync(frames,key,"fixture")
    row["status"]="COMPLETE";row.pop("error",None);row["counts"]["failed_candidates"]=0
    for item in row["local_rows"]:
        item.update(status="SCORED",q=1.0);item.pop("error",None)
    for item in row["candidate_rows"]:
        item.update(status="SCORED",score=float(scores.get(item["source_offset"],-1.0)));item.pop("error",None)
    best=max(x["score"] for x in row["candidate_rows"])
    top=[x["source_offset"] for x in row["candidate_rows"] if abs(x["score"]-best)<=align.PUBLIC.tie_atol]
    row["summary"].update(top_offsets=top,canonical_offset=top[0] if len(top)==1 else None,
        unique=len(top)==1,best_score=best,sync_accepted=False)
    return row

def rules():
    return dict(status="FROZEN",rule_sha256="a"*64,thresholds={
        "181":dict(tau_M=1.0,tau_m=None,tau_I=.2),
        "177":dict(tau_M=1.0,tau_m=.1,tau_I=.2),
        "89":dict(tau_M=1.0,tau_m=.1,tau_I=.2)})

def sync(frames=177,M=2,m=.5,actions=1,status="COMPLETE"):
    return dict(status=status,M=M,m=None if frames==181 else m,unique_action=actions==1,
        top_action_count=actions,chosen_action=dict(received_index_map=list(range(frames))) if actions==1 else None)

def identity(I=.8,exact=True,status="COMPLETE"):
    return dict(status=status,I=I,exact32=exact)

def test_fixed_roster_and_denominator():
    rows=method.query_roster()
    assert len(rows)==192 and len({x["query_id"] for x in rows})==192
    assert sum(x["source_id"]=="DEV" for x in rows)==64
    assert {len(x) for x in method.VIEW_MAPS.values()}=={181,177,89}
    d=method.fixed_denominator()
    assert d==dict(sources=3,native_trajectories=9,received_conditions=12,
        physical_observations=96,logical_queries=192,sync_candidates=74784,
        framewise_frames=13872,framewise_batches=1812,max_payload_reads=192,
        logical_payload_votes=6589440,logical_time_bit_rows=219648,logical_final_bits=6144)

def test_same_action_tie_resolves_but_different_action_tie_does_not():
    key="watermark";offsets=list(align.sync.candidate_offsets(89,align.PUBLIC))
    a=offsets[0];b=next(x for x in offsets if x%4==a%4 and x!=a)
    same=method.summarize_sync(fake_global(89,key,{a:5,b:5}),89,key)
    assert same["status"]=="COMPLETE" and same["unique_action"] and len(same["top_paths"])==2
    c=next(x for x in offsets if x%4!=a%4)
    different=method.summarize_sync(fake_global(89,key,{a:5,c:5}),89,key)
    assert different["status"]=="COMPLETE" and not different["unique_action"] and different["top_action_count"]==2

def test_low_sync_precedes_action_tie_and_payload():
    decision=method.decide(sync(M=1.0,actions=2),identity(status="TECHNICAL_INCOMPLETE"),177,rules())
    assert decision["decision"]=="REJECT" and decision["reason"]=="REJECT_LOW_SYNC"
    decision=method.decide(sync(M=2,actions=2),identity(),177,rules())
    assert decision["decision"]=="UNCERTAIN" and decision["reason"]=="UNCERTAIN_SYNC"

def test_identity_gray_negative_accept_and_inconsistency():
    assert method.decide(sync(),identity(I=-.01,exact=False),177,rules())["reason"]=="REJECT_IDENTITY"
    assert method.decide(sync(),identity(I=.1,exact=False),177,rules())["reason"]=="UNCERTAIN_IDENTITY_WEAK"
    assert method.decide(sync(),identity(I=.8,exact=True),177,rules())["decision"]=="ACCEPT"
    assert method.decide(sync(),identity(I=.8,exact=False),177,rules())["reason"]=="UNCERTAIN_INTERNAL_INCONSISTENCY"
    assert method.decide(sync(status="TECHNICAL_INCOMPLETE"),identity(),177,rules())["reason"]=="UNCERTAIN_TECHNICAL_SYNC"

def development_rows():
    rows=[]
    for row in method.query_roster(("DEV",)):
        N=row["frames"];null=row["video_id"] in ("OFF","A_P1") or row["key_label"]=="K1"
        positive=row["video_id"]=="A_M05" and row["key_label"]=="K0"
        s=sync(N,M=1 if null else 3,m=.1 if null else .5)
        ident=identity(I=.8 if positive else .1,exact=positive)
        rows.append(dict(row,sync=s,identity=ident))
    return rows

def test_freeze_once_from_complete_dev64_and_fail_closed():
    frozen=method.freeze_rules(development_rows(),"b"*64,"c"*64)
    assert frozen["status"]=="FROZEN" and set(frozen["thresholds"])=={"181","177","89"}
    assert frozen["thresholds"]["177"]["tau_M"]==1 and frozen["thresholds"]["177"]["tau_m"]==.1
    assert frozen["thresholds"]["177"]["tau_I"]==.1
    assert method.freeze_rules(development_rows()[:-1],"b"*64,"c"*64)["status"]=="NOT_FREEZABLE"
    broken=development_rows();broken[0]["sync"]["status"]="TECHNICAL_INCOMPLETE"
    assert method.freeze_rules(broken,"b"*64,"c"*64)["status"]=="NOT_FREEZABLE"

def test_false_claim_aggregation_tri_state():
    false=dict(posthoc=dict(required=True,false_claim=True,technical_complete=True))
    missing=dict(posthoc=dict(required=True,false_claim=False,technical_complete=False))
    clean=dict(posthoc=dict(required=True,false_claim=False,technical_complete=True))
    assert method.aggregate_false_claim([false,missing]) is True
    assert method.aggregate_false_claim([missing,clean])=="UNRESOLVED"
    assert method.aggregate_false_claim([clean]) is False

def test_action_truth_is_posthoc_and_paths_remain_separate():
    assert method.true_action_for_view("CROP177_P2")==method.deletion.correction(dict(family="H0",b=2,k=None))["received_index_map"]
    assert len(method.true_action_for_view("DELETE177_B2K88"))==177
    source=(ROOT/"main/tube_state/video_trajectory_attribution_v1.py").read_text()
    assert "runtime" not in source and "experiments" not in source
    runner=(ROOT/"experiments/wan_state_clock/video_trajectory_attribution_v1_run.py").read_text()
    assert runner.index('sync_seal=dump')<runner.index('payload_rows={}')
    assert runner.index('payload_seal=dump')<runner.index('def posthoc_roles')
    ast.parse(source);ast.parse(runner)

def test_config_freezes_sources_arms_maps_and_public_claim():
    cfg=json.loads((ROOT/"experiments/wan_state_clock/configs/video_trajectory_attribution_v1.json").read_text())
    assert list(cfg["sources"])==["DEV","C1","C2"]
    assert [cfg["sources"][x]["seed"] for x in cfg["sources"]]==[2026100701,2026100802,2026100803]
    assert cfg["trajectories"]["OFF"]==dict(payload=False,message=None,received=["OFF"],m05=False)
    assert cfg["trajectories"]["A"]["received"]==["A_P1","A_M05"]
    assert cfg["trajectories"]["B"]["received"]==["B_M05"]
    assert cfg["claim"]==dict(message="OKOK",bits=list(method.CLAIM_BITS))
    assert {k:tuple(v["received_index_map"]) for k,v in cfg["views"].items()}==method.VIEW_MAPS

def test_action_equivalence_baseline_counts():
    actions177={method._action(method.deletion.correction(h)["received_index_map"]) for h in method.deletion.hypotheses()}
    assert len(method.deletion.hypotheses())==709 and len(actions177)==699
    offsets=list(align.sync.candidate_offsets(89,align.PUBLIC))
    actions89={method._action(align.phase_map(89,x%4)) for x in offsets}
    assert len(offsets)==93 and len(actions89)==4

def test_noneligible_identity_null_is_retained_but_attempted_failure_blocks():
    rows=development_rows()
    tied=next(r for r in rows if r["frames"]==177 and r["video_id"]=="OFF" and r["key_label"]=="K0")
    tied["sync"].update(unique_action=False,top_action_count=2,chosen_action=None)
    tied["identity"]=dict(status="NOT_ELIGIBLE",attempted=False,I=None,exact32=None)
    assert method.freeze_rules(rows,"b"*64,"c"*64)["status"]=="FROZEN"
    rows=development_rows()
    failed=next(r for r in rows if r["frames"]==177 and r["video_id"]=="OFF" and r["key_label"]=="K0")
    failed["identity"]=dict(status="TECHNICAL_INCOMPLETE",attempted=True,I=None,exact32=None)
    assert method.freeze_rules(rows,"b"*64,"c"*64)["status"]=="NOT_FREEZABLE"

def test_unsupported_missing_and_nonfinite_are_stable_uncertain():
    assert method.decide(sync(),identity(),90,rules())["decision"]=="UNCERTAIN"
    broken=rules();del broken["thresholds"]["177"]
    assert method.decide(sync(),identity(),177,broken)["reason"]=="UNCERTAIN_UNSUPPORTED_PROTOCOL"
    assert method.decide(sync(M=float("nan")),identity(),177,rules())["reason"]=="UNCERTAIN_TECHNICAL_SYNC"
    assert method.decide(sync(),identity(I=float("nan")),177,rules())["reason"]=="UNCERTAIN_TECHNICAL_PAYLOAD"

def test_identity_rejects_fractional_counter_evidence():
    rows=[]
    for i,bit in enumerate(method.CLAIM_BITS):
        rows.append(dict(bit_index=i,ones=9.0 if bit else 1.0,zeros=1.0 if bit else 9.0,count=10.0,decoded=bit))
    result=method.identity_evidence(dict(status="READ",truth_inputs=False,bit_rows=rows))
    assert result["status"]=="TECHNICAL_INCOMPLETE"

def test_runner_config_and_notebook_draft_schema():
    from experiments.wan_state_clock import video_trajectory_attribution_v1_run as runner
    cfg=runner.load_config()
    assert cfg["fixed_denominator"]==method.fixed_denominator(3)
    nb=json.loads((ROOT/"notebooks/video_trajectory_attribution_v1_colab.ipynb").read_text())
    assert nb["metadata"]["candidate_binding"]==dict(candidate="trajectory-attribution-v1",source_sha=None,status="UNPUBLISHED_DRAFT")
    code=[c for c in nb["cells"] if c["cell_type"]=="code"]
    assert "".join(code[0]["source"])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in code:
        assert cell["outputs"]==[] and cell["execution_count"] is None
        ast.parse("".join(cell["source"]))
    setup="".join(code[1]["source"])
    assert setup.index("if SOURCE_SHA is None:")<setup.index("OUTPUT.mkdir")
    assert "CONFIG_PATH=REPO/'experiments/wan_state_clock/configs/video_trajectory_attribution_v1.json'" in "".join(code[3]["source"])

def _fill_fake_source(store,source_id,rules_value=None,fail_attempt=False):
    for query in store.data["queries"].values():
        if query["source_id"]!=source_id:continue
        N=query["frames"];null=query["video_id"] in ("OFF","A_P1") or query["key_label"]=="K1"
        positive=query["video_id"]=="A_M05" and query["key_label"]=="K0"
        query["sync"]=sync(N,M=1 if null else 3,m=.1 if null else .5)
        query["sync"]["chosen_action"]=dict(received_index_map=method.true_action_for_view(query["view_id"]))
        query["identity"]=identity(I=.8 if positive else .1,exact=positive)
        query["identity"]["attempted"]=True
        query["status"]="PAYLOAD_READ"
    if fail_attempt and source_id=="DEV":
        row=next(q for q in store.data["queries"].values() if q["source_id"]=="DEV" and q["frames"]==177 and q["video_id"]=="OFF" and q["key_label"]=="K0")
        row["identity"]=dict(status="TECHNICAL_INCOMPLETE",attempted=True,I=None,exact32=None,error="fixture")
    store.data["seals"][source_id+"/sync"]=dict(path="fixture",sha256="a"*64)
    store.data["seals"][source_id+"/payload"]=dict(path="fixture",sha256="b"*64)
    store.save()

def test_fake_control_plane_full_dev_freeze_and_confirmation(tmp_path,monkeypatch):
    from experiments.wan_state_clock import video_trajectory_attribution_v1_run as runner
    calls=[]
    def prepared(store,source_id):
        calls.append(source_id);store.data["sources"][source_id]["status"]="COMPLETE";store.save()
        return dict(received={x:{} for x in method.VIDEO_IDS},status="COMPLETE")
    monkeypatch.setattr(runner,"prepare_source",prepared)
    monkeypatch.setattr(runner,"collect_source",lambda store,source_id,record,rules=None:_fill_fake_source(store,source_id,rules))
    result=runner.run(tmp_path/"complete",runner.load_config())
    assert result["status"]=="COMPLETE" and result["rules"]["status"]=="FROZEN"
    assert calls==["DEV","C1","C2"] and len(result["queries"])==192
    assert sum(q["decision"]["decision"]=="ACCEPT" for q in result["queries"].values())==24
    assert all(result["sources"][s]["any_false_claim"] is False for s in method.SOURCE_IDS)

def test_fake_control_plane_rule_failure_retains_confirmation128_not_run(tmp_path,monkeypatch):
    from experiments.wan_state_clock import video_trajectory_attribution_v1_run as runner
    calls=[]
    def prepared(store,source_id):
        calls.append(source_id);store.data["sources"][source_id]["status"]="COMPLETE";store.save()
        return dict(received={x:{} for x in method.VIDEO_IDS},status="COMPLETE")
    monkeypatch.setattr(runner,"prepare_source",prepared)
    monkeypatch.setattr(runner,"collect_source",lambda store,source_id,record,rules=None:_fill_fake_source(store,source_id,rules,fail_attempt=True))
    result=runner.run(tmp_path/"blocked",runner.load_config())
    assert result["rules"]["status"]=="NOT_FREEZABLE" and calls==["DEV"]
    confirmation=[q for q in result["queries"].values() if q["source_id"] in ("C1","C2")]
    assert len(confirmation)==128
    assert all(q["status"]=="NOT_RUN" and q["decision"]["reason"]=="UNCERTAIN_RULE_NOT_FREEZABLE" for q in confirmation)
    assert all(result["sources"][s]["status"]=="NOT_RUN" for s in ("C1","C2"))

def test_fake_confirmation_partial_failure_retains64_and_continues(tmp_path,monkeypatch):
    from experiments.wan_state_clock import video_trajectory_attribution_v1_run as runner
    calls=[]
    def prepared(store,source_id):
        calls.append(source_id)
        if source_id=="C1":raise RuntimeError("fixture source failure")
        store.data["sources"][source_id]["status"]="COMPLETE";store.save()
        return dict(received={x:{} for x in method.VIDEO_IDS},status="COMPLETE")
    monkeypatch.setattr(runner,"prepare_source",prepared)
    monkeypatch.setattr(runner,"collect_source",lambda store,source_id,record,rules=None:_fill_fake_source(store,source_id,rules))
    result=runner.run(tmp_path/"partial",runner.load_config())
    assert calls==["DEV","C1","C2"] and result["status"]=="RETAINED_INCOMPLETE"
    failed=[q for q in result["queries"].values() if q["source_id"]=="C1"]
    assert len(failed)==64 and all(q["decision"]["reason"]=="UNCERTAIN_TECHNICAL_SOURCE" for q in failed)
    assert result["sources"]["C2"]["status"]=="COMPLETE"
