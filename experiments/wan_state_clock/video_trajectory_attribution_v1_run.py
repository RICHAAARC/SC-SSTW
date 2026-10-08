"""Fixed development-freeze then unseen-confirmation trajectory attribution run."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from main.tube_state import video_trajectory_attribution_v1 as method
from runtime.wan import video_trajectory_attribution_v1 as runtime
from runtime.wan.provenance import source_identity
from experiments.wan_state_clock import video_trajectory_attribution_v1_prepare as prepare

ROOT=Path(__file__).resolve().parents[2]
ENTRY="experiments.wan_state_clock.video_trajectory_attribution_v1_run"
CONFIG=ROOT/"experiments/wan_state_clock/configs/video_trajectory_attribution_v1.json"

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    path=Path(path);raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix==".gz" else raw)
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,indent=2,allow_nan=False)+"\n").encode()
    if path.suffix==".gz":raw=gzip.compress(raw,mtime=0)
    temp=path.with_suffix(path.suffix+".tmp");temp.write_bytes(raw);os.replace(temp,path)
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def load_config(path=CONFIG):
    path=Path(path).resolve();cfg=read(path);fixed=read(CONFIG)
    if cfg!=fixed or cfg["formula_version"]!=method.RULE_FORMULA_VERSION or cfg["fixed_denominator"]!=method.fixed_denominator(3):
        raise ValueError("fixed attribution config required")
    cfg["_config_path"]=str(path);return cfg

class Store:
    def __init__(self,output,cfg):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False);self.cfg=cfg
        provenance=source_identity(ROOT)
        roster=method.query_roster()
        self.data=dict(status="RUNNING",stage="INITIALIZED",source_sha=provenance["git_commit"],
            source_provenance=provenance,config_sha256=digest(cfg["_config_path"]),
            source_roster_sha256=hashlib.sha256(json.dumps(cfg["sources"],sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            fixed_denominator=cfg["fixed_denominator"],rules=dict(status="PENDING"),
            sources={s:dict(status="PENDING",role=cfg["sources"][s]["role"]) for s in method.SOURCE_IDS},
            queries={r["query_id"]:dict(r,status="PENDING",sync=None,identity=None,decision=None) for r in roster},
            aggregates={},seals={},failures=[],evidence_ceiling=cfg["evidence_ceiling"])
        self.save()
    def save(self):prepare.dump(self.output/"result.json",self.data)
    def failure(self,stage,exc):
        self.data["failures"].append(dict(stage=stage,error=f"{type(exc).__name__}: {exc}"));self.save()

def run_worker(store,source_id,phase):
    out=store.output/"prepared"/source_id;log=store.output/"workers"/(source_id+"."+phase+".log")
    log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,"-B","-m",ENTRY,"--config",store.cfg["_config_path"],"--output",str(out),
         "--worker",phase,"--source-id",source_id]
    with log.open("wb") as stream:
        child=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
    if child.returncode:raise RuntimeError(f"{source_id}/{phase} worker exited {child.returncode}")
    return read(out/"source_preparation.json")

def prepare_source(store,source_id):
    store.data["sources"][source_id]["status"]="RUNNING";store.save()
    for phase in ("generation","native_decode","framewise_media"):
        store.data["stage"]=source_id+"/"+phase;store.save();run_worker(store,source_id,phase)
    record=read(store.output/"prepared"/source_id/"source_preparation.json")
    if record["status"]!="COMPLETE" or set(record["received"])!=set(method.VIDEO_IDS):
        raise RuntimeError("source preparation incomplete")
    store.data["sources"][source_id].update(status="COMPLETE",
        preparation_receipt=dump(store.output/"receipts"/(source_id+".json"),record))
    store.save();return record

def _observation(runtime_record,source_id,video_id,view_id,input_type):
    full=input_type.read(runtime_record["received"][video_id]);mapping=method.VIEW_MAPS[view_id]
    observed=input_type.construct(full,mapping)
    receipt=input_type.receipt(observed)
    return observed,dict(source_id=source_id,video_id=video_id,view_id=view_id,
        frames=len(mapping),received_index_map=list(mapping),**receipt)

def collect_source(store,source_id,record,rules=None,input_type=runtime.Inputs,
                   framewise_type=runtime.FramewiseBackend,wan_type=runtime.WanBackend):
    keys=store.cfg["keys"];sync_rows={};observation_receipts={};framewise=None
    try:
        framewise=framewise_type(store.cfg)
        for video_id in method.VIDEO_IDS:
            for view_id in method.VIEW_MAPS:
                observed,receipt=_observation(record,source_id,video_id,view_id,input_type)
                oid="/".join((source_id,video_id,view_id));observation_receipts[oid]=receipt
                latent=framewise.encode(observed)
                for key_label in method.KEY_LABELS:
                    qid=oid+"/"+key_label
                    try:
                        raw=runtime.score_received(framewise,latent,keys[key_label],receipt["frames"])
                        raw_receipt=dump(store.output/"blind_sync"/(qid+".json.gz"),raw)
                        summary=method.summarize_sync(raw,receipt["frames"],keys[key_label])
                        sync_rows[qid]=dict(raw=raw_receipt,summary=summary)
                        store.data["queries"][qid].update(status="SYNC_READ",sync=summary)
                    except Exception as exc:
                        summary=dict(status="TECHNICAL_INCOMPLETE",error=str(exc),M=None,m=None,
                            top_action_count=0,unique_action=False,chosen_action=None)
                        sync_rows[qid]=dict(error=str(exc),summary=summary)
                        store.data["queries"][qid].update(status="SYNC_FAILED",sync=summary)
                    store.save()
                observed=latent=None
    finally:
        if framewise is not None:framewise.close()
    sync_seal=dump(store.output/"seals"/(source_id+".sync.json"),
        dict(source_id=source_id,observations=observation_receipts,queries=sync_rows,truth_inputs=False))
    store.data["seals"][source_id+"/sync"]=sync_seal;store.save()
    payload_rows={};wan=None
    try:
        wan=wan_type(store.cfg)
        for video_id in method.VIDEO_IDS:
            for view_id in method.VIEW_MAPS:
                observed,receipt=_observation(record,source_id,video_id,view_id,input_type)
                oid="/".join((source_id,video_id,view_id))
                for key_label in method.KEY_LABELS:
                    qid=oid+"/"+key_label;query=store.data["queries"][qid];sync=query["sync"]
                    eligible=sync.get("status")=="COMPLETE" and sync.get("unique_action")
                    if rules is not None and eligible:
                        t=rules["thresholds"][str(receipt["frames"])]
                        eligible=sync["M"]>t["tau_M"] and (receipt["frames"]==181 or sync["m"]>t["tau_m"])
                    if not eligible:
                        identity=dict(status="NOT_ELIGIBLE",attempted=False,error="PAYLOAD_NOT_RUN_WITHOUT_QUALIFIED_UNIQUE_ACTION",I=None,exact32=None,truth_inputs=False)
                    else:
                        try:
                            corrected=runtime.apply_action(observed,sync["chosen_action"])
                            latent=wan.encode(corrected);detail=wan.read(latent,keys[key_label],receipt["frames"])
                            detail_receipt=dump(store.output/"blind_payload"/(qid+".json.gz"),detail)
                            identity=method.identity_evidence(detail)
                            payload_rows[qid]=dict(detail=detail_receipt,identity=identity)
                            corrected=latent=None
                        except Exception as exc:
                            identity=dict(status="TECHNICAL_INCOMPLETE",attempted=True,error=str(exc),I=None,exact32=None,truth_inputs=False)
                    query.update(status="PAYLOAD_READ" if identity["status"]=="COMPLETE" else "PAYLOAD_UNAVAILABLE",identity=identity)
                    payload_rows.setdefault(qid,dict(identity=identity));store.save()
                observed=None
    finally:
        if wan is not None:wan.close()
    payload_seal=dump(store.output/"seals"/(source_id+".payload.json"),
        dict(source_id=source_id,queries=payload_rows,truth_inputs=False))
    store.data["seals"][source_id+"/payload"]=payload_seal;store.save()

def posthoc_roles(store,source_id):
    rows=[]
    for qid,query in store.data["queries"].items():
        if query["source_id"]!=source_id:continue
        expected=query["video_id"]=="A_M05" and query["key_label"]=="K0"
        role=dict(expected_accept=expected,required_for_false_claim=True,
            control="SYNC_NULL" if query["video_id"]=="A_P1" else "PRIMARY_POSITIVE" if expected else "IDENTITY_OR_SYNC_NULL")
        query["posthoc"]=dict(role=role,action_map_correct=None,absolute_source_path_ambiguous=len(query["sync"].get("top_paths",[]))>1)
        if query["sync"].get("chosen_action") is not None:
            query["posthoc"]["action_map_correct"]=query["sync"]["chosen_action"]["received_index_map"]==method.true_action_for_view(query["view_id"])
        query["posthoc"].update(method.posthoc_false_claim(query["decision"],role))
        rows.append(query)
    def aggregate(selected):
        return method.aggregate_false_claim(selected)
    observation={}
    for video_id in method.VIDEO_IDS:
        for view_id in method.VIEW_MAPS:
            group=[r for r in rows if r["video_id"]==video_id and r["view_id"]==view_id]
            observation[video_id+"/"+view_id]=aggregate(group)
    any_edit={}
    for video_id in method.VIDEO_IDS:
        for key_label in method.KEY_LABELS:
            group=[r for r in rows if r["video_id"]==video_id and r["key_label"]==key_label]
            any_edit[video_id+"/"+key_label]=aggregate(group)
    any_key={}
    for video_id in method.VIDEO_IDS:
        group=[r for r in rows if r["video_id"]==video_id]
        any_key[video_id]=aggregate(group)
    source_value=aggregate(rows)
    store.data["aggregates"][source_id]=dict(observation_any_key=observation,
        condition_key_any_edit=any_edit,condition_any_key_any_edit=any_key,
        source_any_false_claim=source_value)
    store.data["sources"][source_id]["any_false_claim"]=source_value
    store.save()

def settle_source(store,source_id,reason):
    for query in store.data["queries"].values():
        if query["source_id"]!=source_id:continue
        if query.get("sync") is None:
            query["sync"]=dict(status="TECHNICAL_INCOMPLETE",error=reason,M=None,m=None,
                unique_action=False,top_action_count=0,chosen_action=None,top_paths=[])
        if query.get("identity") is None:
            query["identity"]=dict(status="TECHNICAL_INCOMPLETE",attempted=False,error=reason,I=None,exact32=None)
        if query.get("decision") is None:
            query["decision"]=dict(decision="UNCERTAIN",reason="UNCERTAIN_TECHNICAL_SOURCE",
                rule_sha256=store.data["rules"].get("rule_sha256"),frames=query["frames"],
                claim_message=method.CLAIM_MESSAGE)
        if query["status"]=="PENDING":query["status"]="TECHNICAL_INCOMPLETE"
    store.data["sources"][source_id].update(status="RETAINED_INCOMPLETE",error=reason)
    store.save()

def not_run_confirmation(store):
    for source_id in ("C1","C2"):
        store.data["sources"][source_id].update(status="NOT_RUN",reason="UNCERTAIN_RULE_NOT_FREEZABLE")
        for query in store.data["queries"].values():
            if query["source_id"]==source_id:
                query.update(status="NOT_RUN",decision=dict(decision="UNCERTAIN",reason="UNCERTAIN_RULE_NOT_FREEZABLE",
                    rule_sha256=None,frames=query["frames"],claim_message=method.CLAIM_MESSAGE))
    store.save()

def run(output,cfg=None):
    cfg=load_config() if cfg is None else cfg;store=Store(output,cfg)
    try:
        store.data["stage"]="DEV_PREPARATION";store.save();dev=prepare_source(store,"DEV")
        store.data["stage"]="DEV_BLIND";store.save();collect_source(store,"DEV",dev,rules=None)
        development=[q for q in store.data["queries"].values() if q["source_id"]=="DEV"]
        rules=method.freeze_rules(development,store.data["source_roster_sha256"],store.data["config_sha256"])
        store.data["rules"]=rules;store.data["seals"]["rules"]=dump(store.output/"seals"/"frozen_rules.json",rules);store.save()
        for query in development:query["decision"]=method.decide(query["sync"],query["identity"],query["frames"],rules)
        posthoc_roles(store,"DEV")
        if rules["status"]!="FROZEN":
            not_run_confirmation(store);store.data.update(status="RETAINED_INCOMPLETE",stage="FINISHED");store.save();return store.data
        for source_id in ("C1","C2"):
            try:
                store.data["stage"]=source_id+"_PREPARATION";store.save();record=prepare_source(store,source_id)
                store.data["stage"]=source_id+"_BLIND";store.save();collect_source(store,source_id,record,rules=rules)
                for query in store.data["queries"].values():
                    if query["source_id"]==source_id:
                        query["decision"]=method.decide(query["sync"],query["identity"],query["frames"],rules)
                decision_seal=dump(store.output/"seals"/(source_id+".decisions.json"),
                    dict(source_id=source_id,rule_sha256=rules["rule_sha256"],
                        decisions={q["query_id"]:q["decision"] for q in store.data["queries"].values() if q["source_id"]==source_id},
                        truth_inputs=False))
                store.data["seals"][source_id+"/decisions"]=decision_seal;posthoc_roles(store,source_id)
            except BaseException as exc:
                store.failure(store.data["stage"],exc);settle_source(store,source_id,f"{type(exc).__name__}: {exc}")
                posthoc_roles(store,source_id)
                if not isinstance(exc,Exception):raise
        store.data.update(status="COMPLETE" if not store.data["failures"] else "RETAINED_INCOMPLETE",stage="FINISHED");store.save()
        return store.data
    except BaseException as exc:
        store.failure(store.data["stage"],exc)
        settle_source(store,"DEV",f"{type(exc).__name__}: {exc}")
        if store.data["rules"].get("status")!="FROZEN":not_run_confirmation(store)
        store.data.update(status="RETAINED_INCOMPLETE",stage="FINISHED");store.save();raise

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);p.add_argument("--config",type=Path,default=CONFIG)
    p.add_argument("--worker",choices=("generation","native_decode","framewise_media"));p.add_argument("--source-id",choices=method.SOURCE_IDS)
    args=p.parse_args();cfg=load_config(args.config)
    if args.worker:
        if args.source_id is None:raise ValueError("worker source-id required")
        prepare.run_phase(args.output,args.worker,cfg,args.source_id)
    else:run(args.output,cfg)
if __name__=="__main__":main()
