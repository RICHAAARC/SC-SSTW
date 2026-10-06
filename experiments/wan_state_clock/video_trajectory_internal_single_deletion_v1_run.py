"""Fixed C/D single-deletion diagnostic. Oracle construction follows the blind payload seal."""
from pathlib import Path
import argparse,copy,hashlib,json,subprocess,sys
from experiments.wan_state_clock import receiver_records as shared
from main.tube_state import grow_video_reference as payload
from main.tube_state import video_trajectory_internal_single_deletion_v1 as method
from runtime.wan.video_trajectory_internal_single_deletion_v1 import FramewiseBackend,WanBackend
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"experiments/wan_state_clock/configs/video_trajectory_internal_single_deletion_v1.json"
PREPARATION=CONFIG.with_name(CONFIG.stem+"_preparation.json")
ORACLE=CONFIG.with_name(CONFIG.stem+"_oracle.json")
POSTHOC=CONFIG.with_name(CONFIG.stem+"_posthoc.json")
digest,read,dump,keys=shared.digest,shared.read,shared.dump,shared.keys
BLIND=method.MODES[:3]
def load_config(path=CONFIG):return shared.public_config(path)
def checked(path,h):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=h:raise RuntimeError("configuration identity mismatch: "+path.name)
    return json.loads(raw)
class Store(shared.Store):
    def __init__(self,output,cfg):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False);self.cfg=cfg
        self.data=dict(status="RUNNING",stage="INITIALIZED",
            **shared.source_metadata(ROOT,cfg,CONFIG),
            fixed_denominator=cfg["fixed_denominator"],environment=shared.environment(cfg),evidence_ceiling=cfg["evidence_ceiling"],
            calls={k:dict(attempted=0,completed=0) for k in (*cfg["planned_calls"],"wan_receiver_encode","payload_read")},
            stage_calls={stage:{k:dict(attempted=0,completed=0) for k in ("wan_receiver_encode","payload_read")} for stage in ("blind","oracle")},
            observations={oid:dict(status="PENDING",**x) for oid,x in cfg["inputs"].items()},
            sync_reads={},estimates={},physical_reads={},alignment_receipts={},payload_reads={},payload_posthoc={},path_posthoc={},differences={},failures=[])
        for oid in cfg["inputs"]:
            for key,_ in keys(cfg):
                sid=oid+"/"+key
                self.data["sync_reads"][sid]=dict(status="PENDING",path=str(self.output/"sync"/(sid+".json.gz")))
                self.data["path_posthoc"][sid]=dict(status="PENDING")
                for mode in method.MODES:
                    lid=sid+"/"+mode
                    self.data["payload_reads"][lid]=dict(status="PENDING",observation_id=oid,key_label=key,mode=mode,received_frames=177,R=44,
                        planned_detailed_votes=42240,planned_time_bit_rows=1408,planned_final_bits=32,oracle=mode=="TRUTH_PATH")
                    self.data["payload_posthoc"][lid]=dict(status="PENDING",planned_final_bits=32)
                    if mode!="RAW":self.data["differences"][lid]=dict(status="PENDING")
        self.save()
    def stage_call(self,stage,name,fn):
        self.data["stage_calls"][stage][name]["attempted"]+=1
        result=self.call(name,fn)
        self.data["stage_calls"][stage][name]["completed"]+=1;self.save();return result
def construct(full,indices):
    import torch
    if full.dtype!=torch.uint8 or full.device.type!="cpu" or len(full)!=181 or len(indices)!=177 or any(type(x) is not int or x not in range(181) for x in indices):
        raise ValueError("fixed full181 ->177 preparation map required")
    return full.index_select(0,torch.tensor(indices)).contiguous().clone()
def prepare(store,sources,backend):
    record=dict(status="PENDING",receiver_must_not_consume=True,observations={oid:dict(status="PENDING") for oid in store.cfg["inputs"]})
    try:
        cfg=checked(Path(store.cfg.get("preparation_config",PREPARATION)),store.cfg["preparation_config_sha256"])
        if set(cfg["source_frame_maps"])!=set(store.cfg["inputs"]):raise ValueError("preparation roster")
        if list(cfg["source_frame_maps"].values())!=[list(range(2,179)),list(range(2,90))+list(range(91,180))]:raise ValueError("fixed C/D maps required")
        record["source"]=cfg["source"]
        full=store.call("source_read",lambda:backend.read_source(cfg["source"]))
        record["verified_full"]=backend.pixel_receipt(full)
        if record["verified_full"]["sha256"]!=cfg["source"]["sha256"]:raise ValueError("full identity")
        for oid,indices in cfg["source_frame_maps"].items():
            try:
                clip=store.call("source_slice",lambda:construct(full,indices));receipt=backend.pixel_receipt(clip)
                if receipt["shape"]!=store.cfg["inputs"][oid]["shape"] or receipt["bytes"]!=store.cfg["inputs"][oid]["bytes"]:raise ValueError("clip geometry")
                sources[oid]=clip;store.cfg["inputs"][oid]["sha256"]=receipt["sha256"]
                store.data["observations"][oid].update(status="VERIFIED",**receipt)
                record["observations"][oid]=dict(status="PREPARED",source_frame_map=indices,received=receipt)
            except Exception as exc:
                store.failure(oid+"/PREPARE",exc);record["observations"][oid].update(status="FAILED",error=str(exc))
        record["status"]="COMPLETE" if len(sources)==2 else "INCOMPLETE"
    except Exception as exc:record.update(status="FAILED",error=str(exc));store.failure("PREPARATION",exc)
    finally:
        for oid,row in record["observations"].items():
            if row["status"]=="PENDING":row.update(status="FAILED",error="PREPARATION_NOT_COMPLETED")
            if oid not in sources:store.data["observations"][oid].update(status="FAILED",error=row.get("error","missing"))
        if record["status"]=="PENDING":record.update(status="INTERRUPTED")
        store.data["preparation_receipt"]=dump(store.output/"preparation_receipt.json",record);store.save()
def seal_sync(store):
    for oid in store.cfg["inputs"]:
        for key,label in keys(store.cfg):
            sid=oid+"/"+key;row=store.data["sync_reads"][sid]
            if row["status"]!="SAVED":
                data=method.missing(label,row.get("error","NOT_COMPLETED"));receipt=dump(row["path"],dict(readout=data,truth_inputs=False))
                row.update(status="FAILED",sha256=receipt["sha256"],error=data["error"])
            if digest(row["path"])!=row["sha256"]:raise RuntimeError("sync sidecar identity")
            store.data["estimates"][sid]=method.estimates(read(row["path"])["readout"],label)
    store.data["blind_sync_seal"]=dump(store.output/"blind_sync_readouts.json",dict(observations=store.data["observations"],sync_reads=store.data["sync_reads"],estimates=store.data["estimates"],truth_inputs=False))
    plan=method.plan_blind(store.cfg["inputs"],store.data["estimates"])
    store.data["blind_plan_seal"]=dump(store.output/"blind_plan.json",plan);store.data["blind_plan"]=copy.deepcopy(plan)
    install_plan(store,plan);store.save();return plan
def install_plan(store,plan,modes=BLIND):
    for rid,item in plan["reads"].items():
        store.data["physical_reads"].setdefault(rid,dict(status="PENDING",**copy.deepcopy(item),detail_path=str(store.output/"physical"/(rid+".votes.json.gz"))))
        row=store.data["physical_reads"][rid]
        row["logical_slots"]=list(dict.fromkeys(row["logical_slots"]+item["logical_slots"]))
    for lid,item in plan["logical_slots"].items():
        if lid.rsplit("/",1)[-1] in modes:store.data["payload_reads"][lid].update(copy.deepcopy(item))
def check_seal(receipt):
    if digest(receipt["path"])!=receipt["sha256"]:raise RuntimeError("seal identity mismatch")
def run_plan(store,sources,backend,cache,plan,stage):
    for gid,item in plan["encodes"].items():
        pending=[(rid,r) for rid,r in store.data["physical_reads"].items() if r["encode_id"]==gid and r["status"]=="PENDING"]
        if not pending:continue
        receipt=store.data["alignment_receipts"].get(gid)
        z=None;received=None
        try:
            if receipt is None:
                if item["observation_id"] not in sources or backend is None:raise ValueError("received RGB/backend unavailable")
                received=backend.operate_map(sources[item["observation_id"]],item["received_index_map"])
                receipt=dict(status="ENCODING",**item,before_sha256=store.cfg["inputs"][item["observation_id"]]["sha256"],
                    after=backend.pixel_receipt(received),first_stage=stage)
                store.data["alignment_receipts"][gid]=receipt;store.save()
                z=store.stage_call(stage,"wan_receiver_encode",lambda:backend.encode(received))
                cache[gid]=backend.cache_latent(z);receipt.update(status="ENCODED",normalized_shape=list(z.shape));store.save()
            elif receipt["status"]!="ENCODED":raise ValueError("cached encode failure; no retry")
            if gid not in cache:raise ValueError("cached latent unavailable; no re-encode")
            z=backend.restore_latent(cache[gid])
            for rid,row in pending:
                try:
                    row["status"]="RUNNING";store.save()
                    detail=store.stage_call(stage,"payload_read",lambda k=dict(keys(store.cfg))[row["key_label"]]:backend.read(z,k,177))
                    r=dump(row["detail_path"],detail)
                    row.update(status="READ",detail_sha256=r["sha256"],detail_bytes=r["bytes"],first_stage=stage,
                        decoded_bits=detail["original_readout"]["decoded_bits"],votes=detail["original_readout"]["votes"],
                        detailed_votes=detail["detailed_vote_count"],time_bit_rows=detail["time_bit_count"],
                        original_reader_match=detail["original_reader_match"],truth_inputs=False);store.save()
                except Exception as exc:row.update(status="FAILED",error=str(exc));store.failure(rid+"/READ",exc)
        except Exception as exc:
            if receipt is None:store.data["alignment_receipts"][gid]=dict(status="FAILED",**item,error=str(exc),first_stage=stage)
            elif receipt["status"]=="ENCODING":receipt.update(status="FAILED",error=str(exc))
            for _,row in pending:
                if row["status"]=="PENDING":row.update(status="FAILED",error=str(exc))
            store.failure(gid+"/ENCODE",exc)
        finally:z=None;received=None
def seal_payload(store,modes,name,oracle):
    for row in store.data["alignment_receipts"].values():
        if row["status"]=="ENCODING":row.update(status="FAILED",error="INTERRUPTED")
    selected={lid:x for lid,x in store.data["payload_reads"].items() if x["mode"] in modes}
    for lid,row in selected.items():
        rid=row.get("physical_read");physical=store.data["physical_reads"].get(rid)
        if physical is not None:
            if physical["status"]!="READ" and "detail_sha256" not in physical:
                physical.update(status="FAILED",error=physical.get("error","NOT_COMPLETED"))
                r=dump(physical["detail_path"],method.prior.missing_detail(177,physical["error"]))
                physical.update(detail_sha256=r["sha256"],detail_bytes=r["bytes"])
            for k in ("status","error","detail_path","detail_sha256","detail_bytes","decoded_bits","votes","detailed_votes","time_bit_rows"):
                if k in physical:row[k]=physical[k]
        else:
            row.update(status="FAILED",error=row.get("error","NO_RESOLVED_OR_COMPLETED_PLAN"))
            r=dump(store.output/"unresolved"/(lid+".votes.json.gz"),method.prior.missing_detail(177,row["error"]))
            row.update(detail_path=r["path"],detail_sha256=r["sha256"],detail_bytes=r["bytes"])
    refs={x.get("physical_read") for x in selected.values()}
    receipt=dump(store.output/(name+".json"),dict(logical_reads=selected,
        physical_reads={rid:x for rid,x in store.data["physical_reads"].items() if rid in refs},
        alignment_receipts=store.data["alignment_receipts"],oracle=oracle,truth_inputs=oracle,
        blind_sync_sha256=store.data["blind_sync_seal"]["sha256"],blind_plan_sha256=store.data["blind_plan_seal"]["sha256"]))
    store.data[name+"_seal"]=receipt;store.save()
def oracle_plan(store,blind):
    check_seal(store.data["blind_sync_seal"]);check_seal(store.data["blind_payload_seal"])
    truth=checked(Path(store.cfg.get("oracle_config",ORACLE)),store.cfg["oracle_config_sha256"])
    if set(truth["paths"])!=set(store.cfg["inputs"]):raise ValueError("oracle roster")
    plan=copy.deepcopy(blind);plan["truth_inputs"]=True
    for oid,spec in store.cfg["inputs"].items():
        op=method.correction(truth["paths"][oid])
        for key,_ in keys(store.cfg):method.add_slot(plan,oid,key,"TRUTH_PATH",op,spec,oracle=True)
    store.data["oracle_config_sha256"]=store.cfg["oracle_config_sha256"]
    store.data["oracle_plan_seal"]=dump(store.output/"oracle_plan.json",plan)
    install_plan(store,plan,("TRUTH_PATH",));store.save();return plan
def posthoc(store):
    check_seal(store.data["blind_payload_seal"]);check_seal(store.data["oracle_payload_seal"])
    cfg=checked(Path(store.cfg.get("posthoc_config",POSTHOC)),store.cfg["posthoc_config_sha256"]);expected=payload.message_bits(cfg["message"]);joined={}
    store.data["posthoc_config_sha256"]=store.cfg["posthoc_config_sha256"]
    for lid,row in store.data["payload_reads"].items():
        oid,key,mode=lid.split("/")
        if digest(row["detail_path"])!=row["detail_sha256"]:raise RuntimeError("votes identity")
        h=method.prior.join_truth(read(row["detail_path"]),expected);joined[lid]=h
        truth=cfg["truth"][oid];indices=row.get("operation",{}).get("received_index_map")
        h.update(key_role="REGISTERED" if key=="K0" else "WRONG_KEY",mode=mode,view=truth["view"],oracle=mode=="TRUTH_PATH",
            alias_of=row.get("alias_of"),source_frame_map_posthoc=[truth["source_frame_map"][i] for i in indices] if indices is not None else None)
        rec=dump(store.output/"posthoc"/(lid+".json.gz"),h)
        store.data["payload_posthoc"][lid]={k:v for k,v in h.items() if k!="time_bit_rows"}
        store.data["payload_posthoc"][lid].update(path=rec["path"],sha256=rec["sha256"],time_bit_rows=len(h["time_bit_rows"]),summary=method.prior.summarize(h));store.save()
    for sid,est in store.data["estimates"].items():
        oid,key=sid.split("/");truth=cfg["truth"][oid]
        store.data["path_posthoc"][sid]=dict(status="EVALUATED_POSTSEAL",view=truth["view"],key_role="REGISTERED" if key=="K0" else "WRONG_KEY",
            H0=method.path_evaluation(est["H0"],truth),joint=method.path_evaluation(est["joint"],truth))
        for mode in method.MODES[1:]:
            lid=sid+"/"+mode;a=joined[sid+"/RAW"];b=joined[lid]
            if a["status"]==b["status"]=="EVALUATED_TRUTH":
                deltas=[dict(bit_index=x["bit_index"],**({"receiver_latent_index":x["receiver_latent_index"]} if "receiver_latent_index" in x else {}),
                    signed_normalized_margin_delta=y["signed_normalized_margin"]-x["signed_normalized_margin"],decoded_error_delta=y["bit_error"]-x["bit_error"])
                    for x,y in zip(a["bit_rows"]+a["time_bit_rows"],b["bit_rows"]+b["time_bit_rows"])]
                store.data["differences"][lid]=dict(status="DESCRIPTIVE_POSTSEAL",oracle=mode=="TRUTH_PATH",alias_of=store.data["payload_reads"][lid].get("alias_of"),
                    bit_error_delta=b["bit_errors"]-a["bit_errors"],bit_rows=deltas[:32],time_bit_rows=deltas[32:],
                    comparison_limit="Nominal receiver index; not matched source receptive fields")
            else:store.data["differences"][lid]=dict(status="FAILED",error="paired read unavailable")
    store.save()
def run(output,*,cfg=None,framewise_type=FramewiseBackend,wan_type=WanBackend):
    cfg=copy.deepcopy(load_config() if cfg is None else cfg);store=Store(output,cfg);sources={};cache={};fw=wan=None;aborted=None
    try:
        prepare(store,sources,wan_type)
        if sources:fw=store.call("framewise_vae_load",lambda:framewise_type(cfg["framewise_model"]))
        for oid in cfg["inputs"]:
            if oid not in sources:continue
            latent=None
            try:
                latent,enc=store.call("framewise_receiver_encode",lambda:fw.encode(sources[oid]))
                for key,label in keys(cfg):
                    sid=oid+"/"+key;row=store.data["sync_reads"][sid]
                    try:
                        a=store.call("sync_score",lambda k=label:fw.score(latent,k))
                        r=dump(row["path"],dict(readout=a,encode_receipt=enc,truth_inputs=False))
                        row.update(status="SAVED",sha256=r["sha256"],**a["counts"]);store.save()
                    except Exception as exc:row.update(status="FAILED",error=str(exc));store.failure(sid+"/SYNC",exc)
            except Exception as exc:store.failure(oid+"/FRAMEWISE",exc)
            finally:latent=None
    except BaseException as exc:aborted=exc;store.failure("SYNC_INTERRUPTED",exc)
    finally:
        if fw is not None:
            try:fw.close()
            except BaseException as exc:store.failure("FRAMEWISE_RELEASE",exc)
    blind=seal_sync(store);plan=blind
    try:
        try:
            if aborted is None and sources:wan=store.call("wan_vae_load",lambda:wan_type(cfg["model"]))
            if aborted is None:run_plan(store,sources,wan,cache,blind,"blind")
        except BaseException as exc:aborted=exc;store.failure("BLIND_PAYLOAD_INTERRUPTED",exc)
        finally:seal_payload(store,BLIND,"blind_payload",False)
        try:
            if aborted is None:
                plan=oracle_plan(store,blind);run_plan(store,sources,wan,cache,plan,"oracle")
        except BaseException as exc:aborted=exc;store.failure("ORACLE_INTERRUPTED",exc)
        finally:seal_payload(store,("TRUTH_PATH",),"oracle_payload",True)
    finally:
        cache.clear();sources.clear()
        if wan is not None:
            try:wan.close()
            except BaseException as exc:store.failure("WAN_RELEASE",exc)
    try:posthoc(store)
    except BaseException as exc:
        store.failure("POSTHOC",exc)
        for kind in ("payload_posthoc","path_posthoc","differences"):
            for row in store.data[kind].values():
                if row["status"]=="PENDING":row.update(status="FAILED",error=str(exc),
                    bit_rows=[dict(bit_index=b,status="FAILED") for b in range(32)] if kind=="payload_posthoc" else [])
    def counts(rows):
        good=[r for r in rows if r["status"]=="READ"]
        return dict(reads=len(good),votes=sum(r["detailed_votes"] for r in good),time_bit_rows=sum(r["time_bit_rows"] for r in good),final_bits=32*len(good))
    store.data["counts"]=dict(logical=counts(store.data["payload_reads"].values()),physical=counts(store.data["physical_reads"].values()),
        blind=counts([r for r in store.data["payload_reads"].values() if not r["oracle"]]),oracle=counts([r for r in store.data["payload_reads"].values() if r["oracle"]]))
    store.data["planned_physical_calls"]=dict(wan_receiver_encode=len(plan["encodes"]),payload_read=len(plan["reads"]))
    assert len(plan["encodes"])<=12 and len(plan["reads"])<=16
    targets=dict(cfg["planned_calls"],**store.data["planned_physical_calls"])
    store.data["call_integrity"]={k:dict(expected=n,actual=store.data["calls"][k],match=store.data["calls"][k]==dict(attempted=n,completed=n)) for k,n in targets.items()}
    store.data["status"]="COMPLETE" if all(r["status"]=="READ" for r in store.data["payload_reads"].values()) and not store.data["failures"] else "INCOMPLETE"
    store.data["stage"]="FINISHED";store.save()
    if aborted is not None:raise aborted
    return store.data
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--config",type=Path,required=True);a=p.parse_args()
    r=run(a.output,cfg=load_config(a.config));print(json.dumps(dict(status=r["status"],counts=r["counts"],result=str(a.output/"result.json"))))
    return 0 if r["status"]=="COMPLETE" else 1
if __name__=="__main__":sys.exit(main())
