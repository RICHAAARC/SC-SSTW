"""Milestone1A: one saved FULL read, four CPU crops, unchanged Stage2 receiver."""
from pathlib import Path
import argparse,copy,hashlib,json,subprocess,sys
from experiments.wan_state_clock import video_trajectory_receiver_estimated_align_v1_run as shared
from main.tube_state import grow_video_reference as payload
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as method
from runtime.wan.video_trajectory_receiver_estimated_align_v1 import FramewiseBackend,WanBackend
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"experiments/wan_state_clock/configs/video_trajectory_receiver_phase23_v1.json"
POSTHOC=CONFIG.with_name("video_trajectory_receiver_phase23_v1_posthoc.json")
PREPARATION=CONFIG.with_name("video_trajectory_receiver_phase23_v1_preparation.json")
digest=shared.digest
read=shared.read
dump=shared.dump
keys=shared.keys
environment=shared.environment
run_payload=shared.run_payload
seal_payload=shared.seal_payload
def load_config():return read(CONFIG)

class Store(shared.Store):
    def __init__(self,output,cfg):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False);self.cfg=cfg
        sha=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
        clean=not subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain"],text=True)
        self.data=dict(status="RUNNING",stage="INITIALIZED",source_sha=sha,source_clean=clean,
            source_files={p:digest(ROOT/p) for p in cfg["source_files"]},config_sha256=digest(CONFIG),
            fixed_denominator=cfg["fixed_denominator"],planned_calls=cfg["planned_calls"],
            physical_upper_bounds=cfg["physical_upper_bounds"],environment=environment(cfg),
            calls={k:dict(attempted=0,completed=0) for k in (*cfg["planned_calls"],"wan_receiver_encode","payload_read")},
            observations={o:dict(status="PENDING",**s) for o,s in cfg["inputs"].items()},
            sync_reads={},estimates={},physical_reads={},payload_reads={},payload_posthoc={},
            sync_posthoc={},same_run_differences={},failures=[],
            evidence_ceiling=cfg["evidence_ceiling"])
        for oid,spec in cfg["inputs"].items():
            n=spec["frames"];R=method.support(n)
            for key,_ in keys(cfg):
                sid=oid+"/"+key
                self.data["sync_reads"][sid]=dict(status="PENDING",path=str(self.output/"sync"/oid/(key+".json.gz")))
                self.data["sync_posthoc"][sid]=dict(status="PENDING")
                self.data["same_run_differences"][sid]=dict(status="PENDING")
                for mode in ("BASELINE","EST_ALIGN"):
                    lid=sid+"/"+mode
                    self.data["payload_reads"][lid]=dict(status="PENDING",observation_id=oid,key_label=key,mode=mode,
                        received_frames=n,R=R,planned_detailed_votes=32*30*R,planned_time_bit_rows=32*R,planned_final_bits=32)
                    self.data["payload_posthoc"][lid]=dict(status="PENDING",planned_final_bits=32)
        self.save()

def construct_received(full,start,length):
    """Fixture preparation only; start never enters receiver method calls."""
    import torch
    if full.dtype!=torch.uint8 or full.device.type!="cpu" or full.ndim!=4 or full.shape[0]!=181 or full.shape[-1]!=3:
        raise ValueError("verified CPU FULL181 RGB required")
    if type(start) is not int or length not in (177,89) or start<0 or start+length>181:
        raise ValueError("fixed in-range crop required")
    return full[start:start+length].contiguous().clone()

def prepare_inputs(store,sources,backend_type):
    record=dict(status="PENDING",receiver_must_not_consume=True,source=None,
        crops={oid:dict(status="PENDING") for oid in store.cfg["inputs"]})
    full=None
    try:
        raw=PREPARATION.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=store.cfg["preparation_config_sha256"]:raise RuntimeError("preparation config mismatch")
        prep=json.loads(raw)
        if set(prep["crops"])!=set(store.cfg["inputs"]):raise ValueError("preparation roster mismatch")
        record["source"]=prep["source"];record["preparation_config_sha256"]=hashlib.sha256(raw).hexdigest()
        for oid,x in prep["crops"].items():
            if x["length"]!=store.cfg["inputs"][oid]["frames"]:raise ValueError("public length mismatch")
            record["crops"][oid].update(source_start=x["start"],source_stop=x["start"]+x["length"])
        full=store.call("source_read",lambda:backend_type.read_source(prep["source"]))
        record["source_verified"]=backend_type.pixel_receipt(full)
        if record["source_verified"]["sha256"]!=prep["source"]["sha256"]:raise ValueError("FULL identity mismatch")
        for oid,item in prep["crops"].items():
            try:
                clip=store.call("source_slice",lambda x=item:construct_received(full,x["start"],x["length"]))
                receipt=backend_type.pixel_receipt(clip);spec=store.cfg["inputs"][oid]
                if receipt["shape"]!=spec["shape"] or receipt["bytes"]!=spec["bytes"]:raise ValueError("crop shape/size mismatch")
                spec["sha256"]=receipt["sha256"];sources[oid]=clip
                store.data["observations"][oid].update(status="VERIFIED",**receipt)
                record["crops"][oid].update(status="PREPARED",received=receipt,
                    source_frame_map=list(range(item["start"],item["start"]+item["length"])))
                store.save()
            except Exception as exc:
                record["crops"][oid].update(status="FAILED",error=str(exc))
                store.data["observations"][oid].update(status="FAILED",error=str(exc))
                store.failure(oid+"/SLICE",exc)
        record["status"]="COMPLETE" if len(sources)==len(prep["crops"]) else "INCOMPLETE"
    except Exception as exc:
        record.update(status="FAILED",error=str(exc));store.failure("SOURCE_PREPARATION",exc)
    finally:
        full=None
        for oid,row in record["crops"].items():
            if row["status"]=="PENDING":
                row.update(status="FAILED",error=record.get("error","PREPARATION_NOT_COMPLETED"))
                store.data["observations"][oid].update(status="FAILED",error=row["error"])
        if record["status"]=="PENDING":record.update(status="INTERRUPTED",error="PREPARATION_NOT_COMPLETED")
        receipt=dump(store.output/"preparation_receipt.json",record)
        store.data["source_preparation"]=dict(status=record["status"],path=receipt["path"],sha256=receipt["sha256"],
            receiver_inputs="Only opaque observation RGB, public lengths and per-key estimates; crop truth excluded",
            clip_storage="CPU copies; no new codec or separate received-media save")
        store.save()

def seal_sync_and_plan(store):
    for oid,spec in store.cfg["inputs"].items():
        for key,label in keys(store.cfg):
            sid=oid+"/"+key;row=store.data["sync_reads"][sid]
            if row["status"]!="SAVED":
                detail=method.missing_sync(spec["frames"],label,row.get("error","NOT_COMPLETED"))
                receipt=dump(row["path"],dict(readout=detail,truth_inputs=False))
                row.update(status="FAILED",sha256=receipt["sha256"],error=detail["error"])
            saved=read(row["path"])["readout"]
            assert digest(row["path"])==row["sha256"]
            store.data["estimates"][sid]=method.estimate(saved,spec["frames"],label)
    for row in store.data["observations"].values():
        if row["status"]=="PENDING":row.update(status="FAILED",error="NOT_COMPLETED")
    receipt=dump(store.output/"blind_sync_readouts.json",dict(observations=store.data["observations"],
        sync_reads=store.data["sync_reads"],estimates=store.data["estimates"],truth_inputs=False))
    store.data["blind_sync_sha256"]=receipt["sha256"]
    plan=method.physical_plan(store.cfg["inputs"],store.data["estimates"])
    assert all(plan["planned_calls"][k]<=v for k,v in store.cfg["physical_upper_bounds"].items())
    receipt=dump(store.output/"physical_plan.json",plan)
    store.data["physical_plan_sha256"]=receipt["sha256"];store.data["physical_plan"]=plan
    store.data["physical_reads"]={rid:dict(status="PENDING",**x,detail_path=str(store.output/"physical"/(rid+".votes.json.gz")))
                                  for rid,x in plan["reads"].items()}
    for lid,x in plan["logical_slots"].items():store.data["payload_reads"][lid].update(x)
    store.data["stage"]="PHYSICAL_PLAN_FROZEN";store.save()
    return plan

def load_posthoc(store):
    """Reporting truth is first read after the final payload seal; no history lookup."""
    if digest(store.output/"blind_payload_readouts.json")!=store.data["blind_payload_sha256"]:raise RuntimeError("payload seal mismatch")
    raw=POSTHOC.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=store.cfg["posthoc_config_sha256"]:raise RuntimeError("posthoc config identity mismatch")
    store.data["posthoc_config_sha256"]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def join_posthoc(store):
    reporting=load_posthoc(store);expected=payload.message_bits(reporting["message"]);joined={}
    for lid,row in store.data["payload_reads"].items():
        oid,key,mode=lid.split("/")
        if digest(row["detail_path"])!=row["detail_sha256"]:raise RuntimeError("detail hash mismatch")
        h=method.join_truth(read(row["detail_path"]),expected);joined[lid]=h
        truth=reporting["truth"][oid]
        est=store.data["estimates"][oid+"/"+key]
        phase=0 if mode=="BASELINE" else est.get("phase")
        h.update(key_role="REGISTERED" if key=="K0" else "WRONG_KEY",mode=mode,alias_of=row.get("alias_of"),
            **truth,estimated_offset=est.get("offset"),phase=phase,
            source_frame_map_posthoc=[truth["source_start"]+i for i in method.phase_map(row["received_frames"],phase)] if phase is not None else None)
        receipt=dump(store.output/"posthoc"/(lid+".json.gz"),h)
        store.data["payload_posthoc"][lid]={k:v for k,v in h.items() if k!="time_bit_rows"}
        store.data["payload_posthoc"][lid].update(path=receipt["path"],sha256=receipt["sha256"],time_bit_rows=len(h["time_bit_rows"]),
            summary=method.summarize(h));store.save()
    for oid,spec in store.cfg["inputs"].items():
        truth=reporting["truth"][oid]
        for key,_ in keys(store.cfg):
            sid=oid+"/"+key;est=store.data["estimates"][sid]
            store.data["sync_posthoc"][sid]=dict(status="EVALUATED_POSTSEAL",key_role="REGISTERED" if key=="K0" else "WRONG_KEY",
                estimated_offset=est.get("offset"),truth_offset=truth["source_start"],
                offset_correct=est.get("offset")==truth["source_start"] if est["status"]=="ESTIMATED" else None,
                phase_correct=est.get("phase")==truth["source_start"]%4 if est["status"]=="ESTIMATED" else None,
                geometry_only=spec["frames"]==181)
            base=joined[sid+"/BASELINE"];changed=joined[sid+"/EST_ALIGN"]
            if base["status"]==changed["status"]=="EVALUATED_TRUTH":
                store.data["same_run_differences"][sid]=dict(status="DESCRIPTIVE_POSTSEAL",
                    bit_error_delta=changed["bit_errors"]-base["bit_errors"],
                    bit_rows=[dict(bit_index=a["bit_index"],signed_normalized_margin_delta=z["signed_normalized_margin"]-a["signed_normalized_margin"],
                        decoded_error_delta=z["bit_error"]-a["bit_error"]) for a,z in zip(base["bit_rows"],changed["bit_rows"])],
                    time_bit_rows=[dict(receiver_latent_index=a["receiver_latent_index"],bit_index=a["bit_index"],payload_channel=a["payload_channel"],
                        signed_normalized_margin_delta=z["signed_normalized_margin"]-a["signed_normalized_margin"],decoded_error_delta=z["bit_error"]-a["bit_error"])
                        for a,z in zip(base["time_bit_rows"],changed["time_bit_rows"])],alias_of=store.data["payload_reads"][sid+"/EST_ALIGN"].get("alias_of"))
            else:store.data["same_run_differences"][sid]=dict(status="FAILED",error="fixed paired read unavailable")
    store.save()

def run(output,*,cfg=None,framewise_type=FramewiseBackend,wan_type=WanBackend):
    cfg=copy.deepcopy(load_config() if cfg is None else cfg);store=Store(output,cfg);sources={};fw=None;aborted=None
    try:
        prepare_inputs(store,sources,wan_type)
        if sources:fw=store.call("framewise_vae_load",lambda:framewise_type(cfg["framewise_model"]))
        for oid,spec in cfg["inputs"].items():
            if oid not in sources:continue
            latent=None
            try:
                latent,enc=store.call("framewise_receiver_encode",lambda:fw.encode(sources[oid]))
                for key,label in keys(cfg):
                    sid=oid+"/"+key;row=store.data["sync_reads"][sid]
                    try:
                        result=store.call("sync_score",lambda k=label:fw.score(latent,k))
                        receipt=dump(row["path"],dict(readout=result,encode_receipt=enc,truth_inputs=False))
                        row.update(status="SAVED",sha256=receipt["sha256"],summary=result["summary"],**result["counts"]);store.save()
                    except Exception as exc:row.update(status="FAILED",error=str(exc));store.failure(sid+"/SYNC",exc)
            except Exception as exc:store.failure(oid+"/FRAMEWISE",exc)
            finally:latent=None
    except BaseException as exc:aborted=exc;store.failure("FRESH_SYNC_INTERRUPTED",exc)
    finally:
        if fw is not None:
            try:fw.close()
            except BaseException as exc:store.failure("FRAMEWISE_RELEASE",exc)
        plan=seal_sync_and_plan(store)
    try:
        if aborted is None:run_payload(store,sources,wan_type,plan)
    except BaseException as exc:aborted=exc;store.failure("PAYLOAD_INTERRUPTED",exc)
    finally:sources.clear();seal_payload(store)
    try:join_posthoc(store)
    except BaseException as exc:
        store.failure("POSTHOC",exc)
        for kind in ("payload_posthoc","sync_posthoc","same_run_differences"):
            for row in store.data[kind].values():
                if row["status"]=="PENDING":row.update(status="FAILED",error=str(exc),
                    bit_rows=[dict(bit_index=i,status="FAILED") for i in range(32)] if kind=="payload_posthoc" else [])
    logical=list(store.data["payload_reads"].values());physical=list(store.data["physical_reads"].values())
    def counts(rows):
        good=[x for x in rows if x["status"]=="READ"]
        return dict(reads=len(good),votes=sum(x["detailed_votes"] for x in good),time_bit_rows=sum(x["time_bit_rows"] for x in good),final_bits=32*len(good))
    store.data["counts"]=dict(logical=counts(logical),physical=counts(physical),
        sync_reads=sum(x["status"]=="SAVED" for x in store.data["sync_reads"].values()))
    targets=dict(cfg["planned_calls"],**plan["planned_calls"])
    store.data["call_integrity"]={k:dict(expected=n,actual=store.data["calls"][k],match=store.data["calls"][k]==dict(attempted=n,completed=n)) for k,n in targets.items()}
    store.data["status"]="COMPLETE" if all(x["status"]=="READ" for x in logical) and not store.data["failures"] else "INCOMPLETE"
    store.data["stage"]="FINISHED";store.save()
    if aborted is not None:raise aborted
    return store.data

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();r=run(args.output)
    print(json.dumps(dict(status=r["status"],counts=r["counts"],result=str(args.output/"result.json"))))
    return 0 if r["status"]=="COMPLETE" else 1
if __name__=="__main__":sys.exit(main())
