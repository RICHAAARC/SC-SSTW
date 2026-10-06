"""Fixed saved-M05 fresh-sync -> deterministic alignment -> Wan receiver."""
from pathlib import Path
import argparse,gzip,hashlib,importlib.metadata,json,os,subprocess,sys
from main.tube_state import grow_video_reference as payload
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as method
from runtime.wan.video_trajectory_receiver_estimated_align_v1 import FramewiseBackend,WanBackend
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"experiments/wan_state_clock/configs/video_trajectory_receiver_estimated_align_v1.json"
POSTHOC=CONFIG.with_name("video_trajectory_receiver_estimated_align_v1_posthoc.json")
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p=Path(p);raw=p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix==".gz" else raw)
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(obj,indent=2,allow_nan=False)+"\n").encode()
    if p.suffix==".gz":raw=gzip.compress(raw,mtime=0)
    tmp=p.with_suffix(p.suffix+".tmp");tmp.write_bytes(raw);os.replace(tmp,p)
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
def load_config():return read(CONFIG)
def keys(cfg):return (("K0",cfg["key"]),("K1",cfg["wrong_key"]))
def environment(cfg):
    result={}
    for name in cfg["environment_pins"]:
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    return result

class Store:
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
            sync_posthoc={},historical_comparisons={},same_run_differences={},failures=[],
            evidence_ceiling=cfg["evidence_ceiling"])
        for oid,spec in cfg["inputs"].items():
            n=spec["frames"];R=method.support(n)
            for key,_ in keys(cfg):
                sid=oid+"/"+key
                self.data["sync_reads"][sid]=dict(status="PENDING",path=str(self.output/"sync"/oid/(key+".json.gz")))
                self.data["sync_posthoc"][sid]=dict(status="PENDING")
                self.data["historical_comparisons"][sid]=dict(status="PENDING")
                self.data["same_run_differences"][sid]=dict(status="PENDING")
                for mode in ("BASELINE","EST_ALIGN"):
                    lid=sid+"/"+mode
                    self.data["payload_reads"][lid]=dict(status="PENDING",observation_id=oid,key_label=key,mode=mode,
                        received_frames=n,R=R,planned_detailed_votes=32*30*R,planned_time_bit_rows=32*R,planned_final_bits=32)
                    self.data["payload_posthoc"][lid]=dict(status="PENDING",planned_final_bits=32)
        self.save()
    def save(self):dump(self.output/"result.json",self.data)
    def failure(self,stage,exc):
        self.data["failures"].append(dict(stage=stage,error=f"{type(exc).__name__}: {exc}"));self.save()
    def call(self,name,fn):
        self.data["calls"][name]["attempted"]+=1;self.save()
        value=fn();self.data["calls"][name]["completed"]+=1;self.save();return value

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
    assert plan["planned_calls"]["wan_receiver_encode"]<=7 and plan["planned_calls"]["payload_read"]<=10
    receipt=dump(store.output/"physical_plan.json",plan)
    store.data["physical_plan_sha256"]=receipt["sha256"];store.data["physical_plan"]=plan
    store.data["physical_reads"]={rid:dict(status="PENDING",**x,detail_path=str(store.output/"physical"/(rid+".votes.json.gz")))
                                  for rid,x in plan["reads"].items()}
    for lid,x in plan["logical_slots"].items():store.data["payload_reads"][lid].update(x)
    store.data["stage"]="PHYSICAL_PLAN_FROZEN";store.save()
    return plan

def run_payload(store,sources,backend_type,plan):
    if digest(store.output/"blind_sync_readouts.json")!=store.data["blind_sync_sha256"]:raise RuntimeError("sync seal mismatch")
    if digest(store.output/"physical_plan.json")!=store.data["physical_plan_sha256"]:raise RuntimeError("plan seal mismatch")
    backend=None
    try:
        if sources:backend=store.call("wan_vae_load",lambda:backend_type(store.cfg["model"]))
        for gid,item in plan["encodes"].items():
            oid=item["observation_id"];normalized=None;received=None
            receipts=store.data.setdefault("alignment_receipts",{})
            try:
                if oid not in sources:raise ValueError("fixed received RGB unavailable")
                received=backend_type.operate_clip(sources[oid],item["phase"])
                receipts[gid]=dict(status="ENCODING",before_sha256=store.cfg["inputs"][oid]["sha256"],
                    after=backend_type.pixel_receipt(received),**item);store.save()
                normalized=store.call("wan_receiver_encode",lambda:backend.encode(received))
                receipts[gid].update(status="ENCODED",normalized_shape=list(normalized.shape));store.save()
                for rid,row in store.data["physical_reads"].items():
                    if row["encode_id"]!=gid:continue
                    try:
                        row["status"]="RUNNING";store.save()
                        key=dict(keys(store.cfg))[row["key_label"]]
                        detail=store.call("payload_read",lambda k=key:backend.read(normalized,k,item["frames"]))
                        receipt=dump(row["detail_path"],detail)
                        row.update(status="READ",detail_sha256=receipt["sha256"],detail_bytes=receipt["bytes"],
                            decoded_bits=detail["original_readout"]["decoded_bits"],votes=detail["original_readout"]["votes"],
                            detailed_votes=detail["detailed_vote_count"],time_bit_rows=detail["time_bit_count"],
                            original_reader_match=detail["original_reader_match"],truth_inputs=False);store.save()
                    except Exception as exc:
                        row.update(status="FAILED",error=str(exc));store.failure(rid+"/READ",exc)
            except Exception as exc:
                receipts.setdefault(gid,dict(**item)).update(status="FAILED",error=str(exc))
                for row in store.data["physical_reads"].values():
                    if row["encode_id"]==gid and row["status"]=="PENDING":row.update(status="FAILED",error=str(exc))
                store.failure(gid+"/ENCODE",exc)
            finally:normalized=None;received=None
    finally:
        if backend is not None:
            try:backend.close()
            except BaseException as exc:store.failure("WAN_RELEASE",exc)

def seal_payload(store):
    for row in store.data.get("alignment_receipts",{}).values():
        if row["status"]=="ENCODING":row.update(status="FAILED",error="NOT_COMPLETED")
    for row in store.data["physical_reads"].values():
        if row["status"]!="READ":
            row.update(status="FAILED",error=row.get("error","NOT_COMPLETED"))
            receipt=dump(row["detail_path"],method.missing_detail(row["frames"],row["error"]))
            row.update(detail_sha256=receipt["sha256"],detail_bytes=receipt["bytes"])
    for lid,row in store.data["payload_reads"].items():
        rid=row.get("physical_read")
        if rid is not None:
            physical=store.data["physical_reads"][rid]
            for k in ("status","error","detail_path","detail_sha256","detail_bytes","decoded_bits","votes","detailed_votes","time_bit_rows"):
                if k in physical:row[k]=physical[k]
        else:
            row.update(status="FAILED",error=row.get("error","NO_UNIQUE_COMPLETE_ESTIMATE"))
            receipt=dump(store.output/"unresolved"/(lid+".votes.json.gz"),method.missing_detail(row["received_frames"],row["error"]))
            row.update(detail_path=receipt["path"],detail_sha256=receipt["sha256"],detail_bytes=receipt["bytes"])
    receipt=dump(store.output/"blind_payload_readouts.json",dict(
        logical_reads=store.data["payload_reads"],physical_reads=store.data["physical_reads"],
        alignment_receipts=store.data.get("alignment_receipts",{}),physical_plan_sha256=store.data["physical_plan_sha256"],
        blind_sync_sha256=store.data["blind_sync_sha256"],truth_inputs=False))
    store.data["blind_payload_sha256"]=receipt["sha256"];store.data["stage"]="PAYLOAD_BLIND_SEALED";store.save()

def load_posthoc(store):
    """Only called after the final blind seal. No earlier truth/history file read."""
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
    # History is first touched here, after both new blind seals and truth reporting.
    try:
        ref=reporting["history"]
        if digest(ref["result_path"])!=ref["result_sha256"] or digest(ref["blind_path"])!=ref["blind_sha256"]:
            raise RuntimeError("historical fixed identity mismatch")
        old=read(ref["result_path"]);oldseal=read(ref["blind_path"])
        for sid,row in store.data["sync_reads"].items():
            prior=oldseal["sync_reads"][sid]
            if digest(prior["path"])!=prior["sha256"]:raise RuntimeError("historical sync detail mismatch")
            previous=read(prior["path"])["readout"];now=read(row["path"])["readout"]
            base=store.data["payload_reads"][sid+"/BASELINE"];pb=old["payload_reads"][sid]
            store.data["historical_comparisons"][sid]=dict(status="COMPARED_POSTSEAL",
                sync_summary_equal=now["summary"]==previous["summary"],
                sync_candidate_rows_equal=now["candidate_rows"]==previous["candidate_rows"],
                baseline_decoded_equal=base.get("decoded_bits")==pb.get("decoded_bits") if base["status"]=="READ" else None,
                baseline_aggregate_equal=base.get("votes")==pb.get("votes") if base["status"]=="READ" else None,
                prior_sync_sha256=prior["sha256"],selection_or_retry=False)
    except Exception as exc:
        for row in store.data["historical_comparisons"].values():
            if row["status"]=="PENDING":row.update(status="REFERENCE_UNAVAILABLE",error=str(exc))
    store.save()

def run(output,*,cfg=None,framewise_type=FramewiseBackend,wan_type=WanBackend):
    cfg=load_config() if cfg is None else cfg;store=Store(output,cfg);sources={};fw=None;aborted=None
    try:
        for oid,spec in cfg["inputs"].items():
            try:
                sources[oid]=store.call("source_read",lambda s=spec:wan_type.read_source(s))
                store.data["observations"][oid]["status"]="VERIFIED";store.save()
            except Exception as exc:
                store.data["observations"][oid].update(status="FAILED",error=str(exc));store.failure(oid+"/SOURCE",exc)
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
        for kind in ("payload_posthoc","sync_posthoc","historical_comparisons","same_run_differences"):
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
