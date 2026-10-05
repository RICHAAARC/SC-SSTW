"""Fixed four-observation saved-RGB receiver diagnostic; no generation or codec."""
from pathlib import Path
import argparse,copy,gzip,hashlib,importlib.metadata,json,os,subprocess,sys
from main.tube_state import grow_video_reference as payload
from main.tube_state import video_trajectory_receiver_origin_v1 as method
from runtime.wan.video_trajectory_receiver_origin_v1 import ReceiverBackend
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"experiments/wan_state_clock/configs/video_trajectory_receiver_origin_v1.json"

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(data,indent=2,allow_nan=False)+"\n").encode()
    if path.suffix==".gz":raw=gzip.compress(raw,mtime=0)
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_bytes(raw);os.replace(tmp,path)
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
def load_config():return json.loads(CONFIG.read_text())
def environment(cfg):
    out={}
    for name in cfg["environment_pins"]:
        try:out[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:out[name]=None
    return out
def roster(cfg):
    return [(f"obs_{i:02d}",condition,start) for i,(condition,start) in
            enumerate((c,s) for c in cfg["inputs"] for s in (0,1))]
class Store:
    def __init__(self,output,cfg):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.cfg=cfg
        try:
            source=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
            clean=not subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain"],text=True)
        except Exception:source=None;clean=False
        self.data=dict(status="RUNNING",stage="INITIALIZED",source_sha=source,source_clean=clean,
            source_files={p:digest(ROOT/p) for p in cfg["source_files"]},
            config_sha256=digest(CONFIG),fixed_denominator=cfg["fixed_denominator"],
            planned_calls=cfg["planned_calls"],calls={name:dict(attempted=0,completed=0) for name in cfg["planned_calls"]},
            environment=environment(cfg),model=cfg["model"],source_inputs={c:dict(status="PENDING",**s) for c,s in cfg["inputs"].items()},
            observations={},payload_reads={},payload_posthoc={},historical_comparisons={},failures=[],
            counts=dict(observations=0,payload_reads=0,detailed_votes=0,time_bit_rows=0,final_bit_rows=0),
            evidence_ceiling=cfg["evidence_ceiling"])
        for oid,c,start in roster(cfg):
            self.data["observations"][oid]=dict(status="PENDING",received_frames=177)
            for key in ("K0","K1"):
                sid=oid+"/"+key
                self.data["payload_reads"][sid]=dict(status="PENDING",observation_id=oid,key_label=key,
                    detail_path=str(self.output/"blind"/oid/(key+".votes.json.gz")),
                    planned_detailed_votes=42240,planned_time_bit_rows=1408,planned_final_bit_rows=32)
                self.data["payload_posthoc"][sid]=dict(status="PENDING",planned_final_bit_rows=32)
                if start==1:self.data["historical_comparisons"][sid]=dict(status="PENDING")
        self.save()
    def save(self):dump(self.output/"result.json",self.data)
    def failure(self,stage,exc):
        self.data["failures"].append(dict(stage=stage,error=f"{type(exc).__name__}: {exc}"));self.save()
    def call(self,name,fn):
        row=self.data["calls"][name];row["attempted"]+=1;self.save()
        value=fn();row["completed"]+=1;self.save();return value

def seal(store):
    # Every failed read gets an explicit artifact with all 1408 time-bit failure slots.
    for sid,row in store.data["payload_reads"].items():
        if row["status"] in ("PENDING","RUNNING"):
            row.update(status="FAILED",error="NOT_COMPLETED; fixed slot retained")
        if row["status"]!="READ":
            receipt=dump(row["detail_path"],method.missing_detail(row.get("error","read failed")))
            row.update(detail_sha256=receipt["sha256"],detail_bytes=receipt["bytes"])
    for row in store.data["observations"].values():
        if row["status"] in ("PENDING","RUNNING","ENCODING"):row.update(status="FAILED",error="NOT_COMPLETED")
    for row in store.data["source_inputs"].values():
        if row["status"]=="PENDING":row.update(status="FAILED",error="NOT_COMPLETED")
    blind=dict(observations=store.data["observations"],payload_reads=store.data["payload_reads"],
               truth_inputs=False,receiver_inputs="received uint8 clip, key, fixed R44 protocol only")
    receipt=dump(store.output/"blind_receiver_readouts.json",blind)
    store.data["blind_receiver_sha256"]=receipt["sha256"]
    store.data["stage"]="BLIND_SEALED";store.save()

def read_detail(path):return json.loads(gzip.decompress(Path(path).read_bytes()))
def join_posthoc(store):
    if digest(store.output/"blind_receiver_readouts.json")!=store.data["blind_receiver_sha256"]:
        raise RuntimeError("blind seal mismatch")
    cfg=store.cfg;expected=payload.message_bits(cfg["message"])
    for oid,condition,start in roster(cfg):
        for key in ("K0","K1"):
            sid=oid+"/"+key;row=store.data["payload_reads"][sid]
            detail=read_detail(row["detail_path"])
            if digest(row["detail_path"])!=row["detail_sha256"]:raise RuntimeError("blind detail mismatch")
            joined=method.join_truth(detail,expected)
            joined.update(condition=condition,source_start_posthoc_only=start,
                          key_role="REGISTERED" if key=="K0" else "WRONG_KEY",
                          blind_detail_sha256=row["detail_sha256"])
            receipt=dump(store.output/"posthoc"/oid/(key+".json.gz"),joined)
            store.data["payload_posthoc"][sid]={k:v for k,v in joined.items() if k!="time_bit_rows"}
            store.data["payload_posthoc"][sid].update(time_bit_path=receipt["path"],sha256=receipt["sha256"],
                time_bit_rows=len(joined["time_bit_rows"]))
            store.save()
    # Historical payload is first read here, after the current blind evidence is sealed.
    try:
        ref=cfg["historical_result"]
        if digest(ref["path"])!=ref["sha256"]:raise RuntimeError("historical result SHA mismatch")
        old=json.loads(Path(ref["path"]).read_text())
        for oid,condition,start in roster(cfg):
            if start!=1:continue
            for key in ("K0","K1"):
                sid=oid+"/"+key;new=store.data["payload_reads"][sid]
                prior=old["payload_reads"][cfg["inputs"][condition]["historical_observation_id"]+"/"+key]
                if new["status"]!="READ":
                    store.data["historical_comparisons"][sid]=dict(status="CURRENT_READ_FAILED")
                else:
                    store.data["historical_comparisons"][sid]=dict(status="COMPARED_POSTSEAL",
                        decoded_equal=new["decoded_bits"]==prior["decoded_bits"],
                        votes_equal=new["votes"]==prior["votes"],
                        prior_decoded_bits=prior["decoded_bits"],prior_votes=prior["votes"],
                        vote_delta=[dict(bit_index=i,ones=new["votes"][i]["ones"]-v["ones"],zeros=new["votes"][i]["zeros"]-v["zeros"])
                                    for i,v in enumerate(prior["votes"])],
                        prior_result_sha256=ref["sha256"],selection_or_retry=False)
    except Exception as exc:
        for row in store.data["historical_comparisons"].values():
            if row["status"]=="PENDING":row.update(status="REFERENCE_UNAVAILABLE",error=f"{type(exc).__name__}: {exc}")
        store.data["historical_reference_error"]=f"{type(exc).__name__}: {exc}"
    store.save()

def run(output,*,cfg=None,backend_type=ReceiverBackend):
    cfg=load_config() if cfg is None else cfg
    store=Store(output,cfg);backend=None;sources={};aborted=None
    try:
        for condition,spec in cfg["inputs"].items():
            try:
                sources[condition]=store.call("source_read",lambda s=spec:backend_type.read_source(s))
                store.data["source_inputs"][condition]["status"]="VERIFIED";store.save()
            except Exception as exc:
                store.data["source_inputs"][condition].update(status="FAILED",error=str(exc));store.failure(condition+"/SOURCE",exc)
        if sources:
            backend=store.call("wan_vae_load",lambda:backend_type(cfg["model"]))
        for oid,condition,start in roster(cfg):
            obs=store.data["observations"][oid]
            if condition not in sources:
                obs.update(status="FAILED",error="fixed source unavailable");store.save();continue
            normalized=None;received=None
            try:
                received=backend_type.slice_source(sources[condition],start)
                identity=backend_type.pixel_receipt(received)
                if start==1 and identity["sha256"]!=cfg["inputs"][condition]["start1_sha256"]:
                    raise RuntimeError("start1 slice does not match original saved crop SHA")
                obs.update(status="ENCODING",**identity);store.save()
                # No condition, crop start, history or truth enters the backend.
                normalized=store.call("wan_receiver_encode",lambda:backend.encode(received))
                obs.update(status="ENCODED",normalized_shape=list(normalized.shape));store.save()
                for key_label,key in (("K0",cfg["key"]),("K1",cfg["wrong_key"])):
                    sid=oid+"/"+key_label;row=store.data["payload_reads"][sid]
                    try:
                        row["status"]="RUNNING";store.save()
                        detail=store.call("payload_read",lambda k=key:backend.read(normalized,k))
                        receipt=dump(row["detail_path"],detail)
                        reference=detail["original_readout"]
                        row.update(status="READ",detail_sha256=receipt["sha256"],detail_bytes=receipt["bytes"],
                            decoded_bits=reference["decoded_bits"],votes=reference["votes"],R=44,
                            original_reader_match=detail["original_reader_match"],truth_used=False,
                            detailed_votes=detail["detailed_vote_count"],time_bit_rows=detail["time_bit_count"])
                        store.save()
                    except Exception as exc:
                        row.update(status="FAILED",error=f"{type(exc).__name__}: {exc}");store.failure(sid+"/READ",exc)
            except Exception as exc:
                obs.update(status="FAILED",error=f"{type(exc).__name__}: {exc}")
                for key in ("K0","K1"):
                    row=store.data["payload_reads"][oid+"/"+key]
                    if row["status"]=="PENDING":row.update(status="FAILED",error=obs["error"])
                store.failure(oid+"/ENCODE",exc)
            finally:
                normalized=None;received=None
    except BaseException as exc:
        aborted=exc;store.failure("RECEIVER_INTERRUPTED",exc)
    finally:
        sources.clear()
        if backend is not None:
            try:backend.close()
            except BaseException as exc:store.failure("MODEL_RELEASE",exc)
        seal(store)
    try:join_posthoc(store)
    except BaseException as exc:
        store.failure("POSTHOC",exc)
        for row in store.data["historical_comparisons"].values():
            if row["status"]=="PENDING":row.update(status="NOT_EVALUATED",error="posthoc failed: "+str(exc))
        for row in store.data["payload_posthoc"].values():
            if row["status"]=="PENDING":row.update(status="FAILED",error=str(exc),
                bit_rows=[dict(bit_index=i,status="FAILED") for i in range(32)])
    reads=list(store.data["payload_reads"].values())
    store.data["counts"]=dict(observations=sum(x["status"]=="ENCODED" for x in store.data["observations"].values()),
        payload_reads=sum(x["status"]=="READ" for x in reads),
        detailed_votes=sum(x.get("detailed_votes",0) for x in reads if x["status"]=="READ"),
        time_bit_rows=sum(x.get("time_bit_rows",0) for x in reads if x["status"]=="READ"),
        final_bit_rows=sum(32 for x in reads if x["status"]=="READ"))
    store.data["call_integrity"]={k:dict(expected=n,actual=store.data["calls"][k],
        match=store.data["calls"][k]==dict(attempted=n,completed=n)) for k,n in cfg["planned_calls"].items()}
    store.data["status"]="COMPLETE" if all(x["status"]=="READ" for x in reads) and not store.data["failures"] else "INCOMPLETE"
    store.data["stage"]="FINISHED";store.save()
    if aborted is not None:raise aborted
    return store.data

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    result=run(args.output)
    print(json.dumps(dict(status=result["status"],counts=result["counts"],result=str(args.output/"result.json"))))
    return 0 if result["status"]=="COMPLETE" else 1
if __name__=="__main__":sys.exit(main())

