"""One fixed conditional joint entry; source/matched are separate child processes."""
from pathlib import Path
import argparse,copy,gzip,hashlib,importlib.metadata,json,os,signal,subprocess,sys
from main.tube_state import video_trajectory_conditional_joint_v1 as method
from main.tube_state import video_trajectory_receiver_origin_v1 as origin
from runtime.wan import video_trajectory_conditional_joint_v1 as runtime
from runtime.wan.conditional_joint.provenance import source_identity
from experiments.wan_state_clock import video_trajectory_conditional_joint_v1_prepare as prepare
ROOT=Path(__file__).resolve().parents[2]
ENTRY="experiments.wan_state_clock.video_trajectory_conditional_joint_v1_run"
CONFIG=ROOT/"experiments/wan_state_clock/configs/video_trajectory_conditional_joint_v1.json"
QUALITY=("PRE/P1_P0","PRE/M05_P1","PRE/M05_P0","POST/P1_P0","POST/M05_P1","POST/M05_P0")
dump=prepare.dump;digest=prepare.digest

def read(p):
    p=Path(p);b=p.read_bytes();return json.loads(gzip.decompress(b) if p.suffix==".gz" else b)
def keys(cfg):return dict(K0=cfg["key"],K1=cfg["wrong_key"])
def load_config(path=CONFIG):
    path=Path(path).resolve();cfg=read(path);fixed=read(CONFIG)
    if cfg!=fixed or cfg["fixed_denominator"]!=method.FIXED or cfg["physical_upper_bounds"]!=method.UPPER or cfg["views"]!=method.VIEWS:raise ValueError("fixed joint public config required")
    cfg["_config_path"]=str(path)
    for name in ("preparation","oracle","posthoc"):cfg[name+"_config"]=str(path.parent/cfg[name+"_config"])
    return cfg

def companion(cfg,name):
    p=Path(cfg[name+"_config"])
    if digest(p)!=cfg[name+"_config_sha256"]:raise ValueError(name+" config SHA mismatch")
    return read(p) # semantic parse; identity-only reads can precede the appropriate seal

def check_seal(row):
    if digest(row["path"])!=row["sha256"]:raise ValueError("sealed bytes changed")

class Store:
    def __init__(self,output,cfg):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False);self.cfg=cfg
        ident=source_identity(ROOT,"experiments/wan_state_clock/video_trajectory_conditional_joint_v1_run.py",[cfg["_config_path"]]+[cfg[n+"_config"] for n in ("preparation","oracle","posthoc")])
        env={}
        for k in cfg["environment_pins"]:
            try:env[k]=importlib.metadata.version(k)
            except importlib.metadata.PackageNotFoundError:env[k]=None
        self.data=dict(status="RUNNING",stage="INITIALIZED",**ident,config_sha256=digest(cfg["_config_path"]),fixed_denominator=cfg["fixed_denominator"],physical_upper_bounds=cfg["physical_upper_bounds"],environment=env,calls={},workers={},inputs={k:dict(v,status="PENDING",sha256=None) for k,v in cfg["inputs"].items()},sync_reads={v+"/"+k:dict(view_id=v,input_id=x["input_id"],protocol=x["protocol"],frames=method.INPUT_FRAMES[x["input_id"]],key_label=k,status="PENDING") for v,x in method.VIEWS.items() for k in method.KEY_LABELS},estimates={},payload_reads=method.logical_slots(),physical_reads={},alignment_receipts={},posthoc={},quality={k:dict(status="PENDING") for k in QUALITY},failures=[],actual_generation_calls=False,evidence_ceiling=cfg["evidence_ceiling"],receiver_scope="Conditional blind recovery within preregistered public protocol families; not arbitrary-attack routing or presence detection.")
        self.save()
    def save(self):dump(self.output/"result.json",self.data)
    def failure(self,stage,exc):self.data["failures"].append(dict(stage=stage,error=f"{type(exc).__name__}: {exc}"));self.save()
    def call(self,name,fn):
        row=self.data["calls"].setdefault(name,dict(attempted=0,completed=0));row["attempted"]+=1;self.save();value=fn();row["completed"]+=1;self.save();return value

def handle_termination(signum,frame):
    # Notebook SIGTERM reaches this parent; unwind and reap isolated worker groups.
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    raise KeyboardInterrupt("process-group termination requested")

def worker(store,phase):
    path=store.output/(phase+".worker.log");cmd=[sys.executable,"-B","-m",ENTRY,"--config",store.cfg["_config_path"],"--output",str(store.output/"source_preparation"),"--worker",phase]
    row=dict(status="RUNNING",command=cmd,log_path=str(path));store.data["workers"][phase]=row;store.save();child=None
    try:
        with path.open("wb") as f:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            row["pid"]=child.pid;store.save();code=child.wait()
        row.update(returncode=code,status="COMPLETE" if code==0 else "FAILED",log_sha256=digest(path));store.save()
        if code:raise RuntimeError(phase+" worker exited "+str(code))
    except BaseException as exc:
        if child is not None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:pass
            finally:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait() # reaped before Notebook's five-second parent-group grace expires
        row.update(status="FAILED",error=f"{type(exc).__name__}: {exc}",log_sha256=digest(path) if path.exists() else None);store.save();raise

def prepare_source(store,worker_fn=worker):
    companion(store.cfg,"preparation") # writer/config and fixture indices legal only in preparation
    try:
        for phase in ("source","matched"):worker_fn(store,phase)
        p=store.output/"source_preparation/source_preparation.json";x=read(p)
        if x["status"]!="COMPLETE":raise ValueError("source preparation incomplete")
        store.data["source_preparation_receipt"]=dict(path=str(p),sha256=digest(p));store.data["actual_generation_calls"]=True;store.save();return x
    except BaseException as exc:
        p=store.output/"source_preparation/source_preparation.json"
        if p.exists():
            x=read(p)
            if x.get("status")!="COMPLETE":x.update(status="INCOMPLETE",parent_failure=f"{type(exc).__name__}: {exc}");dump(p,x)
            store.data["source_preparation_receipt"]=dict(path=str(p),sha256=digest(p));store.data["actual_generation_calls"]=bool(x.get("source_calls",{}).get("trajectory",{}).get("attempted"));store.save()
        raise

def prepare_inputs(store,prepared,input_type):
    config=companion(store.cfg,"preparation");maps=config["input_maps"]
    if set(maps)!=set(method.INPUT_FRAMES):raise ValueError("fixed physical input roster")
    full=store.call("source_read",lambda:input_type.read(prepared["received_sources"]["M05"]));sources={}
    for oid,n in method.INPUT_FRAMES.items():
        try:
            indices=maps[oid]
            if len(indices)!=n:raise ValueError("received public length mismatch")
            rgb=store.call("source_construct",lambda:input_type.construct(full,indices));identity=input_type.receipt(rgb)
            sources[oid]=rgb;store.data["inputs"][oid].update(status="READ",**identity)
        except Exception as exc:store.data["inputs"][oid].update(status="FAILED",error=str(exc));store.failure(oid,exc)
        store.save()
    # Source-frame maps live in preparation receipt, never in selector input metadata.
    store.data["input_preparation_receipt"]=dump(store.output/"input_preparation.json",dict(source=prepared["received_sources"]["M05"],source_frame_maps=maps,physical_inputs=store.data["inputs"],aliases={"path_00":"global_03"}));store.save();return sources

def score_inputs(store,sources,framewise_type):
    model=None
    try:
        if not sources:return
        model=store.call("framewise_vae_load",lambda:framewise_type(store.cfg))
        for oid,rgb in sources.items():
            try:z=store.call("framewise_receiver_encode",lambda:model.encode(rgb))
            except Exception as exc:store.failure(oid+"/FW",exc);continue
            for sid,row in store.data["sync_reads"].items():
                if row["input_id"]!=oid:continue
                try:
                    row["status"]="RUNNING";store.save()
                    raw=store.call("sync_score",lambda:model.score(z,keys(store.cfg)[row["key_label"]],row["protocol"]))
                    row.update(status="SCORED",readout=dump(store.output/"sync"/(sid+".json.gz"),raw))
                    store.data["estimates"][sid]=method.estimates(raw,row["protocol"],row["frames"],keys(store.cfg)[row["key_label"]])
                except Exception as exc:row.update(status="FAILED",error=str(exc));store.failure(sid+"/SCORE",exc)
                store.save()
            z=None
    finally:
        if model is not None:model.close()

def seal_sync(store):
    for sid,row in store.data["sync_reads"].items():
        if row["status"]!="SCORED":
            row.update(status="FAILED",error=row.get("error","NOT_COMPLETED"))
            missing=(method.align.missing_sync(row["frames"],keys(store.cfg)[row["key_label"]],row["error"]) if row["protocol"]=="GLOBAL" else method.deletion.missing(keys(store.cfg)[row["key_label"]],row["error"]))
            row["readout"]=dump(store.output/"sync"/(sid+".json.gz"),missing)
        store.data["estimates"].setdefault(sid,{})
    store.data["blind_sync_seal"]=dump(store.output/"blind_sync.json",dict(sync_reads=store.data["sync_reads"],estimates=store.data["estimates"],truth_inputs=False));store.save()

def install_plan(store,plan,modes):
    for lid,row in plan["logical_slots"].items():
        if store.data["payload_reads"][lid]["mode"] in modes:store.data["payload_reads"][lid].update(row)
    for rid,row in plan["reads"].items():
        if rid not in store.data["physical_reads"]:store.data["physical_reads"][rid]=dict(row,status="PENDING")
        else:store.data["physical_reads"][rid]["logical_slots"]=list(dict.fromkeys(store.data["physical_reads"][rid]["logical_slots"]+row["logical_slots"]))
    store.save()

def make_blind_plan(store):
    check_seal(store.data["blind_sync_seal"]);plan=method.empty_plan()
    for lid,row in store.data["payload_reads"].items():
        if row["oracle"]:continue
        op=method.blind_operation(row,store.data["estimates"][row["view_id"]+"/"+row["key_label"]])
        method.add_slot(plan,lid,row,op,store.data["inputs"][row["input_id"]],keys(store.cfg)[row["key_label"]])
    if len(plan["encodes"])>28 or len(plan["reads"])>36:raise ValueError("blind physical budget")
    store.data["blind_plan_seal"]=dump(store.output/"blind_plan.json",plan);install_plan(store,plan,{"BASELINE","EST_ALIGN","RAW","GLOBAL_ALIGN","PATH_ALIGN"});return plan

def make_oracle_plan(store,blind):
    check_seal(store.data["blind_payload_seal"]);check_seal(store.data["blind_sync_seal"])
    oracle=companion(store.cfg,"oracle");plan=copy.deepcopy(blind);plan["truth_inputs"]=True
    for lid,row in store.data["payload_reads"].items():
        if row["oracle"]:method.add_slot(plan,lid,row,method.deletion.correction(oracle["paths"][row["view_id"]]),store.data["inputs"][row["input_id"]],keys(store.cfg)[row["key_label"]])
    if len(plan["encodes"])>30 or len(plan["reads"])>40:raise ValueError("total physical budget")
    store.data["oracle_plan_seal"]=dump(store.output/"oracle_plan.json",plan);install_plan(store,plan,{"TRUTH_PATH"});return plan

def run_plan(store,sources,model,cache,plan,stage):
    for gid,item in plan["encodes"].items():
        pending=[(rid,row) for rid,row in store.data["physical_reads"].items() if row["encode_id"]==gid and row["status"]=="PENDING"]
        if not pending:continue
        receipt=store.data["alignment_receipts"].get(gid);z=rgb=None
        try:
            if receipt is None:
                if item["input_id"] not in sources or model is None:raise ValueError("received input/backend unavailable")
                rgb=model.operate(sources[item["input_id"]],item["received_index_map"])
                receipt=dict(item,status="ENCODING",output_identity=model.receipt(rgb),first_stage=stage);store.data["alignment_receipts"][gid]=receipt;store.save()
                z=store.call(stage+"/wan_receiver_encode",lambda:model.encode(rgb));cache[gid]=model.cache(z);receipt.update(status="ENCODED",normalized_shape=list(z.shape));store.save()
            elif receipt["status"]!="ENCODED":raise ValueError("cached encode failure; no retry")
            if gid not in cache:raise ValueError("cached latent unavailable; no retry")
            z=model.restore(cache[gid])
            for rid,row in pending:
                try:
                    row["status"]="RUNNING";store.save()
                    detail=store.call(stage+"/payload_read",lambda:model.read(z,keys(store.cfg)[row["key_label"]],row["frames"]))
                    row.update(status="READ",detail=dump(store.output/"payload"/(rid+".votes.json.gz"),detail),first_stage=stage);store.save()
                except Exception as exc:row.update(status="FAILED",error=str(exc));store.failure(rid+"/READ",exc)
        except Exception as exc:
            if receipt is None:store.data["alignment_receipts"][gid]=dict(item,status="FAILED",error=str(exc),first_stage=stage)
            elif receipt["status"]=="ENCODING":receipt.update(status="FAILED",error=str(exc))
            for _,row in pending:
                if row["status"]=="PENDING":row.update(status="FAILED",error=str(exc))
            store.failure(gid+"/ENCODE",exc)
        finally:z=rgb=None

def seal_payload(store,oracle):
    for row in store.data["alignment_receipts"].values():
        if row["status"]=="ENCODING":row.update(status="FAILED",error="INTERRUPTED")
    chosen={lid:row for lid,row in store.data["payload_reads"].items() if row["oracle"]==oracle}
    for lid,row in chosen.items():
        physical=store.data["physical_reads"].get(row.get("physical_read"))
        if physical is not None:
            if physical["status"]!="READ":
                physical.update(status="FAILED",error=physical.get("error","NOT_COMPLETED"))
                if "detail" not in physical:physical["detail"]=dump(store.output/"payload"/(row["physical_read"]+".votes.json.gz"),method.missing_final(row["frames"],physical["error"]))
            for k in ("status","detail","error"):
                if k in physical:row[k]=physical[k]
        else:
            row.update(status="UNRESOLVED" if row["status"]=="UNRESOLVED" else "FAILED",error=row.get("error","NOT_COMPLETED"))
            row["detail"]=dump(store.output/"missing"/(lid+".votes.json.gz"),method.missing_final(row["frames"],row["error"]))
    name="oracle_payload_seal" if oracle else "blind_payload_seal"
    store.data[name]=dump(store.output/(name+".json"),dict(logical_reads=chosen,physical_reads=store.data["physical_reads"],alignment_receipts=store.data["alignment_receipts"],oracle=oracle,truth_inputs=oracle));store.save()

def quality_rows(store,prepared,input_type):
    native=prepared["source_protocol"];pre=prepared["rasters"];post=prepared["received_sources"]
    for space,pairs in (("PRE",(("P1_P0",pre["P1"],native),("M05_P1",pre["M05"],pre["P1"]),("M05_P0",pre["M05"],native))),("POST",(("P1_P0",post["P1"],native),("M05_P1",post["M05"],post["P1"]),("M05_P0",post["M05"],native)))):
        for label,candidate,reference in pairs:
            qid=space+"/"+label
            try:
                c=input_type.read(candidate);b=input_type.read(reference)
                metrics=store.call("quality_compute",lambda:runtime.quality.rgb_quality_metrics(b.float().div(255),c.float().div(255)))
                if label=="M05_P1":metrics["residual"]=runtime.quality.residual_temporal_metrics(b,c,source_start=0)
                store.data["quality"][qid]=dict(status="MEASURED",candidate=candidate,reference=reference,**metrics)
            except Exception as exc:store.data["quality"][qid]=dict(status="FAILED",error=str(exc));store.failure(qid,exc)
            store.save()

def posthoc(store):
    check_seal(store.data["oracle_payload_seal"]);check_seal(store.data["blind_payload_seal"])
    cfg=companion(store.cfg,"posthoc");expected=method.payload.message_bits(cfg["message"])
    if len(expected)!=32:raise ValueError("posthoc exactly32 bits required")
    for lid,row in store.data["payload_reads"].items():
        detail=read(row["detail"]["path"])
        if digest(row["detail"]["path"])!=row["detail"]["sha256"]:raise ValueError("payload detail changed")
        if detail["status"]=="READ" and (len(detail["bit_rows"])!=32 or len(detail["time_bit_rows"])!=32*row["R"]):raise ValueError("fixed payload support")
        joined=origin.join_truth(detail,expected);truth=cfg["true_paths"][row["view_id"]];e=store.data["estimates"][row["view_id"]+"/"+row["key_label"]]
        estimate=e.get("global" if row["protocol"]=="GLOBAL" else "global_" if row["mode"]=="GLOBAL_ALIGN" else "joint",{})
        path=(dict(family="H0",b=estimate["offset"],k=None) if row["protocol"]=="GLOBAL" and estimate.get("status")=="ESTIMATED" else estimate.get("path"))
        if row["mode"] in ("RAW","BASELINE"):path=None
        if row["oracle"]:path=truth
        true_map=cfg["input_maps"][row["input_id"]]
        inferred=([j+path["b"] for j in range(row["frames"])] if path and path["family"]=="H0" else method.deletion.path_map(path) if path else None)
        path_correct=(path==truth and inferred==true_map) if path else None
        summary=dict(status=joined["status"],key_role="CORRECT_KEY" if row["key_label"]=="K0" else "WRONG_KEY",bit_errors=joined.get("bit_errors"),error_bits=joined.get("error_bits"),truth_path=truth,estimated_path=path,absolute_path_correct=path_correct,phase_correct=(path["b"]%4==truth["b"]%4) if path else None,source_map_correct=sum(a==b for a,b in zip(inferred,true_map)) if inferred else None,source_map_denominator=row["frames"],geometry_only=row["frames"]==181,oracle=row["oracle"],alias_of=row.get("alias_of"))
        summary.update(family_correct=path["family"]==truth["family"] if path else None,b_correct=path["b"]==truth["b"] if path else None,
            k_signed_error=path["k"]-truth["k"] if path and path["family"]==truth["family"]=="H1" else None)
        summary["k_absolute_error"]=abs(summary["k_signed_error"]) if summary["k_signed_error"] is not None else None
        if row.get("operation"):
            summary["output_source_map"]=[true_map[j] for j in row["operation"]["received_index_map"]]
        if joined["status"]=="EVALUATED_TRUTH":
            summary["channel_margins"]=[dict(payload_channel=c,negative=sum(x["signed_normalized_margin"]<0 for x in joined["time_bit_rows"] if x["payload_channel"]==c),zero=sum(x["signed_normalized_margin"]==0 for x in joined["time_bit_rows"] if x["payload_channel"]==c),positive=sum(x["signed_normalized_margin"]>0 for x in joined["time_bit_rows"] if x["payload_channel"]==c),errors=sum(x["bit_error"] for x in joined["time_bit_rows"] if x["payload_channel"]==c)) for c in range(4)]
            for name,rows in (("final",joined["bit_rows"]),("time",joined["time_bit_rows"])):
                v=[x["signed_normalized_margin"] for x in rows];summary[name+"_margins"]=dict(min=min(v),negative=sum(x<0 for x in v),zero=sum(x==0 for x in v),positive=sum(x>0 for x in v),errors=sum(x["bit_error"] for x in rows))
        primary=row["key_label"]=="K0" and row["frames"]!=181 and row["mode"] in ("EST_ALIGN","PATH_ALIGN")
        summary["primary"]=primary
        if primary:
            summary["conditional_interpretation"]="INSUFFICIENT_UNIQUE_PATH" if estimate.get("reason")=="TIED_TOP" else "VALID_COUNTEREXAMPLE" if path is not None and not path_correct else "TECHNICAL_UNRESOLVED" if joined["status"]!="EVALUATED_TRUTH" or path is None else "VALID_COUNTEREXAMPLE" if joined["bit_errors"] else "FIXED_CASE_SUPPORTED"
        summary["detail"]=dump(store.output/"posthoc"/(lid+".json.gz"),joined);store.data["posthoc"][lid]=summary;store.save()

def differences(store):
    # Reporting only: comparison at nominal receiver indices, not matched receptive fields.
    out={}
    for lid,row in store.data["payload_reads"].items():
        baseline="BASELINE" if row["protocol"]=="GLOBAL" else "RAW"
        if row["mode"]==baseline:continue
        reference=row["view_id"]+"/"+row["key_label"]+"/"+baseline
        a,b=store.data["posthoc"].get(reference,{}),store.data["posthoc"].get(lid,{})
        result=dict(reference=reference,candidate=lid,status="FAILED",receiver_time_comparison="same nominal index, not matched source receptive fields")
        if a.get("status")==b.get("status")=="EVALUATED_TRUTH":
            x,y=read(a["detail"]["path"]),read(b["detail"]["path"])
            result.update(status="DESCRIPTIVE_POSTSEAL",final_bit_error_delta=b["bit_errors"]-a["bit_errors"])
            for name,field in (("final","bit_rows"),("time","time_bit_rows")):
                values=[v["signed_normalized_margin"]-u["signed_normalized_margin"] for u,v in zip(x[field],y[field])]
                result[name+"_margin_delta"]=dict(count=len(values),negative=sum(v<0 for v in values),zero=sum(v==0 for v in values),positive=sum(v>0 for v in values),min=min(values),max=max(values))
        else:result["error"]="reference or candidate payload/posthoc unavailable"
        out[lid]=result
    store.data["mode_differences"]=out;store.save()

def finish(store):
    for lid,row in store.data["payload_reads"].items():
        if lid not in store.data["posthoc"]:
            detail=method.missing_final(row["frames"],"POSTHOC_NOT_COMPLETED");store.data["posthoc"][lid]=dict(status="FAILED",key_role="CORRECT_KEY" if row["key_label"]=="K0" else "WRONG_KEY",detail=dump(store.output/"posthoc"/(lid+".json.gz"),detail))
    for row in store.data["quality"].values():
        if row["status"]=="PENDING":row.update(status="FAILED",error="NOT_COMPLETED")
    for row in store.data["inputs"].values():
        if row["status"]=="PENDING":row.update(status="FAILED",error="NOT_COMPLETED")
    store.data["counts"]=dict(logical_payload_reads=len(store.data["payload_reads"]),logical_votes=sum(x["planned_votes"] for x in store.data["payload_reads"].values()),logical_time_bit_rows=sum(x["planned_time_bits"] for x in store.data["payload_reads"].values()),logical_final_bits=sum(x["planned_final_bits"] for x in store.data["payload_reads"].values()),physical_reads=len(store.data["physical_reads"]),physical_planned_votes=sum(32*30*method.align.support(x["frames"]) for x in store.data["physical_reads"].values()),physical_planned_time_bit_rows=sum(32*method.align.support(x["frames"]) for x in store.data["physical_reads"].values()),physical_planned_final_bits=32*len(store.data["physical_reads"]),planned_encodes=len(store.data["alignment_receipts"]),read_completed=sum(x["status"]=="READ" for x in store.data["physical_reads"].values()))
    store.data["status"]="COMPLETE" if not store.data["failures"] and all(x["status"]=="READ" for x in store.data["payload_reads"].values()) and all(x["status"]=="EVALUATED_TRUTH" for x in store.data["posthoc"].values()) and all(x["status"]=="MEASURED" for x in store.data["quality"].values()) else "RETAINED_INCOMPLETE"
    store.data["stage"]="FINISHED";store.save()

def run(output,*,cfg=None,source_fn=prepare_source,input_type=runtime.Inputs,framewise_type=runtime.FramewiseBackend,wan_type=runtime.WanBackend,quality_fn=quality_rows):
    cfg=load_config() if cfg is None else cfg;store=Store(output,cfg);sources={};prepared={};model=None;cache={};abort=None;blind=method.empty_plan()
    try:
        store.data["stage"]="PREPARATION";store.save()
        prepared=source_fn(store);sources=prepare_inputs(store,prepared,input_type)
        store.data["stage"]="BLIND_SYNC";store.save();score_inputs(store,sources,framewise_type)
    except BaseException as exc:
        store.failure(store.data["stage"],exc)
        if not isinstance(exc,Exception):abort=exc
    finally:seal_sync(store)
    try:
        blind=make_blind_plan(store)
        if sources and abort is None:model=store.call("wan_vae_load",lambda:wan_type(cfg))
        if abort is None:run_plan(store,sources,model,cache,blind,"blind")
    except BaseException as exc:
        store.failure("BLIND_PAYLOAD",exc)
        if not isinstance(exc,Exception):abort=exc
    finally:seal_payload(store,False)
    try:
        if abort is None:
            oracle=make_oracle_plan(store,blind);run_plan(store,sources,model,cache,oracle,"oracle")
    except BaseException as exc:
        store.failure("ORACLE",exc)
        if not isinstance(exc,Exception):abort=exc
    finally:
        seal_payload(store,True)
        if model is not None:
            try:model.close()
            except BaseException as exc:
                store.failure("WAN_RELEASE",exc)
                if not isinstance(exc,Exception):abort=exc
        model=None;cache.clear()
    try:
        if abort is None:posthoc(store)
    except BaseException as exc:
        store.failure("POSTHOC",exc)
        if not isinstance(exc,Exception):abort=exc
    try:
        if abort is None:quality_fn(store,prepared,input_type)
    except BaseException as exc:
        store.failure("QUALITY",exc)
        if not isinstance(exc,Exception):abort=exc
    finish(store)
    differences(store)
    if abort is not None:raise abort
    return store.data

def main():
    signal.signal(signal.SIGTERM,handle_termination)
    p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);p.add_argument("--config",type=Path,default=CONFIG);p.add_argument("--worker",choices=("source","matched"));a=p.parse_args();cfg=load_config(a.config)
    if a.worker:prepare.run_phase(a.output,a.worker,companion(cfg,"preparation"))
    else:run(a.output,cfg=cfg)
if __name__=="__main__":main()
