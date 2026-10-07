"""Fixed two-family calibration -> one confirmation source. No receiver payload."""
from pathlib import Path
import argparse,copy,gzip,hashlib,importlib.metadata,json,subprocess,sys,time
from main.tube_state import video_terminal_sync_path_decision_v1 as method
from runtime.wan.video_terminal_sync_path_decision_v1 import FramewiseBackend,InputBackend
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"experiments/wan_state_clock/configs/video_terminal_sync_path_decision_v1.json"
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p=Path(p);raw=p.read_bytes();return json.loads(gzip.decompress(raw) if p.suffix==".gz" else raw)
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);raw=(json.dumps(x,indent=2,allow_nan=False)+"\n").encode()
    if p.suffix==".gz":raw=gzip.compress(raw,mtime=0)
    tmp=p.with_suffix(p.suffix+".tmp");tmp.write_bytes(raw);tmp.replace(p)
    return dict(path=str(p),sha256=digest(p),bytes=len(raw))
def check_seal(receipt):
    if digest(receipt["path"])!=receipt["sha256"]:raise ValueError("seal identity")
def companion(cfg,suffix):
    p=CONFIG.with_name(CONFIG.stem+"_"+suffix+".json")
    if digest(p)!=cfg[suffix+"_sha256"]:raise ValueError("companion identity: "+suffix)
    return read(p)
def keys(cfg):return (("K0",cfg["key"]),("K1",cfg["wrong_key"]))
def validate_config(cfg):
    if cfg["name"]!="video_terminal_sync_path_decision_v1":raise ValueError("fixed entrypoint")
    for kind in ("development_inputs","confirmation_inputs"):
        if list(cfg[kind])!=[('cal_' if kind.startswith('development') else 'obs_')+f'{i:02}' for i in range(1,9)]:raise ValueError("opaque fixed observation roster")
        for i,x in enumerate(cfg[kind].values()):
            n=(177,177,89,89)[i%4]
            if x["frames"]!=n or x["shape"]!=[n,320,512,3] or x["bytes"]!=n*320*512*3:raise ValueError("fixed geometry")
    if list(cfg["cached_development"])!=[f'cal_{j:02}/{key}' for j in (9,10) for key in ("K0","K1")]:raise ValueError("cached fixed roster")
    if cfg["fixed_denominator"]["confirmation_decisions"]!=16 or cfg["fixed_denominator"]["payload_reads"]!=0:raise ValueError("fixed denominator")
class Store:
    def __init__(self,output,cfg):
        validate_config(cfg);self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False);self.cfg=cfg
        env={}
        for k in {**cfg["environment_pins"],**cfg["generation_dependency_pins"]}:
            try:env[k]=importlib.metadata.version(k)
            except importlib.metadata.PackageNotFoundError:env[k]=None
        self.data=dict(status="RUNNING",stage="INITIALIZED",source_sha=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),source_clean=not subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain"],text=True),source_files={p:digest(ROOT/p) for p in cfg["source_files"]},config_sha256=digest(CONFIG),fixed_denominator=cfg["fixed_denominator"],environment=env,calls={},development={},confirmation={},decisions={},posthoc={},failures=[],evidence_ceiling=cfg["evidence_ceiling"],thresholds=dict(status="UNAVAILABLE",families={}))
        for stage,inputs in (("development",cfg["development_inputs"]),("confirmation",cfg["confirmation_inputs"])):
            for oid,spec in inputs.items():
                for key,_ in keys(cfg):self.data[stage][oid+"/"+key]=dict(status="PENDING",execution_status="NOT_RUN",frames=spec["frames"])
        for sid,spec in cfg["cached_development"].items():self.data["development"][sid]=dict(status="PENDING",execution_status="NOT_RUN",frames=spec["frames"],reuse=True)
        for sid,row in self.data["confirmation"].items():
            self.data["decisions"][sid]=dict(state="UNCERTAIN",reason="NOT_RUN",frames=row["frames"],path=None,received_index_map=None,truth_inputs=False)
            self.data["posthoc"][sid]=dict(status="PENDING")
        for stage in ("development","confirmation"):
            for name in ("source_read","source_slice","framewise_vae_load","framewise_receiver_encode","sync_score"):
                self.data["calls"][stage+"/"+name]=dict(attempted=0,completed=0)
        for name in ("receiver_wan_encode","payload_read"):self.data["calls"][name]=dict(attempted=0,completed=0)
        self.save()
    def save(self):dump(self.output/"result.json",self.data)
    def call(self,k,fn):
        row=self.data["calls"][k];row["attempted"]+=1;self.save();x=fn();row["completed"]+=1;self.save();return x
    def failure(self,stage,exc):self.data["failures"].append(dict(stage=stage,error=f"{type(exc).__name__}: {exc}"));self.save()
def prepare_inputs(store,stage,specs,backend):
    prep=companion(store.cfg,"preparation");inputs=store.cfg[stage+"_inputs"];result={};record={}
    for source,spec in specs.items():
        try:
            full=store.call(stage+"/source_read",lambda:backend.read_source(spec));receipt=backend.receipt(full)
            if receipt["sha256"]!=spec["sha256"] or receipt["shape"]!=[181,320,512,3] or receipt["bytes"]!=88965120:raise ValueError("FULL bytes/identity")
            for oid,pub in inputs.items():
                conf=pub.get("conf_observation",oid);item=prep["observations"][conf]
                if item["source"]!=source:continue
                try:
                    clip=store.call(stage+"/source_slice",lambda:backend.construct(full,item["source_frame_map"]));cr=backend.receipt(clip)
                    if cr["shape"]!=pub["shape"] or cr["bytes"]!=pub["bytes"]:raise ValueError("received geometry")
                    result[oid]=clip;record[oid]=dict(status="PREPARED",source=source,source_frame_map=item["source_frame_map"],received=cr)
                except Exception as exc:store.failure(stage+"/"+oid+"/SLICE",exc)
            full=None
        except Exception as exc:store.failure(stage+"/SOURCE/"+source,exc)
    for oid in inputs:record.setdefault(oid,dict(status="FAILED",reason="PREPARATION_NOT_COMPLETED"))
    store.data[stage+"_preparation"]=dump(store.output/(stage+"_preparation.json"),dict(receiver_must_not_consume=True,observations=record));store.save()
    return result

def score_stage(store,stage,sources,backend_type):
    backend=None;primary=None
    try:
        if sources:backend=store.call(stage+"/framewise_vae_load",lambda:backend_type(store.cfg["framewise_model"]))
        for oid,rgb in sources.items():
            z=None
            try:
                z,enc=store.call(stage+"/framewise_receiver_encode",lambda:backend.encode(rgb))
                for key,label in keys(store.cfg):
                    sid=oid+"/"+key;row=store.data[stage][sid];row["execution_status"]="ATTEMPTED";store.save()
                    try:
                        raw=store.call(stage+"/sync_score",lambda:backend.score(z,label))
                        saved=dump(store.output/stage/(sid+".json.gz"),dict(readout=raw,encode_receipt=enc,truth_inputs=False))
                        row.update(status="SAVED",execution_status="SCORED",**saved);store.save()
                    except Exception as exc:store.failure(stage+"/"+sid,exc);row.update(status="FAILED",error=str(exc))
            except Exception as exc:store.failure(stage+"/"+oid+"/ENCODE",exc)
            finally:z=None
    except BaseException as exc:primary=exc;raise
    finally:
        sources.clear()
        if backend is not None:
            try:backend.close()
            except BaseException as exc:
                store.failure(stage+"/RELEASE",exc)
                if primary is None:raise

def cached_read(spec):
    if digest(spec["path"])!=spec["sha256"]:raise ValueError("cached read SHA mismatch")
    return read(spec["path"])["readout"]
def install_cached(store,loader):
    for sid,spec in store.cfg["cached_development"].items():
        try:
            raw=loader(spec);saved=dump(store.output/"development"/(sid+".json.gz"),dict(readout=raw,truth_inputs=False))
            store.data["development"][sid].update(status="SAVED",execution_status="REUSED",**saved)
        except Exception as exc:store.failure("CACHE/"+sid,exc);store.data["development"][sid].update(status="FAILED",error=str(exc))
        store.save()
def seal_scores(store,stage):
    for sid,row in store.data[stage].items():
        if row["status"]!="SAVED":
            key=dict(keys(store.cfg))[sid.rsplit("/",1)[1]];n=row["frames"];reason=row.get("error","NOT_COMPLETED")
            raw=method.deletion.missing(key,reason) if n==177 else method.prior.missing_sync(n,key,reason)
            receipt=dump(store.output/stage/(sid+".json.gz"),dict(readout=raw,truth_inputs=False))
            row.update(status="FAILED",error=reason,**receipt)
        check_seal(row)
    store.data[stage+"_score_seal"]=dump(store.output/(stage+"_blind_scores.json"),dict(reads=store.data[stage],truth_inputs=False));store.save()
def analyses(store,stage):
    check_seal(store.data[stage+"_score_seal"])
    return {sid:method.analyze(read(row["path"])["readout"],row["frames"],dict(keys(store.cfg))[sid.rsplit("/",1)[1]]) for sid,row in store.data[stage].items()}
def load_calibration_labels(store):
    check_seal(store.data["development_score_seal"])
    labels=companion(store.cfg,"calibration_labels")
    nulls=labels["null_rosters"]
    if any(sid not in store.data["development"] for ids in nulls.values() for sid in ids):raise ValueError("null roster outside sealed reads")
    return labels

def launch_worker(store,phase):
    command=[sys.executable,"-u","-m","experiments.wan_state_clock.video_terminal_sync_path_decision_v1_prepare","--output",str(store.output/"source_preparation"),"--phase",phase]
    child=None;code=None;primary=None;start=time.perf_counter()
    store.data.setdefault("workers",{})[phase]=dict(status="RUNNING",command=command);store.save()
    try:
        with (store.output/(phase+".log")).open("w") as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:print(line,end="",flush=True);log.write(line);log.flush()
            code=child.wait()
        if code!=0:raise RuntimeError(f"{phase} worker exit {code}")
    except BaseException as exc:
        primary=exc
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        raise
    finally:
        if child is not None and child.stdout is not None:child.stdout.close()
        store.data["workers"][phase]=dict(status="COMPLETE" if code==0 and primary is None else "FAILED",command=command,returncode=code,error=str(primary) if primary else None,elapsed_seconds=time.perf_counter()-start);store.save()
def prepare_new_source(store,worker=launch_worker):
    check_seal(store.data["threshold_seal"])
    if store.data["thresholds"]["status"]!="AVAILABLE":raise RuntimeError("both calibrations required before generation")
    p=store.output/"source_preparation/source_preparation.json";ready=False
    try:
        worker(store,"source");x=read(p)
        if x["status"]!="SOURCE_READY":raise ValueError("source identity not saved")
        worker(store,"matched");x=read(p)
        if x["status"]!="COMPLETE" or set(x["received_sources"])!={"P1","M05"}:raise ValueError("matched preparation incomplete")
        if not all(v["match"] for name in ("source_calls_integrity","calls_integrity") for v in x[name].values()):raise ValueError("preparation calls incomplete")
        ready=True
        return x["received_sources"]
    finally:
        if p.exists():
            saved=read(p)
            if not ready and saved.get("status")!="FAILED":
                reason="Preparation worker did not complete; no retry or replacement"
                saved["status"]="INCOMPLETE";saved.setdefault("failures",[]).append(dict(stage="WORKER_NOT_COMPLETED",error=reason))
                def finish(value):
                    if isinstance(value,dict):
                        if value.get("status") in ("PENDING","RUNNING"):value.update(status="NOT_COMPLETED",error=reason)
                        for child in value.values():finish(child)
                    elif isinstance(value,list):
                        for child in value:finish(child)
                finish(saved);dump(p,saved)
            store.data["source_preparation_receipt"]=dict(path=str(p),sha256=digest(p),status=saved["status"])
        else:store.data["source_preparation_receipt"]=dict(status="NOT_COMPLETED")
        store.save()
def posthoc(store):
    check_seal(store.data["decision_seal"]);check_seal(store.data["operation_plan_seal"])
    truth=companion(store.cfg,"posthoc")["truth"]
    for sid,d in store.data["decisions"].items():
        oid,key=sid.split("/");t=truth[oid];n=d["frames"];positive=t["condition"]=="M05" and key=="K0";path=d.get("path")
        target=method.operation(n,t["path"]);op=d.get("received_index_map")
        inferred=method.deletion.path_map(path) if path and n==177 else list(range(path["b"],path["b"]+n)) if path else None
        kerr=path["k"]-t["path"]["k"] if path and path["family"]==t["path"]["family"]=="H1" else None
        store.data["posthoc"][sid]=dict(status="EVALUATED_POSTSEAL",positive=positive,key_role="REGISTERED" if key=="K0" else "WRONG_KEY",truth=t,state=d["state"],execution_status=store.data["confirmation"][sid]["execution_status"],null_false_authorization=(not positive and d["state"]=="ACCEPT_ACTION"),positive_wrong_correction=(positive and op is not None and op!=target),positive_authorized=positive and op is not None,operation_cost=store.data["operation_plan"]["operations"][sid]["GATED"]["operation_cost"],operation_correct=op==target if op is not None else None,source_map_correct=sum(a==b for a,b in zip(inferred,t["source_frame_map"])) if inferred else None,source_map_denominator=n,absolute_path_correct=path==t["path"] if path else None,k_signed_error=kerr,k_absolute_error=abs(kerr) if kerr is not None else None)
    store.save()
def family_summary(store):
    result={}
    for n in (177,89):
        ids=[sid for sid,d in store.data["decisions"].items() if d["frames"]==n]
        rows=[store.data["posthoc"][sid] for sid in ids]
        false=sum(bool(x.get("null_false_authorization")) for x in rows)
        wrong=sum(bool(x.get("positive_wrong_correction")) for x in rows)
        covered=sum(bool(x.get("positive_authorized")) for x in rows)
        technical=sum(store.data["confirmation"][sid]["execution_status"]!="SCORED" or store.data["confirmation_analyses"][sid]["status"]!="SCORED" or store.data["posthoc"][sid]["status"]!="EVALUATED_POSTSEAL" for sid in ids)
        not_run=sum(store.data["confirmation"][sid]["execution_status"]=="NOT_RUN" for sid in ids)
        result[str(n)]=dict(decision_denominator=8,null_false_authorizations=false,null_denominator=6,positive_authorized=covered,positive_denominator=2,positive_wrong_corrections=wrong,technical_failures=technical,not_run=not_run,
            states={state:sum(store.data["decisions"][sid]["state"]==state for sid in ids) for state in ("ACCEPT_ACTION","REJECT","UNCERTAIN")},
            fixed_case_interpretation="VALID_COUNTEREXAMPLE" if false or wrong else "INSUFFICIENT_EVIDENCE" if technical or covered!=2 else "FIXED_CASE_SUPPORT",
            claim_limit="One fixed source/roster only; neither all reject nor all uncertain closes the method; no FPR/coverage/scientific PASS")
    return result

def run(output,*,cfg=None,framewise_type=FramewiseBackend,input_type=InputBackend,cache_loader=cached_read,source_fn=prepare_new_source):
    cfg=copy.deepcopy(read(CONFIG) if cfg is None else cfg);store=Store(output,cfg);abort=None
    try:
        prep=companion(cfg,"preparation")
        sources=prepare_inputs(store,"development",prep["development_sources"],input_type)
        score_stage(store,"development",sources,framewise_type);install_cached(store,cache_loader)
    except BaseException as exc:abort=exc;store.failure("DEVELOPMENT_INTERRUPTED",exc)
    finally:seal_scores(store,"development")
    try:
        stats=analyses(store,"development");store.data["development_analyses"]=stats
        for sid,a in stats.items():
            if a["status"]!="SCORED":store.failure("DEVELOPMENT_ANALYSIS/"+sid,ValueError(a.get("reason","invalid analysis")))
        labels=load_calibration_labels(store)
        store.data["thresholds"]=method.calibrate(stats,labels["null_rosters"])
    except BaseException as exc:
        if not isinstance(exc,Exception):abort=abort or exc
        store.failure("CALIBRATION",exc)
    store.data["threshold_seal"]=dump(store.output/"frozen_thresholds.json",store.data["thresholds"]);store.save()
    if store.data["thresholds"]["status"]=="AVAILABLE" and abort is None:
        try:
            specs=source_fn(store)
            sources=prepare_inputs(store,"confirmation",specs,input_type)
            score_stage(store,"confirmation",sources,framewise_type)
        except BaseException as exc:
            if not isinstance(exc,Exception):abort=exc
            store.failure("CONFIRMATION_PREPARATION_OR_SCORE",exc)
    else:
        store.data["generation_gate"]=dict(status="NOT_RUN",reason="CALIBRATION_UNAVAILABLE_OR_INTERRUPTED")
    seal_scores(store,"confirmation")
    stats=analyses(store,"confirmation")
    store.data["confirmation_analyses"]=stats
    for sid,s in stats.items():
        store.data["decisions"][sid]=method.decide(s,store.data["thresholds"])
        store.data["decisions"][sid]["execution_status"]=store.data["confirmation"][sid]["execution_status"]
    store.data["decision_seal"]=dump(store.output/"blind_decisions.json",dict(decisions=store.data["decisions"],truth_inputs=False))
    plan=method.cpu_plan(store.data["decisions"]);store.data["operation_plan"]=plan
    store.data["operation_plan_seal"]=dump(store.output/"blind_operations.json",plan);store.save()
    try:posthoc(store)
    except BaseException as exc:
        if not isinstance(exc,Exception):abort=abort or exc
        store.failure("POSTHOC",exc)
        for row in store.data["posthoc"].values():
            if row["status"]=="PENDING":row.update(status="FAILED",reason=str(exc))
    store.data["family_summary"]=family_summary(store)
    store.data["counts"]=dict(development_slots=len(store.data["development"]),confirmation_slots=len(store.data["confirmation"]),decisions=len(store.data["decisions"]),states={s:sum(x["state"]==s for x in store.data["decisions"].values()) for s in ("ACCEPT_ACTION","REJECT","UNCERTAIN")},receiver_wan_encode=0,payload_read=0)
    store.data["stage"]="FINISHED"
    store.data["status"]="COMPLETE" if not store.data["failures"] and store.data["thresholds"]["status"]=="AVAILABLE" and all(x["status"]=="SCORED" for x in stats.values()) and len(store.data.get("development_analyses",{}))==20 and all(x["status"]=="SCORED" for x in store.data.get("development_analyses",{}).values()) else "RETAINED_INCOMPLETE"
    store.save()
    if abort is not None:raise abort
    return store.data
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);a=p.parse_args();result=run(a.output);print(json.dumps(dict(status=result["status"],counts=result["counts"])));sys.exit(0 if result["status"]=="COMPLETE" else 1)
