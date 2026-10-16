"""Durable fixed-unit accounting and report-only recovery for one P48 task."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import threading
import time
from main.tube_state import local_joint_p48_joint_control_v1 as method
from main.tube_state.local_joint_p0_input_vjp_v1 import finite_tensor
from runtime.wan.local_joint_p0_input_vjp_v1 import Journal as ResourceJournal, write_json, tensor, resource_snapshot

LIMITS = dict(transformer_prepare=1,dit_ordinary_forward=6,dit_gradient_forward=8,dit_input_vjp=8,
    native_update=6,vae_load=1,gradient_decode=4,decoder_vjp=4,ordinary_decode=3)
STATES = ("RETURNED","FAILED","INTERRUPTED_COMPLETION_UNKNOWN","RUNNING")


def recount(r):
    r["counts"] = {k:dict(attempted=sum(c["kind"]==k for c in r["calls"]),
        **{s:sum(c["kind"]==k and c["status"]==s for c in r["calls"]) for s in STATES}) for k in LIMITS}
    r["block_counts"] = {p:dict(attempted=sum(b[p]!="NOT_RUN" for v in r["dit_replay"].values() for b in v.get("blocks",[])),
        **{s:sum(b[p]==s for v in r["dit_replay"].values() for b in v.get("blocks",[])) for s in STATES}) for p in ("forward","recompute")}
    r["chunk_counts"] = {p:{k:sum(v.get("counts",{}).get("vae_chunk",{}).get(p,{}).get(k,0) for v in r["vae_replay"].values())
        + (sum(v[k] for v in r["ordinary_chunks"].values()) if p=="forward" else 0)
        for k in ("attempted","completed")} for p in ("forward","recompute")}


def initial(config):
    return dict(status="RUNNING",config=copy.deepcopy(config),calls=[],failures=[],resources=[],dit_replay={},vae_replay={},ordinary_chunks={},
        endpoints={n:dict(status="MISSING",reason="not_entered") for n in method.VIEWS},
        objectives={n:dict(status="MISSING",reason="not_entered",branches={b:dict(status="MISSING") for b in ("conditional","unconditional")}) for n in method.OBJECTIVES},
        expected_cost=dict(**LIMITS,tracked_gradient_block_forward=240,block_recompute_upper=240,vae_original_chunks=322,vae_recompute_upper=184,
            new_source=0,new_baseline=0,encode=0,codec=0,retry=0),
        block_count_scope="block_counts tracks only 8 gradient forwards x30 blocks=240; ordinary 6 forwards add180 uninstrumented native blocks; total original blocks420",
        fixed_denominators=dict(endpoints=3,endpoint_windows=264,endpoint_chips=4224,endpoint_metrics=165,
            calibration_units=4,calibration_windows=352,calibration_chips=5632,calibration_metrics=220,independent_sources=1),
        p0_sunk_cost=dict(transformer_prepare=1,dit_forward=2,dit_input_vjp=2,block_recompute=60,separate_from_P48=True),
        actual_model_execution="NOT_STARTED",scientific_pass=False,automatic_retry=False,ceiling=method.CEILING,
        created_at_unix=time.time())


def read_state(output):
    output=Path(output)
    issues=[]
    for name in ("result.json","result.previous.json"):
        path=output/name
        if not path.exists(): continue
        try:
            r=json.loads(path.read_text())
            if set(r["endpoints"]) != set(method.VIEWS) or set(r["objectives"]) != set(method.OBJECTIVES):
                raise ValueError("fixed endpoint/objective rows missing")
            rows={c["id"]:c for c in r["calls"]}
            for p in (output/"calls").glob("*.json"):
                c=json.loads(p.read_text()); rows[c["id"]]=c
            r["calls"]=sorted(rows.values(),key=lambda c:c["id"])
            r.setdefault("recovery_notes",[]).extend(issues)
            # Preserve contradictions before recomputing counters.
            if any(v.get("attempted",0)>sum(c["kind"]==k for c in r["calls"]) for k,v in r.get("counts",{}).items()):
                r["physical_history_uncertain"]="stored attempts exceed readable call receipts"
            return r
        except (OSError,ValueError,KeyError,TypeError) as exc: issues.append(dict(path=str(path),reason=repr(exc)))
    if issues or (output.exists() and any(output.iterdir())):
        raise OSError("existing task state unreadable; no new model work: "+str(issues))
    return None


def zero_entry(r):
    return (r.get("actual_model_execution")=="NOT_STARTED" and r.get("calls")==[]
        and r.get("resources")==[] and r.get("dit_replay")=={} and r.get("vae_replay")=={} and r.get("ordinary_chunks")=={}
        and not r.get("recovery_notes") and not r.get("physical_history_uncertain")
        and set(r.get("counts",{}))==set(LIMITS) and all(all(type(n) is int and n==0 for n in v.values()) for v in r["counts"].values())
        and all(v.get("status")=="MISSING" for v in r["endpoints"].values())
        and all(v.get("status")=="MISSING" for v in r["objectives"].values()))


def precheck_can_continue(output,r):
    if not zero_entry(r) or r["status"]!="ENGINEERING_FAILURE" or not r["failures"] or any(f["stage"]!="saved_source" for f in r["failures"]): return False
    try:
        for p in Path(output).glob("result*.tmp"): return False
        for name in ("result.json","result.previous.json","report.json"):
            p=Path(output)/name
            if p.exists() and not zero_entry(json.loads(p.read_text())): return False
        if any((Path(output)/"calls").glob("*.json")): return False
    except (OSError,ValueError,KeyError,TypeError): return False
    return True


def needs_model_work(output):
    r=read_state(output)
    return r is None or precheck_can_continue(output,r)


class Journal(ResourceJournal):
    def sample(self):
        with self.lock:
            s=resource_snapshot(self.device)
            global_row=self.result.setdefault("resource_lifetime",dict(samples=0))
            global_row["samples"]+=1; global_row["latest"]=s
            for key in ("host_rss_bytes","host_process_max_rss_bytes","cuda_allocated_bytes","cuda_reserved_bytes","cuda_peak_allocated_bytes","cuda_peak_reserved_bytes"):
                if s.get(key) is not None: global_row[key+"_max"]=max(global_row.get(key+"_max",0),s[key])
            if self.stage is not None:
                self.stage["latest"]=s
                if s["host_rss_bytes"] is not None:
                    self.stage["sampled_host_rss_peak_bytes"]=max(self.stage.get("sampled_host_rss_peak_bytes",0),s["host_rss_bytes"])
            return s

    def start_stage(self,name):
        # Capture the preceding gap's short-lived CUDA peak before any reset.
        self.result.setdefault("resource_boundaries",[]).append(dict(before_stage=name,snapshot=self.sample()))
        self.save()
        super().start_stage(name)

    def __exit__(self,*args):
        self.stop.set(); self.thread.join(timeout=2)
        self.result.setdefault("resource_boundaries",[]).append(dict(final=True,snapshot=self.sample()))
        self.end_stage()
        save_report(self.output,self.result)

    def save(self):
        with self.lock:
            recount(self.result)
            path=self.output/"result.json"
            if path.exists():
                try: old=json.loads(path.read_text())
                except (OSError,ValueError): old=None
                if old is not None: write_json(self.output/"result.previous.json",old)
            write_json(path,self.result)

    def __enter__(self):
        def monitor():
            last=time.monotonic()
            while not self.stop.wait(.5):
                try:
                    self.sample()
                    if time.monotonic()-last>=60:
                        self.save(); print("P48 checkpoint",self.result["status"],self.result.get("counts"),flush=True); last=time.monotonic()
                except Exception as exc:
                    self.result.setdefault("monitor_errors",[]).append(repr(exc)); return
        self.thread=threading.Thread(target=monitor,daemon=True); self.thread.start(); return self

    def call(self,kind,unit,function):
        r=self.result
        if sum(c["kind"]==kind for c in r["calls"])>=LIMITS[kind] or any(c["kind"]==kind and c["unit"]==unit for c in r["calls"]):
            raise RuntimeError("fixed physical call already consumed: "+kind+"/"+unit)
        c=dict(id=f"{len(r['calls']):03d}",kind=kind,unit=unit,status="RUNNING",entered_at_unix=time.time())
        r["actual_model_execution"]="STARTED"; r["calls"].append(c)
        path=self.output/"calls"/(c["id"]+".json")
        write_json(path,c); self.start_stage(kind+"/"+unit)
        try:
            value=function(); c["status"]="RETURNED"; return value
        except BaseException as exc:
            c.update(status="FAILED" if isinstance(exc,Exception) else "INTERRUPTED_COMPLETION_UNKNOWN",reason=repr(exc)); raise
        finally:
            c["finished_at_unix"]=time.time(); write_json(path,c); self.end_stage()


def save_report(output,r):
    recount(r)
    write_json(Path(output)/"result.json",r)
    write_json(Path(output)/"report.json",r)
    return r


def check_json(path,kind,shape=None):
    """Read method data with its real cardinalities; no digest or source identity."""
    import math
    v=json.loads(Path(path).read_text())
    def finite(x):
        if isinstance(x,float) and not math.isfinite(x): raise ValueError("nonfinite JSON number")
        if isinstance(x,dict):
            for a in x.values(): finite(a)
        elif isinstance(x,list):
            for a in x: finite(a)
    finite(v)
    def numbers(values,n):
        if not isinstance(values,list) or len(values)!=n or any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
            raise ValueError("invalid numeric cardinality")
    if kind=="raw":
        if not isinstance(v,list) or len(v)!=88: raise ValueError("raw receiver must retain88 windows")
        for row in v:
            for name in ("state_chips","payload_chips"):
                chips=row[name]
                if len(chips)!=8 or any("q" not in x for x in chips): raise ValueError("raw receiver chip cardinality")
                if any(x["q"] is not None and type(x["q"]) not in (int,float) for x in chips): raise ValueError("invalid raw chip value")
    elif kind=="scores":
        numbers([x["value"] for x in v["state"]["correlations"]],22)
        numbers([x["signed_mean"] for x in v["payload"]["metrics"]],32)
    elif kind=="values":
        numbers(v["correlations"],22); numbers(v["margins"],32)
    elif kind=="quality":
        if set(v["energies"])!=set(method.ENERGIES) or any(type(x) not in (int,float) or x<0 for x in v["energies"].values()): raise ValueError("eight nonnegative energies required")
        numbers(v["frame_mse"],shape[0]); numbers(v["transition_mse"],shape[0]-1)
        numbers(v["frame_max_abs"],shape[0]); numbers(v["transition_max_abs"],shape[0]-1)
    elif kind=="objective":
        if type(v["value"]) not in (int,float) or v["tie_rule"]!="equal active subgradient": raise ValueError("objective receipt malformed")
    elif kind=="comparison":
        if type(v["exact_equal"]) is not bool or "shape" not in v or "finite" not in v: raise ValueError("comparison receipt malformed")
    elif kind=="control":
        if v["sampling_step"]!=48 or v["total_sampling_steps"]!=50 or v["control"]["dtype"]!="torch.float32": raise ValueError("control receipt malformed")
    elif kind=="arithmetic":
        if v["input_dtype"]!="torch.bfloat16" or v["saved_dtype"]!="torch.float32" or "actual_output_dtype" not in v: raise ValueError("arithmetic receipt malformed")
    elif kind=="solution":
        if not isinstance(v.get("status"),str) or v.get("physical_candidates") not in (0,1): raise ValueError("solution receipt malformed")
    elif kind=="object":
        if not isinstance(v,dict): raise ValueError("object JSON required")
    else: raise ValueError("unknown method JSON kind")
    return v


def resume_report(output,r,reason="existing physical task; no repeated model work"):
    import torch
    output=Path(output)
    for c in r["calls"]:
        if c["status"]=="RUNNING":
            c.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason=reason)
            write_json(output/"calls"/(c["id"]+".json"),c)
    for v in r["dit_replay"].values():
        for b in v.get("blocks",[]):
            for p in ("forward","recompute"):
                if b[p]=="RUNNING": b[p]="INTERRUPTED_COMPLETION_UNKNOWN"
    r["resume_policy"]="report-only: live models/graphs cannot be recovered under one preparation"
    # Validate saved actual values, not labels, digests or source versions.
    issues=[]
    def json_rows(directory,row,unit):
        for record in row.get("json_evidence",[]):
            try: check_json(directory/record["file"],record["kind"],record.get("shape"))
            except Exception as exc:
                issues.append(dict(stage="saved_evidence",unit=unit,file=record["file"],reason=repr(exc)))
                row.update(status="MISSING",reason="saved method JSON unreadable or invalid")
    for unit,row in list(r["endpoints"].items())+list(r["objectives"].items()):
        directory=output/(unit if unit in method.VIEWS else "objectives/"+unit)
        json_rows(directory,row,unit)
        for record in row.get("evidence",[]):
            try:
                finite_tensor(tensor(directory/record["file"]),record["shape"],record["file"],getattr(torch,record["dtype"]))
            except Exception as exc:
                issues.append(dict(stage="saved_evidence",unit=unit,file=record["file"],reason=repr(exc)))
                row.update(status="MISSING",reason="saved numerical evidence unreadable or invalid")
        for b,br in row.get("branches",{}).items():
            json_rows(directory/b,br,unit+"/"+b)
            if br["status"]=="MISSING" and br.get("reason")=="saved method JSON unreadable or invalid": row["status"]="MISSING"
            for record in br.get("evidence",[]):
                try: finite_tensor(tensor(directory/b/record["file"]),record["shape"],record["file"],getattr(torch,record["dtype"]))
                except Exception as exc:
                    issues.append(dict(stage="saved_evidence",unit=unit+"/"+b,file=record["file"],reason=repr(exc)))
                    br.update(status="MISSING",reason="saved numerical evidence unreadable or invalid"); row["status"]="MISSING"
    json_rows(output,r,"root")
    if issues:
        r["failures"].extend(issues); r["status"]="ENGINEERING_FAILURE"
    if r["status"]=="RUNNING": r["status"]="INTERRUPTED"
    if any(c["status"]=="INTERRUPTED_COMPLETION_UNKNOWN" for c in r["calls"]): r["status"]="INTERRUPTED"
    if r["status"] not in ("COMPLETE","CONSTRUCTION_UNDEFINED","NO_RESOLVED_CANDIDATE"):
        r["comparison"]=dict(status="INCOMPLETE_EVIDENCE",scientific_pass=False,ceiling=method.CEILING)
    return save_report(output,r)


def finalize_interrupted(output,reason):
    r=read_state(output)
    if r is None: return None
    # Fresh Run all validates files via resume_report. Child takeover only needs
    # recovery when the child was interrupted; avoid rereading 7 full RGBs twice.
    if r["status"]!="RUNNING" and not any(c["status"]=="RUNNING" for c in r["calls"]): return r
    return resume_report(output,r,reason)
