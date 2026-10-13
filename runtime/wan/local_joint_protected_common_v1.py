"""Four independent protection VJPs, one optional decode, durable unit resume."""
from __future__ import annotations

import copy
import gc
import os
from pathlib import Path
import shutil
import time

from main.tube_state import local_joint_protected_common_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_state_payload_posthoc_v1 as posthoc
from main.tube_state import local_joint_terminal_bridge_v1 as observer
from runtime.wan import generation, vae as adapter
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from runtime.wan import local_joint_all_bit_quality_v1 as visual
from runtime.wan.local_joint_all_bit_v1 import validate_raw, evaluation_complete, compare_values, loss_from_values
from runtime.wan.local_joint_readout_checkpoint_v1 import ReplayLedger, checkpoint_decode
from runtime.wan.local_joint_readout_m0_v1 import masked, metric_values
from runtime.wan.local_joint_state_payload_provider_v1 import PUBLIC_LATENT_SUPPORT, WanPosteriorBackend
from runtime.wan.local_joint_terminal_bridge_v1 import KEY, LATENT_SHAPE, tensor_contract

EXPECTED = dict(vae_load=1,decode=5,decoder_vjp=4,ordinary_decode=1,encode=0,dit=0,native=0,
    scheduler=0,codec=0,ordinary_chunks=46,gradient_forward_chunks=184,nominal_replay_chunks=184,
    windows=352,chips=5632,metrics=220,gradient_probe_windows=352,gradient_probe_chips=5632,
    gradient_probe_metrics=220,reused_view_count=3,new_view_count=1,planned_pause_extra_load=1)
STEPS = ("direction","step")
FINISHED = ("COMPLETE","COMPLETE_NO_CANDIDATE")


def initial_result(config):
    calls = {"load":dict(kind="vae_load",status="NOT_RUN"),"PROTECTED":dict(kind="decode",status="NOT_RUN")}
    for name in method.PROTECTIONS:
        calls[name+"_gradient_decode"] = dict(kind="decode",status="NOT_RUN")
        calls[name+"_vjp"] = dict(kind="decoder_vjp",status="NOT_RUN")
    return dict(schema="local-joint-protected-common-v1",status="RUNNING",config=copy.deepcopy(config),
        views={name:dict(status="NOT_RUN",reason="not_run",expected_windows=88,expected_chips=1408,
            expected_metrics=55,observed_windows=0,observed_chips=0,observed_metrics=0,
            origin="reused_saved_view" if name != "PROTECTED" else "new_candidate") for name in method.VIEWS},
        gradient_units={name:dict(status="NOT_RUN",observed_windows=0,observed_chips=0,observed_metrics=0)
                        for name in method.PROTECTIONS},steps={name:dict(status="NOT_RUN") for name in STEPS},
        model_calls=calls,call_history=[],replay={},failures=[],failure_history=[],cleanup_errors=[],
        counts={**{kind+"_"+phase:0 for kind in ("vae_load","decode","decoder_vjp")
                   for phase in ("attempted","completed")},
                **{kind:0 for kind in ("encode","dit","native","scheduler","codec")}},
        ordinary_chunks=dict(attempted=0,completed=0),expected_cost=EXPECTED,
        ceiling=method.CEILING,scientific_pass=False,automatic_retry=False,actual_model_execution="NOT_STARTED")


def read_state(output):
    result = resume.read_state(output,views=method.VIEWS,steps=STEPS)
    if result is None:
        return None
    failures = {row["id"]:row for row in result.get("failure_history",[])}
    for path in (Path(output)/"failures").glob("*.json"):
        try:
            row = resume.read_json(path)
            if not isinstance(row,dict) or not isinstance(row.get("id"),str):
                raise ValueError("failure sidecar requires an object and string id")
            failures[row["id"]] = row
        except (OSError,ValueError,TypeError,KeyError) as exc:
            result.setdefault("recovery_events",[]).append(dict(path=str(path),reason=repr(exc)))
    result["failure_history"] = list(failures.values())
    return result


def refresh_counts(result):
    resume.recount(result)
    own = [r for r in result["call_history"] if r.get("attempt_id") == result.get("attempt_id")]
    result["current_attempt_counts"] = {kind+"_"+phase:sum(r["kind"]==kind and
        (phase=="attempted" or r["status"]=="RETURNED") for r in own)
        for kind in ("vae_load","decode","decoder_vjp") for phase in ("attempted","completed")}
    result["totals"] = {field:sum(x[field] for x in result["views"].values()) for field in
        ("expected_windows","expected_chips","expected_metrics","observed_windows","observed_chips","observed_metrics")}
    result["gradient_probe_totals"] = {"observed_"+field:sum(x["observed_"+field]
        for x in result["gradient_units"].values()) for field in ("windows","chips","metrics")}
    result["totals"].update(planned_source_points=1,independent_samples=1,reused_views=3)
    result["physical_replay_chunks"] = {phase:sum(r.get("replay",{}).get("counts",{}).get("vae_chunk",{})
        .get(phase,{}).get("attempted",0) for r in result["call_history"]) for phase in ("forward","recompute")}


def record_failure(output,result,stage,reason):
    row = dict(id=str(time.time_ns()),attempt_id=result.get("attempt_id"),stage=stage,reason=reason)
    result["failures"].append(row); result["failure_history"].append(row)
    resume.atomic_bytes(Path(output)/"failures"/(row["id"]+".json"),resume.json_bytes(row))


def finalize_interrupted(output,reason):
    result = read_state(output)
    if result is None or result["status"] != "RUNNING":
        return result
    result["status"] = "INTERRUPTED"
    record_failure(output,result,"external_exit",reason)
    for row in result["call_history"]:
        if row["status"] == "RUNNING": row.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason=reason)
    refresh_counts(result)
    journal = resume.Journal(output,views=method.VIEWS,steps=STEPS)
    journal.save(result); journal.checkpoint("report-only child cleanup")
    return result


def model_work_remaining(output):
    output = Path(output)
    state = read_state(output)
    if state is None and output.exists() and any(output.iterdir()):
        raise OSError("nonempty output has no recoverable task state")
    if state is not None and state["status"] in FINISHED:
        return False
    if gradient_file_problem(output,state) is not None:
        return False  # Concrete saved-data fault; runner records it without a model.
    if any(not (output/"units"/(name+".pt")).is_file() for name in method.PROTECTIONS):
        return True
    direction = output/"direction.json"
    if direction.exists() and resume.read_json(direction)["status"] != "POSITIVE_COMMON_DIRECTION":
        return False
    return candidate_decode_pending(output,state)


def gradient_was_committed(output,state,name):
    """A VJP RETURNED receipt alone is not an atomic gradient-unit commit."""
    output = Path(output)
    state = state or {}
    pause = state.get("controlled_pause",{})
    if not pause:
        try: pause = resume.read_json(output/"controlled_pause.json")
        except (OSError,ValueError): pause = {}
    return ((output/"gradient_commits"/(name+".json")).exists()
            or name in state.get("completed_gradient_units",[])
            or state.get("gradient_units",{}).get(name,{}).get("status") == "COMPLETE"
            or (isinstance(pause,dict) and pause.get("after_unit") == name))


def gradient_file_problem(output,state):
    """CPU preflight shared by the model-work hint and the actual runner."""
    output = Path(output)
    for name in method.PROTECTIONS:
        path = output/"units"/(name+".pt")
        if not path.is_file():
            if gradient_was_committed(output,state,name):
                return name,"missing committed gradient unit: "+str(path)+"; no replacement VJP"
            continue
        try:
            import torch
        except ImportError:
            # The notebook's ordinary CPU dependency probe can install torch.
            # Do not claim a numerical file failure before it can be read.
            continue
        try:
            unit = torch.load(path,map_location="cpu",weights_only=True)
            if not isinstance(unit,dict) or unit.get("name") != name:
                raise ValueError("gradient unit has a different or missing scalar name")
            terminal,gradient,q = unit["input_terminal"],unit["gradient"],unit["q"]
            tensor_contract(gradient,tuple(terminal.shape))
            tensor_contract(terminal,tuple(terminal.shape))
            if tuple(q.shape) != (22,4,16) or not bool(torch.isfinite(q).all()):
                raise ValueError("gradient observation must have finite shape (22,4,16)")
            method.validate_values(unit["readout"]["values"])
        except Exception as exc:
            return name,"unreadable or invalid saved gradient unit: "+str(path)+"; "+repr(exc)
    return None


def rgb_was_committed(output,state,name):
    """Completed data lost later must not become permission to repeat a model call.

The return receipt of decode is deliberately excluded: RGB might not yet have
been committed when the process died. Prior scored/finished state supports
recovery of tasks created before the per-view completion marker was added.
"""
    output = Path(output)
    state = state or {}
    return ((output/"rgb_commits"/(name+".json")).exists()
            or name in state.get("completed_rgb_views",[])
            or state.get("views",{}).get(name,{}).get("status") == "SCORED"
            or (name == "PROTECTED" and state.get("status") == "COMPLETE")
            or (output/name/"raw.json").exists())


def candidate_decode_pending(output,state):
    return not (Path(output)/"PROTECTED/float_rgb.pt").is_file() and not rgb_was_committed(output,state,"PROTECTED")


def raw_q(rows):
    import torch
    validate_raw(rows)
    return torch.tensor([[c["q"] for c in row["state_chips"]+row["payload_chips"]]
                        for row in rows],dtype=torch.float64).reshape(22,4,16)


def q_matches_raw(path,expected):
    """Actual numerical observation validation, without identity/provenance checks."""
    import torch
    try:
        q = torch.load(path,map_location="cpu",weights_only=True)
        return (isinstance(q,torch.Tensor) and tuple(q.shape) == (22,4,16)
                and bool(torch.isfinite(q).all()) and torch.equal(q.double(),expected))
    except Exception:
        return False


def runtime_observation():
    """Observations only: no runtime identity admission or equality requirement."""
    try: boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError: boot = None
    return dict(boot_id=boot,hostname=os.environ.get("HOSTNAME"),process_id=os.getpid())


def copy_saved_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = target.with_name(target.name+".tmp")
    with Path(source).open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src,dst,1024*1024); dst.flush(); os.fsync(dst.fileno())
    os.replace(temporary,target)


def reports_complete(output,available,frame_count):
    from PIL import Image
    output = Path(output)
    try:
        for name in available:
            directory = output/name
            expected_q = raw_q(resume.read_json(directory/"raw.json"))
            if not q_matches_raw(directory/"q.pt",expected_q): return False
            if not evaluation_complete(resume.read_json(directory/"metrics.json")): return False
            if not all((directory/f).is_file() for f in ("float_rgb.pt","terminal_latent.pt","q.pt")): return False
            for ref in method.VIEWS[:2]:
                row = resume.read_json(directory/("quality_vs_"+ref+".json"))
                if len(row["framewise"]) != frame_count or len(row["temporal_framewise"]) != frame_count-1: return False
                index = resume.read_json(directory/("frames_vs_"+ref+".json"))
                if not index["frames"]: return False
                for item in index["frames"]+index["crops"]:
                    with Image.open(directory/item["path"]) as image: image.verify()
        return len(resume.read_json(output/"comparison.json")["bits_32"]) == 32
    except (OSError,ValueError,TypeError,KeyError):
        return False


def refresh_reports(output,result,parameters):
    output = Path(output)
    values = {}
    for name in method.VIEWS:
        directory = output/name
        try: evaluation = resume.read_json(directory/"metrics.json")
        except (OSError,ValueError): evaluation = posthoc.missing_evaluation("missing_readout")
        values[name] = metric_values(evaluation)
        try: rows = resume.read_json(directory/"raw.json")
        except (OSError,ValueError): rows = []
        count = sum(x is not None for x in values[name]["correlations"]+values[name]["margins"]+[values[name]["state_gap"]])
        result["views"][name].update(**values[name],loss=loss_from_values(values[name]),observed_metrics=count,
            observed_windows=len(rows),observed_chips=sum(c.get("q") is not None for row in rows
                for c in row.get("state_chips",[])+row.get("payload_chips",[])))
        if count == 55: result["views"][name].update(status="SCORED",reason=None)
    predicted = result["steps"]["step"].get("predictions",{})
    table = []
    for i in range(32):
        margins = {name:values[name]["margins"][i] for name in method.VIEWS}
        start,old,new = margins["START_FINAL"],margins["TANGENT"],margins["PROTECTED"]
        old_gain = None if start is None or old is None else old-start
        gain = None if start is None or new is None else new-start
        table.append(dict(index=i,fragment=i//8,bit=i%8,margins=margins,
            tangent_gain=old_gain,protected_gain=gain,
            retained_tangent_gain_ratio=None if gain is None or old_gain in (None,0) else gain/old_gain,
            predicted=predicted.get(f"bit_{i:02d}" if i != 29 else "bit29"),
            protected_minus_mu=None if new is None else new-method.MU))
    guards = {}
    for name,view in values.items():
        margins = view["margins"]
        if any(x is None for x in margins):
            guards[name] = dict(status="MISSING",barrier=None); continue
        guards[name] = dict(status="SCORED",barrier=method.barrier_value(margins,parameters),
            bit29_minus_mu=margins[29]-method.MU,
            other25_lost_positive=[i for i in parameters["indices"] if margins[i] <= 0],
            other25_below_mu=[i for i in parameters["indices"] if margins[i] < method.MU],
            other25_decreased=[i for i in parameters["indices"] if margins[i] < values["START_FINAL"]["margins"][i]],
            state_minus_gamma=view["state_gap"]-method.GAMMA)
    quality = {}
    for name in method.VIEWS:
        try:
            row = resume.read_json(output/name/"quality_vs_BASE.json")
            quality[name] = dict(mse=row["rmse"]**2,rmse=row["rmse"],max_abs=row["max_abs"],
                temporal_residual_rmse=row["temporal_residual_rmse"],
                temporal_residual_max_abs=row["temporal_residual_max_abs"],no_quality_pass=True)
        except (OSError,ValueError,KeyError): quality[name] = None
    resume.atomic_bytes(output/"comparison.json",resume.json_bytes(dict(bits_32=table,
        versus={ref:compare_values(values[ref],values["PROTECTED"]) for ref in method.VIEWS[:3]},
        states={name:dict(correlations=v["correlations"],state_gap=v["state_gap"]) for name,v in values.items()},
        positive_bits={name:sum(x is not None and x>0 for x in v["margins"]) for name,v in values.items()},
        protections=guards,quality_vs_BASE=quality,quality_references=list(method.VIEWS[:2]),
        predicted=predicted,ceiling=method.CEILING)))
    refresh_counts(result)


def run(config,output,*,loader=None,protocol=carrier.PUBLIC,latent_shape=LATENT_SHAPE,
        support=PUBLIC_LATENT_SUPPORT,source_identity=None):
    import torch
    output = Path(output)
    for name,location in config["inputs"].items():
        source_directory = Path(location).resolve() if name in ("gradient_units","tangent_run") else Path(location).resolve().parent
        if output.resolve().is_relative_to(source_directory):
            raise ValueError("independent output must be outside source input directories")
    prior = read_state(output)
    if prior is None and output.exists() and any(output.iterdir()):
        raise OSError("nonempty output has no recoverable task state")
    output.mkdir(parents=True,exist_ok=True)
    result = prior or initial_result(config)
    # Preserve completion facts before RUNNING replaces the previous status and
    # before a missing artifact changes a formerly SCORED view to FAILED.
    result["completed_rgb_views"] = [name for name in method.VIEWS if rgb_was_committed(output,result,name)]
    result["completed_gradient_units"] = [name for name in method.PROTECTIONS if gradient_was_committed(output,result,name)]
    attempt = str(time.time_ns())
    if prior is not None:
        resume.atomic_bytes(output/"attempts"/(attempt+"_prior.json"),resume.json_bytes(prior))
        result.setdefault("attempt_history",[]).append(dict(attempt_id=prior.get("attempt_id"),status=prior["status"],
            failures=copy.deepcopy(prior["failures"]),runtime=prior.get("runtime")))
    for row in result["call_history"]:
        if row["status"] == "RUNNING": row.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason="previous process ended before durable return")
    result.update(status="RUNNING",attempt_id=attempt,failures=[],source=source_identity or {},
                  runtime=runtime_observation(),started_at_unix=time.time(),resources={})
    journal = resume.Journal(output,views=method.VIEWS,steps=STEPS)
    model = None

    def flush(): refresh_counts(result); journal.save(result)
    def save_json(path,value): resume.atomic_bytes(path,resume.json_bytes(value))
    def failure(stage,exc): record_failure(output,result,stage,repr(exc)); flush()
    def load_tensor(path): return torch.load(path,map_location="cpu",weights_only=True)
    def mark_gradient_committed(name):
        marker = output/"gradient_commits"/(name+".json")
        if not marker.exists():
            save_json(marker,dict(unit=name,path="units/"+name+".pt",
                committed_at_unix=time.time(),attempt_id=attempt))
        if name not in result["completed_gradient_units"]: result["completed_gradient_units"].append(name)
    def call(name,operation):
        row = dict(id=attempt+"_"+str(len(result["call_history"])).zfill(4),attempt_id=attempt,
            unit=name,kind=result["model_calls"][name]["kind"],status="RUNNING",started_at_unix=time.time())
        result["call_history"].append(row); result["model_calls"][name] = row; flush()
        try: value = operation()
        except BaseException as exc:
            row.update(status="FAILED" if isinstance(exc,Exception) else "INTERRUPTED_COMPLETION_UNKNOWN",reason=repr(exc)); flush(); raise
        row.update(status="RETURNED",finished_at_unix=time.time()); flush()
        return value

    def require_model():
        nonlocal model
        if model is None:
            if str(config.get("device","cuda")).startswith("cuda"): torch.cuda.reset_peak_memory_stats()
            model = call("load",lambda:(loader or generation.load_frozen_vae)(config,device=config.get("device","cuda")))
            WanPosteriorBackend(model)
            result["actual_model_execution"] = "REAL" if loader is None else "INJECTED_TEST_DOUBLE"
        return model

    def validate_gradient(g):
        tensor_contract(g,latent_shape)
        if torch.count_nonzero(g-masked(g,support)): raise ValueError("gradient outside adopted support")

    def fixed_inputs():
        path = output/"inputs/fixed_inputs.pt"
        if path.exists():
            bundle = load_tensor(path)
        else:
            if result["call_history"] or any((output/"units").glob("*.pt")):
                raise OSError("previous model work exists but fixed input backup is missing")
            locations = config["inputs"]
            six = load_tensor(locations["six_bit_inputs"])
            tangent = load_tensor(Path(locations["tangent_run"])/"inputs/fixed_inputs.pt")
            if not torch.equal(six["base"],tangent["base"]) or not torch.equal(six["final"],tangent["start"]):
                raise ValueError("saved six derivatives and tangent comparator use different numerical BASE/START")
            reference = metric_values(resume.read_json(Path(locations["tangent_run"])/"START_FINAL/metrics.json"))
            gradients = []
            for bit in method.BITS:
                unit = load_tensor(Path(locations["gradient_units"])/f"bit_{bit:02d}.pt")
                if unit["bit_index"] != bit or not torch.equal(unit["input_terminal"],six["final"]):
                    raise ValueError("saved derivative is paired with a different bit or numerical START")
                if unit["readout"]["derivative_sign"] != "positive_margin":
                    raise ValueError("old gradient must differentiate the positive margin")
                validate_gradient(unit["gradient"]); gradients.append(unit["gradient"])
            bundle = dict(base=six["base"],start=six["final"],tangent=tangent["candidate"],
                old_gradients=gradients,reference=reference,barrier=method.barrier_parameters(reference),source_locators=locations)
            resume.atomic_tensor(path,bundle)
        for name in ("base","start","tangent"): tensor_contract(bundle[name],latent_shape)
        support.validate_for(tuple(bundle["base"].shape))
        delta = bundle["start"].double()-bundle["base"].double()
        if not 0 < float(delta.norm()) <= 1 or torch.count_nonzero(delta-masked(delta,support)):
            raise ValueError("START must use the adopted support and original BASE ball")
        for g in bundle["old_gradients"]: validate_gradient(g)
        if len(bundle["old_gradients"]) != 6: raise ValueError("six saved gradients required")
        if method.barrier_parameters(bundle["reference"]) != bundle["barrier"]:
            raise ValueError("saved barrier coefficients do not describe numerical START")
        result["fixed_input"] = dict(backup="inputs/fixed_inputs.pt",source_locators=bundle["source_locators"],source_arm="CONT_FINAL")
        flush()
        return bundle

    def observe_saved(name,terminal):
        directory = output/name
        if name != "PROTECTED":
            source = Path(bundle["source_locators"]["tangent_run"])/name
            for filename in ("terminal_latent.pt","float_rgb.pt","raw.json","q.pt","metrics.json"):
                if not (directory/filename).exists():
                    if filename in ("terminal_latent.pt","float_rgb.pt") or (source/filename).is_file():
                        copy_saved_file(source/filename,directory/filename)
        if not (directory/"terminal_latent.pt").exists(): resume.atomic_tensor(directory/"terminal_latent.pt",terminal)
        if not torch.equal(load_tensor(directory/"terminal_latent.pt"),terminal):
            raise ValueError("saved RGB view is paired with a different numerical terminal")
        path = directory/"float_rgb.pt"
        if path.exists(): rgb = load_tensor(path)
        else:
            if name != "PROTECTED" or not candidate_decode_pending(output,result):
                raise OSError("missing completed RGB; no replacement decode for a saved view")
            rgb = decode(terminal)
            tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
            resume.atomic_tensor(path,rgb)
        tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
        # Written only after a complete, readable float RGB has been committed.
        # A failed marker write leaves the actual RGB available for recovery.
        marker = output/"rgb_commits"/(name+".json")
        if not marker.exists():
            save_json(marker,dict(view=name,path=str(path.relative_to(output)),
                committed_at_unix=time.time(),attempt_id=attempt))
        if name not in result["completed_rgb_views"]: result["completed_rgb_views"].append(name)
        flush()
        try:
            rows = resume.read_json(directory/"raw.json"); validate_raw(rows)
        except (OSError,ValueError,TypeError,KeyError):
            rows = observer.read_rgb(rgb,key=KEY,rho=.5,protocol=protocol)
            save_json(directory/"raw.json",rows); validate_raw(rows); flush()
        expected_q = raw_q(rows)
        if not q_matches_raw(directory/"q.pt",expected_q):
            resume.atomic_tensor(directory/"q.pt",expected_q)
            result.setdefault("recovery_events",[]).append(dict(path=str(directory/"q.pt"),
                action="CPU rebuilt q from validated raw observations",attempt_id=attempt))
        try: evaluation = resume.read_json(directory/"metrics.json")
        except (OSError,ValueError): evaluation = None
        if not evaluation_complete(evaluation):
            save_json(directory/"metrics.json",observer.evaluate_saved(rows,key=KEY,message=bytes.fromhex("8001a55a"),rho=.5,protocol=protocol))
        flush()

    def validate_unit(unit,name):
        if unit["name"] != name or not torch.equal(unit["input_terminal"],bundle["start"]):
            raise ValueError("protection derivative is paired with a different scalar or numerical START")
        validate_gradient(unit["gradient"])
        method.validate_values(unit["readout"]["values"])
        if tuple(unit["q"].shape) != (22,4,16) or not bool(torch.isfinite(unit["q"]).all()):
            raise ValueError("complete finite protection observation required")

    def measure(name):
        require_model()
        def attach(summary):
            result["replay"][name] = summary
            for row in reversed(result["call_history"]):
                if row["unit"] == name+"_gradient_decode" and row.get("attempt_id") == attempt:
                    row["replay"] = copy.deepcopy(summary); break
        def progress(summary): attach(summary); flush()
        ledger = ReplayLedger(callback=progress)
        z = bundle["start"].to(config.get("device","cuda")).clone().requires_grad_(True)
        original_decode = model.decode
        try:
            adapter._clear_cache(model)
            model.decode = lambda *a,**k:call(name+"_gradient_decode",lambda:original_decode(*a,**k))
            mean,std = adapter._scale_tensors(model,z)
            decoded = checkpoint_decode(model,z*std+mean,ledger)
            rgb = (decoded[0].permute(1,2,3,0).float()/2+.5).clamp(0,1)
            tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
            read = method.tensor_readout(rgb,key=KEY,message=bytes.fromhex("8001a55a"),protocol=protocol)
            q,values = read["q"].detach().cpu(),method.values55(read)
            resume.atomic_tensor(output/"observations"/(name+"_q.pt"),q)
            report = dict(name=name,values=values,differences_from_saved_START={
                key:[a-b for a,b in zip(values[key],bundle["reference"][key])] for key in ("margins","correlations")})
            save_json(output/"observations"/(name+"_readout.json"),report)
            result["gradient_units"][name].update(observed_windows=88,observed_chips=1408,observed_metrics=55,readout=report)
            base_rgb = load_tensor(output/"BASE/float_rgb.pt") if name == "quality_Q" else None
            scalar = method.protection_scalar(name,read,rgb,base_rgb,bundle["barrier"])
            report.update(scalar_value=float(scalar.detach()),definition="fixed START barrier derivative surrogate" if name == "barrier25" else name)
            save_json(output/"observations"/(name+"_readout.json"),report); flush()
            raw = call(name+"_vjp",lambda:torch.autograd.grad(scalar,z)[0]).detach()
            if not bool(torch.isfinite(raw).all()): raise FloatingPointError("nonfinite protection derivative")
            gradient = masked(raw,support).cpu()
            unit = dict(name=name,input_terminal=bundle["start"],gradient=gradient,q=q,readout=report,
                gradient_norms=dict(raw_l2=float(raw.double().norm()),masked_l2=float(gradient.double().norm())),
                call_ids=[r["id"] for r in result["call_history"] if r.get("attempt_id")==attempt and r["unit"].startswith(name)])
            resume.atomic_tensor(output/"units"/(name+".pt"),unit)
            mark_gradient_committed(name)
            return unit
        finally:
            model.decode = original_decode
            try: adapter._clear_cache(model)
            finally:
                ledger.callback = None; ledger.release_boundary_storage(); attach(ledger.summary())

    def decode(terminal):
        require_model()
        original,original_decode = model.decoder.forward,model.decode
        def counted(*a,**k):
            active = next(r for r in reversed(result["call_history"]) if r["unit"] == "PROTECTED" and r["attempt_id"] == attempt)
            chunks = active.setdefault("ordinary_chunks",dict(attempted=0,completed=0))
            chunks["attempted"] += 1; flush()
            value = original(*a,**k)
            chunks["completed"] += 1; flush()
            return value
        model.decoder.forward = counted
        model.decode = lambda *a,**k:call("PROTECTED",lambda:original_decode(*a,**k))
        try: return adapter.decode_normalized_latent(model,terminal.to(config.get("device","cuda"))).cpu()
        finally: model.decoder.forward,model.decode = original,original_decode

    def quality_and_visuals(available):
        # Retain only one reference and one target RGB at a time.
        for ref in method.VIEWS[:2]:
            try:
                reference = load_tensor(output/ref/"float_rgb.pt")
                tensor_contract(reference,protocol.video_shape,bounded_rgb=True)
            except Exception as exc:
                failure("quality_reference_"+ref,exc); continue
            reports = {}
            for name in available:
                path = output/name/("quality_vs_"+ref+".json")
                try:
                    try: row = resume.read_json(path)
                    except (OSError,ValueError): row = None
                    if not isinstance(row,dict) or len(row.get("framewise",[])) != len(reference) or len(row.get("temporal_framewise",[])) != len(reference)-1:
                        rgb = reference if name == ref else load_tensor(output/name/"float_rgb.pt")
                        tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
                        row = visual.quality(rgb,reference,protocol); del rgb
                        row.update(reference=ref,target=name,no_quality_threshold=True)
                        save_json(path,row); flush()
                    reports[name] = row
                except Exception as exc: failure(name+"_quality_vs_"+ref,exc)
            frames,crops = visual.selection(reports,len(reference),views=("REFERENCE",)+tuple(available))
            for name in available:
                try:
                    index_path = output/name/("frames_vs_"+ref+".json")
                    try: old = resume.read_json(index_path)
                    except (OSError,ValueError): old = None
                    if old is not None and old.get("frame_indices") == frames and old.get("crop_requests") == crops:
                        from PIL import Image
                        try:
                            for item in old["frames"]+old["crops"]:
                                with Image.open(output/name/item["path"]) as image: image.verify()
                            continue
                        except (OSError,ValueError,KeyError): pass
                    rgb = reference if name == ref else load_tensor(output/name/"float_rgb.pt")
                    rendered,crop_images = [],[]
                    for t in frames:
                        rendered.append(visual.render_pair(output/name/("frames_vs_"+ref)/f"frame_{t:03d}.png",name,t,rgb[t],reference[t]))
                    for n,request in enumerate(crops):
                        t,y,x = request["frame"],request["y"],request["x"]
                        box = (max(0,y-visual.CROP_RADIUS),min(rgb.shape[1],y+visual.CROP_RADIUS+1),
                               max(0,x-visual.CROP_RADIUS),min(rgb.shape[2],x+visual.CROP_RADIUS+1))
                        crop_images.append(visual.render_pair(output/name/("crops_vs_"+ref)/f"crop_{n:02d}_{t:03d}.png",name,t,rgb[t],reference[t],box=box))
                    save_json(index_path,dict(reference=ref,target=name,frame_indices=frames,crop_requests=crops,
                        frames=rendered,crops=crop_images,rgb_range=[0,1],residual_range=[-.25,.25],numeric_metrics_unclipped=True))
                    del rgb; flush()
                except Exception as exc: failure(name+"_visuals_vs_"+ref,exc)
            del reference; gc.collect()


    with journal.periodic():
        stage = "inputs"
        try:
            flush()
            c = config["carrier"]
            if (c["key"],c["message_hex"],c["rho"],c["cap"],config["step_size"]) != (KEY,"8001a55a",.5,1.,method.STEP):
                raise ValueError("adopted key/message/rho/cap and one .10 step required")
            bundle = fixed_inputs()
            problem = gradient_file_problem(output,result)
            if problem is not None:
                stage,reason = problem
                raise OSError(reason)
            terminals = dict(BASE=bundle["base"],START_FINAL=bundle["start"],TANGENT=bundle["tangent"])
            for name in method.VIEWS[:3]:
                stage = "reuse_"+name; observe_saved(name,terminals[name])
            refresh_reports(output,result,bundle["barrier"]); flush()
            units = {}
            for name in method.PROTECTIONS:
                path = output/"units"/(name+".pt")
                if path.exists():
                    unit = load_tensor(path); validate_unit(unit,name); units[name] = unit
                    mark_gradient_committed(name)
                    result["gradient_units"][name].update(status="COMPLETE",readout=unit["readout"],
                        observed_windows=88,observed_chips=1408,observed_metrics=55)
            unresolved = None
            for name in (() if (output/"direction_unit.pt").exists() else method.PROTECTIONS):
                stage = name
                if name not in units:
                    result["gradient_units"][name]["status"] = "RUNNING"; flush()
                    try: units[name] = measure(name)
                    except method.UndefinedProtection as exc:
                        unresolved = dict(status="UNDEFINED_PROTECTION",name=name,reason=str(exc))
                        result["gradient_units"][name].update(status="UNDEFINED",reason=str(exc)); break
                validate_unit(units[name],name)
                result["gradient_units"][name].update(status="COMPLETE",readout=units[name]["readout"],
                    observed_windows=88,observed_chips=1408,observed_metrics=55)
                flush(); gc.collect()
                if name == "bit29" and not (output/"controlled_pause.json").exists():
                    # Read back the completed unit before scheduling the one-time stop.
                    validate_unit(load_tensor(output/"units/bit29.pt"),name)
                    pause = dict(status="PAUSED_FOR_SESSION_RESTART",after_unit=name,attempt_id=attempt,
                        runtime=result["runtime"],instruction="Delete/disconnect the Colab runtime, reconnect, then Run all.",
                        no_live_autograd_checkpoint=True)
                    save_json(output/"controlled_pause.json",pause)
                    result.update(status="PAUSED_FOR_SESSION_RESTART",controlled_pause=pause)
                    flush(); return result
            stage = "direction"
            direction_path = output/"direction_unit.pt"
            if unresolved is not None:
                direction_unit = dict(direction=None,report=unresolved)
                resume.atomic_tensor(direction_path,direction_unit)
            elif direction_path.exists():
                direction_unit = load_tensor(direction_path)
            else:
                gradients = bundle["old_gradients"]+[units[name]["gradient"] for name in method.PROTECTIONS]
                direction,report = method.solve_direction(gradients,bundle["base"],bundle["start"])
                direction_unit = dict(direction=direction,report=report)
                resume.atomic_tensor(direction_path,direction_unit)
            result["steps"]["direction"] = direction_unit["report"]
            save_json(output/"direction.json",direction_unit["report"]); flush()
            available = list(method.VIEWS[:3])
            if direction_unit["direction"] is not None:
                stage = "candidate"
                direction = direction_unit["direction"]
                if torch.count_nonzero(direction-masked(direction,support)):
                    raise ValueError("direction outside adopted support")
                gradients = bundle["old_gradients"]+[units[name]["gradient"] for name in method.PROTECTIONS]
                path = output/"candidate_unit.pt"
                terminal,step = method.candidate(bundle["base"],bundle["start"],direction_unit["direction"],gradients)
                if path.exists():
                    saved = load_tensor(path)
                    if not torch.equal(saved["terminal"],terminal): raise ValueError("saved candidate differs from the one fixed numerical step")
                else: resume.atomic_tensor(path,dict(terminal=terminal,step=step))
                result["steps"]["step"] = step; save_json(output/"step.json",step); flush()
                stage = "PROTECTED"; observe_saved("PROTECTED",terminal); available.append("PROTECTED")
            else:
                result["views"]["PROTECTED"].update(status="NOT_RUN",reason=direction_unit["report"]["status"])
                result["steps"]["step"] = dict(status="NOT_RUN",reason=direction_unit["report"]["status"])
            stage = "quality_and_reports"
            if not reports_complete(output,available,protocol.video_shape[0]): quality_and_visuals(available)
            else: result["last_resume_action"] = "report_only; zero model calls"
            refresh_reports(output,result,bundle["barrier"])
            result["status"] = ("COMPLETE" if "PROTECTED" in available else "COMPLETE_NO_CANDIDATE") if not result["failures"] and reports_complete(output,available,protocol.video_shape[0]) else "ENGINEERING_FAILURE"
            pause = resume.read_json(output/"controlled_pause.json")
            result["resume_observation"] = dict(pause_runtime=pause["runtime"],current_runtime=result["runtime"],
                different_boot_observed=None if not pause["runtime"]["boot_id"] or not result["runtime"]["boot_id"]
                else pause["runtime"]["boot_id"] != result["runtime"]["boot_id"],
                note="runtime identifiers are observations only; same-process continuation is not a cross-runtime test")
        except BaseException as exc:
            result["status"] = "ENGINEERING_FAILURE" if isinstance(exc,Exception) else "INTERRUPTED"
            if stage in result["gradient_units"]: result["gradient_units"][stage].update(status="FAILED",reason=repr(exc))
            if stage in result["views"]: result["views"][stage].update(status="FAILED",reason=repr(exc))
            failure(stage,exc)
            if not isinstance(exc,Exception): raise
        finally:
            model = None; gc.collect()
            if str(config.get("device","cuda")).startswith("cuda") and torch.cuda.is_initialized():
                result["resources"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                torch.cuda.empty_cache()
            result["finished_at_unix"] = time.time(); flush()
    return result
