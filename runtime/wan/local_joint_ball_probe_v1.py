"""One fixed guarded full-ball candidate, one ordinary decode, durable resume."""
from __future__ import annotations

import copy
import gc
import os
from pathlib import Path
import time

from main.tube_state import local_joint_ball_probe_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_state_payload_posthoc_v1 as posthoc
from main.tube_state import local_joint_terminal_bridge_v1 as observer
from runtime.wan import generation, vae as adapter
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from runtime.wan import local_joint_all_bit_quality_v1 as visual
from runtime.wan import local_joint_protected_common_v1 as prior
from runtime.wan.local_joint_all_bit_v1 import validate_raw, evaluation_complete, compare_values, loss_from_values
from runtime.wan.local_joint_readout_m0_v1 import masked, metric_values
from runtime.wan.local_joint_state_payload_provider_v1 import PUBLIC_LATENT_SUPPORT, WanPosteriorBackend
from runtime.wan.local_joint_terminal_bridge_v1 import KEY, LATENT_SHAPE, tensor_contract

EXPECTED = dict(vae_load=1,decode=1,decoder_vjp=0,ordinary_decode=1,encode=0,dit=0,
    native=0,scheduler=0,codec=0,ordinary_chunks=46,reused_gradient_count=10,
    windows=352,chips=5632,metrics=220,reused_view_count=3,new_view_count=1)
STEPS = ("solve","candidate")
FINISHED = ("COMPLETE",)


def initial_result(config):
    calls = {"load":dict(kind="vae_load",status="NOT_RUN"),
             "BALL_PROBE":dict(kind="decode",status="NOT_RUN")}
    return dict(schema="local-joint-ball-probe-v1",status="RUNNING",config=copy.deepcopy(config),
        views={name:dict(status="NOT_RUN",reason="not_run",expected_windows=88,expected_chips=1408,
            expected_metrics=55,observed_windows=0,observed_chips=0,observed_metrics=0,
            origin="new_candidate" if name == "BALL_PROBE" else "reused_saved_view") for name in method.VIEWS},
        steps={name:dict(status="NOT_RUN") for name in STEPS},model_calls=calls,call_history=[],
        failures=[],failure_history=[],cleanup_errors=[],
        counts={**{kind+"_"+phase:0 for kind in ("vae_load","decode","decoder_vjp")
                   for phase in ("attempted","completed")},
                **{kind:0 for kind in ("encode","dit","native","scheduler","codec")}},
        ordinary_chunks=dict(attempted=0,completed=0),expected_cost=EXPECTED,
        ceiling=method.CEILING,scientific_pass=False,automatic_retry=False,
        actual_model_execution="NOT_STARTED")


def read_state(output):
    result = resume.read_state(output,views=method.VIEWS,steps=STEPS)
    if result is None: return None
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
    own = [row for row in result["call_history"] if row.get("attempt_id") == result.get("attempt_id")]
    result["current_attempt_counts"] = {kind+"_"+phase:sum(row["kind"]==kind and
        (phase=="attempted" or row["status"]=="RETURNED") for row in own)
        for kind in ("vae_load","decode","decoder_vjp") for phase in ("attempted","completed")}
    result["totals"] = {field:sum(row[field] for row in result["views"].values()) for field in
        ("expected_windows","expected_chips","expected_metrics","observed_windows","observed_chips","observed_metrics")}
    result["totals"].update(planned_source_points=1,independent_samples=1,reused_views=3)


def record_failure(output,result,stage,reason):
    row = dict(id=str(time.time_ns()),attempt_id=result.get("attempt_id"),stage=stage,reason=reason)
    result["failures"].append(row); result["failure_history"].append(row)
    resume.atomic_bytes(Path(output)/"failures"/(row["id"]+".json"),resume.json_bytes(row))


def finalize_interrupted(output,reason):
    result = read_state(output)
    if result is None or result["status"] != "RUNNING": return result
    result["status"] = "INTERRUPTED"
    record_failure(output,result,"external_exit",reason)
    for row in result["call_history"]:
        if row["status"] == "RUNNING": row.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason=reason)
    refresh_counts(result)
    journal = resume.Journal(output,views=method.VIEWS,steps=STEPS)
    journal.save(result); journal.checkpoint("report-only child cleanup")
    return result


def rgb_was_committed(output,state,name="BALL_PROBE"):
    output,state = Path(output),state or {}
    return ((output/"rgb_commits"/(name+".json")).exists()
            or name in state.get("completed_rgb_views",[])
            or state.get("views",{}).get(name,{}).get("status") == "SCORED"
            or (name == "BALL_PROBE" and state.get("status") == "COMPLETE")
            or (output/name/"raw.json").exists())


def candidate_decode_pending(output,state):
    return not (Path(output)/"BALL_PROBE/float_rgb.pt").is_file() and not rgb_was_committed(output,state)


def model_work_remaining(output):
    output = Path(output)
    state = read_state(output)
    if state is None and output.exists() and any(output.iterdir()):
        raise OSError("nonempty output has no recoverable task state")
    if state is not None and state["status"] in FINISHED: return False
    return candidate_decode_pending(output,state)


def raw_q(rows): return prior.raw_q(rows)
def q_matches_raw(path,expected): return prior.q_matches_raw(path,expected)
def runtime_observation(): return prior.runtime_observation()
def copy_saved_file(source,target): return prior.copy_saved_file(source,target)


def reports_complete(output,frame_count):
    from PIL import Image
    output = Path(output)
    try:
        for name in method.VIEWS:
            directory = output/name
            expected_q = raw_q(resume.read_json(directory/"raw.json"))
            if not q_matches_raw(directory/"q.pt",expected_q): return False
            if not evaluation_complete(resume.read_json(directory/"metrics.json")): return False
            if not all((directory/f).is_file() for f in ("float_rgb.pt","terminal_latent.pt","q.pt")): return False
            for reference in method.VIEWS[:2]:
                row = resume.read_json(directory/("quality_vs_"+reference+".json"))
                if len(row["framewise"]) != frame_count or len(row["temporal_framewise"]) != frame_count-1: return False
                index = resume.read_json(directory/("frames_vs_"+reference+".json"))
                if not index["frames"]: return False
                for item in index["frames"]+index["crops"]:
                    with Image.open(directory/item["path"]) as image: image.verify()
        return len(resume.read_json(output/"comparison.json")["bits_32"]) == 32
    except (OSError,ValueError,TypeError,KeyError): return False


def refresh_reports(output,result,parameters):
    output = Path(output); values = {}
    for name in method.VIEWS:
        directory = output/name
        try: evaluation = resume.read_json(directory/"metrics.json")
        except (OSError,ValueError): evaluation = posthoc.missing_evaluation("missing_readout")
        values[name] = metric_values(evaluation)
        try: rows = resume.read_json(directory/"raw.json")
        except (OSError,ValueError): rows = []
        count = sum(value is not None for value in values[name]["correlations"]+values[name]["margins"]+[values[name]["state_gap"]])
        result["views"][name].update(**values[name],loss=loss_from_values(values[name]),observed_metrics=count,
            observed_windows=len(rows),observed_chips=sum(chip.get("q") is not None for row in rows
                for chip in row.get("state_chips",[])+row.get("payload_chips",[])))
        if count == 55: result["views"][name].update(status="SCORED",reason=None)
    predictions = result["steps"]["candidate"].get("fp32",{}).get("predictions",{})
    target_predictions = predictions.get("target_margins",{})
    table = []
    for index in range(32):
        margins = {name:values[name]["margins"][index] for name in method.VIEWS}
        start,protected,ball = margins["START_FINAL"],margins["PROTECTED"],margins["BALL_PROBE"]
        table.append(dict(index=index,fragment=index//8,bit=index%8,margins=margins,
            ball_minus_START=None if start is None or ball is None else ball-start,
            ball_minus_PROTECTED=None if protected is None or ball is None else ball-protected,
            predicted_endpoint_margin=target_predictions.get(f"bit_{index:02d}"),
            ball_minus_mu=None if ball is None else ball-method.MU))
    protections = {}
    for name,view in values.items():
        margins = view["margins"]
        if any(value is None for value in margins):
            protections[name] = dict(status="MISSING",barrier=None); continue
        protections[name] = dict(status="SCORED",barrier=method.barrier_value(margins,parameters),
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
        versus={reference:compare_values(values[reference],values["BALL_PROBE"]) for reference in method.VIEWS[:3]},
        states={name:dict(correlations=view["correlations"],state_gap=view["state_gap"]) for name,view in values.items()},
        positive_bits={name:sum(value is not None and value>0 for value in view["margins"]) for name,view in values.items()},
        protections=protections,quality_vs_BASE=quality,quality_references=list(method.VIEWS[:2]),
        predicted=predictions,ceiling=method.CEILING)))
    refresh_counts(result)


def run(config,output,*,loader=None,protocol=carrier.PUBLIC,latent_shape=LATENT_SHAPE,
        support=PUBLIC_LATENT_SUPPORT,source_identity=None):
    import torch
    output = Path(output)
    source = Path(config["inputs"]["protected_run"]).resolve()
    if output.resolve().is_relative_to(source):
        raise ValueError("independent output must be outside the protected source run")
    prior_state = read_state(output)
    if prior_state is None and output.exists() and any(output.iterdir()):
        raise OSError("nonempty output has no recoverable task state")
    output.mkdir(parents=True,exist_ok=True)
    result = prior_state or initial_result(config)
    result["completed_rgb_views"] = [name for name in method.VIEWS if rgb_was_committed(output,result,name)]
    attempt = str(time.time_ns())
    if prior_state is not None:
        resume.atomic_bytes(output/"attempts"/(attempt+"_prior.json"),resume.json_bytes(prior_state))
        result.setdefault("attempt_history",[]).append(dict(attempt_id=prior_state.get("attempt_id"),
            status=prior_state["status"],failures=copy.deepcopy(prior_state["failures"]),runtime=prior_state.get("runtime")))
    for row in result["call_history"]:
        if row["status"] == "RUNNING": row.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason="previous process ended before durable return")
    result.update(status="RUNNING",attempt_id=attempt,failures=[],source=source_identity or {},
                  runtime=runtime_observation(),started_at_unix=time.time(),resources={})
    journal = resume.Journal(output,views=method.VIEWS,steps=STEPS); model = None

    def flush(): refresh_counts(result); journal.save(result)
    def save_json(path,value): resume.atomic_bytes(path,resume.json_bytes(value))
    def failure(stage,exc): record_failure(output,result,stage,repr(exc)); flush()
    def load_tensor(path): return torch.load(path,map_location="cpu",weights_only=True)
    def call(name,operation):
        row = dict(id=attempt+"_"+str(len(result["call_history"])).zfill(4),attempt_id=attempt,
            unit=name,kind=result["model_calls"][name]["kind"],status="RUNNING",started_at_unix=time.time())
        result["call_history"].append(row); result["model_calls"][name]=row; flush()
        try: value=operation()
        except BaseException as exc:
            row.update(status="FAILED" if isinstance(exc,Exception) else "INTERRUPTED_COMPLETION_UNKNOWN",reason=repr(exc)); flush(); raise
        row.update(status="RETURNED",finished_at_unix=time.time()); flush(); return value
    def require_model():
        nonlocal model
        if model is None:
            if str(config.get("device","cuda")).startswith("cuda"): torch.cuda.reset_peak_memory_stats()
            model = call("load",lambda:(loader or generation.load_frozen_vae)(config,device=config.get("device","cuda")))
            WanPosteriorBackend(model)
            result["actual_model_execution"] = "REAL" if loader is None else "INJECTED_TEST_DOUBLE"
        return model
    def validate_gradient(gradient):
        tensor_contract(gradient,latent_shape)
        if torch.count_nonzero(gradient-masked(gradient,support)): raise ValueError("gradient outside adopted support")

    def fixed_inputs():
        path = output/"inputs/fixed_inputs.pt"
        if path.exists(): bundle=load_tensor(path)
        else:
            if result["call_history"] or (output/"candidate_unit.pt").exists():
                raise OSError("previous work exists but fixed input backup is missing")
            saved = load_tensor(source/"inputs/fixed_inputs.pt")
            gradients = list(saved["old_gradients"])
            for name in method.PROTECTIONS:
                unit = load_tensor(source/"units"/(name+".pt"))
                if unit["name"] != name or not torch.equal(unit["input_terminal"],saved["start"]):
                    raise ValueError("saved protection derivative uses a different scalar or numerical START")
                gradients.append(unit["gradient"])
            protected = load_tensor(source/"candidate_unit.pt")["terminal"]
            bundle = dict(base=saved["base"],start=saved["start"],protected=protected,
                gradients=gradients,reference=saved["reference"],barrier=saved["barrier"],
                protected_run=str(source))
            resume.atomic_tensor(path,bundle)
        for name in ("base","start","protected"): tensor_contract(bundle[name],latent_shape)
        support.validate_for(tuple(bundle["base"].shape))
        delta=bundle["start"].double()-bundle["base"].double()
        if not 0<float(delta.norm())<=method.CAP or torch.count_nonzero(delta-masked(delta,support)):
            raise ValueError("START must use the adopted support and original BASE ball")
        if len(bundle["gradients"]) != 10: raise ValueError("exactly ten saved gradients required")
        for gradient in bundle["gradients"]: validate_gradient(gradient)
        if method.barrier_parameters(bundle["reference"]) != bundle["barrier"]:
            raise ValueError("saved barrier coefficients do not describe numerical START")
        result["fixed_input"] = dict(backup="inputs/fixed_inputs.pt",protected_run=bundle["protected_run"],source_arm="CONT_FINAL")
        flush(); return bundle

    def decode(terminal):
        require_model(); original,original_decode=model.decoder.forward,model.decode
        def counted(*args,**kwargs):
            active=next(row for row in reversed(result["call_history"]) if row["unit"]=="BALL_PROBE" and row["attempt_id"]==attempt)
            chunks=active.setdefault("ordinary_chunks",dict(attempted=0,completed=0)); chunks["attempted"]+=1; flush()
            value=original(*args,**kwargs); chunks["completed"]+=1; flush(); return value
        model.decoder.forward=counted; model.decode=lambda *a,**k:call("BALL_PROBE",lambda:original_decode(*a,**k))
        try: return adapter.decode_normalized_latent(model,terminal.to(config.get("device","cuda"))).cpu()
        finally: model.decoder.forward,model.decode=original,original_decode

    def observe(name,terminal):
        directory=output/name
        if name != "BALL_PROBE":
            source_directory=source/name
            for filename in ("terminal_latent.pt","float_rgb.pt","raw.json","q.pt","metrics.json"):
                if not (directory/filename).exists():
                    if filename in ("terminal_latent.pt","float_rgb.pt") or (source_directory/filename).is_file():
                        copy_saved_file(source_directory/filename,directory/filename)
        if not (directory/"terminal_latent.pt").exists(): resume.atomic_tensor(directory/"terminal_latent.pt",terminal)
        if not torch.equal(load_tensor(directory/"terminal_latent.pt"),terminal):
            raise ValueError("saved RGB view is paired with a different numerical terminal")
        rgb_path=directory/"float_rgb.pt"
        if rgb_path.exists(): rgb=load_tensor(rgb_path)
        else:
            if name != "BALL_PROBE" or not candidate_decode_pending(output,result):
                raise OSError("missing completed RGB; no replacement decode for a saved view")
            rgb=decode(terminal); tensor_contract(rgb,protocol.video_shape,bounded_rgb=True); resume.atomic_tensor(rgb_path,rgb)
        tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
        marker=output/"rgb_commits"/(name+".json")
        if not marker.exists(): save_json(marker,dict(view=name,path=str(rgb_path.relative_to(output)),committed_at_unix=time.time(),attempt_id=attempt))
        if name not in result["completed_rgb_views"]: result["completed_rgb_views"].append(name)
        flush()
        try: rows=resume.read_json(directory/"raw.json"); validate_raw(rows)
        except (OSError,ValueError,TypeError,KeyError):
            rows=observer.read_rgb(rgb,key=KEY,rho=.5,protocol=protocol); save_json(directory/"raw.json",rows); flush()
        expected_q=raw_q(rows)
        if not q_matches_raw(directory/"q.pt",expected_q): resume.atomic_tensor(directory/"q.pt",expected_q); flush()
        try: evaluation=resume.read_json(directory/"metrics.json")
        except (OSError,ValueError): evaluation=None
        if not evaluation_complete(evaluation):
            save_json(directory/"metrics.json",observer.evaluate_saved(rows,key=KEY,message=bytes.fromhex("8001a55a"),rho=.5,protocol=protocol)); flush()
        score_marker=output/"scoring_commits"/(name+".json")
        if not score_marker.exists(): save_json(score_marker,dict(view=name,raw="raw.json",q="q.pt",metrics="metrics.json",committed_at_unix=time.time(),attempt_id=attempt))

    def quality_and_visuals():
        for reference_name in method.VIEWS[:2]:
            try:
                reference=load_tensor(output/reference_name/"float_rgb.pt"); tensor_contract(reference,protocol.video_shape,bounded_rgb=True)
            except Exception as exc: failure("quality_reference_"+reference_name,exc); continue
            reports={}
            for name in method.VIEWS:
                path=output/name/("quality_vs_"+reference_name+".json")
                try:
                    try: row=resume.read_json(path)
                    except (OSError,ValueError): row=None
                    if not isinstance(row,dict) or len(row.get("framewise",[]))!=len(reference) or len(row.get("temporal_framewise",[]))!=len(reference)-1:
                        rgb=reference if name==reference_name else load_tensor(output/name/"float_rgb.pt")
                        tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
                        row=visual.quality(rgb,reference,protocol); del rgb
                        row.update(reference=reference_name,target=name,no_quality_threshold=True); save_json(path,row); flush()
                    reports[name]=row
                except Exception as exc: failure(name+"_quality_vs_"+reference_name,exc)
            frames,crops=visual.selection(reports,len(reference),views=("REFERENCE",)+method.VIEWS)
            for name in method.VIEWS:
                try:
                    index_path=output/name/("frames_vs_"+reference_name+".json")
                    try: old=resume.read_json(index_path)
                    except (OSError,ValueError): old=None
                    if old is not None and old.get("frame_indices")==frames and old.get("crop_requests")==crops:
                        from PIL import Image
                        try:
                            for item in old["frames"]+old["crops"]:
                                with Image.open(output/name/item["path"]) as image: image.verify()
                            continue
                        except (OSError,ValueError,KeyError): pass
                    rgb=reference if name==reference_name else load_tensor(output/name/"float_rgb.pt")
                    rendered,crop_images=[],[]
                    for frame in frames:
                        rendered.append(visual.render_pair(output/name/("frames_vs_"+reference_name)/f"frame_{frame:03d}.png",name,frame,rgb[frame],reference[frame]))
                    for number,request in enumerate(crops):
                        frame,y,x=request["frame"],request["y"],request["x"]
                        box=(max(0,y-visual.CROP_RADIUS),min(rgb.shape[1],y+visual.CROP_RADIUS+1),
                             max(0,x-visual.CROP_RADIUS),min(rgb.shape[2],x+visual.CROP_RADIUS+1))
                        crop_images.append(visual.render_pair(output/name/("crops_vs_"+reference_name)/f"crop_{number:02d}_{frame:03d}.png",name,frame,rgb[frame],reference[frame],box=box))
                    save_json(index_path,dict(reference=reference_name,target=name,frame_indices=frames,crop_requests=crops,
                        frames=rendered,crops=crop_images,rgb_range=[0,1],residual_range=[-.25,.25],numeric_metrics_unclipped=True))
                    del rgb; flush()
                except Exception as exc: failure(name+"_visuals_vs_"+reference_name,exc)
            del reference; gc.collect()

    with journal.periodic():
        stage="inputs"
        try:
            flush(); c=config["carrier"]
            if (config.get("schema"),c["key"],c["message_hex"],c["rho"],c["cap"]) != (
                    "local-joint-ball-probe-v1",KEY,"8001a55a",.5,method.CAP):
                raise ValueError("adopted schema/key/message/rho and BASE radius cap 1 required")
            bundle=fixed_inputs(); candidate_path=output/"candidate_unit.pt"
            stage="solve"
            if candidate_path.exists(): candidate_unit=load_tensor(candidate_path)
            else:
                terminal,report=method.solve_candidate(bundle["gradients"],bundle["base"],bundle["start"],bundle["reference"])
                if terminal is None: raise RuntimeError("fixed ball candidate unresolved: "+repr(report))
                candidate_unit=dict(terminal=terminal,report=report); resume.atomic_tensor(candidate_path,candidate_unit)
                save_json(output/"candidate_commits/fixed_ball.json",dict(path="candidate_unit.pt",committed_at_unix=time.time(),attempt_id=attempt))
            terminal,report=candidate_unit["terminal"],candidate_unit["report"]
            tensor_contract(terminal,latent_shape)
            endpoint=terminal.double()-bundle["base"].double()
            if float(endpoint.norm())>method.CAP or torch.count_nonzero(endpoint-masked(endpoint,support)):
                raise ValueError("stored fixed candidate violates support or original BASE radius 1")
            actual=method.linear_predictions(bundle["gradients"],bundle["reference"],bundle["start"],terminal)
            if actual["max_normalized_guard_violation"]>method.RESIDUAL_TOL:
                raise ValueError("stored FP32 candidate violates a fixed linear protection constraint")
            report["fp32"]["predictions"]=actual; result["steps"]["solve"]=report; result["steps"]["candidate"]=report
            save_json(output/"solve.json",report); save_json(output/"candidate.json",report); flush()
            terminals=dict(BASE=bundle["base"],START_FINAL=bundle["start"],PROTECTED=bundle["protected"],BALL_PROBE=terminal)
            for name in method.VIEWS:
                stage=name; observe(name,terminals[name])
            stage="quality_and_reports"
            if not reports_complete(output,protocol.video_shape[0]): quality_and_visuals()
            else: result["last_resume_action"]="report_only; zero model calls"
            refresh_reports(output,result,bundle["barrier"])
            result["status"]="COMPLETE" if not result["failures"] and reports_complete(output,protocol.video_shape[0]) else "ENGINEERING_FAILURE"
        except BaseException as exc:
            result["status"]="ENGINEERING_FAILURE" if isinstance(exc,Exception) else "INTERRUPTED"
            if stage in result["views"]: result["views"][stage].update(status="FAILED",reason=repr(exc))
            failure(stage,exc)
            if not isinstance(exc,Exception): raise
        finally:
            model=None; gc.collect()
            if str(config.get("device","cuda")).startswith("cuda") and torch.cuda.is_initialized():
                result["resources"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                torch.cuda.empty_cache()
            result["finished_at_unix"]=time.time(); flush()
    return result
