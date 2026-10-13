"""Saved tangent direction, three ordinary decodes, durable complete-unit recovery."""
from __future__ import annotations

import copy
import gc
from pathlib import Path
import time

from main.tube_state import local_joint_tangent_common_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_state_payload_posthoc_v1 as posthoc
from main.tube_state import local_joint_terminal_bridge_v1 as observer
from runtime.wan import generation, vae as adapter
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from runtime.wan import local_joint_all_bit_quality_v1 as visual
from runtime.wan.local_joint_all_bit_v1 import validate_raw, evaluation_complete, compare_values, loss_from_values
from runtime.wan.local_joint_readout_m0_v1 import masked, metric_values
from runtime.wan.local_joint_state_payload_provider_v1 import PUBLIC_LATENT_SUPPORT, WanPosteriorBackend
from runtime.wan.local_joint_terminal_bridge_v1 import KEY, LATENT_SHAPE, tensor_contract

EXPECTED = dict(vae_load=1,decode=3,decoder_vjp=0,encode=0,dit=0,native=0,scheduler=0,codec=0,
                ordinary_chunks=138,gradient_forward_chunks=0,nominal_replay_chunks=0,
                windows=264,chips=4224,metrics=165)


def initial_result(config):
    calls = {"load":dict(kind="vae_load",status="NOT_RUN")}
    calls.update({name:dict(kind="decode",status="NOT_RUN") for name in method.VIEWS})
    return dict(schema="local-joint-tangent-common-v1",status="RUNNING",config=copy.deepcopy(config),
        views={name:dict(status="NOT_RUN",reason="not_run",expected_windows=88,expected_chips=1408,
                        expected_metrics=55,observed_windows=0,observed_chips=0,observed_metrics=0)
               for name in method.VIEWS},steps={"step":dict(status="NOT_RUN")},model_calls=calls,
        call_history=[],replay={},failures=[],failure_history=[],cleanup_errors=[],
        counts={**{kind+"_"+phase:0 for kind in ("vae_load","decode","decoder_vjp")
                   for phase in ("attempted","completed")},
                **{kind:0 for kind in ("encode","dit","native","scheduler","codec")}},
        ordinary_chunks=dict(attempted=0,completed=0),expected_cost=EXPECTED,
        ceiling=method.CEILING,scientific_pass=False,automatic_retry=False,
        actual_model_execution="NOT_STARTED")


def read_state(output):
    result = resume.read_state(output,views=method.VIEWS,steps=("step",))
    if result is None:
        return None
    failures = {row["id"]:row for row in result.get("failure_history",[])}
    for path in (Path(output)/"failures").glob("*.json"):
        try:
            row = resume.read_json(path)
            if not isinstance(row,dict) or not isinstance(row.get("id"),str):
                raise ValueError("failure sidecar must be an object with a string id")
            failures[row["id"]] = row
        except (OSError,ValueError,KeyError,TypeError) as exc:
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
    result["totals"].update(planned_source_points=1,independent_samples=1)


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
    journal = resume.Journal(output,views=method.VIEWS,steps=("step",))
    journal.save(result); journal.checkpoint("report-only child cleanup")
    return result


def reports_complete(output,frame_count=181):
    from PIL import Image
    output = Path(output)
    try:
        for name in method.VIEWS:
            directory = output/name
            validate_raw(resume.read_json(directory/"raw.json"))
            if not evaluation_complete(resume.read_json(directory/"metrics.json")): return False
            if not all((directory/file).is_file() for file in ("float_rgb.pt","terminal_latent.pt","q.pt")): return False
            for ref in method.VIEWS[:2]:
                row = resume.read_json(directory/("quality_vs_"+ref+".json"))
                if len(row["framewise"]) != frame_count or len(row["temporal_framewise"]) != frame_count-1: return False
                images = resume.read_json(directory/("frames_vs_"+ref+".json"))
                if not images["frames"] or not all((directory/item["path"]).is_file() for item in images["frames"]+images["crops"]): return False
                for item in images["frames"]+images["crops"]:
                    with Image.open(directory/item["path"]) as image: image.verify()
        return len(resume.read_json(output/"comparison.json")["bits_32"]) == 32
    except (OSError,ValueError,TypeError,KeyError):
        return False


def model_work_remaining(output):
    output = Path(output)
    prior = read_state(output)
    if prior is None and output.exists() and any(output.iterdir()):
        raise OSError("Nonempty output has no recoverable tangent state; refusing silent restart")
    # Scores alone are not permission to replace missing completed float RGB.
    return any(not (output/name/"float_rgb.pt").is_file() and
        not (output/name/"raw.json").is_file() for name in method.VIEWS)


def refresh_reports(output,result):
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
    comparisons = {ref:compare_values(values[ref],values["TANGENT"]) for ref in method.VIEWS[:2]}
    predicted = result["steps"]["step"].get("predicted_margin_changes",{})
    table = []
    for i in range(32):
        margins = {name:values[name]["margins"][i] for name in method.VIEWS}
        start,end = margins["START_FINAL"],margins["TANGENT"]
        table.append(dict(index=i,fragment=i//8,bit=i%8,margins=margins,
            predicted_change=predicted.get(str(i)),actual_change=None if start is None or end is None else end-start,
            prediction_scope="saved six-bit first derivative only" if i in method.BITS else "unmeasured"))
    resume.atomic_bytes(output/"comparison.json",resume.json_bytes(dict(bits_32=table,versus=comparisons,
        states={name:dict(correlations=v["correlations"],state_gap=v["state_gap"]) for name,v in values.items()},
        positive_bits={name:sum(x is not None and x>0 for x in v["margins"]) for name,v in values.items()},
        quality_references=list(method.VIEWS[:2]),ceiling=method.CEILING)))
    refresh_counts(result)


def run(config,output,*,loader=None,protocol=carrier.PUBLIC,latent_shape=LATENT_SHAPE,
        support=PUBLIC_LATENT_SUPPORT,source_identity=None):
    import torch
    output = Path(output)
    for name,location in config["inputs"].items():
        source_directory = Path(location).resolve() if name == "gradient_units" else Path(location).resolve().parent
        if output.resolve().is_relative_to(source_directory):
            raise ValueError("independent output must be outside source input directories")
    prior = read_state(output)
    if prior is None and output.exists() and any(output.iterdir()):
        raise OSError("Nonempty output has no recoverable tangent state; refusing silent restart")
    output.mkdir(parents=True,exist_ok=True)
    result = prior or initial_result(config)
    attempt = str(time.time_ns())
    if prior is not None:
        resume.atomic_bytes(output/"attempts"/(attempt+"_prior.json"),resume.json_bytes(prior))
        result.setdefault("attempt_history",[]).append(dict(attempt_id=prior.get("attempt_id"),
            status=prior["status"],failures=copy.deepcopy(prior["failures"])))
    for row in result["call_history"]:
        if row["status"] == "RUNNING": row.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason="previous process ended before durable return")
    result.update(status="RUNNING",attempt_id=attempt,failures=[],source=source_identity or {},
                  started_at_unix=time.time(),resources={})
    journal = resume.Journal(output,views=method.VIEWS,steps=("step",))
    model = None

    def flush(): refresh_counts(result); journal.save(result)
    def save_json(path,value): resume.atomic_bytes(path,resume.json_bytes(value))
    def failure(stage,exc): record_failure(output,result,stage,repr(exc)); flush()
    def call(name,operation):
        row = dict(id=attempt+"_"+str(len(result["call_history"])).zfill(4),attempt_id=attempt,
            unit=name,kind=result["model_calls"][name]["kind"],status="RUNNING",started_at_unix=time.time())
        result["call_history"].append(row); result["model_calls"][name] = row; flush()
        try: value = operation()
        except BaseException as exc:
            row.update(status="FAILED" if isinstance(exc,Exception) else "INTERRUPTED_COMPLETION_UNKNOWN",reason=repr(exc)); flush(); raise
        row.update(status="RETURNED",finished_at_unix=time.time()); flush()
        return value
    def load_tensor(path): return torch.load(path,map_location="cpu",weights_only=True)

    def fixed_inputs():
        path = output/"inputs/fixed_inputs.pt"
        if path.exists():
            bundle = load_tensor(path)
        else:
            if result["call_history"] or any((output/name/"float_rgb.pt").exists() for name in method.VIEWS):
                raise OSError("previous model work exists but durable fixed inputs are missing")
            source = config["inputs"]
            saved = load_tensor(source["six_bit_inputs"])
            bundle = dict(base=saved["base"],start=saved["final"],
                direction=load_tensor(source["directions"])["tangent_common"],source_locators=source,
                step_size=float(config["step_size"]))
            for name in ("base","start"): tensor_contract(bundle[name],latent_shape)
            terminal,report = method.candidate(bundle["base"],bundle["start"],bundle["direction"],bundle["step_size"])
            predictions, prediction_notes = {}, []
            for bit in method.BITS:
                try:
                    unit = load_tensor(Path(source["gradient_units"])/("bit_"+str(bit).zfill(2)+".pt"))
                    tensor_contract(unit["gradient"],latent_shape)
                    if unit["bit_index"] != bit or not torch.equal(unit["input_terminal"],bundle["start"]):
                        raise ValueError("saved derivative is paired with a different bit or numerical START")
                    if torch.count_nonzero(unit["gradient"]-masked(unit["gradient"],support)):
                        raise ValueError("saved gradient outside adopted support")
                    g = unit["gradient"].double()
                    predictions[str(bit)] = dict(slope=float((g*bundle["direction"]).sum()),
                        planned=float((g*(bundle["step_size"]*bundle["direction"])).sum()),
                        actual_terminal_increment=float((g*(terminal.double()-bundle["start"].double())).sum()))
                except Exception as exc:
                    prediction_notes.append(dict(bit=bit,status="UNAVAILABLE",reason=repr(exc)))
            bundle.update(candidate=terminal,step_report=dict(report,predicted_margin_changes=predictions,
                prediction_notes=prediction_notes,predictions_are_optional=True))
            validate_bundle(bundle)
            resume.atomic_tensor(path,bundle)
        validate_bundle(bundle)
        result["steps"]["step"] = bundle["step_report"]
        result["fixed_input"] = dict(backup="inputs/fixed_inputs.pt",source_locators=bundle["source_locators"],
                                    source_arm="CONT_FINAL",direction_field="tangent_common")
        save_json(output/"step.json",bundle["step_report"]); flush()
        return bundle

    def validate_bundle(bundle):
        for name in ("base","start","candidate"): tensor_contract(bundle[name],latent_shape)
        support.validate_for(tuple(bundle["base"].shape))
        for tensor in (bundle["start"].double()-bundle["base"].double(),bundle["direction"],
                       bundle["candidate"].double()-bundle["base"].double()):
            if torch.count_nonzero(tensor-masked(tensor,support)): raise ValueError("input or candidate outside adopted support")
        terminal,_ = method.candidate(bundle["base"],bundle["start"],bundle["direction"],bundle["step_size"])
        if not torch.equal(terminal,bundle["candidate"]): raise ValueError("saved candidate does not represent the saved fixed step")

    def decode(name,terminal):
        nonlocal model
        if model is None:
            if str(config.get("device","cuda")).startswith("cuda"): torch.cuda.reset_peak_memory_stats()
            model = call("load",lambda:(loader or generation.load_frozen_vae)(config,device=config.get("device","cuda")))
            WanPosteriorBackend(model)
            result["actual_model_execution"] = "REAL" if loader is None else "INJECTED_TEST_DOUBLE"
        original, original_decode = model.decoder.forward,model.decode
        def counted(*args,**kwargs):
            active = next(r for r in reversed(result["call_history"]) if r["unit"] == name and r["attempt_id"] == attempt)
            chunks = active.setdefault("ordinary_chunks",dict(attempted=0,completed=0))
            chunks["attempted"] += 1; flush()
            value = original(*args,**kwargs)
            chunks["completed"] += 1; flush()
            return value
        model.decoder.forward = counted
        model.decode = lambda *a,**k:call(name,lambda:original_decode(*a,**k))
        try: return adapter.decode_normalized_latent(model,terminal.to(config.get("device","cuda"))).cpu()
        finally: model.decoder.forward,model.decode = original,original_decode

    def observe(name,terminal):
        directory = output/name
        terminal_path = directory/"terminal_latent.pt"
        if terminal_path.exists():
            if not torch.equal(load_tensor(terminal_path),terminal): raise ValueError("saved view is paired with a different terminal")
        else: resume.atomic_tensor(terminal_path,terminal)
        path = directory/"float_rgb.pt"
        if path.exists():
            rgb = load_tensor(path)
        else:
            if (directory/"raw.json").exists(): raise OSError("completed observation has lost its float RGB; no replacement decode")
            rgb = decode(name,terminal)
            tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
            resume.atomic_tensor(path,rgb); flush()
        tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
        # Raw q is committed before message truth enters evaluation.
        try:
            rows = resume.read_json(directory/"raw.json"); validate_raw(rows)
        except (OSError,ValueError,KeyError,TypeError):
            rows = observer.read_rgb(rgb,key=KEY,rho=.5,protocol=protocol)
            save_json(directory/"raw.json",rows); flush(); validate_raw(rows)
        if not (directory/"q.pt").exists():
            resume.atomic_tensor(directory/"q.pt",torch.tensor([[c["q"] for c in row["state_chips"]+row["payload_chips"]]
                for row in rows],dtype=torch.float64).reshape(22,4,16))
        path = directory/"metrics.json"
        try: evaluation = resume.read_json(path)
        except (OSError,ValueError): evaluation = None
        if not evaluation_complete(evaluation):
            evaluation = observer.evaluate_saved(rows,key=KEY,message=bytes.fromhex("8001a55a"),rho=.5,protocol=protocol)
            save_json(path,evaluation)
        flush()
        del rgb

    def quality_and_visuals():
        # Retain only one reference and one target RGB at a time.
        for ref in method.VIEWS[:2]:
            try:
                reference = load_tensor(output/ref/"float_rgb.pt")
                tensor_contract(reference,protocol.video_shape,bounded_rgb=True)
            except Exception as exc:
                failure("quality_reference_"+ref,exc); continue
            reports = {}
            for name in method.VIEWS:
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
            frames,crops = visual.selection(reports,len(reference),views=("REFERENCE",)+method.VIEWS)
            for name in method.VIEWS:
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
            if (c["key"],c["message_hex"],c["rho"],c["cap"]) != (KEY,"8001a55a",.5,1.):
                raise ValueError("adopted key/message/rho/original BASE cap required")
            bundle = fixed_inputs()
            terminals = dict(BASE=bundle["base"],START_FINAL=bundle["start"],TANGENT=bundle["candidate"])
            if not reports_complete(output,protocol.video_shape[0]):
                for name in method.VIEWS:
                    stage = name
                    try: observe(name,terminals[name])
                    except Exception as exc:
                        result["views"][name].update(status="FAILED",reason=repr(exc)); failure(name,exc)
                stage = "quality_and_visuals"
                quality_and_visuals()
            else: result["last_resume_action"] = "report_only; zero model calls"
            refresh_reports(output,result)
            result["status"] = "COMPLETE" if reports_complete(output,protocol.video_shape[0]) and not result["failures"] else "ENGINEERING_FAILURE"
        except BaseException as exc:
            result["status"] = "ENGINEERING_FAILURE" if isinstance(exc,Exception) else "INTERRUPTED"
            failure(stage,exc)
            if not isinstance(exc,Exception): raise
        finally:
            model = None; gc.collect()
            if str(config.get("device","cuda")).startswith("cuda") and torch.cuda.is_initialized():
                result["resources"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                torch.cuda.empty_cache()
            result["finished_at_unix"] = time.time(); flush()
    return result
