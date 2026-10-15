"""Serial fresh same-point gradients and two ordinary decodes, durable unit resume."""
from __future__ import annotations

import copy
import gc
from pathlib import Path
import time

from main.tube_state import local_joint_same_point_dual_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_terminal_bridge_v1 as observer
from runtime.wan import generation, vae as adapter
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from runtime.wan import local_joint_all_bit_quality_v1 as visual
from runtime.wan import local_joint_posterior_bridge_v1 as semantics
from runtime.wan import local_joint_gradient_sum_v1 as saved
from runtime.wan import local_joint_protected_common_v1 as prior
from runtime.wan.local_joint_all_bit_v1 import validate_raw, evaluation_complete
from runtime.wan.local_joint_readout_m0_v1 import masked, metric_values
from runtime.wan.local_joint_readout_checkpoint_v1 import ReplayLedger, checkpoint_decode
from runtime.wan.local_joint_state_payload_provider_v1 import PUBLIC_LATENT_SUPPORT, WanPosteriorBackend
from runtime.wan.local_joint_terminal_bridge_v1 import KEY, LATENT_SHAPE, tensor_contract

EXPECTED = dict(vae_load=1, gradient_decode=2, decoder_vjp=2, ordinary_decode=2, decode=4,
    gradient_forward_chunks=92, ordinary_chunks=92, total_forward_chunks=184, nominal_replay_chunks=92,
    encode=0, dit=0, native=0, scheduler=0, codec=0, new_source_endpoints=2, independent_samples=0,
    new_windows=176, new_chips=2816, new_metrics=110, gradient_windows=176, gradient_chips=2816,
    gradient_metrics=110, reused_endpoint_views=2, historical_reference_views=1, quality_pairs=4,
    endpoint_windows=352, endpoint_chips=5632, endpoint_metrics=220)
FINISHED = ("COMPLETE", "COMPLETE_WITH_UNDEFINED_ARMS", "COMPLETE_WITH_MISSING_EVIDENCE")
UNDEFINED = ("UNDEFINED_ACTIVE_TIE", "ZERO_GRADIENT")


def tensor(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def write_json(path, value):
    resume.atomic_bytes(path, resume.json_bytes(value))


def slot():
    return dict(status="MISSING", reason="not_run", expected_windows=88, expected_chips=1408,
                expected_metrics=55, observed_windows=0, observed_chips=0, observed_metrics=0)


def initial_result(config):
    calls = {"load": dict(kind="vae_load", status="NOT_RUN")}
    for name in method.ARMS:
        for suffix, kind in (("_gradient_decode", "decode"), ("_vjp", "decoder_vjp"), ("_ordinary_decode", "decode")):
            calls[name+suffix] = dict(kind=kind, status="NOT_RUN")
    return dict(schema="local-joint-same-point-dual-v1", config=copy.deepcopy(config), status="RUNNING",
        units={name: slot() for name in method.ARMS}, steps={name: dict(status="NOT_RUN") for name in method.ARMS},
        views={name: slot() for name in method.VIEWS},
        quality={name+"_vs_"+ref: dict(status="MISSING", reason="not_run") for name in method.ARMS for ref in ("BASE","SUM_STEP")},
        model_calls=calls, call_history=[], replay={}, failures=[], failure_history=[], cleanup_errors=[],
        counts={kind+"_"+phase: 0 for kind in ("vae_load","decode","decoder_vjp") for phase in ("attempted","completed")},
        ordinary_chunks=dict(attempted=0,completed=0), expected_cost=EXPECTED, scientific_pass=False,
        actual_model_execution="NOT_STARTED", automatic_retry=False, ceiling=method.CEILING)


def read_state(output):
    value = resume.read_state(output, views=method.VIEWS, steps=method.ARMS)
    if value is not None:
        rows = {x["id"]:x for x in value.get("failure_history",[])}
        for path in (Path(output)/"failures").glob("*.json"):
            try:
                row=resume.read_json(path); rows[row["id"]]=row
            except (OSError,ValueError,TypeError,KeyError) as exc:
                value.setdefault("recovery_events",[]).append(dict(path=str(path),reason=repr(exc)))
        value["failure_history"]=list(rows.values())
    return value


def refresh_counts(result):
    resume.recount(result)
    own=[r for r in result["call_history"] if r["attempt_id"]==result.get("attempt_id")]
    result["current_attempt_counts"]={kind+"_"+phase:sum(r["kind"]==kind and (phase=="attempted" or r["status"]=="RETURNED") for r in own)
        for kind in ("vae_load","decode","decoder_vjp") for phase in ("attempted","completed")}
    for field, rows in (("physical_replay_chunks",result["call_history"]),("current_attempt_replay_chunks",own)):
        result[field]={phase:{outcome:sum(r.get("replay",{}).get("counts",{}).get("vae_chunk",{}).get(phase,{}).get(outcome,0) for r in rows)
            for outcome in ("attempted","completed")} for phase in ("forward","recompute")}
    result["current_attempt_ordinary_chunks"]={phase:sum(r.get("ordinary_chunks",{}).get(phase,0) for r in own) for phase in ("attempted","completed")}
    result["counts"].update({k:0 for k in ("encode","dit","native","scheduler","codec")})
    result["observation_accounting"]={}
    for label,names,collection in (("four_endpoints",("BASE","SUM_STEP")+method.ARMS,result["views"]),
            ("new_ordinary_endpoints",method.ARMS,result["views"]),
            ("repeated_S_gradient_forwards",method.ARMS,result["units"]),
            ("historical_saved_reference",("HISTORICAL_26",),result["views"])):
        result["observation_accounting"][label]={field:sum(collection[n].get(field,0) for n in names)
            for field in ("expected_windows","expected_chips","expected_metrics","observed_windows","observed_chips","observed_metrics")}


def record_failure(output,result,stage,reason):
    row=dict(id=str(time.time_ns()),attempt_id=result.get("attempt_id"),stage=stage,reason=reason)
    result["failures"].append(row);result["failure_history"].append(row)
    write_json(Path(output)/"failures"/(row["id"]+".json"),row)


def committed(output,result,kind,name):
    if (Path(output)/(kind+"_commits")/(name+".json")).exists():
        return True
    if kind=="unit":
        return result.get("units",{}).get(name,{}).get("status") in ("COMPLETE",)+UNDEFINED
    return result.get("views",{}).get(name,{}).get("status")=="SCORED"


def model_work_remaining(output):
    output=Path(output); state=read_state(output)
    if state is None:
        if output.exists() and any(output.iterdir()):
            raise OSError("nonempty output has no recoverable same-point task state")
        return True
    for name in method.ARMS:
        path=output/"units"/(name+".pt")
        if not path.exists():
            if not committed(output,state,"unit",name): return True
            continue
        unit=tensor(path)
        if unit.get("status") in UNDEFINED: continue
        if state.get("steps",{}).get(name,{}).get("status")=="ZERO_GRADIENT": continue
        if not (output/name/"float_rgb.pt").exists() and not committed(output,state,"rgb",name): return True
    return False


def finalize_interrupted(output, reason):
    result=read_state(output)
    if result is None or result["status"]!="RUNNING": return result
    result["status"]="INTERRUPTED"
    record_failure(output,result,"external_exit",reason)
    for row in result["call_history"]:
        if row["status"]=="RUNNING": row.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason=reason)
    refresh_counts(result)
    journal=resume.Journal(output,views=method.VIEWS,steps=method.ARMS)
    journal.save(result);journal.checkpoint("report-only child cleanup")
    return result


def values_from_report(report):
    if "payload" in report and "state" in report:
        return metric_values(report)
    return report.get("values",report)


def refresh_report(output,result):
    output=Path(output)
    report=dict(ceiling=method.CEILING,scientific_pass=False,views={},arms={},quality=result["quality"],
        historical_reference="HISTORICAL_26 is preserved evidence, not a same-S/same-step control")
    for name in method.VIEWS:
        row=result["views"][name]
        summary=row.get("summary") if row["status"]=="SCORED" else None
        status="SCORED" if summary else "FAILED" if row["status"]=="FAILED" else "MISSING"
        report["views"][name]=dict(status=row["status"],reason=row.get("reason"),summary=summary,
            payload_rows=[dict(index=i,status=status,margin=None if summary is None else summary["margins"][i]) for i in range(32)],
            state_rows=[dict(offset=i,status=status,correlation=None if summary is None else summary["correlations"][i]) for i in range(22)])
    for name in method.ARMS:
        current=report["views"][name]["summary"]; comparisons={}
        for ref in ("BASE","SUM_STEP"):
            before=report["views"][ref]["summary"]
            comparisons[ref]=dict(
                margins=[None if current is None or before is None else b-a for a,b in zip(before["margins"] if before else [None]*32,current["margins"] if current else [None]*32)],
                correlations=[None if current is None or before is None else b-a for a,b in zip(before["correlations"] if before else [None]*22,current["correlations"] if current else [None]*22)],
                gained_positive=[] if current is None or before is None else [i for i in range(32) if before["margins"][i]<=0<current["margins"][i]],
                lost_positive=[] if current is None or before is None else [i for i in range(32) if current["margins"][i]<=0<before["margins"][i]],
                comparison_available=current is not None and before is not None)
        report["arms"][name]=dict(construction=result["steps"][name],gradient=result["units"][name],comparisons=comparisons)
    refresh_counts(result);report["observation_accounting"]=result["observation_accounting"]
    write_json(output/"comparison.json",report)
    return report


def run(config,output,*,loader=None,protocol=carrier.PUBLIC,latent_shape=LATENT_SHAPE,support=PUBLIC_LATENT_SUPPORT,source_identity=None):
    import torch
    output=Path(output); source=Path(config["inputs"]["sum_run"])
    if output.resolve()==source.resolve() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("output must be separate from the completed source run")
    previous=read_state(output)
    if previous is None and output.exists() and any(output.iterdir()):
        raise OSError("nonempty output has no recoverable task state")
    output.mkdir(parents=True,exist_ok=True)
    result=previous or initial_result(config); attempt=str(time.time_ns())
    if previous:
        write_json(output/"attempts"/(attempt+"_prior.json"),previous)
    for row in result["call_history"]:
        if row["status"]=="RUNNING": row.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason="previous process ended before durable return")
    result.update(status="RUNNING",attempt_id=attempt,failures=[],source=source_identity or {},started_at_unix=time.time())
    journal=resume.Journal(output,views=method.VIEWS,steps=method.ARMS)
    model=None; load_failed=False

    def flush():
        refresh_counts(result);journal.save(result)

    def failure(stage,exc):
        record_failure(output,result,stage,repr(exc));flush()

    def commit(kind,name):
        write_json(output/(kind+"_commits")/(name+".json"),dict(name=name,attempt_id=attempt,committed_at_unix=time.time()))
        flush()

    def call(name,operation):
        row=dict(id=attempt+"_"+str(len(result["call_history"])).zfill(4),attempt_id=attempt,unit=name,
                 kind=result["model_calls"][name]["kind"],status="RUNNING",started_at_unix=time.time())
        result["call_history"].append(row);flush()
        try: value=operation()
        except BaseException as exc:
            row.update(status="FAILED" if isinstance(exc,Exception) else "INTERRUPTED_COMPLETION_UNKNOWN",reason=repr(exc));flush();raise
        row.update(status="RETURNED",finished_at_unix=time.time());flush();return value

    def require_model():
        nonlocal model,load_failed
        if load_failed: raise RuntimeError("shared model load failed in this attempt; no automatic retry")
        if model is None:
            try:
                model=call("load",lambda:(loader or generation.load_frozen_vae)(config,device=config.get("device","cuda")))
                WanPosteriorBackend(model)
            except BaseException:
                load_failed=True;raise
            result["actual_model_execution"]="REAL" if loader is None else "INJECTED_TEST_DOUBLE"
        return model

    def inputs():
        path=output/"inputs/fixed_inputs.pt"
        if path.exists(): bundle=tensor(path)
        else:
            if result["call_history"]: raise OSError("fixed numerical input backup missing after physical work; no replacement source")
            bundle=dict(base=tensor(source/"BASE/terminal_latent.pt"),start=tensor(source/"candidate_unit.pt")["terminal"],
                        source_run=str(source), historical_readout=config["inputs"]["historical_allbit_metrics"])
            for key in ("base","start"): tensor_contract(bundle[key],latent_shape)
            method.construct("NEXT_MEAN",bundle["base"],bundle["start"],torch.zeros_like(bundle["base"]),support)
            resume.atomic_tensor(path,bundle)
        for key in ("base","start"): tensor_contract(bundle[key],latent_shape)
        method.construct("NEXT_MEAN",bundle["base"],bundle["start"],torch.zeros_like(bundle["base"]),support)
        return bundle

    def references(bundle):
        src=Path(bundle["source_run"])
        for name in method.REFERENCES:
            path=output/name/"readout.json"
            try:
                report=resume.read_json(path if path.exists() else (Path(bundle["historical_readout"]) if name=="HISTORICAL_26" else src/name/"readout.json"))
                values=values_from_report(report);summary=method.summarize(values)
                write_json(path,dict(values=values,source="saved reference; no new model call"))
                result["views"][name].update(status="SCORED",reason=None,summary=summary,observed_windows=88,observed_chips=1408,observed_metrics=55)
            except (OSError,ValueError,TypeError,KeyError) as exc:
                result["views"][name].update(status="MISSING",reason=repr(exc))
            flush()

    def measure(name,bundle):
        require_model()
        def attach(summary):
            result["replay"][name]=summary
            for row in reversed(result["call_history"]):
                if row["unit"]==name+"_gradient_decode" and row["attempt_id"]==attempt:
                    row["replay"]=copy.deepcopy(summary);break
        ledger=ReplayLedger(callback=lambda summary:(attach(summary),flush()))
        z=bundle["start"].to(config.get("device","cuda")).clone().requires_grad_(True)
        original_decode=model.decode
        try:
            adapter._clear_cache(model)
            model.decode=lambda *args,**kwargs:call(name+"_gradient_decode",lambda:original_decode(*args,**kwargs))
            mean,std=adapter._scale_tensors(model,z)
            decoded=checkpoint_decode(model,z*std+mean,ledger)
            rgb=(decoded[0].permute(1,2,3,0).float()/2+.5).clamp(0,1)
            tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
            read=method.tensor_readout(rgb,key=KEY,message=bytes.fromhex("8001a55a"),protocol=protocol)
            q=read["q"].detach().cpu();report=method.read_report(name,read)
            resume.atomic_tensor(output/"observations"/(name+"_q.pt"),q)
            write_json(output/"observations"/(name+"_readout.json"),report)
            result["units"][name].update(readout=report,observed_windows=88,observed_chips=1408,observed_metrics=55);flush()
            unit=dict(name=name,status="COMPLETE",input_terminal=bundle["start"],q=q,readout=report)
            try: objective=method.scalar(name,read)
            except method.UndefinedActiveTie as exc:
                unit.update(status="UNDEFINED_ACTIVE_TIE",reason=str(exc));return unit
            raw=call(name+"_vjp",lambda:torch.autograd.grad(objective,z)[0]).detach()
            tensor_contract(raw,latent_shape)
            gradient=masked(raw,support).cpu()
            unit.update(raw_gradient=raw.cpu(),gradient=gradient,gradient_normalized=False,
                raw_gradient_norm=float(raw.double().norm()),masked_gradient_norm=float(gradient.double().norm()),independent_graph=True)
            if unit["masked_gradient_norm"]==0: unit["status"]="ZERO_GRADIENT"
            return unit
        finally:
            model.decode=original_decode
            try: adapter._clear_cache(model)
            finally:
                ledger.callback=None;ledger.release_boundary_storage();attach(ledger.summary());flush()

    def ordinary(name,terminal):
        require_model();original=model.decoder.forward;original_decode=model.decode
        def counted(*args,**kwargs):
            row=next(r for r in reversed(result["call_history"]) if r["unit"]==name+"_ordinary_decode" and r["attempt_id"]==attempt)
            chunks=row.setdefault("ordinary_chunks",dict(attempted=0,completed=0))
            chunks["attempted"]+=1;flush()
            value=original(*args,**kwargs);chunks["completed"]+=1;flush();return value
        model.decoder.forward=counted
        model.decode=lambda *args,**kwargs:call(name+"_ordinary_decode",lambda:original_decode(*args,**kwargs))
        try: return adapter.decode_normalized_latent(model,terminal.to(config.get("device","cuda"))).cpu()
        finally: model.decoder.forward,model.decode=original,original_decode

    def observe(name,terminal):
        directory=output/name;path=directory/"float_rgb.pt"
        if committed(output,result,"scoring",name):
            try:
                raw=resume.read_json(directory/"raw.json");validate_raw(raw)
                q=tensor(directory/"q.pt")
                if not torch.equal(q.double(),prior.raw_q(raw)):
                    raise ValueError("saved q does not numerically match saved raw observations")
                metrics=resume.read_json(directory/"metrics.json")
                if not evaluation_complete(metrics): raise ValueError("saved metrics lack complete fixed rows")
                values=values_from_report(resume.read_json(directory/"readout.json"))
                summary=method.summarize(values)
                stored_metrics=method.summarize(metric_values(metrics))
                for field in ("margins","correlations"):
                    if any(abs(a-b)>1e-14 for a,b in zip(summary[field],stored_metrics[field])):
                        raise ValueError("saved readout and saved metrics numerically disagree")
            except (OSError,ValueError,TypeError,KeyError,RuntimeError,AttributeError) as exc:
                raise OSError("committed scoring missing/invalid; no replacement scoring: "+str(exc)) from exc
            result["views"][name].update(status="SCORED",reason=None,summary=summary,
                observed_windows=88,observed_chips=1408,observed_metrics=55)
            flush()
            return
        if path.exists(): rgb=tensor(path)
        else:
            if committed(output,result,"rgb",name): raise OSError("committed RGB missing; no replacement decode")
            rgb=ordinary(name,terminal);tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
            resume.atomic_tensor(path,rgb)
        tensor_contract(rgb,protocol.video_shape,bounded_rgb=True);commit("rgb",name)
        try:
            raw=resume.read_json(directory/"raw.json");validate_raw(raw)
        except (OSError,ValueError,TypeError,KeyError):
            raw=observer.read_rgb(rgb,key=KEY,rho=.5,protocol=protocol)
            write_json(directory/"raw.json",raw);flush()
        q=prior.raw_q(raw);resume.atomic_tensor(directory/"q.pt",q)
        metrics=observer.evaluate_saved(raw,key=KEY,message=bytes.fromhex("8001a55a"),rho=.5,protocol=protocol)
        write_json(directory/"metrics.json",metrics)
        values=metric_values(metrics);summary=method.summarize(values)
        write_json(directory/"readout.json",dict(values=values,source="ordinary saved RGB; raw committed before truth scoring"))
        result["views"][name].update(status="SCORED",reason=None,summary=summary,observed_windows=88,observed_chips=1408,observed_metrics=55)
        commit("scoring",name)

    def quality_pair(name,ref,bundle):
        key=name+"_vs_"+ref;directory=output/"quality"/key
        try:
            row=resume.read_json(directory/"quality.json");index=resume.read_json(directory/"frames.json")
            from PIL import Image
            frames,crops=visual.selection({name:row},protocol.video_shape[0],views=(ref,name))
            if saved.quality_complete(row,protocol.video_shape[0]) and saved.visual_index_complete(index,frames,crops):
                for item in index["frames"]+index["crops"]:
                    with Image.open(directory/item["path"]) as image: image.verify()
                result["quality"][key]=dict(status="SCORED",report=str((directory/"quality.json").relative_to(output)),reused=True);return
        except (OSError,ValueError,TypeError,KeyError): pass
        try:
            rgb=tensor(output/name/"float_rgb.pt");reference=tensor(Path(bundle["source_run"])/ref/"float_rgb.pt")
            for value in (rgb,reference): tensor_contract(value,protocol.video_shape,bounded_rgb=True)
            row=visual.quality(rgb,reference,protocol);row.update(reference=ref,target=name,no_quality_threshold=True)
            semantics.add_rgb_semantics(row);write_json(directory/"quality.json",row)
            frames,crops=visual.selection({name:row},len(rgb),views=(ref,name))
            images=[visual.render_pair(directory/"frames"/f"frame_{i:03d}.png",key,i,rgb[i],reference[i]) for i in frames]
            crop_images=[]
            for n,request in enumerate(crops):
                i,y,x=request["frame"],request["y"],request["x"]
                box=(max(0,y-visual.CROP_RADIUS),min(rgb.shape[1],y+visual.CROP_RADIUS+1),max(0,x-visual.CROP_RADIUS),min(rgb.shape[2],x+visual.CROP_RADIUS+1))
                crop_images.append(visual.render_pair(directory/"crops"/f"crop_{n:02d}_{i:03d}.png",key,i,rgb[i],reference[i],box=box))
            write_json(directory/"frames.json",dict(status="SAVED",frame_indices=frames,crop_requests=crops,frames=images,crops=crop_images))
            result["quality"][key]=dict(status="SCORED",report=str((directory/"quality.json").relative_to(output)),reused=False)
        except (OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:
            result["quality"][key]=dict(status="MISSING",reason=repr(exc),replacement_decode=False)
        flush()

    stage="config"
    with journal.periodic():
        try:
            flush()
            c=config["carrier"]
            if (config["schema"],c["key"],c["message_hex"],c["rho"],c["cap"],config["step"],config["arms"]) != (
                "local-joint-same-point-dual-v1",KEY,"8001a55a",.5,1.,.1,list(method.ARMS)):
                raise ValueError("adopted dual arm/key/message/rho/step/original cap semantics required")
            stage="inputs";bundle=inputs();references(bundle)
            for name in method.ARMS:
                stage=name+"_gradient"
                try:
                    path=output/"units"/(name+".pt")
                    if path.exists(): unit=tensor(path)
                    else:
                        if committed(output,result,"unit",name): raise OSError("committed gradient missing; no replacement VJP")
                        unit=measure(name,bundle)
                        resume.atomic_tensor(path,unit)
                    if unit.get("name")!=name or not torch.equal(unit["input_terminal"],bundle["start"]):
                        raise ValueError("saved arm gradient is not paired with this numerical S")
                    if unit["status"] not in ("COMPLETE",)+UNDEFINED: raise ValueError("invalid saved gradient unit status")
                    if tuple(unit["q"].shape)!=(22,4,16) or not bool(unit["q"].isfinite().all()):
                        raise ValueError("saved gradient observation must retain finite 88-window q")
                    summary=method.summarize(unit["readout"]["values"])
                    if unit["status"]=="UNDEFINED_ACTIVE_TIE":
                        tied=len(summary["actual_max_wrong_offsets"])>1
                        if not tied or (name=="NEXT_ALLBIT" and not summary["allbit_loss"]["state_active"]):
                            raise ValueError("saved undefined tie does not match this arm's actual objective")
                    else:
                        tensor_contract(unit["raw_gradient"],latent_shape);tensor_contract(unit["gradient"],latent_shape)
                        if torch.count_nonzero(unit["gradient"]-masked(unit["raw_gradient"],support)):
                            raise ValueError("saved gradient mask mismatch")
                        if (unit["status"]=="ZERO_GRADIENT") != (float(unit["gradient"].double().norm())==0):
                            raise ValueError("saved zero-gradient status disagrees with actual gradient")
                    result["units"][name].update(status=unit["status"],reason=unit.get("reason"),readout=unit["readout"],
                        observed_windows=88,observed_chips=1408,observed_metrics=55)
                    commit("unit",name)
                    if unit["status"] in UNDEFINED:
                        result["steps"][name]=dict(status=unit["status"],reason=unit.get("reason"));result["views"][name].update(status="MISSING",reason=unit["status"]);continue
                    terminal,construction=method.construct(name,bundle["base"],bundle["start"],unit["gradient"],support)
                    if terminal is None: raise ValueError("zero gradient must retain the ZERO_GRADIENT unit status")
                    candidate_path=output/name/"terminal_latent.pt"
                    if candidate_path.exists():
                        old=tensor(candidate_path);tensor_contract(old,latent_shape)
                        if not torch.equal(old,terminal): raise ValueError("saved candidate violates its fixed numerical formula")
                    else: resume.atomic_tensor(candidate_path,terminal)
                    result["steps"][name]=construction;commit("candidate",name)
                except Exception as exc:
                    result["units"][name]["status"]="FAILED";result["views"][name].update(status="MISSING",reason=repr(exc));failure(stage,exc)
                finally:
                    gc.collect()
                    if str(config.get("device","cuda")).startswith("cuda") and torch.cuda.is_initialized(): torch.cuda.empty_cache()
            for name in method.ARMS:
                if result["units"][name]["status"]!="COMPLETE": continue
                stage=name+"_ordinary"
                try: observe(name,tensor(output/name/"terminal_latent.pt"))
                except Exception as exc:
                    result["views"][name].update(status="FAILED",reason=repr(exc));failure(stage,exc)
            stage="quality"
            for name in method.ARMS:
                for ref in ("BASE","SUM_STEP"):
                    if result["views"][name]["status"]=="SCORED": quality_pair(name,ref,bundle)
                    else: result["quality"][name+"_vs_"+ref]=dict(status="MISSING",reason=result["views"][name].get("reason"))
            undefined=any(result["units"][n]["status"] in UNDEFINED for n in method.ARMS)
            missing=any(result["views"][n]["status"]!="SCORED" for n in method.REFERENCES) or any(r["status"]!="SCORED" for r in result["quality"].values())
            result["status"]="ENGINEERING_FAILURE" if result["failures"] else "COMPLETE_WITH_UNDEFINED_ARMS" if undefined else "COMPLETE_WITH_MISSING_EVIDENCE" if missing else "COMPLETE"
        except BaseException as exc:
            result["status"]="ENGINEERING_FAILURE" if isinstance(exc,Exception) else "INTERRUPTED"
            failure(stage,exc)
            if not isinstance(exc,Exception): raise
        finally:
            if model is not None:
                try: adapter._clear_cache(model)
                except Exception as exc: result["cleanup_errors"].append(repr(exc))
                model=None
            gc.collect();result["finished_at_unix"]=time.time();refresh_report(output,result);flush()
    return result
