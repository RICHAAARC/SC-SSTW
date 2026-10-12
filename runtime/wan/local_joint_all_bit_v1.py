"""Adopted all-bit: four fixed views, six decode calls, two fresh scalar VJPs."""
from __future__ import annotations

import gc
import copy
import json
from pathlib import Path
import time

from main.tube_state import local_joint_all_bit_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_state_payload_posthoc_v1 as posthoc
from main.tube_state import local_joint_terminal_bridge_v1 as observer
from runtime.wan import generation, vae as adapter
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from runtime.wan.local_joint_readout_checkpoint_v1 import ReplayLedger, checkpoint_decode
from runtime.wan.local_joint_readout_m0_v1 import masked, metric_values
from runtime.wan.local_joint_all_bit_quality_v1 import quality, export_visuals
from runtime.wan.local_joint_state_payload_provider_v1 import PUBLIC_LATENT_SUPPORT, WanPosteriorBackend
from runtime.wan.local_joint_terminal_bridge_v1 import KEY, LATENT_SHAPE, tensor_contract, write_json

CEILING = ("One seen saved JOINT terminal; 32-bit soft margin deficit, two fresh half steps, "
           "total BASE-relative L2<=1. Lower loss does not guarantee bit coverage, state floor, "
           "quality or original descriptive condition. No trajectory/CFG/native causality, blind "
           "decoding, codec, FPR or scientific PASS. No COMMON refresh or M1/M2.")
EXPECTED = dict(vae_load=1, decode=6, decoder_vjp=2, encode=0, dit=0, native=0, scheduler=0, codec=0,
                ordinary_chunks=184, gradient_forward_chunks=92, nominal_replay_chunks=92,
                windows=352, chips=5632, metrics=220)


def save_tensor(path, tensor):
    resume.atomic_tensor(path, tensor.detach().cpu())


def initial_result(config):
    calls = {"load": dict(kind="vae_load", status="NOT_RUN")}
    calls.update({name: dict(kind="decode", status="NOT_RUN") for name in method.VIEWS})
    for k in (1,2):
        calls[f"step_{k}_gradient_decode"] = dict(kind="decode", status="NOT_RUN")
        calls[f"step_{k}_vjp"] = dict(kind="decoder_vjp", status="NOT_RUN")
    return dict(schema="local-joint-all-bit-v1", status="RUNNING", config=config,
        ceiling=CEILING, scientific_pass=False, expected_cost=EXPECTED, automatic_retry=False,
        counts={**{f"{k}_{phase}": 0 for k in ("vae_load", "decode", "decoder_vjp")
                   for phase in ("attempted", "completed")}, "encode":0, "dit":0, "native":0, "scheduler":0, "codec":0},
        ordinary_chunks=dict(attempted=0, completed=0), replay={}, model_calls=calls,
        views={name: dict(status="MISSING", reason="not_run", expected_windows=88,
                         direction_status="UNDEFINED" if name in method.VIEWS[2:] else "FIXED_INPUT",
                         expected_chips=1408, expected_metrics=55, observed_windows=0,
                         observed_chips=0, observed_metrics=0) for name in method.VIEWS},
        steps={f"step_{k}":dict(status="NOT_RUN") for k in (1,2)}, gradient_readouts={},
        failures=[], cleanup_errors=[], actual_model_execution="NOT_STARTED",
        direction_status="NOT_RUN", quality_frames=list(method.FRAMES))


def loss_from_values(values):
    if any(x is None for x in values["margins"]+[values["state_gap"]]):
        return None
    return method.loss_summary(values["margins"],values["state_gap"],values["active"]["other_offsets"])


def compare_values(reference, target):
    def changes(field):
        return [None if a is None or b is None else b-a
                for a,b in zip(reference[field],target[field])]
    before, after = reference["margins"], target["margins"]
    return dict(objective_change=changes("objectives"), margin_change=changes("margins"),
        correlation_change=changes("correlations"),
        gained_bits=[dict(fragment=i//8, bit=i%8) for i,(a,b) in enumerate(zip(before,after))
                     if a is not None and b is not None and a <= 0 < b],
        lost_bits=[dict(fragment=i//8, bit=i%8) for i,(a,b) in enumerate(zip(before,after))
                   if a is not None and b is not None and b <= 0 < a],
        positive_bits=None if any(x is None for x in after) else sum(x>0 for x in after),
        nonpositive_bits=None if any(x is None for x in after) else
            [dict(fragment=i//8,bit=i%8,remaining_gap=-x) for i,x in enumerate(after) if x<=0],
        active_change=None if any(x is None for x in reference["objectives"]+target["objectives"])
                      else reference["active"] != target["active"])


def refresh_reports(output, result):
    """Durable raw/metrics only: no pixel loading, rescoring, VJP or model calls."""
    output = Path(output)
    metrics,losses = {},{}
    for name in method.VIEWS:
        path = output/name/"metrics.json"
        try:
            evaluation = resume.read_json(path)
            metric_values(evaluation)
        except (OSError,ValueError,KeyError,TypeError,AttributeError):
            evaluation = posthoc.missing_evaluation("unreadable_or_missing_metrics")
        values = metrics[name] = metric_values(evaluation)
        losses[name] = loss_from_values(values)
        raw_path = output/name/"raw.json"
        try:
            rows = resume.read_json(raw_path)
            if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
                raise ValueError("raw rows must be objects")
        except (OSError,ValueError,TypeError):
            rows = []
        count = sum(x is not None for x in values["correlations"]+values["margins"]+[values["state_gap"]])
        view = result["views"][name]
        view.update(observed_windows=len(rows), observed_metrics=count, loss=losses[name],
            observed_chips=sum(c.get("q") is not None for r in rows for c in r.get("state_chips", [])+r.get("payload_chips", [])),
            artifacts={label:"SAVED" if (output/name/file).exists() else "MISSING"
                       for label,file in (("quality","quality.json"),("frames","frames.json"),
                                          ("float_rgb","float_rgb.pt"),("terminal","terminal_latent.pt"),("delta","delta.pt"))},
            **values)
        if count == 55:
            view.update(status="SCORED", reason=None)
        elif view["status"] == "SCORED":
            view.update(status="MISSING",reason="unreadable_or_incomplete_metrics")
    def compare(a,b):
        answer=compare_values(metrics[a],metrics[b])
        answer.update(reference=a,target=b,
            actual_loss_change=None if losses[a] is None or losses[b] is None else losses[b]["loss"]-losses[a]["loss"],
            reference_summary=losses[a],target_summary=losses[b])
        return answer
    step_changes={}
    for k,(a,b) in enumerate((("BASE","ALLBIT_HALF"),("ALLBIT_HALF","ALLBIT_TWO")),1):
        row=compare(a,b)
        row["predicted_loss_change"]=result["steps"][f"step_{k}"].get("predicted_loss_change")
        row["prediction_meaning"]="one scalar loss slope; not 32 component slopes"
        step_changes[f"step_{k}"]=row
    table=[]
    for i in range(32):
        margins={name:metrics[name]["margins"][i] for name in method.VIEWS}
        def change(a,b):
            return None if margins[a] is None or margins[b] is None else margins[b]-margins[a]
        table.append(dict(fragment=i//8,bit=i%8,margins=margins,
            versus_base={name:change("BASE",name) for name in method.VIEWS[1:]},
            versus_saved_mean={name:change("TWO_MEAN_SAVED",name) for name in ("ALLBIT_HALF","ALLBIT_TWO")},
            half_to_two=change("ALLBIT_HALF","ALLBIT_TWO"),
            nonpositive_deficits={name:None if x is None else max(0.,-x) for name,x in margins.items()}))
    write_json(output/"comparison.json",dict(metrics=metrics,losses=losses,
        versus_base={name:compare("BASE",name) for name in method.VIEWS[1:]},
        primary_comparison=compare("TWO_MEAN_SAVED","ALLBIT_TWO"),
        half_to_two=compare("ALLBIT_HALF","ALLBIT_TWO"),steps=step_changes,bits_32=table,ceiling=CEILING))
    result["totals"] = {field:sum(x[field] for x in result["views"].values())
                       for field in ("expected_windows","expected_chips","expected_metrics",
                                     "observed_windows","observed_chips","observed_metrics")}


def finalize_interrupted(output, reason):
    output = Path(output)
    result = resume.read_state(output)
    if result is None:
        return None
    if result["status"] != "RUNNING":
        return result
    result["status"] = "INTERRUPTED"
    result["failures"].append(dict(stage="external_termination",reason=reason))
    for row in result["model_calls"].values():
        if row["status"] == "RUNNING":
            row.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason=reason)
        elif row["status"] == "NOT_RUN":
            row.update(status="MISSING_DEPENDENCY",reason=reason)
    for name,row in result["views"].items():
        if row["status"] != "SCORED":
            row.update(status="MISSING",reason=reason)
            if not (output/name/"metrics.json").exists():
                write_json(output/name/"metrics.json",posthoc.missing_evaluation(reason))
    for row in result["steps"].values():
        if row["status"] == "NOT_RUN":
            row.update(status="MISSING_DEPENDENCY",reason=reason)
    refresh_reports(output,result)
    for row in result.get("call_history", []):
        if row["status"] == "RUNNING":
            row.update(status="INTERRUPTED_COMPLETION_UNKNOWN", reason=reason)
    resume.Journal(output).save(result)
    return result


def validate_raw(rows):
    import math
    if not isinstance(rows,list) or len(rows)!=88:
        raise ValueError("raw observation requires 88 windows")
    chips=[c for row in rows for c in row["state_chips"]+row["payload_chips"]]
    if len(chips)!=1408 or any(c["q"] is None or not math.isfinite(c["q"]) for c in chips):
        raise ValueError("raw observation requires 1408 finite chips")


def evaluation_complete(evaluation):
    import math
    try:
        values=metric_values(evaluation)
        return all(x is not None and math.isfinite(x) for x in values["correlations"]+values["margins"]+[values["state_gap"]])
    except (TypeError,KeyError,AttributeError):
        return False


def observation_saved(directory):
    directory=Path(directory)
    try:
        if evaluation_complete(resume.read_json(directory/"metrics.json")):
            return True
    except (OSError,ValueError):
        pass
    try:
        validate_raw(resume.read_json(directory/"raw.json"))
        return True
    except (OSError,ValueError,TypeError,KeyError):
        return False


def terminal_missing_views(output):
    """Method-defined terminal outcomes keep dependent slots missing permanently."""
    state=resume.read_state(output)
    if state is None:
        return set()
    terminal={"ZERO_GRADIENT","UNDEFINED_ACTIVE_STATE_MAX_TIE"}
    if state["steps"]["step_1"]["status"] in terminal:
        return {"ALLBIT_HALF","ALLBIT_TWO"}
    if state["steps"]["step_2"]["status"] in terminal:
        return {"ALLBIT_TWO"}
    return set()


def reports_complete(output):
    """Small-artifact check for report-only re-entry, without loading model tensors."""
    output=Path(output)
    try:
        skipped=terminal_missing_views(output)
        for name in method.VIEWS:
            if name in skipped:
                continue
            validate_raw(resume.read_json(output/name/"raw.json"))
            if not evaluation_complete(resume.read_json(output/name/"metrics.json")):
                return False
            quality_row=resume.read_json(output/name/"quality.json")
            if len(quality_row["framewise"])!=181 or len(quality_row["temporal_framewise"])!=180:
                return False
            images=resume.read_json(output/name/"frames.json")
            if not images["frames"] or not all((output/name/row["path"]).is_file() for row in images["frames"]+images["crops"]):
                return False
        return True
    except (OSError,ValueError,TypeError,KeyError):
        return False


def model_work_remaining(output):
    """Notebook dependency hint only. The runner validates actual cached units."""
    output=Path(output)
    if reports_complete(output):
        return False
    skipped=terminal_missing_views(output)
    if any(not (output/name/"float_rgb.pt").is_file() and not observation_saved(output/name)
           for name in method.VIEWS if name not in skipped):
        return True
    return any(not (output/(step+"_unit.pt")).is_file() and
               not (output/"gradients"/(step+"_checkpoint.pt")).is_file()
               for step,target in (("step_1","ALLBIT_HALF"),("step_2","ALLBIT_TWO"))
               if target not in skipped)


def run(config, output, **kwargs):
    journal = resume.Journal(output)
    with journal.periodic():
        return _run(config, output, journal=journal, **kwargs)


def _run(config, output, *, journal, loader=None, protocol=carrier.PUBLIC, latent_shape=LATENT_SHAPE,
        support=PUBLIC_LATENT_SUPPORT, source_identity=None):
    import torch

    output = Path(output)
    for input_path in config["inputs"].values():
        if output.resolve().is_relative_to(Path(input_path).resolve().parent):
            raise ValueError("fresh output must be outside saved input directories")
    output.mkdir(parents=True,exist_ok=True)
    prior = resume.read_state(output)
    result = prior or initial_result(config)
    # Preserve every physical call and failed process; successful recovery gets a
    # fresh current-attempt status without erasing the historical evidence.
    attempt = str(time.time_ns())
    if prior:
        resume.atomic_bytes(output/"attempts"/(attempt+"_prior.json"), resume.json_bytes(prior))
        result.setdefault("attempt_history", []).append(dict(
            status=prior["status"], failures=copy.deepcopy(prior["failures"]),
            resources=copy.deepcopy(prior.get("resources",{})),
            finished_at_unix=prior.get("finished_at_unix")))
    result.setdefault("call_history", [])
    for row in result["call_history"]:
        if row["status"] == "RUNNING":
            row.update(status="INTERRUPTED_COMPLETION_UNKNOWN", reason="previous process ended before durable return")
    result["failures"] = []
    result["status"] = "RUNNING"
    result["attempt_id"] = attempt
    result["resume_on_reentry"] = True
    result["resources"] = {}
    result["resources_scope"] = "current attempt; earlier attempts retained in attempts/"
    result["source"] = source_identity or {}
    result["started_at_unix"] = time.time()
    model, base_rgb = None, None
    saved = {}

    def flush():
        journal.save(result)

    def cached(path, validate=None, tensor=False):
        if not path.exists():
            return None
        try:
            value = torch.load(path,map_location="cpu",weights_only=True) if tensor else resume.read_json(path)
            if validate is not None:
                validate(value)
            return value
        except Exception as exc:
            result.setdefault("recovery_events", []).append(dict(path=str(path),reason=repr(exc)))
            return None

    def tensor_value(path, shape, bounded=False):
        return cached(path, lambda v:tensor_contract(v,shape,bounded_rgb=bounded), tensor=True)

    def require_model():
        nonlocal model
        if model is None:
            if str(config.get("device","cuda")).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            model=call("load",lambda:(loader or generation.load_frozen_vae)(config,device=config.get("device","cuda")))
            WanPosteriorBackend(model)
            result["actual_model_execution"]="REAL" if loader is None else "INJECTED_TEST_DOUBLE"
        return model

    def failure(stage,exc):
        result["failures"].append(dict(stage=stage,reason=f"{type(exc).__name__}: {exc}",retry=False))
        if stage in result["views"] and result["views"][stage]["status"] != "SCORED":
            result["views"][stage].update(status="MISSING",reason=repr(exc))
        flush()

    def call(ident,operation):
        row=dict(id=attempt+"_"+str(len(result["call_history"])).zfill(4), unit=ident,
                 kind=result["model_calls"][ident]["kind"],status="RUNNING",started_at_unix=time.time())
        result["call_history"].append(row)
        result["model_calls"][ident]=row
        kind=row["kind"]
        result["counts"][kind+"_attempted"]+=1;flush()
        try:
            value=operation()
        except BaseException as exc:
            row.update(status="FAILED" if isinstance(exc,Exception) else "INTERRUPTED_COMPLETION_UNKNOWN",reason=repr(exc));flush();raise
        row.update(status="RETURNED",finished_at_unix=time.time())
        result["counts"][kind+"_completed"]+=1;flush()
        return value

    def ordinary_decode(name,terminal):
        require_model()
        original=model.decoder.forward;original_decode=model.decode
        def counted(*args,**kwargs):
            chunks=result["model_calls"][name].setdefault("ordinary_chunks",dict(attempted=0,completed=0))
            chunks["attempted"]+=1
            result["ordinary_chunks"]["attempted"]+=1;flush()
            value=original(*args,**kwargs)
            chunks["completed"]+=1
            result["ordinary_chunks"]["completed"]+=1;flush()
            return value
        model.decoder.forward=counted
        model.decode=lambda *a,**k:call(name,lambda:original_decode(*a,**k))
        try:
            return adapter.decode_normalized_latent(model,terminal.to(config.get("device","cuda"))).cpu()
        finally:
            model.decoder.forward=original;model.decode=original_decode

    def observe(name,terminal,delta):
        nonlocal base_rgb
        directory=output/name
        saved_terminal=tensor_value(directory/"terminal_latent.pt",latent_shape)
        if saved_terminal is None or not torch.equal(saved_terminal,terminal):
            save_tensor(directory/"terminal_latent.pt",terminal)
        saved_delta=tensor_value(directory/"delta.pt",latent_shape)
        if saved_delta is None or not torch.equal(saved_delta,delta):
            save_tensor(directory/"delta.pt",delta)
        actual=terminal.double()-z0.double()
        result["views"][name]["displacement"]=dict(planned_delta_l2=float(delta.double().norm()),
            actual_terminal_minus_base_l2=float(actual.norm()),
            planned_mask_outside_l2=float((delta-masked(delta,support)).double().norm()),
            actual_mask_outside_l2=float((actual-masked(actual,support)).norm()),
            addition_rounding_l2=float((actual-delta.double()).norm()))
        rows=cached(directory/"raw.json",validate_raw)
        evaluation=cached(directory/"metrics.json")
        completed_observation=rows is not None or evaluation_complete(evaluation)
        rgb=tensor_value(directory/"float_rgb.pt",protocol.video_shape,True)
        if rgb is None and not completed_observation:
            rgb=ordinary_decode(name,terminal)
            tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
            save_tensor(directory/"float_rgb.pt",rgb);flush()
            # This is a new observation, so downstream data must come from this RGB.
            rows,evaluation=None,None
        if name=="BASE":
            base_rgb=rgb  # Remains usable for quality even if the BASE reader fails.
        if rows is None and rgb is not None:
            rows=observer.read_rgb(rgb,key=KEY,rho=.5,protocol=protocol)
            write_json(directory/"raw.json",rows);flush()
            evaluation=None
        if not evaluation_complete(evaluation) and rows is not None:
            evaluation=observer.evaluate_saved(rows,key=KEY,message=bytes.fromhex("8001a55a"),rho=.5,protocol=protocol)
            write_json(directory/"metrics.json",evaluation);flush()
        if rows is None:
            rows=[]
            failure(name+"_raw",OSError("raw unavailable; completed scores retained without a replacement decode"))
        values=metric_values(evaluation)
        count=sum(x is not None for x in values["correlations"]+values["margins"]+[values["state_gap"]])
        result["views"][name].update(status="SCORED" if count==55 else "MISSING",reason=None if count==55 else "incomplete_readout",
            observed_windows=len(rows),observed_metrics=count,
            observed_chips=sum(c["q"] is not None for r in rows for c in r["state_chips"]+r["payload_chips"]),**values,loss=loss_from_values(values))
        write_json(directory/"loss.json",result["views"][name]["loss"])
        try:
            existing_quality=cached(directory/"quality.json")
            if not isinstance(existing_quality,dict) or len(existing_quality.get("framewise",[]))!=protocol.video_shape[0] or len(existing_quality.get("temporal_framewise",[]))!=protocol.video_shape[0]-1:
                if rgb is None or base_rgb is None:
                    raise OSError("quality unavailable: saved RGB or BASE RGB is unreadable; completed observations retained without re-decode")
                write_json(directory/"quality.json",quality(rgb,base_rgb,protocol));flush()
        except Exception as exc:
            # Missing quality does not erase raw or gate the adopted method calls.
            failure(name+"_quality",exc)
        flush()
        return rgb

    def gradient(terminal,step,reference):
        checkpoint=output/"gradients"/(step+"_checkpoint.pt")
        def valid_bundle(bundle):
            tensor_contract(bundle["input_terminal"],latent_shape)
            tensor_contract(bundle["gradient"],latent_shape)
            if torch.count_nonzero(bundle["gradient"]-masked(bundle["gradient"],support)):
                raise ValueError("saved gradient lies outside adopted support")
            if not torch.equal(bundle["input_terminal"],terminal):
                raise ValueError("saved gradient belongs to a different step input")
        bundle=cached(checkpoint,valid_bundle,tensor=True)
        if bundle is not None:
            result["gradient_readouts"][step]=bundle["readout"]
            if not (output/"gradients"/(step+".pt")).exists():
                save_tensor(output/"gradients"/(step+".pt"),bundle["gradient"])
            return bundle["gradient"]
        require_model()
        def replay_progress(summary):
            result["replay"][step]=summary
            result["model_calls"][step+"_gradient_decode"]["replay"]=summary
            flush()
        ledger=ReplayLedger(callback=replay_progress)
        z=terminal.to(config.get("device","cuda")).clone().requires_grad_(True)
        adapter._clear_cache(model)
        original_decode=model.decode
        model.decode=lambda *a,**k:call(step+"_gradient_decode",lambda:original_decode(*a,**k))
        try:
            mean,std=adapter._scale_tensors(model,z)
            decoded=checkpoint_decode(model,z*std+mean,ledger)
            rgb=(decoded[0].permute(1,2,3,0).float()/2+.5).clamp(0,1)
            tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
            read=method.tensor_readout(rgb,key=KEY,message=bytes.fromhex("8001a55a"),protocol=protocol)
            save_tensor(output/"gradients"/(step+"_q.pt"),read["q"])
            values=dict(correlations=read["correlations"].detach().cpu().tolist(),
                state_gap=float(read["objectives"][0].detach()),
                margins=read["margins"].detach().cpu().reshape(-1).tolist(),
                objectives=read["objectives"].detach().cpu().tolist(),active=read["active"])
            ordinary=result["views"][reference]
            changes={field:[a-b for a,b in zip(values[field],ordinary[field])]
                     for field in ("correlations","margins")}
            changes["state_gap"]=values["state_gap"]-ordinary["state_gap"]
            difference55=changes["correlations"]+[changes["state_gap"]]+changes["margins"]
            report=dict(reference=reference,values=values,loss=method.read_summary(read),
                difference_from_ordinary=changes,max_abs_difference_55=max(abs(x) for x in difference55),
                active_change=values["active"]!=ordinary["active"],
                diagnostic_only=True,original_metric_count=55)
            result["gradient_readouts"][step]=report
            write_json(output/"gradients"/(step+"_readout.json"),report);flush()
            scalar=method.scalar_loss(read)
            g=call(step+"_vjp",lambda:torch.autograd.grad(scalar,z)[0])
            g=masked(g.detach(),support).cpu()
            if not bool(torch.isfinite(g).all()):
                raise FloatingPointError("nonfinite masked scalar gradient")
            resume.atomic_tensor(checkpoint,dict(input_terminal=terminal.detach().cpu(),gradient=g,
                readout=report))
            flush()
            save_tensor(output/"gradients"/(step+".pt"),g)
            return g
        finally:
            model.decode=original_decode
            try:
                adapter._clear_cache(model)
            finally:
                ledger.release_boundary_storage()

    try:
        for name in method.VIEWS:
            if not (output/name/"raw.json").exists():
                write_json(output/name/"raw.json",[])
            if not (output/name/"metrics.json").exists():
                write_json(output/name/"metrics.json",posthoc.missing_evaluation("not_run"))
        flush()
        c=config["carrier"]
        if (c["key"],c["message_hex"],c["rho"],c["cap"]) != (KEY,"8001a55a",.5,1.):
            raise ValueError("adopted key/message/rho/total cap required")
        if config.get("objective") != method.PARAMETERS:
            raise ValueError("adopted all-bit mu/gamma/weights/two-half-step parameters required")
        if prior is not None and reports_complete(output):
            result["status"]="COMPLETE"
            result["last_resume_action"]="report_only; no model or tensor work"
            return result
        z0=tensor_value(output/"inputs/terminal_latent.pt",latent_shape)
        if z0 is None:
            z0=torch.load(config["inputs"]["terminal_latent"],map_location="cpu",weights_only=True)
            tensor_contract(z0,latent_shape)
            save_tensor(output/"inputs/terminal_latent.pt",z0);flush()
        result["input_contract"]=dict(terminal=tensor_contract(z0,latent_shape))
        support.validate_for(tuple(z0.shape))
        if config.get("objective") != method.PARAMETERS:
            raise ValueError("adopted all-bit mu/gamma/weights/two-half-step parameters required")
        try:
            delta=tensor_value(output/"inputs/two_mean_delta.pt",latent_shape)
            if delta is None:
                delta=torch.load(config["inputs"]["two_mean_delta"],map_location="cpu",weights_only=True)
            contract=tensor_contract(delta,latent_shape)
            norm=float(delta.double().norm())
            outside=float((delta-masked(delta,support)).double().norm())
            result["input_contract"]["two_mean_delta"]={**contract,"l2":norm,"mask_outside_l2":outside}
            # Validate mathematical semantics; never renormalize or remask the saved comparator.
            if outside!=0 or norm > 1.+8*torch.finfo(delta.dtype).eps:
                raise ValueError("saved TWO_MEAN delta requires original mask and cumulative L2<=1 (FP32 rounding allowed)")
            if tensor_value(output/"inputs/two_mean_delta.pt",latent_shape) is None:
                save_tensor(output/"inputs/two_mean_delta.pt",delta);flush()
            saved["TWO_MEAN_SAVED"]=delta
        except Exception as exc:
            failure("TWO_MEAN_SAVED",exc)
        # Both controls are attempted independently before any new direction is formed.
        for name in method.VIEWS[:2]:
            delta=torch.zeros_like(z0) if name=="BASE" else saved.get(name)
            if delta is None:
                continue
            try:
                observe(name,z0+delta,delta)
            except Exception as exc:
                failure(name,exc)
        delta=torch.zeros_like(z0)
        for k,(reference,target) in enumerate((("BASE","ALLBIT_HALF"),("ALLBIT_HALF","ALLBIT_TWO")),1):
            step=f"step_{k}"
            result["steps"][step].update(reference=reference,target=target)
            if result["steps"][step]["status"] in ("ZERO_GRADIENT","UNDEFINED_ACTIVE_STATE_MAX_TIE"):
                result["direction_status"]=result["steps"][step]["status"]
                break
            if result["views"][reference]["status"] != "SCORED":
                result["steps"][step]["status"]="UNDEFINED_MISSING_START_READ"
                result["direction_status"]="UNDEFINED_MISSING_START_READ"
                break
            try:
                ordinary_loss=result["views"][reference]["loss"]
                if ordinary_loss["active_state_tie"]:
                    raise method.UndefinedActiveTie("ACTIVE_STATE_MAX_TIE in ordinary "+reference+" read")
                terminal=z0+delta
                unit_path=output/(step+"_unit.pt")
                def valid_step(unit):
                    tensor_contract(unit["input_terminal"],latent_shape)
                    if not torch.equal(unit["input_terminal"],terminal):
                        raise ValueError("saved step belongs to a different input terminal")
                    if unit["delta"] is not None:
                        tensor_contract(unit["delta"],latent_shape)
                        tensor_contract(unit["terminal"],latent_shape)
                        if torch.count_nonzero(unit["delta"]-masked(unit["delta"],support)) or float(unit["delta"].double().norm()) > 1.+8*torch.finfo(unit["delta"].dtype).eps:
                            raise ValueError("saved step delta violates adopted mask or cap")
                        if not torch.equal(unit["terminal"],z0+unit["delta"]):
                            raise ValueError("saved terminal does not represent the saved BASE-relative delta")
                unit=cached(unit_path,valid_step,tensor=True)
                if unit is None:
                    g=gradient(terminal,step,reference)
                    updated,report=method.descent_delta(delta,g)
                    report.update(reference=reference,target=target)
                    if updated is not None:
                        actual_increment=(z0+updated).double()-terminal.double()
                        report.update(actual_terminal_increment_l2=float(actual_increment.norm()),
                            predicted_loss_change_actual_terminal_increment=float((g.double()*actual_increment).sum()))
                    resume.atomic_tensor(unit_path,dict(input_terminal=terminal,delta=updated,
                        terminal=None if updated is None else z0+updated,report=report))
                    del g;gc.collect()
                else:
                    updated,report=unit["delta"],unit["report"]
                result["steps"][step]=report
                result["direction_status"]=report["status"]
                write_json(output/(step+".json"),report);flush()
                if updated is None:
                    break
                delta=updated
                result["views"][target]["direction_status"]="DEFINED"
            except method.UndefinedActiveTie as exc:
                result["steps"][step].update(status="UNDEFINED_ACTIVE_STATE_MAX_TIE",reason=str(exc))
                result["direction_status"]="UNDEFINED_ACTIVE_STATE_MAX_TIE"
                break
            except Exception as exc:
                result["steps"][step].update(status="UNDEFINED_STEP_FAILURE",reason=repr(exc))
                result["direction_status"]="UNDEFINED_STEP_FAILURE"
                failure(step,exc)
                break
            # Direction formation already completed and is durable. A scoring decode/read
            # failure cannot retrospectively make that direction undefined.
            try:
                observe(target,z0+delta,delta)
            except Exception as exc:
                failure(target,exc)
                if target=="ALLBIT_HALF":
                    reason="ALLBIT_HALF observation failed: "+repr(exc)
                    result["steps"]["step_2"].update(status="MISSING_DEPENDENCY",reason=reason,
                        reference="ALLBIT_HALF",target="ALLBIT_TWO")
                    result["views"]["ALLBIT_TWO"]["reason"]=reason
                break
        if base_rgb is not None:
            try:
                visuals=export_visuals(output,base_rgb)
                result["quality_frames"]=visuals["frame_indices"]
                for error in visuals["failures"]:
                    failure(error["view"]+"_visuals",OSError(error["reason"]))
            except Exception as exc:
                failure("visuals",exc)
        for name,row in result["views"].items():
            if row["status"]=="MISSING" and row["reason"]=="not_run":
                row["reason"]=result["direction_status"] if name in method.VIEWS[2:] else "missing_input_or_decode"
        result["status"]="COMPLETE" if not result["failures"] else "ENGINEERING_FAILURE"
    except BaseException as exc:
        result["status"]="INTERRUPTED" if not isinstance(exc,Exception) else "ENGINEERING_FAILURE"
        failure("run",exc)
    finally:
        if model is not None:
            try:
                adapter._clear_cache(model)
            except Exception as exc:
                result["cleanup_errors"].append(repr(exc))
        used_model=model is not None
        model=None;gc.collect()
        if used_model and str(config.get("device","cuda")).startswith("cuda"):
            result["resources"]=dict(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            torch.cuda.empty_cache()
        try:
            import resource
            result.setdefault("resources",{})["process_peak_rss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            pass
        for row in result["steps"].values():
            if row["status"]=="NOT_RUN":
                row.update(status="MISSING_DEPENDENCY",reason=result["direction_status"])
        for row in result["model_calls"].values():
            if row["status"]=="NOT_RUN":
                row["status"]="MISSING_DEPENDENCY"
            elif row["status"]=="RUNNING":
                row["status"]="INTERRUPTED_COMPLETION_UNKNOWN"
        refresh_reports(output,result)
        for name,row in result["views"].items():
            if row["observed_metrics"]==0:
                write_json(output/name/"metrics.json",posthoc.missing_evaluation(row["reason"] or "missing"))
        refresh_reports(output,result)
        current_calls=[row for row in result["call_history"] if row["id"].startswith(attempt+"_")]
        result["current_attempt_counts"]={kind+"_"+phase:sum(row["kind"]==kind and
            (phase=="attempted" or row["status"]=="RETURNED") for row in current_calls)
            for kind in ("vae_load","decode","decoder_vjp") for phase in ("attempted","completed")}
        result["finished_at_unix"]=time.time();flush()
        resume.atomic_bytes(output/"attempts"/(attempt+".json"),resume.json_bytes(result))
    return result
