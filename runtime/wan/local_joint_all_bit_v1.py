"""Adopted all-bit: four fixed views, six decode calls, two fresh scalar VJPs."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

from main.tube_state import local_joint_all_bit_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_state_payload_posthoc_v1 as posthoc
from main.tube_state import local_joint_terminal_bridge_v1 as observer
from runtime.wan import generation, vae as adapter
from runtime.wan.local_joint_readout_checkpoint_v1 import ReplayLedger, checkpoint_decode
from runtime.wan.local_joint_readout_m0_v1 import save_tensor, masked, metric_values
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
        evaluation = json.loads(path.read_text()) if path.exists() else posthoc.missing_evaluation("not_run")
        values = metrics[name] = metric_values(evaluation)
        losses[name] = loss_from_values(values)
        raw_path = output/name/"raw.json"
        rows = json.loads(raw_path.read_text()) if raw_path.exists() else []
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
    output = Path(output); path = output/"result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
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
    write_json(path,result)
    return result


def run(config, output, *, loader=None, protocol=carrier.PUBLIC, latent_shape=LATENT_SHAPE,
        support=PUBLIC_LATENT_SUPPORT, source_identity=None):
    import torch

    output = Path(output)
    for input_path in config["inputs"].values():
        if output.resolve().is_relative_to(Path(input_path).resolve().parent):
            raise ValueError("fresh output must be outside saved input directories")
    output.mkdir(parents=True,exist_ok=False)
    result = initial_result(config)
    result["source"] = source_identity or {}
    result["started_at_unix"] = time.time()
    model, base_rgb = None, None
    saved = {}

    def flush():
        write_json(output/"result.json",result)

    def failure(stage,exc):
        result["failures"].append(dict(stage=stage,reason=f"{type(exc).__name__}: {exc}",retry=False))
        if stage in result["views"] and result["views"][stage]["status"] != "SCORED":
            result["views"][stage].update(status="MISSING",reason=repr(exc))
        flush()

    def call(ident,operation):
        row=result["model_calls"][ident];kind=row["kind"]
        row.update(status="RUNNING",started_at_unix=time.time())
        result["counts"][kind+"_attempted"]+=1;flush()
        try:
            value=operation()
        except BaseException as exc:
            row.update(status="FAILED",reason=repr(exc));flush();raise
        row.update(status="RETURNED",finished_at_unix=time.time())
        result["counts"][kind+"_completed"]+=1;flush()
        return value

    def ordinary_decode(name,terminal):
        original=model.decoder.forward;original_decode=model.decode
        def counted(*args,**kwargs):
            result["ordinary_chunks"]["attempted"]+=1;flush()
            value=original(*args,**kwargs)
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
        save_tensor(directory/"terminal_latent.pt",terminal)
        save_tensor(directory/"delta.pt",delta)
        actual=terminal.double()-z0.double()
        result["views"][name]["displacement"]=dict(planned_delta_l2=float(delta.double().norm()),
            actual_terminal_minus_base_l2=float(actual.norm()),
            planned_mask_outside_l2=float((delta-masked(delta,support)).double().norm()),
            actual_mask_outside_l2=float((actual-masked(actual,support)).norm()),
            addition_rounding_l2=float((actual-delta.double()).norm()))
        rgb=ordinary_decode(name,terminal)
        tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
        save_tensor(directory/"float_rgb.pt",rgb)
        rows=observer.read_rgb(rgb,key=KEY,rho=.5,protocol=protocol)
        write_json(directory/"raw.json",rows)
        evaluation=observer.evaluate_saved(rows,key=KEY,message=bytes.fromhex("8001a55a"),rho=.5,protocol=protocol)
        write_json(directory/"metrics.json",evaluation)
        values=metric_values(evaluation)
        count=sum(x is not None for x in values["correlations"]+values["margins"]+[values["state_gap"]])
        result["views"][name].update(status="SCORED" if count==55 else "MISSING",reason=None if count==55 else "incomplete_readout",
            observed_windows=len(rows),observed_metrics=count,
            observed_chips=sum(c["q"] is not None for r in rows for c in r["state_chips"]+r["payload_chips"]),**values,loss=loss_from_values(values))
        write_json(directory/"loss.json",result["views"][name]["loss"])
        if name=="BASE":
            base_rgb=rgb
        if base_rgb is not None:
            try:
                write_json(directory/"quality.json",quality(rgb,base_rgb,protocol))
            except Exception as exc:
                # A missing quality artifact does not erase raw or gate fixed method calls.
                failure(name+"_quality",exc)
        flush()
        return rgb

    def gradient(terminal,step,reference):
        ledger=ReplayLedger(callback=lambda summary:(result["replay"].update({step:summary}),flush()))
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
            write_json(output/name/"raw.json",[])
            write_json(output/name/"metrics.json",posthoc.missing_evaluation("not_run"))
        flush()
        c=config["carrier"]
        if (c["key"],c["message_hex"],c["rho"],c["cap"]) != (KEY,"8001a55a",.5,1.):
            raise ValueError("adopted key/message/rho/total cap required")
        z0=torch.load(config["inputs"]["terminal_latent"],map_location="cpu",weights_only=True)
        result["input_contract"]=dict(terminal=tensor_contract(z0,latent_shape))
        support.validate_for(tuple(z0.shape))
        if config.get("objective") != method.PARAMETERS:
            raise ValueError("adopted all-bit mu/gamma/weights/two-half-step parameters required")
        try:
            delta=torch.load(config["inputs"]["two_mean_delta"],map_location="cpu",weights_only=True)
            contract=tensor_contract(delta,latent_shape)
            norm=float(delta.double().norm())
            outside=float((delta-masked(delta,support)).double().norm())
            result["input_contract"]["two_mean_delta"]={**contract,"l2":norm,"mask_outside_l2":outside}
            # Validate mathematical semantics; never renormalize or remask the saved comparator.
            if outside!=0 or norm > 1.+8*torch.finfo(delta.dtype).eps:
                raise ValueError("saved TWO_MEAN delta requires original mask and cumulative L2<=1 (FP32 rounding allowed)")
            saved["TWO_MEAN_SAVED"]=delta
        except Exception as exc:
            failure("TWO_MEAN_SAVED",exc)
        if torch.cuda.is_available() and str(config.get("device","cuda")).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        model=call("load",lambda:(loader or generation.load_frozen_vae)(config,device=config.get("device","cuda")))
        WanPosteriorBackend(model)
        result["actual_model_execution"]="REAL" if loader is None else "INJECTED_TEST_DOUBLE"
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
            if result["views"][reference]["status"] != "SCORED":
                result["steps"][step]["status"]="UNDEFINED_MISSING_START_READ"
                result["direction_status"]="UNDEFINED_MISSING_START_READ"
                break
            try:
                ordinary_loss=result["views"][reference]["loss"]
                if ordinary_loss["active_state_tie"]:
                    raise method.UndefinedActiveTie("ACTIVE_STATE_MAX_TIE in ordinary "+reference+" read")
                terminal=z0+delta
                g=gradient(terminal,step,reference)
                updated,report=method.descent_delta(delta,g)
                report.update(reference=reference,target=target)
                result["steps"][step]=report
                result["direction_status"]=report["status"]
                if updated is not None:
                    actual_increment=(z0+updated).double()-terminal.double()
                    report.update(actual_terminal_increment_l2=float(actual_increment.norm()),
                        predicted_loss_change_actual_terminal_increment=float((g.double()*actual_increment).sum()))
                write_json(output/(step+".json"),report);flush()
                del g;gc.collect()
                if updated is None:
                    break
                delta=updated
                result["views"][target]["direction_status"]="DEFINED"
                observe(target,z0+delta,delta)
            except method.UndefinedActiveTie as exc:
                result["steps"][step].update(status="UNDEFINED_ACTIVE_STATE_MAX_TIE",reason=str(exc))
                result["direction_status"]="UNDEFINED_ACTIVE_STATE_MAX_TIE"
                break
            except Exception as exc:
                result["steps"][step].update(status="UNDEFINED_STEP_FAILURE",reason=repr(exc))
                result["direction_status"]="UNDEFINED_STEP_FAILURE"
                failure(step,exc)
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
        model=None;gc.collect()
        if torch.cuda.is_available():
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
        result["finished_at_unix"]=time.time();flush()
    return result
