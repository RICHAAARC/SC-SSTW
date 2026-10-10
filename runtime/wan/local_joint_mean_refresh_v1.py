"""R-MEAN-2H: five saved-terminal views, six decode calls and one decoder VJP."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

from main.tube_state import local_joint_mean_refresh_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_state_payload_posthoc_v1 as posthoc
from main.tube_state import local_joint_terminal_bridge_v1 as observer
from runtime.wan import generation, vae as adapter
from runtime.wan.local_joint_readout_checkpoint_v1 import ReplayLedger, checkpoint_decode
from runtime.wan.local_joint_readout_m0_v1 import save_tensor, masked, metric_values, quality, frame_images
from runtime.wan.local_joint_state_payload_provider_v1 import PUBLIC_LATENT_SUPPORT, WanPosteriorBackend
from runtime.wan.local_joint_terminal_bridge_v1 import KEY, LATENT_SHAPE, tensor_contract, write_json

CEILING = ("One seen saved JOINT terminal point; two fixed half steps, total BASE-relative L2<=1. "
           "An average-objective VJP does not guarantee five-objective or all-bit improvement. "
           "No trajectory/CFG/native causality, blind decoding, FPR or scientific PASS. "
           "COMMON refresh and M1/M2 are not adopted by this run.")
EXPECTED = dict(vae_load=1, decode=6, decoder_vjp=1, encode=0, dit=0, native=0, codec=0,
                ordinary_chunks=230, gradient_forward_chunks=46, nominal_replay_chunks=46,
                windows=440, chips=7040, metrics=275)


def initial_result(config):
    calls = {"load": dict(kind="vae_load", status="NOT_RUN")}
    calls.update({name: dict(kind="decode", status="NOT_RUN") for name in method.VIEWS})
    calls["midpoint_gradient_decode"] = dict(kind="decode", status="NOT_RUN")
    calls["midpoint_gradient_vjp"] = dict(kind="decoder_vjp", status="NOT_RUN")
    return dict(schema="local-joint-mean-refresh-v1", status="RUNNING", config=config,
        ceiling=CEILING, scientific_pass=False, expected_cost=EXPECTED, automatic_retry=False,
        counts={**{f"{k}_{phase}": 0 for k in ("vae_load", "decode", "decoder_vjp")
                   for phase in ("attempted", "completed")}, "encode":0, "dit":0, "native":0, "codec":0},
        ordinary_chunks=dict(attempted=0, completed=0), replay={}, model_calls=calls,
        views={name: dict(status="MISSING", reason="not_run", expected_windows=88,
                         direction_status="UNDEFINED" if name == "TWO_MEAN" else "FIXED_INPUT",
                         expected_chips=1408, expected_metrics=55, observed_windows=0,
                         observed_chips=0, observed_metrics=0) for name in method.VIEWS},
        failures=[], cleanup_errors=[], actual_model_execution="NOT_STARTED",
        direction_status="NOT_RUN", quality_frames=list(method.FRAMES))


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
    """Read only durable raw/metrics; no pixel loading, rescoring or model call."""
    output = Path(output)
    metrics = {}
    for name in method.VIEWS:
        path = output/name/"metrics.json"
        evaluation = json.loads(path.read_text()) if path.exists() else posthoc.missing_evaluation("not_run")
        values = metrics[name] = metric_values(evaluation)
        raw_path = output/name/"raw.json"
        rows = json.loads(raw_path.read_text()) if raw_path.exists() else []
        count = sum(x is not None for x in values["correlations"]+values["margins"]+[values["state_gap"]])
        view = result["views"][name]
        view.update(observed_windows=len(rows), observed_metrics=count,
            observed_chips=sum(c.get("q") is not None for r in rows for c in r.get("state_chips", [])+r.get("payload_chips", [])),
            artifacts={label:"SAVED" if (output/name/file).exists() else "MISSING"
                       for label,file in (("quality","quality.json"),("frames","frames.png"),
                                          ("float_rgb","float_rgb.pt"),("terminal","terminal_latent.pt"),("delta","delta.pt"))},
            **values)
        if count == 55:
            view.update(status="SCORED", reason=None)
    comparisons = {name:compare_values(metrics["BASE"],metrics[name]) for name in method.VIEWS[1:]}
    main = compare_values(metrics["ONE_MEAN"],metrics["TWO_MEAN"])
    midpoint = compare_values(metrics["MID_MEAN"],metrics["TWO_MEAN"])
    objective_change = midpoint["objective_change"]
    write_json(output/"comparison.json", dict(metrics=metrics, versus_base=comparisons,
        primary_comparison=dict(reference="ONE_MEAN",target="TWO_MEAN",**main),
        second_half_step=dict(reference="MID_MEAN",target="TWO_MEAN",**midpoint,
            actual_average_change=None if any(x is None for x in objective_change) else sum(objective_change)/5,
            predicted_average_change=result.get("direction",{}).get("predicted_average_change")),
        ceiling=CEILING))
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
        result["views"][name]["delta_l2"]=float(delta.double().norm())
        rgb=ordinary_decode(name,terminal)
        tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
        save_tensor(directory/"float_rgb.pt",rgb)
        rows=observer.read_rgb(rgb,key=KEY,rho=.5,protocol=protocol)
        write_json(directory/"raw.json",rows)
        evaluation=observer.evaluate_saved(rows,key=KEY,message=bytes.fromhex("8001a55a"),rho=.5,protocol=protocol)
        write_json(directory/"metrics.json",evaluation)
        values=metric_values(evaluation)
        count=sum(x is not None for x in values["correlations"]+values["margins"]+[values["state_gap"]])
        result["views"][name].update(status="SCORED" if count==55 else "MISSING",reason=None,
            observed_windows=len(rows),observed_metrics=count,
            observed_chips=sum(c["q"] is not None for r in rows for c in r["state_chips"]+r["payload_chips"]),**values)
        if name=="BASE":
            base_rgb=rgb
        if base_rgb is not None:
            try:
                write_json(directory/"quality.json",quality(rgb,base_rgb,protocol,include_temporal_framewise=True))
                frame_images(output,name,rgb,base_rgb,indices=method.FRAMES)
            except Exception as exc:
                # A missing quality artifact does not erase raw or gate fixed method calls.
                failure(name+"_quality",exc)
        flush()
        return rgb

    def gradient(terminal):
        ledger=ReplayLedger(callback=lambda summary:(result["replay"].update({"midpoint":summary}),flush()))
        z=terminal.to(config.get("device","cuda")).clone().requires_grad_(True)
        adapter._clear_cache(model)
        original_decode=model.decode
        model.decode=lambda *a,**k:call("midpoint_gradient_decode",lambda:original_decode(*a,**k))
        try:
            mean,std=adapter._scale_tensors(model,z)
            decoded=checkpoint_decode(model,z*std+mean,ledger)
            rgb=(decoded[0].permute(1,2,3,0).float()/2+.5).clamp(0,1)
            read=method.tensor_readout(rgb,key=KEY,message=bytes.fromhex("8001a55a"),protocol=protocol)
            values=read["objectives"].detach().cpu().tolist()
            midpoint_read=result["midpoint_readout"]
            result["gradient_readout"]=dict(active=read["active"],tied=read["tied"],objectives=values,
                active_change_from_midpoint=read["active"]!=midpoint_read["active"],
                objective_change_from_midpoint=[a-b for a,b in zip(values,midpoint_read["objectives"])])
            flush()
            scalar=method.mean_objective(read)
            g=call("midpoint_gradient_vjp",lambda:torch.autograd.grad(scalar,z)[0])
            g=masked(g.detach(),support).cpu()
            if not bool(torch.isfinite(g).all()):
                raise FloatingPointError("nonfinite masked mean gradient")
            save_tensor(output/"gradients"/"midpoint_mean.pt",g)
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
        for name,field in (("ONE_MEAN","mean_delta"),("ONE_COMMON","common_delta")):
            try:
                delta=torch.load(config["inputs"][field],map_location="cpu",weights_only=True)
                contract=tensor_contract(delta,latent_shape)
                norm=float(delta.double().norm())
                outside=float((delta-masked(delta,support)).double().norm())
                result["input_contract"][field]={**contract,"l2":norm,"mask_outside_l2":outside}
                # These are actual mathematical inputs, not provenance/identity gates.
                if outside!=0 or abs(norm-1.) > 8*torch.finfo(delta.dtype).eps:
                    raise ValueError("saved M0 direction must have original mask and unit L2 (FP32 rounding allowed)")
                saved[name]=delta
            except Exception as exc:
                failure(name,exc)
        model=call("load",lambda:(loader or generation.load_frozen_vae)(config,device=config.get("device","cuda")))
        WanPosteriorBackend(model)
        result["actual_model_execution"]="REAL" if loader is None else "INJECTED_TEST_DOUBLE"
        if torch.cuda.is_available() and str(config.get("device","cuda")).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        # Independent controls are all attempted before any midpoint degeneration.
        for name in method.VIEWS[:3]:
            delta=torch.zeros_like(z0) if name=="BASE" else saved.get(name)
            if delta is None:
                continue
            try:
                observe(name,z0+delta,delta)
            except Exception as exc:
                failure(name,exc)
        if "ONE_MEAN" not in saved:
            result["direction_status"]="UNDEFINED_MEAN_INPUT"
        else:
            mid=method.midpoint(saved["ONE_MEAN"])
            try:
                rgb=observe("MID_MEAN",z0+mid,mid)
                with torch.no_grad():
                    read=method.tensor_readout(rgb,key=KEY,message=bytes.fromhex("8001a55a"),protocol=protocol)
                result["midpoint_readout"]=dict(active=read["active"],tied=read["tied"],
                                                objectives=read["objectives"].tolist())
                # A tie in either actual midpoint read cannot select a hidden subgradient.
                method.mean_objective(read)
                del read,rgb
                g=gradient(z0+mid)
                delta,report=method.refresh_delta(mid,g)
                result["direction"]=report
                result["direction_status"]=report["status"]
                write_json(output/"direction.json",report);flush()
                del g;gc.collect()
                if delta is not None:
                    result["views"]["TWO_MEAN"]["direction_status"]="DEFINED"
                    try:
                        observe("TWO_MEAN",z0+delta,delta)
                    except Exception as exc:
                        failure("TWO_MEAN",exc)
            except method.UndefinedActiveTie as exc:
                result["direction_status"]="UNDEFINED_ACTIVE_MIN_MAX_TIE"
                result["direction_reason"]=str(exc)
            except Exception as exc:
                result["direction_status"]="UNDEFINED_MIDPOINT_OR_GRADIENT_FAILURE"
                failure("midpoint_or_gradient",exc)
        for name,row in result["views"].items():
            if row["status"]=="MISSING" and row["reason"]=="not_run":
                row["reason"]=result["direction_status"] if name in method.VIEWS[3:] else "missing_input_or_decode"
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
