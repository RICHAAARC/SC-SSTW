"""One saved JOINT, five sequential decoder VJPs and six fixed float-RGB views."""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import time

from main.tube_state import local_joint_readout_m0_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_state_payload_posthoc_v1 as posthoc
from main.tube_state import local_joint_terminal_bridge_v1 as observer
from runtime.wan import generation, vae as adapter
from runtime.wan.local_joint_readout_checkpoint_v1 import ReplayLedger, checkpoint_decode
from runtime.wan.local_joint_state_payload_provider_v1 import PUBLIC_LATENT_SUPPORT, WanPosteriorBackend
from runtime.wan.local_joint_terminal_bridge_v1 import KEY, LATENT_SHAPE, tensor_contract, write_json

CEILING = ("One seen saved JOINT terminal point, fixed readout/mask/cap only. "
           "No trajectory/CFG/native causality, blind decoding, FPR or scientific PASS. "
           "Exact zero common gradient only excludes all-five strictly positive first-order changes; numerical near-zero is uncertain.")
EXPECTED = dict(vae_load=1, decode=11, decoder_vjp=5, encode=0, dit=0, native=0, codec=0,
                ordinary_chunks=276, gradient_forward_chunks=230, nominal_replay_chunks=230,
                windows=528, chips=8448, metrics=330)


def save_tensor(path, tensor):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(tensor.detach().cpu(), temporary)
    os.replace(temporary, path)


def masked(value, support):
    import torch
    result = torch.zeros_like(value)
    start, end = support.time_range
    for y0, y1, x0, x1 in support.rois:
        result[:, :, start:end, y0:y1, x0:x1] = value[:, :, start:end, y0:y1, x0:x1]
    return result


def metric_values(evaluation):
    correlations = [x["value"] for x in evaluation["state"]["correlations"]]
    margins = [x["signed_mean"] for x in evaluation["payload"]["metrics"]]
    objectives = [evaluation["state"]["c0_minus_max_other"]]
    objectives += [None if any(x is None for x in margins[f*8:(f+1)*8])
                   else min(margins[f*8:(f+1)*8]) for f in range(4)]
    other = correlations[1:]
    active = dict(other_offsets=[] if any(x is None for x in other) else
                  [i+1 for i,x in enumerate(other) if x == max(other)],
                  fragment_bits=[[] if objectives[f+1] is None else
                                 [b for b,x in enumerate(margins[f*8:(f+1)*8]) if x == objectives[f+1]]
                                 for f in range(4)])
    return dict(correlations=correlations, state_gap=objectives[0], margins=margins,
                objectives=objectives, active=active)


def initial_result(config):
    calls = {"load": dict(kind="vae_load", status="NOT_RUN")}
    calls.update({name: dict(kind="decode", status="NOT_RUN") for name in method.VIEWS})
    for j in range(5):
        calls[f"gradient_{j}_decode"] = dict(kind="decode", status="NOT_RUN")
        calls[f"gradient_{j}_vjp"] = dict(kind="decoder_vjp", status="NOT_RUN")
    return dict(schema="local-joint-readout-m0-v1", status="RUNNING", config=config,
        ceiling=CEILING, scientific_pass=False, expected_cost=EXPECTED, automatic_retry=False,
        counts={**{f"{k}_{phase}": 0 for k in ("vae_load", "decode", "decoder_vjp")
                   for phase in ("attempted", "completed")}, "encode":0, "dit":0, "native":0, "codec":0},
        ordinary_chunks=dict(attempted=0, completed=0), replay={}, model_calls=calls,
        views={name: dict(status="MISSING", reason="not_run", expected_windows=88,
                         direction_status="FIXED_COMPARATOR" if name in method.VIEWS[:2] else "UNDEFINED",
                         expected_chips=1408, expected_metrics=55, observed_windows=0,
                         observed_chips=0, observed_metrics=0) for name in method.VIEWS},
        failures=[], cleanup_errors=[], actual_model_execution="NOT_STARTED")


def refresh_reports(output, result):
    """Report-only operation, also usable after a reaped child; never calls a model."""
    metrics = {}
    for name in method.VIEWS:
        path = output/name/"metrics.json"
        evaluation = json.loads(path.read_text()) if path.exists() else posthoc.missing_evaluation("not_run")
        metrics[name] = metric_values(evaluation)
        raw_path = output/name/"raw.json"
        rows = json.loads(raw_path.read_text()) if raw_path.exists() else []
        values = metrics[name]
        count = sum(x is not None for x in values["correlations"]+values["margins"]+[values["state_gap"]])
        view = result["views"][name]
        view.update(observed_windows=len(rows), observed_metrics=count,
            observed_chips=sum(c.get("q") is not None for r in rows for c in r.get("state_chips", [])+r.get("payload_chips", [])),
            artifacts={label: "SAVED" if (output/name/file).exists() else "MISSING"
                       for label,file in (("quality","quality.json"),("frames","frames.png"),
                                          ("float_rgb","float_rgb.pt"),("terminal","terminal_latent.pt"),("delta","delta.pt"))},
            **values)
        if count == 55:
            view.update(status="SCORED", reason=None)
    base = metrics["BASE"]
    comparisons = {}
    predicted = result.get("directions", {}).get("predicted", {})
    for name in method.VIEWS[1:]:
        values = metrics[name]
        comparisons[name] = dict(
            objective_change=[None if x is None or b is None else x-b
                              for x, b in zip(values["objectives"], base["objectives"])],
            predicted_change=predicted.get(name),
            correlation_change=[None if x is None or b is None else x-b
                                for x, b in zip(values["correlations"], base["correlations"])],
            margin_change=[None if x is None or b is None else x-b
                           for x, b in zip(values["margins"], base["margins"])],
            gained_bits=[i for i, (b,x) in enumerate(zip(base["margins"],values["margins"]))
                         if b is not None and x is not None and b <= 0 < x],
            lost_bits=[i for i, (b,x) in enumerate(zip(base["margins"],values["margins"]))
                       if b is not None and x is not None and x <= 0 < b])
    fd = [None if a is None or b is None else (b-a)/(2*method.H)
          for a,b in zip(metrics["FD_MINUS"]["objectives"], metrics["FD_PLUS"]["objectives"])]
    active_changes = {name: metrics[name]["active"] != base["active"]
                      if result["views"][name]["observed_metrics"] == 55 and result["views"]["BASE"]["observed_metrics"] == 55
                      else None for name in method.VIEWS[1:]}
    repeated = [x["objectives"] for x in result.get("gradient_readouts", {}).values()]
    repeat_spread = [None if not repeated else max(x[j] for x in repeated)-min(x[j] for x in repeated) for j in range(5)]
    write_json(output/"comparison.json", dict(metrics=metrics, comparisons=comparisons,
        finite_difference=dict(h=method.H, central_slope=fd,
            predicted_slope=predicted.get("COMMON"), active_change_from_base=active_changes,
            gradient_forward_repeat_spread=repeat_spread,
            within_observed_repeat_variation=[None if a is None or b is None or spread is None else abs(b-a) <= spread
                for a,b,spread in zip(metrics["FD_MINUS"]["objectives"],metrics["FD_PLUS"]["objectives"],repeat_spread)],
            numerical_indistinguishable=[None if a is None or b is None else
                abs(b-a) <= 4096*2.220446049250313e-16*max(1.,abs(a),abs(b))
                for a,b in zip(metrics["FD_MINUS"]["objectives"],metrics["FD_PLUS"]["objectives"])],
            numerical_note="FP64 subtraction resolution and existing FP32-decoder repeat spread are diagnostic only; zero observed spread is not a bound on FP32 error. Read active changes alongside central slope."), ceiling=CEILING))
    result["totals"] = {field: sum(x[field] for x in result["views"].values())
                        for field in ("expected_windows", "expected_chips", "expected_metrics",
                                      "observed_windows", "observed_chips", "observed_metrics")}


def finalize_interrupted(output, reason):
    output = Path(output)
    path = output/"result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result["status"] != "RUNNING":
        return result
    result["status"] = "INTERRUPTED"
    result["failures"].append(dict(stage="external_termination", reason=reason))
    for row in result["model_calls"].values():
        if row["status"] == "RUNNING":
            row.update(status="INTERRUPTED_COMPLETION_UNKNOWN", reason=reason)
        elif row["status"] == "NOT_RUN":
            row.update(status="MISSING_DEPENDENCY", reason=reason)
    for name, row in result["views"].items():
        if row["status"] != "SCORED":
            row.update(status="MISSING", reason=reason)
            if not (output/name/"metrics.json").exists():
                write_json(output/name/"metrics.json", posthoc.missing_evaluation(reason))
    refresh_reports(output, result)
    write_json(path, result)
    return result


def quality(rgb, base, protocol, *, include_temporal_framewise=False):
    """Float-RGB quality, per-frame accumulation avoids a full FP64 video copy."""
    import torch
    roi = torch.zeros(rgb.shape[1:3], dtype=torch.bool)
    for y0,y1,x0,x1 in protocol.rois:
        roi[y0:y1,x0:x1] = True
    frames, inside, outside, temporal = [], [], [], []
    transitions = []
    energy_inside = energy_outside = max_abs = peak_inside = peak_outside = 0.
    previous = None
    for index, (frame, reference) in enumerate(zip(rgb, base)):
        difference = frame.double()-reference.double()
        mse = float(difference.square().mean())
        max_abs = max(max_abs, float(difference.abs().max()))
        peak_inside = max(peak_inside, float(difference[roi].abs().max()))
        energy_inside += float(difference[roi].square().sum())
        if bool((~roi).any()):
            peak_outside = max(peak_outside, float(difference[~roi].abs().max()))
            energy_outside += float(difference[~roi].square().sum())
        frames.append(dict(rmse=math.sqrt(mse), psnr_db=None if mse == 0 else -10*math.log10(mse),
                           exact_match=mse == 0))
        inside.append(float(difference[roi].square().mean()))
        outside.append(float(difference[~roi].square().mean()) if bool((~roi).any()) else None)
        if previous is not None:
            change = difference-previous
            temporal.append(float(change.square().mean()))
            if include_temporal_framewise:
                transitions.append(dict(from_frame=index-1, to_frame=index,
                    rmse=math.sqrt(temporal[-1]), max_abs=float(change.abs().max())))
        previous = difference
    mse = sum(x["rmse"]**2 for x in frames)/len(frames)
    report = dict(rmse=math.sqrt(mse), psnr_db=None if mse == 0 else -10*math.log10(mse),
        max_abs=max_abs, roi_inside_energy=energy_inside, roi_outside_energy=energy_outside,
        roi_inside_max_abs=peak_inside, roi_outside_max_abs=None if not bool((~roi).any()) else peak_outside,
        exact_match=mse == 0, psnr_zero_error_representation="null with exact_match=true means infinity",
        framewise=frames, temporal_residual_rmse=math.sqrt(sum(temporal)/len(temporal)),
        roi_inside_rmse=math.sqrt(sum(inside)/len(inside)),
        roi_outside_rmse=None if any(x is None for x in outside) else math.sqrt(sum(outside)/len(outside)),
        roi_scope="spatial union over all frames", temporal_scope="consecutive residual-frame differences")
    if include_temporal_framewise:
        report["temporal_framewise"] = transitions
        report["temporal_peak"] = max(transitions, key=lambda row: row["rmse"]) if transitions else None
    return report


def frame_images(output, name, rgb, base, *, indices=(1, 44, 88, 132, 176)):
    from PIL import Image, ImageDraw
    width, height = rgb.shape[2], rgb.shape[1]
    canvas = Image.new("RGB", (width*len(indices), height*2+40), "white")
    draw = ImageDraw.Draw(canvas)
    for col, index in enumerate(indices):
        frame = (rgb[index].clamp(0, 1)*255).round().byte().numpy()
        residual = ((rgb[index]-base[index])/.2+.5).clamp(0, 1)
        delta = (residual*255).round().byte().numpy()
        canvas.paste(Image.fromarray(frame), (col*width,20))
        canvas.paste(Image.fromarray(delta), (col*width,height+40))
        draw.text((col*width,0), f"{name} frame {index}", fill="black")
        draw.text((col*width,height+20), "residual [-0.1,0.1]", fill="black")
    canvas.save(output/name/"frames.png")


def run(config, output, *, loader=None, protocol=carrier.PUBLIC, latent_shape=LATENT_SHAPE,
        support=PUBLIC_LATENT_SUPPORT, source_identity=None):
    import torch

    output = Path(output)
    for input_path in config["inputs"].values():
        if output.resolve().is_relative_to(Path(input_path).resolve().parent):
            raise ValueError("fresh output must be outside saved input directories")
    output.mkdir(parents=True, exist_ok=False)
    result = initial_result(config)
    result["source"] = source_identity or {}
    result["started_at_unix"] = time.time()
    model, base_rgb, gradients, bridge_delta = None, None, [], None
    def flush():
        write_json(output/"result.json", result)
    def failure(stage, exc):
        result["failures"].append(dict(stage=stage, reason=f"{type(exc).__name__}: {exc}", retry=False))
        if stage in result["views"] and result["views"][stage]["status"] != "SCORED":
            result["views"][stage].update(status="MISSING", reason=repr(exc))
        flush()
    def call(ident, operation):
        row = result["model_calls"][ident]; kind = row["kind"]
        row.update(status="RUNNING", started_at_unix=time.time())
        result["counts"][kind+"_attempted"] += 1; flush()
        try:
            value = operation()
        except BaseException as exc:
            row.update(status="FAILED", reason=repr(exc)); flush(); raise
        row.update(status="RETURNED", finished_at_unix=time.time())
        result["counts"][kind+"_completed"] += 1; flush()
        return value
    def ordinary_decode(name, terminal):
        original = model.decoder.forward
        original_decode = model.decode
        def counted(*args, **kwargs):
            result["ordinary_chunks"]["attempted"] += 1; flush()
            value = original(*args, **kwargs)
            result["ordinary_chunks"]["completed"] += 1; flush()
            return value
        model.decoder.forward = counted
        model.decode = lambda *args, **kwargs: call(name, lambda: original_decode(*args, **kwargs))
        try:
            # The shared adapter preserves actual internal Wan clamp and RGB clamp.
            return adapter.decode_normalized_latent(model, terminal.to(config.get("device", "cuda"))).cpu()
        finally:
            model.decoder.forward = original
            model.decode = original_decode
    def observe(name, terminal, delta):
        nonlocal base_rgb
        directory = output/name
        save_tensor(directory/"terminal_latent.pt", terminal)
        save_tensor(directory/"delta.pt", delta)
        result["views"][name]["delta_l2"] = float(delta.double().norm())
        rgb = ordinary_decode(name, terminal)
        tensor_contract(rgb, protocol.video_shape, bounded_rgb=True)
        save_tensor(directory/"float_rgb.pt", rgb)
        rows = observer.read_rgb(rgb, key=KEY, rho=.5, protocol=protocol)
        # Truth-free raw is persisted before the independent posthoc reducer.
        write_json(directory/"raw.json", rows)
        evaluation = observer.evaluate_saved(rows, key=KEY, message=bytes.fromhex("8001a55a"), rho=.5, protocol=protocol)
        write_json(directory/"metrics.json", evaluation)
        values = metric_values(evaluation)
        count = sum(x is not None for x in values["correlations"]+values["margins"]+[values["state_gap"]])
        result["views"][name].update(status="SCORED" if count == 55 else "MISSING", reason=None,
            observed_windows=len(rows), observed_chips=sum(c["q"] is not None for r in rows for c in r["state_chips"]+r["payload_chips"]),
            observed_metrics=count, **values)
        if name == "BASE":
            base_rgb = rgb
        if base_rgb is not None:
            write_json(directory/"quality.json", quality(rgb, base_rgb, protocol))
            frame_images(output, name, rgb, base_rgb)
        flush()
        return rgb
    def gradient(j, terminal):
        ledger = ReplayLedger(callback=lambda summary: (result["replay"].update({str(j):summary}), flush()))
        z = terminal.to(config.get("device", "cuda")).clone().requires_grad_(True)
        adapter._clear_cache(model)
        original_decode = model.decode
        model.decode = lambda *args, **kwargs: call(f"gradient_{j}_decode", lambda: original_decode(*args, **kwargs))
        try:
            mean, std = adapter._scale_tensors(model, z)
            decoded = checkpoint_decode(model, z*std+mean, ledger)
            rgb = (decoded[0].permute(1,2,3,0).float()/2+.5).clamp(0,1)
            read = method.tensor_readout(rgb, key=KEY, message=bytes.fromhex("8001a55a"), protocol=protocol)
            values = read["objectives"].detach().cpu().tolist()
            baseline = result["base_active"]
            result.setdefault("gradient_readouts", {})[str(j)] = dict(active=read["active"], tied=read["tied"], objectives=values,
                active_change_from_base=read["active"] != baseline["active"],
                objective_change_from_base=[x-b for x,b in zip(values,baseline["objectives"])])
            if read["tied"]:
                raise method.UndefinedActiveTie("ACTIVE_MIN_MAX_TIE: all new directions undefined")
            g = call(f"gradient_{j}_vjp", lambda: torch.autograd.grad(read["objectives"][j], z)[0])
            g = masked(g.detach(), support).cpu()
            if not bool(torch.isfinite(g).all()):
                raise FloatingPointError("nonfinite masked gradient")
            save_tensor(output/"gradients"/f"objective_{j}.pt", g)
            return g
        finally:
            model.decode = original_decode
            try:
                adapter._clear_cache(model)
            finally:
                ledger.release_boundary_storage()
    try:
        for name in method.VIEWS:
            write_json(output/name/"raw.json", [])
            write_json(output/name/"metrics.json", posthoc.missing_evaluation("not_run"))
        flush()
        c = config["carrier"]
        if (c["key"], c["message_hex"], c["rho"], c["cap"]) != (KEY,"8001a55a",.5,1.):
            raise ValueError("adopted M0 key/message/rho/cap required")
        z = torch.load(config["inputs"]["terminal_latent"], map_location="cpu", weights_only=True)
        result["input_contract"] = dict(terminal=tensor_contract(z, latent_shape))
        support.validate_for(tuple(z.shape))
        try:
            bridge_delta = torch.load(config["inputs"]["bridge_delta"], map_location="cpu", weights_only=True)
            result["input_contract"]["bridge_delta"] = tensor_contract(bridge_delta, latent_shape)
            # Read the saved cap delta verbatim; never re-cap/re-mask the comparator.
            result["input_contract"]["bridge_delta_l2"] = float(bridge_delta.double().norm())
        except Exception as exc:
            bridge_delta = None; failure("bridge_input", exc)
        model = call("load", lambda: (loader or generation.load_frozen_vae)(config, device=config.get("device","cuda")))
        WanPosteriorBackend(model)
        result["actual_model_execution"] = "REAL" if loader is None else "INJECTED_TEST_DOUBLE"
        if torch.cuda.is_available() and str(config.get("device", "cuda")).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        try:
            rgb = observe("BASE", z, torch.zeros_like(z))
            with torch.no_grad():
                read = method.tensor_readout(rgb, key=KEY, message=bytes.fromhex("8001a55a"), protocol=protocol)
            result["base_active"] = dict(active=read["active"], tied=read["tied"], objectives=read["objectives"].tolist())
            tied = read["tied"]
            del read, rgb
        except Exception as exc:
            tied = True; failure("BASE", exc)
        if bridge_delta is not None:
            try: observe("BRIDGE", z+bridge_delta, bridge_delta)
            except Exception as exc: failure("BRIDGE", exc)
        if tied:
            result["direction_status"] = "UNDEFINED_BASE_OR_ACTIVE_TIE"
        else:
            try:
                for j in range(5):
                    gradients.append(gradient(j, z))
                    gc.collect()
                candidates, report = method.directions(gradients)
                result["directions"] = report
                result["direction_status"] = "COMPUTED"
                if bridge_delta is not None:
                    report["predicted"]["BRIDGE"] = [float((g.double()*bridge_delta.double()).sum()) for g in gradients]
                common = candidates["COMMON"]
                candidates["FD_MINUS"] = None if common is None else -method.H*common
                candidates["FD_PLUS"] = None if common is None else method.H*common
                for name in ("FD_MINUS", "FD_PLUS"):
                    sign = -1 if name == "FD_MINUS" else 1
                    report["predicted"][name] = None if common is None else [sign*method.H*x for x in report["predicted"]["COMMON"]]
                write_json(output/"directions.json", report); flush()
                for name in method.VIEWS[2:]:
                    delta = candidates[name]
                    if delta is None:
                        result["views"][name]["reason"] = report["mean_status"] if name == "MEAN" else report["simplex"]["status"]
                        continue
                    result["views"][name]["direction_status"] = "DEFINED"
                    try: observe(name, z+delta, delta)
                    except Exception as exc: failure(name, exc)
            except method.UndefinedActiveTie as exc:
                result["direction_status"] = "UNDEFINED_ACTIVE_MIN_MAX_TIE"
                result["direction_reason"] = str(exc)
            except Exception as exc:
                result["direction_status"] = "UNDEFINED_GRADIENT_FAILURE"
                failure("gradients", exc)
        for name, row in result["views"].items():
            if row["status"] == "MISSING" and row["reason"] == "not_run":
                row["reason"] = result.get("direction_status", "missing_dependency") if name in method.VIEWS[2:] else "missing_input_or_decode"
        result["status"] = "COMPLETE" if not result["failures"] else "ENGINEERING_FAILURE"
    except BaseException as exc:
        result["status"] = "INTERRUPTED" if not isinstance(exc, Exception) else "ENGINEERING_FAILURE"
        failure("run", exc)
    finally:
        if model is not None:
            try: adapter._clear_cache(model)
            except Exception as exc: result["cleanup_errors"].append(repr(exc))
        model = None
        gc.collect()
        if torch.cuda.is_available():
            result["resources"] = dict(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            torch.cuda.empty_cache()
        try:
            import resource
            result.setdefault("resources", {})["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            pass
        for row in result["model_calls"].values():
            if row["status"] == "NOT_RUN": row["status"] = "MISSING_DEPENDENCY"
        # Reconstruct from durable evidence before updating missing placeholders.
        # An interruption between metrics write and view update must not erase it.
        refresh_reports(output, result)
        for name, row in result["views"].items():
            if row["observed_metrics"] == 0:
                write_json(output/name/"metrics.json", posthoc.missing_evaluation(row["reason"] or "missing"))
        refresh_reports(output, result)
        result["finished_at_unix"] = time.time()
        flush()
    return result
