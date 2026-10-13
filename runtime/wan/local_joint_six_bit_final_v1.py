"""Six serial independent FINAL graphs on the existing Wan checkpoint/spool path."""
from __future__ import annotations

import copy
import gc
from pathlib import Path
import time

from main.tube_state import local_joint_six_bit_final_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from runtime.wan import generation, vae as adapter
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from runtime.wan.local_joint_readout_checkpoint_v1 import ReplayLedger, checkpoint_decode
from runtime.wan.local_joint_readout_m0_v1 import masked
from runtime.wan.local_joint_state_payload_provider_v1 import PUBLIC_LATENT_SUPPORT, WanPosteriorBackend
from runtime.wan.local_joint_terminal_bridge_v1 import KEY, LATENT_SHAPE, tensor_contract

EXPECTED = dict(vae_load=1,decode=6,decoder_vjp=6,ordinary_decode=0,encode=0,dit=0,native=0,
                scheduler=0,codec=0,ordinary_chunks=0,gradient_forward_chunks=276,
                nominal_replay_chunks=276,windows=528,chips=8448,metrics=330)


def initial_result(config):
    calls = {"load":dict(kind="vae_load",status="NOT_RUN")}
    for name in method.VIEWS:
        calls[name+"_gradient_decode"] = dict(kind="decode",status="NOT_RUN")
        calls[name+"_vjp"] = dict(kind="decoder_vjp",status="NOT_RUN")
    return dict(schema="local-joint-six-bit-final-v1",status="RUNNING",config=copy.deepcopy(config),
                views={name:dict(status="NOT_RUN",bit_index=bit,expected_windows=88,expected_chips=1408,
                                expected_metrics=55,observed_windows=0,observed_chips=0,observed_metrics=0,
                                payload_denominators=[24,24,20,20]) for name,bit in zip(method.VIEWS,method.BITS)},
                steps={},model_calls=calls,call_history=[],replay={},failures=[],cleanup_errors=[],
                counts={**{kind+"_"+phase:0 for kind in ("vae_load","decode","decoder_vjp")
                           for phase in ("attempted","completed")},
                        **{kind:0 for kind in ("ordinary_decode","encode","dit","native","scheduler","codec")}},
                ordinary_chunks=dict(attempted=0,completed=0), expected_cost=EXPECTED,
                ceiling=method.CEILING,scientific_pass=False,automatic_retry=False,
                actual_model_execution="NOT_STARTED",diagnostic_status="NOT_RUN")


def read_state(output):
    result = resume.read_state(output,views=method.VIEWS,steps=())
    if result is None:
        return None
    failures = {row["id"]:row for row in result.get("failure_history",[])}
    for path in (Path(output)/"failures").glob("*.json"):
        try:
            row = resume.read_json(path)
            failures[row["id"]] = row
        except (OSError,ValueError,KeyError,TypeError) as exc:
            result.setdefault("recovery_events",[]).append(dict(path=str(path),reason=repr(exc)))
    result["failure_history"] = list(failures.values())
    for row in failures.values():
        if row.get("attempt_id") == result.get("attempt_id") and row not in result["failures"]:
            result["failures"].append(row)
    return result


def record_failure(output, result, stage, reason):
    row = dict(id=str(time.time_ns()),attempt_id=result.get("attempt_id"),stage=stage,reason=reason)
    result["failures"].append(row)
    result.setdefault("failure_history",[]).append(row)
    resume.atomic_bytes(Path(output)/"failures"/(row["id"]+".json"),resume.json_bytes(row))


def refresh_counts(result):
    resume.recount(result)
    own = [r for r in result["call_history"] if r.get("attempt_id") == result.get("attempt_id")]
    result["current_attempt_counts"] = {kind+"_"+phase:sum(r["kind"]==kind and
        (phase=="attempted" or r["status"]=="RETURNED") for r in own)
        for kind in ("vae_load","decode","decoder_vjp") for phase in ("attempted","completed")}
    result["totals"] = {"expected_"+field:EXPECTED[field] for field in ("windows","chips","metrics")}
    result["totals"].update({"observed_"+field:sum(v["observed_"+field] for v in result["views"].values())
                             for field in ("windows","chips","metrics")})
    result["totals"].update(planned_source_points=1,repeated_reads_are_independent_samples=False)
    result["physical_replay_chunks"] = {phase:sum(r.get("replay",{}).get("counts",{}).get("vae_chunk",{})
        .get(phase,{}).get("attempted",0) for r in result["call_history"]) for phase in ("forward","recompute")}


def model_work_remaining(output):
    """Pre-import hint. Existing units are validated on CPU before any VAE load."""
    output = Path(output)
    prior = read_state(output)
    if prior is None and output.exists() and any(output.iterdir()):
        raise OSError("Nonempty output has no recoverable six-bit state; refusing silent restart")
    return any(not (output/"units"/(name+".pt")).is_file() for name in method.VIEWS)


def finalize_interrupted(output, reason):
    result = read_state(output)
    if result is None:
        return None
    if result["status"] == "RUNNING":
        result["status"] = "INTERRUPTED"
        record_failure(output,result,"external_exit",reason)
    for row in result["call_history"]:
        if row["status"] == "RUNNING":
            row.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason=reason)
    refresh_counts(result)
    journal = resume.Journal(output,views=method.VIEWS,steps=())
    journal.save(result); journal.checkpoint("report-only child cleanup")
    return result


def run(config, output, *, loader=None, protocol=carrier.PUBLIC, latent_shape=LATENT_SHAPE,
        support=PUBLIC_LATENT_SUPPORT, source_identity=None):
    import torch
    output = Path(output)
    for location in config["inputs"].values():
        if output.resolve().is_relative_to(Path(location).resolve().parent):
            raise ValueError("independent output must be outside saved input directories")
    prior = read_state(output)
    if prior is None and output.exists() and any(output.iterdir()):
        raise OSError("Nonempty output has no recoverable six-bit state; refusing silent restart")
    output.mkdir(parents=True,exist_ok=True)
    result = prior or initial_result(config)
    attempt = str(time.time_ns())
    if prior is not None:
        resume.atomic_bytes(output/"attempts"/(attempt+"_prior.json"),resume.json_bytes(prior))
        result.setdefault("attempt_history",[]).append(dict(status=prior["status"],
                    failures=copy.deepcopy(prior["failures"]),attempt_id=prior.get("attempt_id")))
    for row in result["call_history"]:
        if row["status"] == "RUNNING":
            row.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason="previous process ended before durable return")
    result.update(status="RUNNING",attempt_id=attempt,failures=[],source=source_identity or {},
                  started_at_unix=time.time(),resources={})
    model = None
    journal = resume.Journal(output,views=method.VIEWS,steps=())

    def flush():
        refresh_counts(result); journal.save(result)

    def call(unit, operation):
        row = dict(id=attempt+"_"+str(len(result["call_history"])).zfill(4),attempt_id=attempt,
                   unit=unit,kind=result["model_calls"][unit]["kind"],status="RUNNING",started_at_unix=time.time())
        result["call_history"].append(row); result["model_calls"][unit] = row
        flush()
        try:
            value = operation()
        except BaseException as exc:
            row.update(status="FAILED" if isinstance(exc,Exception) else "INTERRUPTED_COMPLETION_UNKNOWN",reason=repr(exc))
            flush(); raise
        row.update(status="RETURNED",finished_at_unix=time.time())
        flush()
        return value

    def require_model():
        nonlocal model
        if model is None:
            if str(config.get("device","cuda")).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            model = call("load",lambda:(loader or generation.load_frozen_vae)(config,device=config.get("device","cuda")))
            WanPosteriorBackend(model)
            result["actual_model_execution"] = "REAL" if loader is None else "INJECTED_TEST_DOUBLE"
        return model

    def validate_inputs(bundle):
        tensor_contract(bundle["base"],latent_shape); tensor_contract(bundle["final"],latent_shape)
        support.validate_for(tuple(bundle["final"].shape))
        delta = bundle["final"].double()-bundle["base"].double()
        if torch.count_nonzero(delta-masked(delta,support)) or float(delta.norm()) > 1:
            raise ValueError("actual FINAL must use adopted support and original BASE radius <=1")
        method.validate_values(bundle["reference"])

    def load_inputs():
        path = output/"inputs/fixed_inputs.pt"
        if path.exists():
            bundle = torch.load(path,map_location="cpu",weights_only=True)
        else:
            if any((output/"units").glob("*.pt")) or result["call_history"]:
                raise OSError("Prior model work exists but its fixed input backup is missing; no model restart")
            locations = config["inputs"]
            saved = torch.load(locations["final_step_unit"],map_location="cpu",weights_only=True)
            # The adopted field is the actual CONT_FINAL terminal, never input_terminal (MID).
            final = saved["terminal"]
            base = torch.load(locations["terminal_latent"],map_location="cpu",weights_only=True)
            reference = resume.read_json(locations["saved_final_result"])["views"]["CONT_FINAL"]
            bundle = dict(base=base,final=final,reference=reference,source_locators=locations,
                          input_field="step_2_unit.pt[terminal]",source_arm="CONT_FINAL")
            validate_inputs(bundle)
            resume.atomic_tensor(path,bundle)
        validate_inputs(bundle)
        result["fixed_input"] = dict(source_arm="CONT_FINAL",input_field="terminal",
                    backup="inputs/fixed_inputs.pt",source_locators=bundle["source_locators"],
                    actual_radius=float((bundle["final"].double()-bundle["base"].double()).norm()))
        return bundle

    def validate_unit(unit, bit, inputs):
        if unit["bit_index"] != bit:
            raise ValueError("saved derivative is paired with a different bit")
        tensor_contract(unit["input_terminal"],latent_shape)
        tensor_contract(unit["gradient"],latent_shape)
        if not torch.equal(unit["input_terminal"],inputs["final"]):
            raise ValueError("saved derivative is paired with a different numerical input")
        if torch.count_nonzero(unit["gradient"]-masked(unit["gradient"],support)):
            raise ValueError("saved derivative lies outside fixed support")
        if tuple(unit["q"].shape) != (22,4,16) or not bool(torch.isfinite(unit["q"]).all()):
            raise ValueError("complete finite 88-window / 1408-chip observation required")
        method.validate_values(unit["readout"]["values"])
        if unit["readout"]["bit_index"] != bit or unit["readout"]["derivative_sign"] != "positive_margin":
            raise ValueError("saved readout has wrong differentiated scalar")

    def measure(name, bit, inputs):
        require_model()
        def attach_replay(summary):
            result["replay"][name] = summary
            for row in reversed(result["call_history"]):
                # new_decode emits before model.decode creates its call receipt.
                # It must never attach the new mutable ledger to an older attempt.
                if row["unit"] == name+"_gradient_decode" and row.get("attempt_id") == attempt:
                    row["replay"] = copy.deepcopy(summary)
                    break
        def replay_progress(summary):
            attach_replay(summary)
            flush()
        ledger = ReplayLedger(callback=replay_progress)
        z = inputs["final"].to(config.get("device","cuda")).clone().requires_grad_(True)
        original_decode = model.decode
        try:
            adapter._clear_cache(model)
            model.decode = lambda *a,**k:call(name+"_gradient_decode",lambda:original_decode(*a,**k))
            mean,std = adapter._scale_tensors(model,z)
            decoded = checkpoint_decode(model,z*std+mean,ledger)
            rgb = (decoded[0].permute(1,2,3,0).float()/2+.5).clamp(0,1)
            tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
            read = method.tensor_readout(rgb,key=KEY,message=bytes.fromhex("8001a55a"),protocol=protocol)
            q = read["q"].detach().cpu()
            resume.atomic_tensor(output/"observations"/(name+"_q.pt"),q)
            report = method.readout_report(read,inputs["reference"],bit)
            resume.atomic_bytes(output/"observations"/(name+"_readout.json"),resume.json_bytes(report))
            result["views"][name].update(readout=report,observed_windows=88,observed_chips=1408,observed_metrics=55)
            flush()
            raw = call(name+"_vjp",lambda:torch.autograd.grad(method.selected_margin(read,bit),z)[0]).detach()
            if not bool(torch.isfinite(raw).all()):
                raise FloatingPointError("nonfinite margin derivative")
            gradient = masked(raw,support).cpu()
            norms = dict(raw_l2=float(raw.double().norm()),support_inside_l2=float(gradient.double().norm()),
                         support_outside_before_mask_l2=float((raw.double()-gradient.to(raw.device).double()).norm()),
                         support_outside_after_mask_l2=0.)
            unit = dict(bit_index=bit,input_terminal=inputs["final"],gradient=gradient,q=q,readout=report,
                        gradient_norms=norms,call_ids=[r["id"] for r in result["call_history"]
                             if r.get("attempt_id")==attempt and r["unit"].startswith(name)],
                        source_arm="CONT_FINAL",unnormalized=True)
            resume.atomic_tensor(output/"units"/(name+".pt"),unit)
            return unit
        finally:
            model.decode = original_decode
            try:
                adapter._clear_cache(model)
            finally:
                # Cleanup cannot depend on a successful Drive write in the callback.
                ledger.callback = None
                ledger.release_boundary_storage()
                attach_replay(ledger.summary())

    with journal.periodic():
        stage = "inputs"
        try:
            flush()
            c = config["carrier"]
            if (c["key"],c["message_hex"],c["rho"],c["cap"]) != (KEY,"8001a55a",.5,1.):
                raise ValueError("adopted key/message/rho/cap required")
            if config.get("bits") != list(method.BITS) or config.get("loss") != method.LOSS_PARAMETERS:
                raise ValueError("adopted six indices and scalar-loss parameters required")
            inputs = load_inputs(); flush()
            units = {}
            # Validate every available durable pair before loading a VAE; malformed
            # committed files are concrete data faults, never permission to rerun all.
            for name,bit in zip(method.VIEWS,method.BITS):
                stage = name
                path = output/"units"/(name+".pt")
                if path.exists():
                    unit = torch.load(path,map_location="cpu",weights_only=True)
                    validate_unit(unit,bit,inputs); units[name] = unit
            for name,bit in zip(method.VIEWS,method.BITS):
                stage = name
                if name not in units:
                    result["views"][name]["status"] = "RUNNING"; flush()
                    units[name] = measure(name,bit,inputs)
                unit = units[name]
                result["views"][name].update(status="COMPLETE",reason=None,readout=unit["readout"],
                            gradient_norms=unit["gradient_norms"],observed_windows=88,observed_chips=1408,observed_metrics=55)
                flush(); gc.collect()
            stage = "CPU_joint_diagnostic"
            report,directions = method.analyze([units[n]["gradient"] for n in method.VIEWS],
                    inputs["base"],inputs["final"],[units[n]["readout"] for n in method.VIEWS])
            resume.atomic_tensor(output/"diagnostic_directions.pt",directions)
            resume.atomic_bytes(output/"diagnostic.json",resume.json_bytes(report))
            result.update(status="COMPLETE",diagnostic_status="COMPLETE",diagnostic="diagnostic.json")
        except BaseException as exc:
            result["status"] = "ENGINEERING_FAILURE" if isinstance(exc,Exception) else "INTERRUPTED"
            record_failure(output,result,stage,repr(exc))
            if stage in result["views"]:
                result["views"][stage].update(status="FAILED",reason=repr(exc))
            if not isinstance(exc,Exception):
                flush(); raise
        finally:
            model = None; gc.collect()
            if str(config.get("device","cuda")).startswith("cuda") and torch.cuda.is_initialized():
                result["resources"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
                torch.cuda.empty_cache()
            result["finished_at_unix"] = time.time()
            flush()
    return result
