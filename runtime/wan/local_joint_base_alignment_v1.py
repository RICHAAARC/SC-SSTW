"""Resumable two-VJP BASE alignment diagnostic on the existing Wan spool path."""
from __future__ import annotations

import copy
import gc
from pathlib import Path
import time

from main.tube_state import local_joint_base_alignment_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from runtime.wan import generation, vae as adapter
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from runtime.wan.local_joint_readout_checkpoint_v1 import ReplayLedger, checkpoint_decode
from runtime.wan.local_joint_readout_m0_v1 import masked
from runtime.wan.local_joint_state_payload_provider_v1 import PUBLIC_LATENT_SUPPORT, WanPosteriorBackend
from runtime.wan.local_joint_terminal_bridge_v1 import KEY, LATENT_SHAPE, tensor_contract

EXPECTED = dict(vae_load=1, decode=2, decoder_vjp=2, ordinary_decode=0, encode=0,
                dit=0, native=0, scheduler=0, codec=0, ordinary_chunks=0,
                gradient_forward_chunks=92, nominal_replay_chunks=92,
                windows=176, chips=2816, metrics=110, independent_source_points=1)
FINISHED = ("COMPLETE",)


def initial_result(config):
    calls = {"load": dict(kind="vae_load", status="NOT_RUN")}
    for name in method.UNITS:
        calls[name+"_gradient_decode"] = dict(kind="decode", status="NOT_RUN")
        calls[name+"_vjp"] = dict(kind="decoder_vjp", status="NOT_RUN")
    return dict(schema="local-joint-base-alignment-v1", status="RUNNING", config=copy.deepcopy(config),
        units={name: dict(status="NOT_RUN", expected_windows=88, expected_chips=1408,
                         expected_metrics=55, observed_windows=0, observed_chips=0,
                         observed_metrics=0) for name in method.UNITS},
        views={}, steps={}, model_calls=calls, call_history=[], replay={}, failures=[],
        failure_history=[], cleanup_errors=[], completed_units=[],
        counts={**{kind+"_"+phase: 0 for kind in ("vae_load", "decode", "decoder_vjp")
                   for phase in ("attempted", "completed")},
                **{kind: 0 for kind in ("ordinary_decode", "encode", "dit", "native", "scheduler", "codec")}},
        ordinary_chunks=dict(attempted=0, completed=0), expected_cost=EXPECTED,
        ceiling=method.CEILING, scientific_pass=False, automatic_retry=False,
        actual_model_execution="NOT_STARTED", diagnostic_status="NOT_RUN")


def read_state(output):
    result = resume.read_state(output, views=(), steps=())
    if result is None:
        return None
    failures = {row["id"]: row for row in result.get("failure_history", [])}
    for path in (Path(output)/"failures").glob("*.json"):
        try:
            row = resume.read_json(path)
            failures[row["id"]] = row
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result.setdefault("recovery_events", []).append(dict(path=str(path), reason=repr(exc)))
    result["failure_history"] = list(failures.values())
    for row in failures.values():
        if row.get("attempt_id") == result.get("attempt_id") and row not in result["failures"]:
            result["failures"].append(row)
    return result


def record_failure(output, result, stage, reason):
    row = dict(id=str(time.time_ns()), attempt_id=result.get("attempt_id"), stage=stage, reason=reason)
    result["failures"].append(row)
    result.setdefault("failure_history", []).append(row)
    resume.atomic_bytes(Path(output)/"failures"/(row["id"]+".json"), resume.json_bytes(row))


def refresh_counts(result):
    resume.recount(result)
    own = [row for row in result["call_history"] if row.get("attempt_id") == result.get("attempt_id")]
    result["current_attempt_counts"] = {kind+"_"+phase: sum(
        row["kind"] == kind and (phase == "attempted" or row["status"] == "RETURNED") for row in own)
        for kind in ("vae_load", "decode", "decoder_vjp") for phase in ("attempted", "completed")}
    result["totals"] = {"expected_"+field: EXPECTED[field] for field in ("windows", "chips", "metrics")}
    result["totals"].update({"observed_"+field: sum(row["observed_"+field]
        for row in result["units"].values()) for field in ("windows", "chips", "metrics")})
    result["totals"].update(planned_source_points=1, repeated_reads_are_independent_samples=False)
    result["physical_replay_chunks"] = {phase: sum(row.get("replay", {}).get("counts", {})
        .get("vae_chunk", {}).get(phase, {}).get("attempted", 0) for row in result["call_history"])
        for phase in ("forward", "recompute")}


def unit_was_committed(output, state, name):
    state = state or {}
    return ((Path(output)/"unit_commits"/(name+".json")).exists()
            or name in state.get("completed_units", [])
            or state.get("units", {}).get(name, {}).get("status") == "COMPLETE")


def unit_file_problem(output, state):
    output = Path(output)
    for name in method.UNITS:
        path = output/"units"/(name+".pt")
        if not path.is_file() and unit_was_committed(output, state, name):
            return name, "missing committed gradient unit: "+str(path)+"; no replacement VJP"
        if not path.is_file():
            continue
        try:
            import torch
            unit = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(unit, dict) or unit.get("name") != name:
                raise ValueError("gradient unit has a different or missing scalar name")
        except Exception as exc:
            return name, "unreadable saved gradient unit: "+str(path)+"; "+repr(exc)
    return None


def model_work_remaining(output):
    output = Path(output)
    state = read_state(output)
    if state is None and output.exists() and any(output.iterdir()):
        raise OSError("nonempty output has no recoverable BASE-alignment state")
    if state is not None and state.get("status") in FINISHED:
        return False
    return any(not (output/"units"/(name+".pt")).is_file()
               and not unit_was_committed(output, state, name) for name in method.UNITS)


def finalize_interrupted(output, reason):
    result = read_state(output)
    if result is None:
        return None
    if result["status"] == "RUNNING":
        result["status"] = "INTERRUPTED"
        record_failure(output, result, "external_exit", reason)
    for row in result["call_history"]:
        if row["status"] == "RUNNING":
            row.update(status="INTERRUPTED_COMPLETION_UNKNOWN", reason=reason)
    refresh_counts(result)
    journal = resume.Journal(output, views=(), steps=())
    journal.save(result)
    journal.checkpoint("report-only child cleanup")
    return result


def run(config, output, *, loader=None, protocol=carrier.PUBLIC, latent_shape=LATENT_SHAPE,
        support=PUBLIC_LATENT_SUPPORT, source_identity=None):
    import torch

    output = Path(output)
    for name, location in config["inputs"].items():
        source = Path(location).resolve() if name == "bridge_run" else Path(location).resolve().parent
        if output.resolve() == source or output.resolve().is_relative_to(source):
            raise ValueError("independent output must be outside saved input locations")
    prior = read_state(output)
    if prior is None and output.exists() and any(output.iterdir()):
        raise OSError("nonempty output has no recoverable BASE-alignment state")
    output.mkdir(parents=True, exist_ok=True)
    result = prior or initial_result(config)
    result["completed_units"] = [name for name in method.UNITS if unit_was_committed(output, result, name)]
    attempt = str(time.time_ns())
    if prior is not None:
        resume.atomic_bytes(output/"attempts"/(attempt+"_prior.json"), resume.json_bytes(prior))
        result.setdefault("attempt_history", []).append(dict(
            status=prior["status"], failures=copy.deepcopy(prior["failures"]),
            attempt_id=prior.get("attempt_id"), runtime=prior.get("runtime")))
    for row in result["call_history"]:
        if row["status"] == "RUNNING":
            row.update(status="INTERRUPTED_COMPLETION_UNKNOWN",
                       reason="previous process ended before durable return")
    result.update(status="RUNNING", attempt_id=attempt, failures=[], source=source_identity or {},
                  started_at_unix=time.time(), resources={})
    journal = resume.Journal(output, views=(), steps=())
    model = None

    def flush():
        refresh_counts(result)
        journal.save(result)

    def save_json(path, value):
        resume.atomic_bytes(path, resume.json_bytes(value))

    def failure(stage, exc):
        record_failure(output, result, stage, repr(exc))
        flush()

    def call(name, operation):
        row = dict(id=attempt+"_"+str(len(result["call_history"])).zfill(4), attempt_id=attempt,
                   unit=name, kind=result["model_calls"][name]["kind"], status="RUNNING",
                   started_at_unix=time.time())
        result["call_history"].append(row)
        result["model_calls"][name] = row
        flush()
        try:
            value = operation()
        except BaseException as exc:
            row.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED_COMPLETION_UNKNOWN",
                       reason=repr(exc))
            flush()
            raise
        row.update(status="RETURNED", finished_at_unix=time.time())
        flush()
        return value

    def require_model():
        nonlocal model
        if model is None:
            if str(config.get("device", "cuda")).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            model = call("load", lambda: (loader or generation.load_frozen_vae)(
                config, device=config.get("device", "cuda")))
            WanPosteriorBackend(model)
            result["actual_model_execution"] = "REAL" if loader is None else "INJECTED_TEST_DOUBLE"
        return model

    def load_inputs():
        path = output/"inputs/fixed_inputs.pt"
        if path.exists():
            bundle = torch.load(path, map_location="cpu", weights_only=True)
        else:
            if result["call_history"] or any((output/"units").glob("*.pt")):
                raise OSError("previous model work exists but fixed input backup is missing")
            locations = config["inputs"]
            six = torch.load(locations["six_bit_inputs"], map_location="cpu", weights_only=True)
            bridge = Path(locations["bridge_run"])
            bundle = dict(base=six["base"], final=six["final"],
                          raw=torch.load(bridge/"tensors/raw.pt", map_location="cpu", weights_only=True),
                          reconstruction_residual=torch.load(
                              bridge/"tensors/reconstruction_residual.pt", map_location="cpu", weights_only=True),
                          source_locators=copy.deepcopy(locations),
                          source_fields=dict(base="six_bit_inputs[base]", final="six_bit_inputs[final]",
                              raw="bridge_run/tensors/raw.pt",
                              reconstruction_residual="bridge_run/tensors/reconstruction_residual.pt"))
            resume.atomic_tensor(path, bundle)
        for name in ("base", "final", "raw", "reconstruction_residual"):
            tensor_contract(bundle[name], latent_shape)
        support.validate_for(tuple(bundle["base"].shape))
        historical = bundle["final"].double()-bundle["base"].double()
        if torch.count_nonzero(historical-masked(historical, support)) or float(historical.norm()) > 1:
            raise ValueError("saved BASE-to-CONT_FINAL direction must retain the fixed support and radius <=1")
        result["fixed_input"] = dict(backup="inputs/fixed_inputs.pt",
            source_locators=bundle["source_locators"], source_fields=bundle["source_fields"],
            base_point="six_bit_inputs[base]", historical_direction="final-base",
            historical_direction_l2=float(historical.norm()))
        flush()
        return bundle

    def validate_unit(unit, name, inputs):
        if unit.get("name") != name or not torch.equal(unit.get("input_terminal"), inputs["base"]):
            raise ValueError("saved derivative is paired with a different scalar or numerical BASE")
        for field in ("raw_gradient", "gradient"):
            tensor_contract(unit[field], latent_shape)
        if torch.count_nonzero(unit["gradient"]-masked(unit["raw_gradient"], support)):
            raise ValueError("saved masked derivative does not equal M(raw gradient)")
        q = unit["q"]
        if tuple(q.shape) != (22, 4, 16) or not bool(torch.isfinite(q).all()):
            raise ValueError("complete finite 88-window / 1408-chip observation required")
        report = unit["readout"]
        if report.get("name") != name or len(report["values"]["margins"]) != 32 \
                or len(report["values"]["correlations"]) != 22:
            raise ValueError("saved readout has the wrong scalar or fixed slots")

    def mark_committed(name):
        marker = output/"unit_commits"/(name+".json")
        if not marker.exists():
            save_json(marker, dict(unit=name, path="units/"+name+".pt",
                                   committed_at_unix=time.time(), attempt_id=attempt))
        if name not in result["completed_units"]:
            result["completed_units"].append(name)

    def measure(name, inputs):
        require_model()

        def attach(summary):
            result["replay"][name] = summary
            for row in reversed(result["call_history"]):
                if row["unit"] == name+"_gradient_decode" and row.get("attempt_id") == attempt:
                    row["replay"] = copy.deepcopy(summary)
                    break

        ledger = ReplayLedger(callback=lambda summary: (attach(summary), flush()))
        z = inputs["base"].to(config.get("device", "cuda")).clone().requires_grad_(True)
        original_decode = model.decode
        try:
            adapter._clear_cache(model)
            model.decode = lambda *args, **kwargs: call(
                name+"_gradient_decode", lambda: original_decode(*args, **kwargs))
            mean, std = adapter._scale_tensors(model, z)
            decoded = checkpoint_decode(model, z*std+mean, ledger)
            rgb = (decoded[0].permute(1, 2, 3, 0).float()/2+.5).clamp(0, 1)
            tensor_contract(rgb, protocol.video_shape, bounded_rgb=True)
            read = method.tensor_readout(rgb, key=KEY, message=bytes.fromhex("8001a55a"), protocol=protocol)
            q = read["q"].detach().cpu()
            report = method.readout_report(name, read)
            resume.atomic_tensor(output/"observations"/(name+"_q.pt"), q)
            save_json(output/"observations"/(name+"_readout.json"), report)
            result["units"][name].update(readout=report, observed_windows=88,
                                          observed_chips=1408, observed_metrics=55)
            flush()
            objective = method.scalar(name, read)
            raw = call(name+"_vjp", lambda: torch.autograd.grad(objective, z)[0]).detach()
            if not bool(torch.isfinite(raw).all()):
                raise FloatingPointError("nonfinite BASE scalar derivative")
            gradient = masked(raw, support).cpu()
            outside = raw.double()-gradient.to(raw.device).double()
            unit = dict(name=name, input_terminal=inputs["base"], raw_gradient=raw.cpu(),
                gradient=gradient, q=q, readout=report,
                gradient_norms=dict(raw_l2=float(raw.double().norm()), masked_l2=float(gradient.double().norm()),
                    outside_before_mask_l2=float(outside.norm()), outside_after_mask_l2=0.0),
                call_ids=[row["id"] for row in result["call_history"]
                          if row.get("attempt_id") == attempt and row["unit"].startswith(name)],
                independent_graph=True, gradient_normalized=False)
            resume.atomic_tensor(output/"units"/(name+".pt"), unit)
            mark_committed(name)
            return unit
        finally:
            model.decode = original_decode
            try:
                adapter._clear_cache(model)
            finally:
                ledger.callback = None
                ledger.release_boundary_storage()
                attach(ledger.summary())

    with journal.periodic():
        stage = "inputs"
        try:
            flush()
            c = config["carrier"]
            if (config.get("schema"), c.get("key"), c.get("message_hex"), c.get("rho"), c.get("cap"),
                    config.get("nonpositive_roster"), config.get("fixed_wrong_offset")) != (
                    "local-joint-base-alignment-v1", KEY, "8001a55a", .5, 1.0,
                    list(method.NONPOSITIVE_ROSTER), method.FIXED_WRONG_OFFSET):
                raise ValueError("fixed BASE point key/message/rho/cap/roster/G16 semantics required")
            inputs = load_inputs()
            units = {}
            for name in method.UNITS:
                stage = name
                path = output/"units"/(name+".pt")
                try:
                    if not path.is_file() and unit_was_committed(output, result, name):
                        raise OSError("missing committed gradient unit: "+str(path)+"; no replacement VJP")
                    if path.is_file():
                        unit = torch.load(path, map_location="cpu", weights_only=True)
                    else:
                        load_row = result["model_calls"]["load"]
                        if (load_row.get("status") == "FAILED"
                                and load_row.get("attempt_id") == attempt):
                            result["units"][name].update(status="BLOCKED_SHARED_MODEL_LOAD",
                                reason="shared VAE load failed before this fixed unit could run")
                            flush()
                            continue
                        result["units"][name]["status"] = "RUNNING"
                        flush()
                        unit = measure(name, inputs)
                    validate_unit(unit, name, inputs)
                    units[name] = unit
                    result["units"][name].update(status="COMPLETE", reason=None,
                        readout=unit["readout"], gradient_norms=unit["gradient_norms"],
                        observed_windows=88, observed_chips=1408, observed_metrics=55)
                    mark_committed(name)
                    flush()
                except Exception as exc:
                    result["units"][name].update(status="FAILED", reason=repr(exc))
                    failure(name, exc)
                finally:
                    gc.collect()
            if len(units) == len(method.UNITS):
                stage = "CPU_projection"
                directions = dict(
                    old_bridge=masked(inputs["raw"].double(), support).cpu(),
                    posterior_bridge=masked(
                        inputs["raw"].double()-inputs["reconstruction_residual"].double(), support).cpu(),
                    base_to_cont_final=masked(inputs["final"].double()-inputs["base"].double(), support).cpu())
                diagnostic = method.analyze(
                    {name: units[name]["gradient"] for name in method.UNITS}, directions)
                diagnostic["direction_sources"] = inputs["source_fields"]
                diagnostic["direction_geometry"] = {
                    "old_bridge": dict(source_l2=float(inputs["raw"].double().norm()),
                        masked_l2=float(directions["old_bridge"].norm()),
                        outside_mask_l2=float((inputs["raw"].double()-directions["old_bridge"]).norm())),
                    "posterior_bridge": dict(
                        source_l2=float((inputs["raw"].double()-inputs["reconstruction_residual"].double()).norm()),
                        masked_l2=float(directions["posterior_bridge"].norm()),
                        outside_mask_l2=float(((inputs["raw"].double()-inputs["reconstruction_residual"].double())
                                               -directions["posterior_bridge"]).norm())),
                    "base_to_cont_final": dict(source_l2=float((inputs["final"].double()-inputs["base"].double()).norm()),
                        masked_l2=float(directions["base_to_cont_final"].norm()), outside_mask_l2=0.0),
                }
                save_json(output/"diagnostic.json", diagnostic)
                result.update(status="COMPLETE", diagnostic_status="COMPLETE", diagnostic="diagnostic.json")
            else:
                result.update(status="ENGINEERING_FAILURE", diagnostic_status="MISSING_FIXED_UNITS",
                              diagnostic=None)
        except BaseException as exc:
            result["status"] = "ENGINEERING_FAILURE" if isinstance(exc, Exception) else "INTERRUPTED"
            if stage in result["units"]:
                result["units"][stage].update(status="FAILED", reason=repr(exc))
            failure(stage, exc)
            if not isinstance(exc, Exception):
                raise
        finally:
            model = None
            gc.collect()
            if str(config.get("device", "cuda")).startswith("cuda") and torch.cuda.is_initialized():
                result["resources"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
                torch.cuda.empty_cache()
            result["finished_at_unix"] = time.time()
            flush()
    return result
