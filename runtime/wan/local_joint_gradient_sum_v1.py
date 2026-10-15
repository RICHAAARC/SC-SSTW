"""One fixed saved-gradient sum candidate, one ordinary decode, durable resume."""
from __future__ import annotations

import copy
import gc
from pathlib import Path
import time

from main.tube_state import local_joint_gradient_sum_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_terminal_bridge_v1 as observer
from runtime.wan import generation, vae as adapter
from runtime.wan import local_joint_all_bit_quality_v1 as visual
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from runtime.wan import local_joint_posterior_bridge_v1 as rgb_semantics
from runtime.wan import local_joint_protected_common_v1 as prior
from runtime.wan.local_joint_all_bit_v1 import evaluation_complete, validate_raw
from runtime.wan.local_joint_readout_m0_v1 import masked, metric_values
from runtime.wan.local_joint_state_payload_provider_v1 import PUBLIC_LATENT_SUPPORT, WanPosteriorBackend
from runtime.wan.local_joint_terminal_bridge_v1 import KEY, LATENT_SHAPE, tensor_contract

EXPECTED = dict(
    vae_load=1, decode=1, decoder_vjp=0, gradient_decode=0,
    ordinary_decode=1, ordinary_chunks=46,
    encode=0, dit=0, native=0, scheduler=0, codec=0,
    new_windows=88, new_chips=1408, new_metrics=55,
    reused_windows=88, reused_chips=1408, reused_metrics=55,
    reported_windows=176, reported_chips=2816, reported_metrics=110,
    reused_gradient_count=2, new_view_count=1, reused_view_count=1,
    source_points=1, new_source_endpoints=1,
    independent_source_videos=0, independent_samples=0,
)
STEPS = ("construct", "candidate")
FINISHED = ("COMPLETE",)
FORMULA_MAX_ABS_ATOL = 2e-6
FORMULA_L2_ATOL = 2e-5


def initial_result(config):
    calls = {"load": dict(kind="vae_load", status="NOT_RUN"),
             "SUM_STEP": dict(kind="decode", status="NOT_RUN")}
    return dict(
        schema="local-joint-gradient-sum-v1", status="RUNNING", config=copy.deepcopy(config),
        views={name: dict(
            status="NOT_RUN", reason="not_run", expected_windows=88,
            expected_chips=1408, expected_metrics=55, observed_windows=0,
            observed_chips=0, observed_metrics=0,
            origin="reused_saved_BASE" if name == "BASE" else "new_fixed_candidate",
        ) for name in method.VIEWS},
        steps={name: dict(status="NOT_RUN") for name in STEPS},
        model_calls=calls, call_history=[], replay={}, failures=[], failure_history=[],
        cleanup_errors=[], completed_rgb_views=[],
        counts={**{kind + "_" + phase: 0 for kind in ("vae_load", "decode", "decoder_vjp")
                   for phase in ("attempted", "completed")},
                **{kind: 0 for kind in ("encode", "dit", "native", "scheduler", "codec")}},
        ordinary_chunks=dict(attempted=0, completed=0), expected_cost=EXPECTED,
        ceiling=method.CEILING, scientific_pass=False, automatic_retry=False,
        actual_model_execution="NOT_STARTED",
    )


def read_state(output):
    result = resume.read_state(output, views=method.VIEWS, steps=STEPS)
    if result is None:
        return None
    failures = {row["id"]: row for row in result.get("failure_history", [])}
    for path in (Path(output) / "failures").glob("*.json"):
        try:
            row = resume.read_json(path)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError("failure sidecar requires an object and string id")
            failures[row["id"]] = row
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result.setdefault("recovery_events", []).append(dict(path=str(path), reason=repr(exc)))
    result["failure_history"] = list(failures.values())
    return result


def refresh_counts(result):
    resume.recount(result)
    own = [row for row in result["call_history"] if row.get("attempt_id") == result.get("attempt_id")]
    result["current_attempt_counts"] = {
        kind + "_" + phase: sum(row["kind"] == kind and
            (phase == "attempted" or row["status"] == "RETURNED") for row in own)
        for kind in ("vae_load", "decode", "decoder_vjp") for phase in ("attempted", "completed")
    }
    result["totals"] = {
        field: sum(row[field] for row in result["views"].values())
        for field in ("expected_windows", "expected_chips", "expected_metrics",
                      "observed_windows", "observed_chips", "observed_metrics")
    }
    result["totals"].update(source_points=1, new_source_endpoints=1,
                            independent_source_videos=0, independent_samples=0,
                            reused_views=1, new_views=1)


def record_failure(output, result, stage, reason):
    row = dict(id=str(time.time_ns()), attempt_id=result.get("attempt_id"),
               stage=stage, reason=reason)
    result["failures"].append(row)
    result.setdefault("failure_history", []).append(row)
    resume.atomic_bytes(Path(output) / "failures" / (row["id"] + ".json"), resume.json_bytes(row))


def finalize_interrupted(output, reason):
    result = read_state(output)
    if result is None or result["status"] != "RUNNING":
        return result
    result["status"] = "INTERRUPTED"
    record_failure(output, result, "external_exit", reason)
    for row in result["call_history"]:
        if row["status"] == "RUNNING":
            row.update(status="INTERRUPTED_COMPLETION_UNKNOWN", reason=reason)
    refresh_counts(result)
    journal = resume.Journal(output, views=method.VIEWS, steps=STEPS)
    journal.save(result)
    journal.checkpoint("report-only child cleanup")
    return result


def rgb_was_committed(output, state, name="SUM_STEP"):
    output, state = Path(output), state or {}
    return ((output / "rgb_commits" / (name + ".json")).exists()
            or name in state.get("completed_rgb_views", [])
            or state.get("views", {}).get(name, {}).get("status") == "SCORED"
            or (name == "SUM_STEP" and state.get("status") == "COMPLETE")
            or (output / name / "readout.json").exists())


def candidate_decode_pending(output, state):
    return not (Path(output) / "SUM_STEP" / "float_rgb.pt").is_file() and not rgb_was_committed(output, state)


def model_work_remaining(output):
    output = Path(output)
    state = read_state(output)
    if state is None and output.exists() and any(output.iterdir()):
        raise OSError("nonempty output has no recoverable gradient-sum task state")
    if state is not None and state["status"] in FINISHED:
        return False
    return candidate_decode_pending(output, state)


def _source_file(source, relative):
    path = source / relative
    if not path.is_file():
        raise OSError("required saved BASE-alignment artifact is missing: " + str(path)
                      + "; no replacement gradient, BASE decode or VJP")
    return path


def _readout_values(report):
    values = report.get("values", report)
    return dict(margins=list(values["margins"]), correlations=list(values["correlations"]))


def quality_complete(row, frame_count):
    """Validate the saved numeric quality and actual-peak event structure."""
    import math

    def finite(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    def peak(value, *, include_frame=False):
        if not isinstance(value, dict):
            return False
        required = ("y", "x", "channel", "max_abs")
        if include_frame:
            required = ("frame",) + required
        if any(key not in value for key in required) or not finite(value["max_abs"]):
            return False
        return all(isinstance(value[key], int) for key in required if key != "max_abs")

    if not isinstance(row, dict) or not rgb_semantics.has_rgb_semantics(row):
        return False
    frames, temporal, events = row.get("framewise"), row.get("temporal_framewise"), row.get("peak_events")
    if not isinstance(frames, list) or len(frames) != frame_count:
        return False
    if not isinstance(temporal, list) or len(temporal) != frame_count - 1:
        return False
    if not peak(row.get("peak_pixel"), include_frame=True) or not isinstance(events, list):
        return False
    if any(not isinstance(item, dict) or not peak(item.get("peak_pixel")) for item in frames + temporal):
        return False
    for event in events:
        if not isinstance(event, dict) or not peak(event.get("peak_pixel")):
            return False
        if not isinstance(event.get("kind"), str) or not finite(event.get("value")):
            return False
        if not isinstance(event.get("from_frame"), int) or not isinstance(event.get("to_frame"), int):
            return False
        neighborhood = event.get("neighborhood")
        if (not isinstance(neighborhood, list) or not neighborhood
                or any(not isinstance(frame, int) or not 0 <= frame < frame_count for frame in neighborhood)):
            return False
    return True


def visual_index_complete(index, expected_frames, expected_crops):
    """Require one unique rendered entry for every selected frame and crop."""
    if not isinstance(index, dict):
        return False
    if index.get("frame_indices") != expected_frames or index.get("crop_requests") != expected_crops:
        return False
    frames, crops = index.get("frames"), index.get("crops")
    if not isinstance(frames, list) or len(frames) != len(expected_frames):
        return False
    if not isinstance(crops, list) or len(crops) != len(expected_crops):
        return False
    expected_frame_paths = [f"frames/frame_{frame:03d}.png" for frame in expected_frames]
    expected_crop_paths = [f"crops/crop_{number:02d}_{request['frame']:03d}.png"
                           for number, request in enumerate(expected_crops)]
    paths = [item.get("path") if isinstance(item, dict) else None for item in frames + crops]
    return (paths == expected_frame_paths + expected_crop_paths
            and len(paths) == len(set(paths)) and all(isinstance(path, str) for path in paths))


def reports_complete(output, frame_count):
    from PIL import Image
    output = Path(output)
    try:
        quality_reports = {}
        indexes = {}
        for name in method.VIEWS:
            directory = output / name
            if not all((directory / filename).is_file() for filename in
                       ("terminal_latent.pt", "float_rgb.pt", "q.pt", "readout.json", "quality.json", "frames.json")):
                return False
            summary = method.summarize(_readout_values(resume.read_json(directory / "readout.json")))
            if len(summary["margins"]) != 32 or len(summary["correlations"]) != 22:
                return False
            quality = resume.read_json(directory / "quality.json")
            if not quality_complete(quality, frame_count):
                return False
            quality_reports[name] = quality
            indexes[name] = resume.read_json(directory / "frames.json")
        expected_frames, expected_crops = visual.selection(
            quality_reports, frame_count, views=method.VIEWS)
        for name, index in indexes.items():
            if not visual_index_complete(index, expected_frames, expected_crops):
                return False
            directory = output / name
            for item in index["frames"] + index["crops"]:
                with Image.open(directory / item["path"]) as image:
                    image.verify()
        return len(resume.read_json(output / "comparison.json")["bits_32"]) == 32
    except (OSError, ValueError, TypeError, KeyError):
        return False


def refresh_reports(output, result):
    output = Path(output)
    summaries = {}
    for name in method.VIEWS:
        try:
            report = resume.read_json(output / name / "readout.json")
            summary = method.summarize(_readout_values(report))
            summaries[name] = summary
            result["views"][name].update(**summary, status="SCORED", reason=None,
                observed_windows=88, observed_chips=1408, observed_metrics=55)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result["views"][name].update(status="FAILED", reason=repr(exc))
    if set(summaries) != set(method.VIEWS):
        refresh_counts(result)
        return
    base, candidate = summaries["BASE"], summaries["SUM_STEP"]
    bits = []
    for index, (left, right) in enumerate(zip(base["margins"], candidate["margins"])):
        bits.append(dict(index=index, fragment=index // 8, bit=index % 8,
            BASE=left, SUM_STEP=right, delta=right - left,
            BASE_minus_mu=left - method.MU, SUM_STEP_minus_mu=right - method.MU))
    quality = {}
    for name in method.VIEWS:
        try:
            row = resume.read_json(output / name / "quality.json")
            quality[name] = dict(rmse=row["rmse"], max_abs=row["max_abs"],
                framewise=row["framewise"], temporal_framewise=row["temporal_framewise"],
                temporal_residual_rmse=row["temporal_residual_rmse"],
                temporal_residual_max_abs=row["temporal_residual_max_abs"],
                peak_pixel=row["peak_pixel"], peak_events=row["peak_events"],
                rgb_input_semantics=row["rgb_input_semantics"],
                numeric_residual_metrics_use_full_range=True,
                display_residual_range=row["display_residual_range"], no_quality_gate=True)
        except (OSError, ValueError, KeyError):
            quality[name] = None
    comparison = dict(
        bits_32=bits,
        states={name: dict(correlations=row["correlations"], state_gap=row["state_gap"],
            actual_max_wrong_correlation=row["actual_max_wrong_correlation"],
            actual_max_wrong_offsets=row["actual_max_wrong_offsets"],
            fixed_G16=row["fixed_G16"], fixed_mean_A=row["fixed_mean_A"],
            state_gap_minus_gamma=row["state_gap_minus_gamma"])
            for name, row in summaries.items()},
        deltas=dict(
            margins=[right - left for left, right in zip(base["margins"], candidate["margins"])],
            correlations=[right - left for left, right in zip(base["correlations"], candidate["correlations"])],
            state_gap=candidate["state_gap"] - base["state_gap"],
            fixed_G16=candidate["fixed_G16"] - base["fixed_G16"],
            fixed_mean_A=candidate["fixed_mean_A"] - base["fixed_mean_A"],
        ),
        counts={name: dict(positive=row["positive_count"], nonpositive=row["nonpositive_count"])
                for name, row in summaries.items()},
        descriptive_distances=dict(mu=method.MU, gamma=method.GAMMA,
            margins_minus_mu={name: row["margin_minus_mu"] for name, row in summaries.items()},
            state_gap_minus_gamma={name: row["state_gap_minus_gamma"] for name, row in summaries.items()},
            no_threshold_gate=True),
        construction=result["steps"].get("candidate"), quality_vs_BASE=quality,
        ceiling=method.CEILING,
    )
    resume.atomic_bytes(output / "comparison.json", resume.json_bytes(comparison))
    refresh_counts(result)


def run(config, output, *, loader=None, protocol=carrier.PUBLIC, latent_shape=LATENT_SHAPE,
        support=PUBLIC_LATENT_SUPPORT, source_identity=None):
    import torch

    output = Path(output)
    source = Path(config["inputs"]["alignment_run"]).resolve()
    if output.resolve() == source or output.resolve().is_relative_to(source):
        raise ValueError("independent output must be outside the BASE-alignment source run")
    prior_state = read_state(output)
    if prior_state is None and output.exists() and any(output.iterdir()):
        raise OSError("nonempty output has no recoverable gradient-sum task state")
    output.mkdir(parents=True, exist_ok=True)
    result = prior_state or initial_result(config)
    result["completed_rgb_views"] = [name for name in method.VIEWS if rgb_was_committed(output, result, name)]
    attempt = str(time.time_ns())
    if prior_state is not None:
        resume.atomic_bytes(output / "attempts" / (attempt + "_prior.json"), resume.json_bytes(prior_state))
        result.setdefault("attempt_history", []).append(dict(
            attempt_id=prior_state.get("attempt_id"), status=prior_state["status"],
            failures=copy.deepcopy(prior_state["failures"]), runtime=prior_state.get("runtime")))
    for row in result["call_history"]:
        if row["status"] == "RUNNING":
            row.update(status="INTERRUPTED_COMPLETION_UNKNOWN",
                       reason="previous process ended before durable return")
    result.update(status="RUNNING", attempt_id=attempt, failures=[], source=source_identity or {},
                  runtime=prior.runtime_observation(), started_at_unix=time.time(), resources={})
    journal = resume.Journal(output, views=method.VIEWS, steps=STEPS)
    model = None

    def flush():
        refresh_counts(result)
        journal.save(result)

    def save_json(path, value):
        resume.atomic_bytes(path, resume.json_bytes(value))

    def failure(stage, exc):
        record_failure(output, result, stage, repr(exc))
        flush()

    def load_tensor(path):
        return torch.load(path, map_location="cpu", weights_only=True)

    def call(name, operation):
        row = dict(id=attempt + "_" + str(len(result["call_history"])).zfill(4),
            attempt_id=attempt, unit=name, kind=result["model_calls"][name]["kind"],
            status="RUNNING", started_at_unix=time.time())
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

    def fixed_inputs():
        path = output / "inputs" / "fixed_inputs.pt"
        if path.exists():
            bundle = load_tensor(path)
        else:
            if result["call_history"] or (output / "candidate_unit.pt").exists():
                raise OSError("previous work exists but fixed input backup is missing")
            saved = load_tensor(_source_file(source, "inputs/fixed_inputs.pt"))
            base = saved["base"]
            gradients, source_readouts, source_q = {}, {}, {}
            for name in ("aggregate_payload", "state_G16"):
                unit = load_tensor(_source_file(source, "units/" + name + ".pt"))
                if unit.get("name") != name or not torch.equal(unit.get("input_terminal"), base):
                    raise ValueError("saved " + name + " gradient is paired with a different numerical BASE")
                if unit.get("gradient_normalized") is not False:
                    raise ValueError("saved " + name + " must be the unnormalized scalar gradient")
                gradients[name] = unit["gradient"]
                source_readouts[name] = unit["readout"]
                source_q[name] = unit["q"]
            bundle = dict(base=base, gradients=gradients,
                base_readout=source_readouts["aggregate_payload"],
                base_q=source_q["aggregate_payload"], alignment_run=str(source),
                base_float_rgb=str(config["inputs"]["base_float_rgb"]))
            resume.atomic_tensor(path, bundle)
        tensor_contract(bundle["base"], latent_shape)
        support.validate_for(tuple(bundle["base"].shape))
        for name, gradient in bundle["gradients"].items():
            tensor_contract(gradient, latent_shape)
            if not bool(torch.isfinite(gradient).all()) or torch.count_nonzero(gradient - masked(gradient, support)):
                raise ValueError("saved " + name + " gradient is nonfinite or outside the adopted support")
        if tuple(bundle["base_q"].shape) != (22, 4, 16) or not bool(torch.isfinite(bundle["base_q"]).all()):
            raise ValueError("saved BASE q must be the complete finite 88-window observation")
        method.summarize(_readout_values(bundle["base_readout"]))
        result["fixed_input"] = dict(backup="inputs/fixed_inputs.pt", alignment_run=str(source),
            base_float_rgb=str(bundle["base_float_rgb"]), gradient_fields={
                "aggregate_payload": "units/aggregate_payload.pt[gradient]",
                "state_G16": "units/state_G16.pt[gradient]"},
            raw_scalar_weights={"aggregate_payload": 1, "state_G16": 1},
            no_per_gradient_normalization=True)
        flush()
        return bundle

    def decode(terminal):
        require_model()
        original, original_decode = model.decoder.forward, model.decode

        def counted(*args, **kwargs):
            active = next(row for row in reversed(result["call_history"])
                          if row["unit"] == "SUM_STEP" and row["attempt_id"] == attempt)
            chunks = active.setdefault("ordinary_chunks", dict(attempted=0, completed=0))
            chunks["attempted"] += 1
            flush()
            value = original(*args, **kwargs)
            chunks["completed"] += 1
            flush()
            return value

        model.decoder.forward = counted
        model.decode = lambda *args, **kwargs: call("SUM_STEP", lambda: original_decode(*args, **kwargs))
        try:
            return adapter.decode_normalized_latent(
                model, terminal.to(config.get("device", "cuda"))).cpu()
        finally:
            model.decoder.forward, model.decode = original, original_decode

    def observe_base(bundle):
        directory = output / "BASE"
        terminal_path, rgb_path = directory / "terminal_latent.pt", directory / "float_rgb.pt"
        if not terminal_path.exists():
            resume.atomic_tensor(terminal_path, bundle["base"])
        if not torch.equal(load_tensor(terminal_path), bundle["base"]):
            raise ValueError("saved BASE RGB is paired with a different numerical terminal")
        if not rgb_path.exists():
            prior.copy_saved_file(bundle["base_float_rgb"], rgb_path)
        rgb = load_tensor(rgb_path)
        tensor_contract(rgb, protocol.video_shape, bounded_rgb=True)
        if not (directory / "q.pt").exists():
            resume.atomic_tensor(directory / "q.pt", bundle["base_q"])
        if not (directory / "readout.json").exists():
            save_json(directory / "readout.json", bundle["base_readout"])
        marker = output / "rgb_commits" / "BASE.json"
        if not marker.exists():
            save_json(marker, dict(view="BASE", path="BASE/float_rgb.pt",
                                   committed_at_unix=time.time(), attempt_id=attempt))
        if "BASE" not in result["completed_rgb_views"]:
            result["completed_rgb_views"].append("BASE")
        flush()

    def observe_candidate(terminal):
        directory = output / "SUM_STEP"
        terminal_path, rgb_path = directory / "terminal_latent.pt", directory / "float_rgb.pt"
        if not terminal_path.exists():
            resume.atomic_tensor(terminal_path, terminal)
        if not torch.equal(load_tensor(terminal_path), terminal):
            raise ValueError("saved candidate RGB is paired with a different numerical terminal")
        if rgb_path.exists():
            rgb = load_tensor(rgb_path)
        else:
            if not candidate_decode_pending(output, result):
                raise OSError("missing completed candidate RGB; no replacement decode permitted")
            rgb = decode(terminal)
            tensor_contract(rgb, protocol.video_shape, bounded_rgb=True)
            resume.atomic_tensor(rgb_path, rgb)
        tensor_contract(rgb, protocol.video_shape, bounded_rgb=True)
        marker = output / "rgb_commits" / "SUM_STEP.json"
        if not marker.exists():
            save_json(marker, dict(view="SUM_STEP", path="SUM_STEP/float_rgb.pt",
                                   committed_at_unix=time.time(), attempt_id=attempt))
        if "SUM_STEP" not in result["completed_rgb_views"]:
            result["completed_rgb_views"].append("SUM_STEP")
        flush()
        try:
            rows = resume.read_json(directory / "raw.json")
            validate_raw(rows)
        except (OSError, ValueError, TypeError, KeyError):
            rows = observer.read_rgb(rgb, key=KEY, rho=.5, protocol=protocol)
            save_json(directory / "raw.json", rows)
            flush()
        expected_q = prior.raw_q(rows)
        if not prior.q_matches_raw(directory / "q.pt", expected_q):
            resume.atomic_tensor(directory / "q.pt", expected_q)
            flush()
        try:
            evaluation = resume.read_json(directory / "metrics.json")
        except (OSError, ValueError):
            evaluation = None
        if not evaluation_complete(evaluation):
            evaluation = observer.evaluate_saved(rows, key=KEY, message=bytes.fromhex("8001a55a"),
                                                 rho=.5, protocol=protocol)
            save_json(directory / "metrics.json", evaluation)
            flush()
        values = metric_values(evaluation)
        save_json(directory / "readout.json", dict(values=dict(
            margins=values["margins"], correlations=values["correlations"], state_gap=values["state_gap"]),
            source="ordinary saved RGB readout"))
        score_marker = output / "scoring_commits" / "SUM_STEP.json"
        if not score_marker.exists():
            save_json(score_marker, dict(view="SUM_STEP", raw="raw.json", q="q.pt",
                metrics="metrics.json", readout="readout.json", committed_at_unix=time.time(),
                attempt_id=attempt))
        flush()

    def quality_and_visuals():
        base = load_tensor(output / "BASE" / "float_rgb.pt")
        tensor_contract(base, protocol.video_shape, bounded_rgb=True)
        quality_reports = {}
        for name in method.VIEWS:
            path = output / name / "quality.json"
            try:
                row = resume.read_json(path)
            except (OSError, ValueError):
                row = None
            if not quality_complete(row, len(base)):
                rgb = base if name == "BASE" else load_tensor(output / name / "float_rgb.pt")
                tensor_contract(rgb, protocol.video_shape, bounded_rgb=True)
                row = visual.quality(rgb, base, protocol)
                row.update(reference="BASE", target=name, no_quality_threshold=True)
            rgb_semantics.add_rgb_semantics(row)
            save_json(path, row)
            quality_reports[name] = row
            flush()
        expected_frames, expected_crops = visual.selection(
            quality_reports, len(base), views=method.VIEWS)
        for name in method.VIEWS:
            index_path = output / name / "frames.json"
            if not index_path.exists():
                continue
            try:
                index = resume.read_json(index_path)
            except (OSError, ValueError, TypeError):
                index = None
            if not visual_index_complete(index, expected_frames, expected_crops):
                index_path.unlink(missing_ok=True)
        visual.export_visuals(output, base, views=method.VIEWS)
        del base
        gc.collect()

    with journal.periodic():
        stage = "inputs"
        try:
            flush()
            carrier_config = config["carrier"]
            if (config.get("schema"), carrier_config.get("key"), carrier_config.get("message_hex"),
                    carrier_config.get("rho"), carrier_config.get("cap"), config.get("step"),
                    config.get("gradient_weights")) != (
                    "local-joint-gradient-sum-v1", KEY, "8001a55a", .5, method.CAP,
                    method.STEP, {"aggregate_payload": 1, "state_G16": 1}):
                raise ValueError("fixed schema/key/message/rho/cap/step and raw scalar weights 1:1 required")
            bundle = fixed_inputs()
            candidate_path = output / "candidate_unit.pt"
            stage = "construct"
            if candidate_path.exists():
                candidate_unit = load_tensor(candidate_path)
            else:
                terminal, report = method.construct_candidate(bundle["base"], bundle["gradients"], support)
                candidate_unit = dict(terminal=terminal, report=report)
                resume.atomic_tensor(candidate_path, candidate_unit)
                save_json(output / "candidate_commits" / "gradient_sum.json", dict(
                    path="candidate_unit.pt", committed_at_unix=time.time(), attempt_id=attempt))
            if not isinstance(candidate_unit, dict) or "terminal" not in candidate_unit:
                raise ValueError("saved candidate unit must contain the fixed terminal")
            terminal = candidate_unit["terminal"]
            tensor_contract(terminal, latent_shape)
            expected_terminal, report = method.construct_candidate(
                bundle["base"], bundle["gradients"], support)
            difference = terminal.double() - expected_terminal.double()
            max_abs_error = float(difference.abs().max())
            l2_error = float(difference.norm())
            report["stored_formula_validation"] = dict(
                status="MATCHES_FIXED_FORMULA" if max_abs_error <= FORMULA_MAX_ABS_ATOL
                    and l2_error <= FORMULA_L2_ATOL else "FORMULA_MISMATCH",
                max_abs_error=max_abs_error, l2_error=l2_error,
                max_abs_atol=FORMULA_MAX_ABS_ATOL, l2_atol=FORMULA_L2_ATOL,
                bitwise_identity_required=False)
            if max_abs_error > FORMULA_MAX_ABS_ATOL or l2_error > FORMULA_L2_ATOL:
                raise ValueError("stored candidate does not match BASE + 0.10*unit(g_A+g_G) "
                                 "within FP32 arithmetic tolerance; no replacement candidate or decode")
            displacement = terminal.double() - bundle["base"].double()
            if torch.count_nonzero(displacement - masked(displacement, support)) or float(displacement.norm()) > method.CAP:
                raise ValueError("stored fixed candidate violates support or original BASE radius 1")
            result["steps"]["construct"] = report
            result["steps"]["candidate"] = report
            save_json(output / "candidate.json", report)
            flush()
            stage = "BASE"
            observe_base(bundle)
            stage = "SUM_STEP"
            observe_candidate(terminal)
            stage = "quality_and_reports"
            if not reports_complete(output, protocol.video_shape[0]):
                quality_and_visuals()
            else:
                result["last_resume_action"] = "report_only; zero model calls"
            refresh_reports(output, result)
            result["status"] = "COMPLETE" if not result["failures"] and reports_complete(
                output, protocol.video_shape[0]) else "ENGINEERING_FAILURE"
        except BaseException as exc:
            result["status"] = "ENGINEERING_FAILURE" if isinstance(exc, Exception) else "INTERRUPTED"
            if stage in result["views"]:
                result["views"][stage].update(status="FAILED", reason=repr(exc))
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
