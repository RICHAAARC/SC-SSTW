"""One residual-compensated posterior bridge candidate with durable resume."""
from __future__ import annotations

import copy
import gc
import os
from pathlib import Path
import shutil
import time

from main.tube_state import local_joint_posterior_bridge_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_state_payload_posthoc_v1 as posthoc
from main.tube_state import local_joint_terminal_bridge_v1 as observer
from runtime.wan import generation, vae as adapter
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from runtime.wan import local_joint_all_bit_quality_v1 as visual
from runtime.wan.local_joint_all_bit_v1 import (
    compare_values,
    evaluation_complete,
    loss_from_values,
    validate_raw,
)
from runtime.wan.local_joint_readout_m0_v1 import metric_values
from runtime.wan.local_joint_state_payload_provider_v1 import (
    PUBLIC_LATENT_SUPPORT,
    WanPosteriorBackend,
)
from runtime.wan.local_joint_terminal_bridge_v1 import KEY, LATENT_SHAPE, tensor_contract


EXPECTED = dict(
    vae_load=1,
    decode=1,
    decoder_vjp=0,
    ordinary_decode=1,
    encode=0,
    dit=0,
    native=0,
    scheduler=0,
    codec=0,
    ordinary_chunks=46,
    reused_saved_tensor_count=3,
    windows=264,
    chips=4224,
    metrics=165,
    reused_view_count=2,
    new_view_count=1,
)
STEPS = ("construct", "candidate")
FINISHED = ("COMPLETE",)
CANDIDATE = "POSTERIOR_BRIDGE"
SOURCE_VIEW_DIRS = {"BASE": "base", "CAPPED": "capped_reinjection"}
RGB_INPUT_SEMANTICS = "decoder_output_clamped_float_[0,1]"
DISPLAY_RESIDUAL_RANGE = [-.25, .25]


def add_rgb_semantics(row):
    """State the numeric/display domains without changing shared quality math."""
    row.update(
        rgb_input_semantics=RGB_INPUT_SEMANTICS,
        numeric_residual_metrics_use_full_range=True,
        display_residual_range=DISPLAY_RESIDUAL_RANGE,
    )
    display = row.get("display")
    if isinstance(display, dict):
        display.update(
            rgb_input_semantics=RGB_INPUT_SEMANTICS,
            numeric_residual_metrics_use_full_range=True,
            display_residual_range=DISPLAY_RESIDUAL_RANGE,
        )
    return row


def has_rgb_semantics(row):
    return (
        isinstance(row, dict)
        and row.get("rgb_input_semantics") == RGB_INPUT_SEMANTICS
        and row.get("numeric_residual_metrics_use_full_range") is True
        and row.get("display_residual_range") == DISPLAY_RESIDUAL_RANGE
    )


def initial_result(config):
    calls = {
        "load": dict(kind="vae_load", status="NOT_RUN"),
        CANDIDATE: dict(kind="decode", status="NOT_RUN"),
    }
    return dict(
        schema="local-joint-posterior-bridge-v1",
        status="RUNNING",
        config=copy.deepcopy(config),
        views={
            name: dict(
                status="NOT_RUN",
                reason="not_run",
                expected_windows=88,
                expected_chips=1408,
                expected_metrics=55,
                observed_windows=0,
                observed_chips=0,
                observed_metrics=0,
                origin=("new_candidate" if name == CANDIDATE else "reused_saved_view"),
                rgb_available=name in method.RGB_VIEWS,
            )
            for name in method.VIEWS
        },
        steps={name: dict(status="NOT_RUN") for name in STEPS},
        model_calls=calls,
        call_history=[],
        failures=[],
        failure_history=[],
        cleanup_errors=[],
        counts={
            **{
                kind + "_" + phase: 0
                for kind in ("vae_load", "decode", "decoder_vjp")
                for phase in ("attempted", "completed")
            },
            **{kind: 0 for kind in ("encode", "dit", "native", "scheduler", "codec")},
        },
        ordinary_chunks=dict(attempted=0, completed=0),
        expected_cost=EXPECTED,
        ceiling=method.CEILING,
        scientific_pass=False,
        automatic_retry=False,
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
        kind + "_" + phase: sum(
            row["kind"] == kind and (phase == "attempted" or row["status"] == "RETURNED")
            for row in own
        )
        for kind in ("vae_load", "decode", "decoder_vjp")
        for phase in ("attempted", "completed")
    }
    result["totals"] = {
        field: sum(row[field] for row in result["views"].values())
        for field in (
            "expected_windows", "expected_chips", "expected_metrics",
            "observed_windows", "observed_chips", "observed_metrics",
        )
    }
    result["totals"].update(planned_source_points=1, independent_samples=1, reused_views=2)


def record_failure(output, result, stage, reason):
    row = dict(id=str(time.time_ns()), attempt_id=result.get("attempt_id"), stage=stage, reason=reason)
    result["failures"].append(row)
    result["failure_history"].append(row)
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


def rgb_was_committed(output, state, name=CANDIDATE):
    output, state = Path(output), state or {}
    return (
        (output / "rgb_commits" / (name + ".json")).exists()
        or name in state.get("completed_rgb_views", [])
        or state.get("views", {}).get(name, {}).get("status") == "SCORED"
        or (name == CANDIDATE and state.get("status") == "COMPLETE")
        or (output / name / "raw.json").exists()
    )


def candidate_decode_pending(output, state):
    return not (Path(output) / CANDIDATE / "float_rgb.pt").is_file() and not rgb_was_committed(output, state)


def model_work_remaining(output):
    output = Path(output)
    state = read_state(output)
    if state is None and output.exists() and any(output.iterdir()):
        raise OSError("nonempty output has no recoverable task state")
    if state is not None and state["status"] in FINISHED:
        return False
    return candidate_decode_pending(output, state)


def raw_q(rows):
    import torch

    validate_raw(rows)
    return torch.tensor(
        [[chip["q"] for chip in row["state_chips"] + row["payload_chips"]] for row in rows],
        dtype=torch.float64,
    ).reshape(22, 4, 16)


def q_matches_raw(path, expected):
    import torch

    try:
        q = torch.load(path, map_location="cpu", weights_only=True)
        return (torch.is_tensor(q) and tuple(q.shape) == (22, 4, 16)
                and bool(torch.isfinite(q).all()) and torch.equal(q.double(), expected))
    except Exception:
        return False


def runtime_observation():
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = None
    return dict(boot_id=boot, hostname=os.environ.get("HOSTNAME"), process_id=os.getpid())


def copy_saved_file(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with Path(source).open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, target)


def roi_margins(q):
    import torch

    signs = torch.tensor(carrier.message_fragments(bytes.fromhex("8001a55a")), dtype=torch.float64) * 2 - 1
    return torch.stack([
        torch.stack([q[fragment::4, roi, 8:].mean(0) * signs[fragment]
                     for fragment in range(4)]).flatten()
        for roi in range(4)
    ])


def reports_complete(output, frame_count):
    from PIL import Image

    output = Path(output)
    try:
        for name in method.VIEWS:
            directory = output / name
            expected_q = raw_q(resume.read_json(directory / "raw.json"))
            if not q_matches_raw(directory / "q.pt", expected_q):
                return False
            if not evaluation_complete(resume.read_json(directory / "metrics.json")):
                return False
        for name in method.RGB_VIEWS:
            directory = output / name
            if not all((directory / filename).is_file()
                       for filename in ("float_rgb.pt", "terminal_latent.pt", "quality_vs_BASE.json")):
                return False
            quality = resume.read_json(directory / "quality_vs_BASE.json")
            if (len(quality["framewise"]) != frame_count
                    or len(quality["temporal_framewise"]) != frame_count - 1
                    or not has_rgb_semantics(quality)):
                return False
            index = resume.read_json(directory / "frames_vs_BASE.json")
            if not has_rgb_semantics(index):
                return False
            for item in index["frames"] + index["crops"]:
                with Image.open(directory / item["path"]) as image:
                    image.verify()
        comparison = resume.read_json(output / "comparison.json")
        return (
            len(comparison["bits_32"]) == 32
            and all(len(comparison["states"][name]["correlations"]) == 22 for name in method.VIEWS)
            and (output / "CAPPED" / "quality_vs_BASE.json").is_file()
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def refresh_reports(output, result):
    output = Path(output)
    values, rois = {}, {}
    for name in method.VIEWS:
        directory = output / name
        try:
            evaluation = resume.read_json(directory / "metrics.json")
        except (OSError, ValueError):
            evaluation = posthoc.missing_evaluation("missing_readout")
        values[name] = metric_values(evaluation)
        try:
            rows = resume.read_json(directory / "raw.json")
            q = raw_q(rows)
            rois[name] = roi_margins(q)
        except (OSError, ValueError, TypeError, KeyError):
            rows, rois[name] = [], None
        count = sum(value is not None for value in
                    values[name]["correlations"] + values[name]["margins"] + [values[name]["state_gap"]])
        result["views"][name].update(
            **values[name],
            loss=loss_from_values(values[name]),
            observed_metrics=count,
            observed_windows=len(rows),
            observed_chips=sum(chip.get("q") is not None for row in rows
                               for chip in row.get("state_chips", []) + row.get("payload_chips", [])),
        )
        if count == 55 and rois[name] is not None:
            result["views"][name].update(status="SCORED", reason=None)

    table = []
    for index in range(32):
        margins = {name: values[name]["margins"][index] for name in method.VIEWS}
        per_roi = {
            name: None if rois[name] is None else [float(rois[name][roi, index]) for roi in range(4)]
            for name in method.VIEWS
        }
        base, capped, new = margins["BASE"], margins["CAPPED"], margins[CANDIDATE]
        table.append(dict(
            index=index,
            fragment=index // 8,
            bit=index % 8,
            margins=margins,
            roi_margins=per_roi,
            candidate_minus_BASE=None if base is None or new is None else new - base,
            candidate_minus_CAPPED=None if capped is None or new is None else new - capped,
        ))

    quality = {}
    for name in method.VIEWS:
        try:
            row = resume.read_json(output / name / "quality_vs_BASE.json")
            quality[name] = row
        except (OSError, ValueError):
            quality[name] = None
    candidate_margins = values[CANDIDATE]["margins"]
    candidate_gap = values[CANDIDATE]["state_gap"]
    complete_readout = all(value is not None for value in candidate_margins) and candidate_gap is not None
    descriptive = dict(
        all_32_payload_margins_positive=complete_readout and all(value > 0 for value in candidate_margins),
        state_G_at_least_0_01=complete_readout and candidate_gap >= 0.01,
        complete_readout=complete_readout,
        delivery_gate=False,
    )
    descriptive["observed_target_met"] = (
        descriptive["all_32_payload_margins_positive"] and descriptive["state_G_at_least_0_01"]
    )
    comparison = dict(
        bits_32=table,
        versus={reference: compare_values(values[reference], values[CANDIDATE])
                 for reference in ("BASE", "CAPPED")},
        states={name: dict(correlations=view["correlations"], state_gap=view["state_gap"])
                for name, view in values.items()},
        positive_bits={name: sum(value is not None and value > 0 for value in view["margins"])
                       for name, view in values.items()},
        quality_vs_BASE=quality,
        descriptive_target=descriptive,
        controls=dict(primary=list(method.VIEWS),
            note="BASE and CAPPED are reused controls; only POSTERIOR_BRIDGE is newly decoded"),
        ceiling=method.CEILING,
    )
    resume.atomic_bytes(output / "comparison.json", resume.json_bytes(comparison))
    refresh_counts(result)


def run(config, output, *, loader=None, protocol=carrier.PUBLIC, latent_shape=LATENT_SHAPE,
        support=PUBLIC_LATENT_SUPPORT, source_identity=None):
    import torch

    output = Path(output)
    bridge_source = Path(config["inputs"]["bridge_run"]).resolve()
    for location in config["inputs"].values():
        source_directory = Path(location).resolve()
        if ((source_directory.is_dir() and output.resolve().is_relative_to(source_directory))
                or output.resolve() == source_directory):
            raise ValueError("independent output must be outside source input directories")
    prior_state = read_state(output)
    if prior_state is None and output.exists() and any(output.iterdir()):
        raise OSError("nonempty output has no recoverable task state")
    output.mkdir(parents=True, exist_ok=True)
    result = prior_state or initial_result(config)
    result["completed_rgb_views"] = [name for name in method.RGB_VIEWS if rgb_was_committed(output, result, name)]
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
                  runtime=runtime_observation(), started_at_unix=time.time(), resources={})
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
            bundle = dict(
                base=load_tensor(config["inputs"]["base_terminal"]),
                raw=load_tensor(bridge_source / "tensors" / "raw.pt"),
                reconstruction_residual=load_tensor(
                    bridge_source / "tensors" / "reconstruction_residual.pt"),
                source_semantics=(
                    "raw and reconstruction_residual reused under completed terminal-bridge field names; "
                    "no independent encoder-identity verification"
                ),
            )
            resume.atomic_tensor(path, bundle)
        for name in ("base", "raw", "reconstruction_residual"):
            tensor_contract(bundle[name], latent_shape)
        support.validate_for(tuple(bundle["base"].shape))
        result["fixed_input"] = dict(
            backup="inputs/fixed_inputs.pt",
            bridge_run=str(bridge_source),
            base_terminal=str(config["inputs"]["base_terminal"]),
            base_float_rgb=str(config["inputs"]["base_float_rgb"]),
            source_semantics=bundle["source_semantics"],
            independent_algebraic_source_validation=False,
        )
        flush()
        return bundle

    def decode(terminal):
        require_model()
        original, original_decode = model.decoder.forward, model.decode

        def counted(*args, **kwargs):
            active = next(row for row in reversed(result["call_history"])
                          if row["unit"] == CANDIDATE and row["attempt_id"] == attempt)
            chunks = active.setdefault("ordinary_chunks", dict(attempted=0, completed=0))
            chunks["attempted"] += 1
            flush()
            value = original(*args, **kwargs)
            chunks["completed"] += 1
            flush()
            return value

        model.decoder.forward = counted
        model.decode = lambda *args, **kwargs: call(CANDIDATE, lambda: original_decode(*args, **kwargs))
        try:
            return adapter.decode_normalized_latent(
                model, terminal.to(config.get("device", "cuda"))).cpu()
        finally:
            model.decoder.forward, model.decode = original, original_decode

    def read_saved_observation(name):
        directory = output / name
        try:
            rows = resume.read_json(directory / "raw.json")
            validate_raw(rows)
            evaluation = resume.read_json(directory / "metrics.json")
            if not evaluation_complete(evaluation):
                raise ValueError("saved evaluation is incomplete")
            return rows, evaluation
        except (OSError, ValueError, TypeError, KeyError):
            source_directory = bridge_source / SOURCE_VIEW_DIRS[name]
            raw_source = resume.read_json(source_directory / "postclip_raw.json")
            rows = raw_source["rows"] if isinstance(raw_source, dict) else raw_source
            validate_raw(rows)
            evaluation = resume.read_json(source_directory / "postclip_metrics.json")
            if not evaluation_complete(evaluation):
                raise ValueError(f"{name} completed-run evaluation is incomplete")
            save_json(directory / "raw.json", rows)
            save_json(directory / "metrics.json", evaluation)
            flush()
            return rows, evaluation

    def score(name, rows, evaluation):
        directory = output / name
        expected_q = raw_q(rows)
        if not q_matches_raw(directory / "q.pt", expected_q):
            resume.atomic_tensor(directory / "q.pt", expected_q)
            flush()
        if not evaluation_complete(evaluation):
            raise ValueError(f"{name} readout is incomplete")
        marker = output / "scoring_commits" / (name + ".json")
        if not marker.exists():
            save_json(marker, dict(view=name, raw="raw.json", q="q.pt", metrics="metrics.json",
                                   committed_at_unix=time.time(), attempt_id=attempt))

    def observe_saved(name, terminal=None):
        directory = output / name
        if name == "BASE":
            if not (directory / "terminal_latent.pt").exists():
                resume.atomic_tensor(directory / "terminal_latent.pt", terminal)
            if not torch.equal(load_tensor(directory / "terminal_latent.pt"), terminal):
                raise ValueError("saved BASE RGB is paired with a different numerical terminal")
            rgb_path = directory / "float_rgb.pt"
            if not rgb_path.exists():
                copy_saved_file(config["inputs"]["base_float_rgb"], rgb_path)
            rgb = load_tensor(rgb_path)
            tensor_contract(rgb, protocol.video_shape, bounded_rgb=True)
            marker = output / "rgb_commits" / "BASE.json"
            if not marker.exists():
                save_json(marker, dict(view="BASE", path="BASE/float_rgb.pt",
                                       committed_at_unix=time.time(), attempt_id=attempt))
            if "BASE" not in result["completed_rgb_views"]:
                result["completed_rgb_views"].append("BASE")
            flush()
        rows, evaluation = read_saved_observation(name)
        score(name, rows, evaluation)

    def observe_candidate(terminal):
        directory = output / CANDIDATE
        if not (directory / "terminal_latent.pt").exists():
            resume.atomic_tensor(directory / "terminal_latent.pt", terminal)
        if not torch.equal(load_tensor(directory / "terminal_latent.pt"), terminal):
            raise ValueError("saved candidate RGB is paired with a different numerical terminal")
        rgb_path = directory / "float_rgb.pt"
        if rgb_path.exists():
            rgb = load_tensor(rgb_path)
        else:
            if not candidate_decode_pending(output, result):
                raise OSError("missing completed candidate RGB; no replacement decode permitted")
            rgb = decode(terminal)
            tensor_contract(rgb, protocol.video_shape, bounded_rgb=True)
            resume.atomic_tensor(rgb_path, rgb)
        tensor_contract(rgb, protocol.video_shape, bounded_rgb=True)
        marker = output / "rgb_commits" / (CANDIDATE + ".json")
        if not marker.exists():
            save_json(marker, dict(view=CANDIDATE, path=CANDIDATE + "/float_rgb.pt",
                                   committed_at_unix=time.time(), attempt_id=attempt))
        if CANDIDATE not in result["completed_rgb_views"]:
            result["completed_rgb_views"].append(CANDIDATE)
        flush()
        try:
            rows = resume.read_json(directory / "raw.json")
            validate_raw(rows)
        except (OSError, ValueError, TypeError, KeyError):
            rows = observer.read_rgb(rgb, key=KEY, rho=.5, protocol=protocol)
            save_json(directory / "raw.json", rows)
            flush()
        try:
            evaluation = resume.read_json(directory / "metrics.json")
        except (OSError, ValueError):
            evaluation = None
        if not evaluation_complete(evaluation):
            evaluation = observer.evaluate_saved(
                rows, key=KEY, message=bytes.fromhex("8001a55a"), rho=.5, protocol=protocol)
            save_json(directory / "metrics.json", evaluation)
            flush()
        score(CANDIDATE, rows, evaluation)

    def capped_quality_summary():
        path = output / "CAPPED" / "quality_vs_BASE.json"
        if path.exists():
            return
        source_result = resume.read_json(bridge_source / "result.json")
        summary = source_result["stages"]["capped_reinjection"]["views"]["postclip"].get("rgb_vs_base")
        if not isinstance(summary, dict):
            raise ValueError("completed CAPPED quality summary is missing")
        save_json(path, dict(status="SAVED_SUMMARY_ONLY", reference="BASE", target="CAPPED",
                             available=summary, framewise=None, temporal_framewise=None,
                             reason="completed terminal-bridge did not persist CAPPED float RGB",
                             no_quality_threshold=True))

    def quality_and_visuals():
        from PIL import Image

        reference = load_tensor(output / "BASE" / "float_rgb.pt")
        tensor_contract(reference, protocol.video_shape, bounded_rgb=True)
        reports = {}
        for name in method.RGB_VIEWS:
            directory = output / name
            path = directory / "quality_vs_BASE.json"
            try:
                row = resume.read_json(path)
            except (OSError, ValueError):
                row = None
            if not isinstance(row, dict) or len(row.get("framewise", [])) != len(reference) \
                    or len(row.get("temporal_framewise", [])) != len(reference) - 1:
                rgb = reference if name == "BASE" else load_tensor(directory / "float_rgb.pt")
                tensor_contract(rgb, protocol.video_shape, bounded_rgb=True)
                row = visual.quality(rgb, reference, protocol)
                row.update(reference="BASE", target=name, no_quality_threshold=True)
                del rgb
            previous = copy.deepcopy(row)
            add_rgb_semantics(row)
            if row != previous or not path.exists():
                save_json(path, row)
                flush()
            reports[name] = row
        capped_quality_summary()
        frames, crops = visual.selection(reports, len(reference), views=method.RGB_VIEWS)
        for name in method.RGB_VIEWS:
            directory = output / name
            index_path = directory / "frames_vs_BASE.json"
            try:
                old = resume.read_json(index_path)
            except (OSError, ValueError):
                old = None
            if old is not None and old.get("frame_indices") == frames and old.get("crop_requests") == crops:
                try:
                    for item in old["frames"] + old["crops"]:
                        with Image.open(directory / item["path"]) as image:
                            image.verify()
                    previous = copy.deepcopy(old)
                    add_rgb_semantics(old)
                    if old != previous:
                        save_json(index_path, old)
                        flush()
                    continue
                except (OSError, ValueError, KeyError):
                    pass
            rgb = reference if name == "BASE" else load_tensor(directory / "float_rgb.pt")
            rendered = [visual.render_pair(
                directory / "frames_vs_BASE" / f"frame_{frame:03d}.png",
                name, frame, rgb[frame], reference[frame]) for frame in frames]
            crop_images = []
            for number, request in enumerate(crops):
                frame, y, x = request["frame"], request["y"], request["x"]
                box = (max(0, y - visual.CROP_RADIUS), min(rgb.shape[1], y + visual.CROP_RADIUS + 1),
                       max(0, x - visual.CROP_RADIUS), min(rgb.shape[2], x + visual.CROP_RADIUS + 1))
                crop_images.append(visual.render_pair(
                    directory / "crops_vs_BASE" / f"crop_{number:02d}_{frame:03d}.png",
                    name, frame, rgb[frame], reference[frame], box=box))
            save_json(index_path, add_rgb_semantics(dict(reference="BASE", target=name, frame_indices=frames,
                crop_requests=crops, frames=rendered, crops=crop_images, rgb_range=[0, 1],
                residual_range=[-.25, .25], numeric_metrics_unclipped=True)))
            del rgb
            flush()
        del reference
        gc.collect()

    with journal.periodic():
        stage = "inputs"
        try:
            flush()
            c = config["carrier"]
            if (config.get("schema"), c["key"], c["message_hex"], c["rho"], c["cap"],
                    c["lambda_residual"], c["numerical_radius"]) != (
                    "local-joint-posterior-bridge-v1", KEY, "8001a55a", .5,
                    method.CAP, method.LAMBDA, method.NUMERICAL_RADIUS):
                raise ValueError("adopted schema/key/message/rho/lambda/mask and numerical radius required")
            bundle = fixed_inputs()
            candidate_path = output / "candidate_unit.pt"
            stage = "construct"
            if candidate_path.exists():
                candidate_unit = load_tensor(candidate_path)
            else:
                terminal, report = method.construct_candidate(
                    bundle["base"], bundle["raw"], bundle["reconstruction_residual"], support)
                candidate_unit = dict(terminal=terminal, report=report)
                resume.atomic_tensor(candidate_path, candidate_unit)
                save_json(output / "candidate_commits" / "posterior_bridge.json",
                          dict(path="candidate_unit.pt", committed_at_unix=time.time(), attempt_id=attempt))
            terminal, report = candidate_unit["terminal"], candidate_unit["report"]
            tensor_contract(terminal, latent_shape)
            actual = terminal.double() - bundle["base"].double()
            outside = actual - method.masked(actual, support)
            if float(actual.norm()) > method.CAP or torch.count_nonzero(outside):
                raise ValueError("stored candidate violates adopted support or BASE radius 1")
            result["steps"]["construct"] = report
            result["steps"]["candidate"] = report
            save_json(output / "construct.json", report)
            save_json(output / "candidate.json", report)
            flush()

            stage = "BASE"
            observe_saved("BASE", bundle["base"])
            stage = "CAPPED"
            observe_saved("CAPPED")
            stage = CANDIDATE
            observe_candidate(terminal)
            stage = "quality_and_reports"
            if not reports_complete(output, protocol.video_shape[0]):
                quality_and_visuals()
                if not any(row.get("attempt_id") == attempt for row in result["call_history"]):
                    result["last_resume_action"] = "report_only; zero model calls"
            else:
                result["last_resume_action"] = "report_only; zero model calls"
            refresh_reports(output, result)
            result["status"] = (
                "COMPLETE" if not result["failures"] and reports_complete(output, protocol.video_shape[0])
                else "ENGINEERING_FAILURE"
            )
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
                result["resources"].update(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
                torch.cuda.empty_cache()
            result["finished_at_unix"] = time.time()
            flush()
    return result
