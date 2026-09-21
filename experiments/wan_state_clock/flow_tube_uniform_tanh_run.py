"""Fixed Uniform-Tanh-44/46 user-run experiment with retained denominators."""
import argparse
import copy
import gc
import hashlib
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from main.tube_state import objective_alignment as objective, projection_margin, state_clock
from runtime.wan import tube_uniform_tanh as runtime
from runtime.wan.generation import load_frozen_vae
from runtime.wan.io import dump, encode_rgb, read_mp4
from runtime.wan.vae import decode_normalized_latent, quantize_rgb8_no_codec, reencode_rgb24_readback, _clear_cache
from .flow_run import quality

MANIFEST = Path(__file__).parent / "configs" / "flow_tube_uniform_tanh.json"
CASES = ("dev_p0_s0", "dev_p1_s0")
ARMS = ("OFF", "SINGLE46_A", "SINGLE46_B", "MULTI44_46_A", "MULTI44_46_B")
LAYERS = ("floatRGB", "RGB8", "MP4")
MODES = ("global_matched", "global_state", "local_matched", "local_without_update", "local_state")
SOURCE_COMMIT = "9fdfac97126fa681c42423089eb342ce85b676a7"
SOURCE_RUN = "flow_tube_response_selection_20260921T013844126172Z"
MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
R_STAR = 0.042943312697648145
HISTORICAL_BUDGETS = (0.04398231690421802, 0.04495011965027108, 0.042943312697648145, 0.04306408796221067)
PLAN = dict(transformer=42, scheduler_step=26, unit_response_probe_step=6,
            second_control_zero_shadow_step=2, clean_leaf_backward=6,
            vae_decode=5, mp4_save=5, vae_encode=60)
RGB_SHAPE = (181, 320, 512, 3)
MODULE = "experiments.wan_state_clock.flow_tube_uniform_tanh_run"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def empty(status):
    return dict(
        status=status,
        videos={arm: dict(status="NOT_RUN", layers={layer: dict(status="NOT_RUN", observations={str(g): dict(status="NOT_RUN") for g in range(4)}) for layer in LAYERS}) for arm in ARMS},
        failures=[],
    )


def guard(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("input/output overlap")
    return source, output


def locate_source_run(drive_root, source_run=SOURCE_RUN):
    """Locate the exact run directory without gating its case/file completeness."""
    drive_root = Path(drive_root)
    preferred = drive_root / "Video-WM" / "FlowTubeResponseSelection" / source_run
    candidates = [path.resolve() for path in drive_root.rglob(source_run) if path.is_dir()]
    unique = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    if len(unique) > 1:
        raise RuntimeError(f"Ambiguous {source_run} directories: {unique}")
    if unique:
        return unique[0]
    # Passing the expected absent path into run_all preserves all fixed failed slots.
    return preferred.resolve()


def validate_manifest(manifest):
    if manifest["name"] != "flow_tube_uniform_tanh" or manifest["protocol"] != "Uniform-Tanh-44/46":
        raise ValueError("wrong fixed protocol")
    if tuple(manifest["arms"]) != ARMS or tuple(row["id"] for row in manifest["cases"]) != CASES:
        raise ValueError("fixed roster mismatch")
    if manifest["model"]["revision"] != MODEL_REVISION or manifest["budget"]["R_star"] != R_STAR:
        raise ValueError("fixed revision/budget mismatch")
    if tuple(manifest["budget"]["historical_complete_dev_native_response_rms"]) != HISTORICAL_BUDGETS or min(HISTORICAL_BUDGETS) != R_STAR:
        raise ValueError("historical oracle provenance mismatch")
    if manifest["fixed_calls_per_case"] != PLAN:
        raise ValueError("fixed call plan mismatch")


def restore(source, case, result, manifest):
    """Hash-verify saved44 inputs and return only z44/v44/history44 plus prompt/book."""
    root = source / case
    generation = load(root / "generation.json")
    if generation.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("wrong historical source commit")
    result["source_inputs"] = [dict(path=str(root / "generation.json"), sha256=sha(root / "generation.json"))]
    files = ("config.json", "codebook.npz", "OFF_nodes.pt", "OFF_snapshots.pt", "prompt.pt", "negative.pt")
    for name in files:
        actual = sha(root / name)
        expected = generation.get("file_sha256", {}).get(name)
        result["source_inputs"].append(dict(path=str(root / name), sha256=actual, expected_sha256=expected, matched=actual == expected))
        if actual != expected:
            raise ValueError("source artifact hash mismatch: " + name)
    for module in (state_clock, projection_margin):
        suffix = "/main/tube_state/" + Path(module.__file__).name
        expected = next((value for key, value in generation.get("source_sha256", {}).items() if key.endswith(suffix)), None)
        if expected is None or sha(module.__file__) != expected:
            raise ValueError("historical carrier/reader source mismatch: " + suffix)
    source_config = load(root / "config.json")
    fixed_case = next(row for row in manifest["cases"] if row["id"] == case)
    generation_config = source_config.get("generation", {})
    expected_generation = manifest["generation"]
    for key in ("height", "width", "frames", "fps", "steps", "guidance_scale"):
        if generation_config.get(key) != expected_generation[key]:
            raise ValueError("historical generation config mismatch: " + key)
    if generation_config.get("negative_prompt") != expected_generation["negative_prompt"]:
        raise ValueError("historical negative prompt mismatch")
    if generation_config.get("prompt") != fixed_case["prompt"] or generation_config.get("seed") != fixed_case["seed"]:
        raise ValueError("historical content/seed mismatch")
    if source_config.get("key_utf8") != manifest["key_utf8"] or source_config.get("model", {}).get("id") != manifest["model"]["id"]:
        raise ValueError("historical carrier/model mismatch")
    nodes_all = torch.load(root / "OFF_nodes.pt", map_location="cpu", weights_only=True)
    z44, v44 = nodes_all[44]["z"], nodes_all[44]["v"]
    del nodes_all
    snapshots = torch.load(root / "OFF_snapshots.pt", weights_only=False)
    snapshot44 = snapshots[44]
    del snapshots
    saved = generation.get("reference_fingerprints", {}).get("44")
    if saved is None or snapshot44.step_index != 44:
        raise ValueError("saved44 reference absent")
    if runtime.fingerprint(vars(snapshot44)) != saved.get("history") or runtime.fingerprint(z44) != saved.get("input"):
        raise ValueError("saved44 history/input mismatch")
    if len(snapshot44.timesteps) != 50 or not snapshot44.predict_x0 or snapshot44.config.prediction_type != "flow_prediction" or snapshot44.config.thresholding or snapshot44.solver_p is not None:
        raise ValueError("original native50 nonthresholded flow scheduler required")
    prompt = torch.load(root / "prompt.pt", map_location="cpu", weights_only=True)
    negative = torch.load(root / "negative.pt", map_location="cpu", weights_only=True)
    book = {key: value for key, value in np.load(root / "codebook.npz").items()}
    config = copy.deepcopy(source_config)
    config["model"]["revision"] = MODEL_REVISION
    result.update(
        original_source=SOURCE_COMMIT,
        original_model_revision=generation.get("resolved_model_revision"),
        original_weight_identity="UNVERIFIED_HISTORICAL_REVISION_NULL",
        original44_fingerprints=saved,
        consumed_saved_nodes=["44.z", "44.v"],
        future_budget_fields_consumed=[],
        budget=dict(R_star=R_STAR, historical_complete_dev_native_response_rms=list(HISTORICAL_BUDGETS),
                    selection="minimum of all four complete historical development budgets",
                    provenance="historical future-LAST oracle; fixed before this run and never read from new sample state",
                    single46_per_step=R_STAR, multi44_46_per_step=R_STAR / 2,
                    nonclaim="equal cumulative RMS is not equal energy or equal terminal displacement"),
    )
    return config, book, z44, v44, snapshot44, prompt, negative


def nominal_record(z, book, message):
    directions = torch.from_numpy(book["directions"]).double()
    codes = torch.from_numpy(book["codes"]).double()
    raw = objective.metrics(z.detach().cpu(), directions, codes, message)
    scores = raw["tanh_message_scores"]
    return dict(correct_score=scores[message], wrong_score=scores[1 - message], gap=scores[message] - scores[1 - message],
                hard_correct_score=raw["hard_clipped_message_scores"][message], hard_wrong_score=raw["hard_clipped_message_scores"][1 - message],
                hard_gap=raw["hard_gap"], nominal_rank=raw["nominal_rank"],
                meaning="temperature-1 nominal latent diagnostic, not the blind media receiver")


def generate_case(case, source, output):
    source, output = guard(source, output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = load(MANIFEST)
    result = empty("RUNNING")
    result.update(case=case, fixed_calls=PLAN, actual_calls={key + "_" + status: 0 for key in PLAN for status in ("attempted", "completed")},
                  file_sha256={}, calls_by_path={}, elapsed_seconds_by_path={}, scientific_pass=None)
    scope = "setup"
    pipe = prompt = negative = snapshot44 = z44 = v44 = off_terminal = None

    def save():
        dump(output / "generation.json", result)

    def fail(stage, exc):
        result["failures"].append(dict(stage=stage, error=repr(exc), traceback=traceback.format_exc()))
        save()

    def count(kind, done):
        key = kind + ("_completed" if done else "_attempted")
        result["actual_calls"][key] += 1
        bucket = result["calls_by_path"].setdefault(scope, {})
        bucket[key] = bucket.get(key, 0) + 1
        save()

    @contextmanager
    def timed(name):
        nonlocal scope
        scope = name
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            result["elapsed_seconds_by_path"][name] = time.perf_counter() - started
            save()

    def artifact(name, value):
        path = output / name
        torch.save(value.detach().cpu(), path)
        result["file_sha256"][name] = sha(path)

    def store(name, terminal, message=None, rows=None):
        if tuple(terminal.shape) != projection_margin.SHAPE or not torch.isfinite(terminal).all():
            raise ValueError("invalid formal terminal")
        artifact(name + "_terminal.pt", terminal)
        item = result["videos"][name]
        item["status"] = "TERMINAL_PERSISTED"
        item["terminal_fingerprint"] = runtime.fingerprint(terminal)
        item["control_steps"] = rows or []
        item["cumulative_native_response"] = runtime.cumulative(rows or [])
        item["terminal_nominal_by_message"] = {str(candidate): nominal_record(terminal, book, candidate) for candidate in (0, 1)}
        if message is not None:
            item["message"] = message
            item["terminal_nominal"] = nominal_record(terminal, book, message)

    try:
        validate_manifest(manifest)
        config, book, z44, v44, snapshot44, prompt, negative = restore(source, case, result, manifest)
        import diffusers
        result["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        result["environment"] = dict(python=platform.python_version(), torch=str(torch.__version__), diffusers=diffusers.__version__)
        result["source_sha256"] = {str(path): sha(path) for path in (Path(__file__), Path(runtime.__file__), MANIFEST, Path(objective.__file__))}
        pipe, dtype, model_record = runtime.load_transformer(config)
        result["new_model"] = model_record
        if model_record.get("requested_revision") != MODEL_REVISION:
            raise ValueError("model loader did not receive fixed revision")
        if model_record.get("resolved_revision") not in (None, MODEL_REVISION) or model_record.get("config_commit_hash") not in (None, MODEL_REVISION):
            raise ValueError("loaded model revision differs from fixed successful revision")
        dump(output / "config.json", config)
        shutil.copyfile(source / case / "codebook.npz", output / "codebook.npz")
        shutil.copyfile(MANIFEST, output / "manifest.json")
        for name in ("config.json", "codebook.npz", "manifest.json"):
            result["file_sha256"][name] = sha(output / name)
        device = next(pipe.transformer.parameters()).device
        z44, v44, prompt, negative = z44.to(device), v44.to(device), prompt.to(device), negative.to(device)
        guidance = config["generation"]["guidance_scale"]

        with timed("formal_OFF"):
            off45, off_s = runtime.zero_step(snapshot44, z44, v44, 44, count)
            current = off45
            single46 = None
            off47 = None
            for index in range(45, 50):
                velocity = runtime.velocity(pipe, current, off_s, prompt, negative, dtype, guidance, index, count)
                if index == 46:
                    single46 = (current.clone(), velocity.clone(), copy.deepcopy(off_s))
                current, off_s = runtime.zero_step(off_s, current, velocity, index, count)
                if index == 46:
                    off47 = current.clone()
            off_terminal = current
            store("OFF", off_terminal)
            result["videos"]["OFF"]["control"] = dict(status="ZERO_CONTROL", start_index=44, terminal_fingerprint=runtime.fingerprint(off_terminal))

        for message, suffix in enumerate(("A", "B")):
            arm = "SINGLE46_" + suffix
            raw = q = terminal = None
            try:
                z, v, snapshot = single46
                with timed("gradient_" + arm):
                    count("clean_leaf_backward", False)
                    raw, gradient = runtime.tanh_direction(z, v, snapshot.sigmas[46], book, message, 46)
                    count("clean_leaf_backward", True)
                    result["videos"][arm]["gradients"] = [gradient]
                with timed("probe_" + arm):
                    q, epsilon, probe = runtime.prepare_direction(snapshot, z, v, off47, raw, R_STAR, 46, count)
                    result["videos"][arm]["unit_probes"] = [probe]
                with timed("formal_" + arm):
                    current, scheduler, row, arrays = runtime.controlled_step(snapshot, z, v, off47, q, epsilon, R_STAR, 46, book, message, count)
                    for label, value in arrays.items():
                        artifact(arm + "_46_" + label + ".pt", value)
                    result["videos"][arm]["control_steps"] = [row]
                    result["videos"][arm]["cumulative_native_response"] = runtime.cumulative([row])
                    save()
                    for index in range(47, 50):
                        current, scheduler, _ = runtime.free_step(pipe, scheduler, current, prompt, negative, dtype, guidance, index, count)
                    terminal = current
                    store(arm, terminal, message, [row])
            except Exception as exc:
                result["videos"][arm]["status"] = "FAILED"
                fail(arm, exc)
            finally:
                raw = q = terminal = None
                release()
                save()

        for message, suffix in enumerate(("A", "B")):
            arm = "MULTI44_46_" + suffix
            raw = q = terminal = None
            try:
                rows, gradients, probes = [], [], []
                result["videos"][arm].update(control_steps=rows, gradients=gradients, unit_probes=probes,
                                               cumulative_native_response=runtime.cumulative(rows))
                save()
                current, velocity, snapshot = z44, v44, snapshot44
                target = R_STAR / 2
                with timed("gradient44_" + arm):
                    count("clean_leaf_backward", False)
                    raw, gradient = runtime.tanh_direction(current, velocity, snapshot.sigmas[44], book, message, 44)
                    count("clean_leaf_backward", True)
                    gradients.append(gradient)
                    result["videos"][arm]["gradients"] = gradients
                with timed("probe44_" + arm):
                    q, epsilon, probe = runtime.prepare_direction(snapshot, current, velocity, off45, raw, target, 44, count)
                    probes.append(probe)
                    result["videos"][arm]["unit_probes"] = probes
                with timed("formal44_" + arm):
                    current, scheduler, row, arrays = runtime.controlled_step(snapshot, current, velocity, off45, q, epsilon, target, 44, book, message, count)
                    rows.append(row)
                    result["videos"][arm]["control_steps"] = rows
                    result["videos"][arm]["cumulative_native_response"] = runtime.cumulative(rows)
                    for label, value in arrays.items():
                        artifact(arm + "_44_" + label + ".pt", value)
                current, scheduler, _ = runtime.free_step(pipe, scheduler, current, prompt, negative, dtype, guidance, 45, count)
                velocity = runtime.velocity(pipe, current, scheduler, prompt, negative, dtype, guidance, 46, count)
                with timed("shadow46_" + arm):
                    zero46, _ = runtime.zero_step(scheduler, current, velocity, 46, count, "second_control_zero_shadow_step")
                with timed("gradient46_" + arm):
                    count("clean_leaf_backward", False)
                    raw, gradient = runtime.tanh_direction(current, velocity, scheduler.sigmas[46], book, message, 46)
                    count("clean_leaf_backward", True)
                    gradients.append(gradient)
                    result["videos"][arm]["gradients"] = gradients
                with timed("probe46_" + arm):
                    q, epsilon, probe = runtime.prepare_direction(scheduler, current, velocity, zero46, raw, target, 46, count)
                    probes.append(probe)
                    result["videos"][arm]["unit_probes"] = probes
                with timed("formal46_" + arm):
                    current, scheduler, row, arrays = runtime.controlled_step(scheduler, current, velocity, zero46, q, epsilon, target, 46, book, message, count)
                    rows.append(row)
                    result["videos"][arm]["control_steps"] = rows
                    result["videos"][arm]["cumulative_native_response"] = runtime.cumulative(rows)
                    for label, value in arrays.items():
                        artifact(arm + "_46_" + label + ".pt", value)
                for index in range(47, 50):
                    current, scheduler, _ = runtime.free_step(pipe, scheduler, current, prompt, negative, dtype, guidance, index, count)
                terminal = current
                result["videos"][arm]["gradients"] = gradients
                result["videos"][arm]["unit_probes"] = probes
                store(arm, terminal, message, rows)
            except Exception as exc:
                result["videos"][arm]["status"] = "FAILED"
                fail(arm, exc)
            finally:
                raw = q = terminal = None
                release()
                save()
        if runtime.fingerprint(vars(snapshot44)) != result["original44_fingerprints"]["history"]:
            raise RuntimeError("original44 snapshot mutated")
    except Exception as exc:
        fail("setup_or_generation", exc)
        for item in result["videos"].values():
            if item["status"] == "NOT_RUN":
                item["status"] = "FAILED"
    finally:
        pipe = prompt = negative = snapshot44 = z44 = v44 = None
        release()
    for arm in ARMS:
        item = result["videos"][arm]
        try:
            terminal = torch.load(output / (arm + "_terminal.pt"), map_location="cpu", weights_only=True)
            off = torch.load(output / "OFF_terminal.pt", map_location="cpu", weights_only=True)
            item["terminal_vs_same_batch_OFF"] = runtime.measures(terminal - off)
        except Exception as exc:
            item["terminal_vs_same_batch_OFF"] = dict(status="MISSING", error=repr(exc))
    result["status"] = "GENERATION_COMPLETE" if all(item["status"] == "TERMINAL_PERSISTED" for item in result["videos"].values()) and not result["failures"] else "WITH_RETAINED_FAILURES"
    save()
    return result


def media_case(case, output):
    output = Path(output)
    result = load(output / "generation.json")
    vae = None
    off_float = None
    off_mp4_valid = False

    def save():
        dump(output / "result.json", result)

    def fail(stage, exc):
        result["failures"].append(dict(stage=stage, error=repr(exc), traceback=traceback.format_exc()))
        save()

    def count(kind, done):
        result["actual_calls"][kind + ("_completed" if done else "_attempted")] += 1
        save()

    try:
        for name in ("config.json", "codebook.npz", "manifest.json"):
            if sha(output / name) != result.get("file_sha256", {}).get(name):
                raise ValueError("local metadata hash mismatch: " + name)
        if result.get("source_commit") != subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip():
            raise ValueError("generation/media source commit differs")
        config = load(output / "config.json")
        book = {key: value for key, value in np.load(output / "codebook.npz").items()}
        vae = load_frozen_vae(config)
        result["new_vae_config_commit_hash"] = getattr(vae.config, "_commit_hash", None)
        for arm in ARMS:
            item = result["videos"][arm]
            rgb = None
            saved = False
            path = output / "videos" / (arm + ".mp4")
            try:
                terminal_path = output / (arm + "_terminal.pt")
                if sha(terminal_path) != result.get("file_sha256", {}).get(terminal_path.name):
                    raise ValueError("terminal hash mismatch")
                terminal = torch.load(terminal_path, map_location="cpu", weights_only=True)
                count("vae_decode", False)
                rgb = decode_normalized_latent(vae, terminal.to(next(vae.parameters()).device)).cpu()
                count("vae_decode", True)
                if tuple(rgb.shape) != RGB_SHAPE or not torch.isfinite(rgb).all():
                    raise ValueError("invalid floatRGB")
                if arm == "OFF":
                    off_float = rgb.clone()
                try:
                    count("mp4_save", False)
                    encode_rgb(rgb, path, 8, 18)
                    count("mp4_save", True)
                    saved = True
                    item["video_path"] = str(path.relative_to(output))
                    result["file_sha256"][item["video_path"]] = sha(path)
                except Exception as exc:
                    fail(arm + "/MP4_save", exc)
            except Exception as exc:
                item["status"] = "FAILED_MEDIA"
                fail(arm + "/decode", exc)
            finally:
                release()
            for layer in LAYERS:
                row = item["layers"][layer]
                observations = {}
                pixels = None
                try:
                    if rgb is None:
                        raise ValueError("decoded floatRGB unavailable")
                    if layer == "floatRGB":
                        pixels = rgb
                    elif layer == "RGB8":
                        pixels = quantize_rgb8_no_codec(rgb).float() / 255
                    else:
                        if not saved:
                            raise ValueError("no complete MP4 saved this attempt")
                        pixels = read_mp4(path)
                        if arm == "OFF" and tuple(pixels.shape) == RGB_SHAPE and torch.isfinite(pixels).all():
                            off_mp4_valid = True
                    if tuple(pixels.shape) != RGB_SHAPE or not torch.isfinite(pixels).all():
                        raise ValueError("invalid actual layer raster")
                    row["actual_raster_dtype"] = str(pixels.dtype)
                    row["actual_raster_range"] = [float(pixels.min()), float(pixels.max())]
                    if arm != "OFF":
                        reference = None
                        if layer == "MP4" and off_mp4_valid:
                            reference = read_mp4(output / "videos" / "OFF.mp4")
                        elif layer == "floatRGB" and off_float is not None:
                            reference = off_float
                        elif layer == "RGB8" and off_float is not None:
                            reference = quantize_rgb8_no_codec(off_float).float() / 255
                        row["quality_vs_same_batch_OFF"] = quality(reference, pixels) if reference is not None else {"status": "MISSING_NEW_OFF_REFERENCE"}
                    for phase in range(4):
                        phase_row = row["observations"][str(phase)]
                        encoded = None
                        try:
                            groups, tail = divmod(len(pixels) - phase - 1, 4)
                            count("vae_encode", False)
                            encoded = reencode_rgb24_readback(vae, pixels[phase:phase + 1 + 4 * groups]).cpu().float()
                            count("vae_encode", True)
                            if tuple(encoded.shape) != (1, 16, 1 + groups, 40, 64) or not torch.isfinite(encoded).all():
                                raise ValueError("invalid layer receiver latent")
                            destination = output / "observations" / arm / layer
                            destination.mkdir(parents=True, exist_ok=True)
                            torch.save(encoded, destination / ("g" + str(phase) + ".pt"))
                            result["file_sha256"][str((destination / ("g" + str(phase) + ".pt")).relative_to(output))] = sha(destination / ("g" + str(phase) + ".pt"))
                            observations[phase] = encoded.numpy()
                            phase_row.update(status="COMPLETE", frames_used=1 + 4 * groups, tail_discarded=tail)
                        except Exception as exc:
                            phase_row.update(status="FAILED", error=repr(exc))
                            fail(arm + "/" + layer + "/g" + str(phase), exc)
                        finally:
                            encoded = None
                            _clear_cache(vae)
                            release()
                            save()
                    detection = state_clock.read(observations, book)
                    dump(output / "detections" / arm / (layer + ".json"), detection)
                    row["rankings"] = detection["rankings"]
                    if arm != "OFF":
                        message = 0 if arm.endswith("A") else 1
                        row["reporting_only"] = state_clock.report(detection, message, 0)
                    row["status"] = "COMPLETE" if len(observations) == 4 else "PARTIAL_OR_FAILED"
                except Exception as exc:
                    row.update(status="FAILED", error=repr(exc))
                    fail(arm + "/" + layer, exc)
                finally:
                    pixels = observations = None
                    _clear_cache(vae)
                    release()
                    save()
            item["status"] = "COMPLETE" if saved and all(row["status"] == "COMPLETE" for row in item["layers"].values()) else "PARTIAL_OR_FAILED"
            rgb = None
            release()
            save()
    except Exception as exc:
        fail("media_setup", exc)
    finally:
        vae = off_float = None
        release()
    result["status"] = "EXECUTION_COMPLETE" if all(item["status"] == "COMPLETE" for item in result["videos"].values()) and not result["failures"] else "WITH_RETAINED_FAILURES"
    result["resources"] = dict(cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                               cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
    save()
    return result


def receiver_record(video, layer, message, mode):
    record = video.get("layers", {}).get(layer, {})
    if record.get("status") != "COMPLETE":
        return None
    ranking = record.get("rankings", {}).get(mode, {})
    by_message = ranking.get("best_by_message", {})
    correct, wrong = by_message.get(str(message)), by_message.get(str(1 - message))
    if correct is None or wrong is None:
        return None
    return dict(correct_score=correct["score"], wrong_score=wrong["score"], gap=correct["score"] - wrong["score"],
                unique_correct=bool(ranking.get("message_unique") and ranking.get("best", {}).get("message") == message))


def summarize(cases):
    rows = []
    for case in CASES:
        videos = cases.get(case, {}).get("videos", {})
        for message, suffix in enumerate(("A", "B")):
            names = ("OFF", "SINGLE46_" + suffix, "MULTI44_46_" + suffix)
            row = dict(case=case, message=message, terminal={}, media={}, controls={})
            for name in names:
                video = videos.get(name, {})
                row["terminal"][name] = nominal_record_from_saved(video, message)
                row["controls"][name] = dict(steps=video.get("control_steps"), cumulative=video.get("cumulative_native_response"), terminal_vs_OFF=video.get("terminal_vs_same_batch_OFF"))
            for layer in LAYERS:
                row["media"][layer] = {mode: {name: receiver_record(videos.get(name, {}), layer, message, mode) for name in names} for mode in MODES}
            rows.append(row)
    return dict(fixed_pair_denominator=4, rows=rows, primary_mode="local_state",
                claim="two fixed development contents; uniform cumulative native-RMS mechanism comparison only; no existence threshold, equal-energy or generalization claim")


def nominal_record_from_saved(video, message):
    return video.get("terminal_nominal_by_message", {}).get(str(message))


def run_all(source, output):
    source, output = guard(source, output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="RUNNING", video_denominator=10, media_layer_denominator=30, receiver_encode_denominator=120,
                  fixed_pair_denominator=4, fixed_calls={key: 2 * value for key, value in PLAN.items()},
                  cases={case: empty("NOT_RUN") for case in CASES}, source_run=SOURCE_RUN,
                  budget=dict(R_star=R_STAR, historical_complete_dev_native_response_rms=list(HISTORICAL_BUDGETS)))
    dump(output / "result.json", result)
    shutil.copyfile(MANIFEST, output / "manifest.json")
    for stage in ("generate", "media"):
        for case in CASES:
            try:
                with (output / (case + "." + stage + ".log")).open("w") as log:
                    child = subprocess.run([sys.executable, "-u", "-m", MODULE, "--source", str(source), "--output", str(output / case), "--case-id", case, "--stage", stage], stdout=log, stderr=subprocess.STDOUT, check=False)
                previous = result["cases"][case]
                result_path = output / case / ("generation.json" if stage == "generate" else "result.json")
                current = load(result_path)
                result["cases"][case] = current | {key: value for key, value in previous.items() if key.endswith("_exit_code")} | {stage + "_exit_code": child.returncode}
            except Exception as exc:
                result["cases"][case].update(status="FAILED_LAUNCH_OR_RESULT", error=repr(exc), **{stage + "_exit_code": None})
            dump(output / "result.json", result)
    result["paired_summary"] = summarize(result["cases"])
    result["actual_calls_observed"] = {key + "_" + status: sum(case.get("actual_calls", {}).get(key + "_" + status, 0) for case in result["cases"].values()) for key in PLAN for status in ("attempted", "completed")}
    result["status"] = "EXECUTION_COMPLETE" if all(case.get("status") == "EXECUTION_COMPLETE" and case.get("generate_exit_code") == 0 and case.get("media_exit_code") == 0 for case in result["cases"].values()) else "WITH_RETAINED_FAILURES"
    dump(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-id", choices=CASES)
    parser.add_argument("--stage", choices=("generate", "media"))
    arguments = parser.parse_args()
    if arguments.case_id and arguments.stage:
        final = generate_case(arguments.case_id, arguments.source, arguments.output) if arguments.stage == "generate" else media_case(arguments.case_id, arguments.output)
    elif not arguments.case_id and not arguments.stage:
        final = run_all(arguments.source, arguments.output)
    else:
        parser.error("paired internal child arguments required")
    if final["status"] not in ("GENERATION_COMPLETE", "EXECUTION_COMPLETE"):
        raise SystemExit(1)
