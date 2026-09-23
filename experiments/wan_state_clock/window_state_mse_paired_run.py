"""Fresh paired legacy/MSE generation with shared media observations and blind receivers."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import torch

from main.tube_state import fixed_key
from main.tube_state import fixed_key_split_receiver as split_receiver
from runtime.wan import (
    fixed_key_control, fixed_key_core, generation, integrated_core, io, payload_control,
    trajectory, vae, window_state_mse,
)
from runtime.wan.generation import load_frozen_vae
from runtime.wan.integrated_core import encode_four_phases, eligible_detection
from runtime.wan.io import dump, encode_rgb, read_mp4
from runtime.wan.vae import _clear_cache, decode_normalized_latent

MANIFEST = Path(__file__).parent / "configs" / "window_state_mse_v1_paired.json"
SPEC = Path(__file__).parents[2] / "docs" / "window_state_mse_v1_paired_spec.md"
MODULE = "experiments.wan_state_clock.window_state_mse_paired_run"
VIEWS = fixed_key.FIXED_VIEWS
PHASES = (0, 1, 2, 3)
RECEIVERS = ("ORIGINAL", *split_receiver.CANDIDATES)
EVALUATION_ARMS = tuple(fixed_key_core.PAIRED_ARM_SPECS)
TOTAL_PLAN = {
    "generation": 4,
    "transformer": 496,
    "scheduler_step": 248,
    "zero_shadow_step": 12,
    "unit_response_probe_step": 12,
    "clean_leaf_backward": 12,
    "vae_decode": 12,
    "mp4_save": 36,
    "mp4_read": 36,
    "vae_encode": 144,
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def case_arms(case):
    return ("OFF",) if case["role"] == "calibration_off" else EVALUATION_ARMS


def empty_receiver():
    return {"status": "NOT_RUN", "detection": None, "decision": {"status": "NOT_RUN", "detected": None}}


def empty_view():
    return {
        "status": "NOT_RUN",
        "observations": {str(phase): {"status": "NOT_RUN"} for phase in PHASES},
        "receivers": {receiver: empty_receiver() for receiver in RECEIVERS},
    }


def empty_arm(arm):
    return {
        "status": "NOT_RUN",
        "arm": arm,
        "views": {view: empty_view() for view in VIEWS},
        "receiver_source_decisions": {
            receiver: {"status": "NOT_RUN", "detected": None} for receiver in RECEIVERS
        },
    }


def empty_case(case):
    return {
        "status": "NOT_RUN",
        "case_id": case["id"],
        "role": case["role"],
        "videos": {arm: empty_arm(arm) for arm in case_arms(case)},
        "actual_calls": {},
        "failures": [],
    }


def validate_manifest(config):
    canonical = load(MANIFEST)
    if config != canonical:
        raise ValueError("fixed paired roster and method configuration must match the canonical manifest")
    protocol = fixed_key_core.load_protocol()
    fixed_key_core.validate_protocol(protocol)
    for field in ("model", "generation"):
        if config[field] != protocol[field]:
            raise ValueError("fixed paired experiment and public protocol mismatch: " + field)
    if config["receiver_protocol_id"] != fixed_key.PROTOCOL_ID:
        raise ValueError("original receiver protocol mismatch")
    if config["split_receiver_protocol_id"] != split_receiver.PROTOCOL_ID:
        raise ValueError("split receiver protocol mismatch")
    if tuple(config["views"]) != VIEWS or tuple(config["receivers"]) != RECEIVERS:
        raise ValueError("fixed view or receiver roster mismatch")
    if tuple(config["calibration_arms"]) != ("OFF",):
        raise ValueError("calibration must be OFF only")
    if tuple(config["evaluation_arms"]) != EVALUATION_ARMS:
        raise ValueError("paired evaluation arm roster mismatch")
    if any(config["control"].get(key) != value for key, value in protocol["control"].items()):
        raise ValueError("fixed control mismatch")
    if config["writer_objectives"] != {
        "LEGACY": fixed_key_control.DEFAULT_OBJECTIVE,
        "MSE": fixed_key_control.WINDOW_STATE_MSE_OBJECTIVE,
    }:
        raise ValueError("paired writer objective mismatch")


def case_config(config, case):
    result = copy.deepcopy(config)
    result["generation"].update(prompt=case["prompt"], seed=case["seed"])
    return result


def generation_call_plan(case):
    return {
        "generation": 1,
        "transformer": 100 if case["role"] == "calibration_off" else 148,
        "scheduler_step": 50 if case["role"] == "calibration_off" else 74,
        "zero_shadow_step": 0 if case["role"] == "calibration_off" else 6,
        "unit_response_probe_step": 0 if case["role"] == "calibration_off" else 6,
        "clean_leaf_backward": 0 if case["role"] == "calibration_off" else 6,
    }


def generate_case_terminals(result, config, case, count):
    if case["role"] == "calibration_off":
        return fixed_key_core.generate_key_terminals(
            result["config"], config["key_utf8"].encode(), ("OFF",), count,
        )
    return fixed_key_core.generate_paired_key_terminals(
        result["config"], config["key_utf8"].encode(), count,
    )


def branch_writer_objective(branch, case, generated):
    return branch.get("writer_objective")


def _counter(result):
    def snapshot():
        return {
            "status": result["status"],
            "current": result.get("progress"),
            "actual_calls": dict(result["actual_calls"]),
        }

    def count(kind, completed):
        key = kind + ("_completed" if completed else "_attempted")
        result["actual_calls"][key] = result["actual_calls"].get(key, 0) + 1
        dump(Path(result["_progress_path"]), snapshot())
        if kind == "transformer" and completed and result["actual_calls"][key] % 20 == 0:
            print(f"progress model_step_pairs_completed={result['actual_calls'][key] // 2}", flush=True)

    return count


def _source_files():
    return (
        Path(__file__), MANIFEST, SPEC, Path(fixed_key.__file__), Path(split_receiver.__file__),
        Path(fixed_key_core.__file__), fixed_key_core.PROTOCOL_PATH, Path(fixed_key_control.__file__),
        Path(window_state_mse.__file__), Path(payload_control.__file__), Path(trajectory.__file__),
        Path(generation.__file__), Path(integrated_core.__file__), Path(io.__file__), Path(vae.__file__),
        Path(fixed_key.carrier.__file__), Path(fixed_key.state_clock.__file__),
    )


def generate_case(case_id, config_path, output):
    output = Path(output)
    config = load(config_path)
    validate_manifest(config)
    case = next(row for row in config["cases"] if row["id"] == case_id)
    output.mkdir(parents=True, exist_ok=False)
    result = empty_case(case)
    result.update(
        status="RUNNING",
        config=case_config(config, case),
        source_commit=None,
        source_files_sha256={},
        fresh_generation={
            "source_cache_inputs": [], "prompt": case["prompt"], "seed": case["seed"],
            "initial_noise": "NEW",
        },
        fixed_calls=generation_call_plan(case),
        _progress_path=str(output / "progress.json"),
    )

    def save():
        dump(output / "generation.json", {key: value for key, value in result.items() if key != "_progress_path"})

    def fail(stage, exc):
        result["failures"].append({"stage": stage, "error": repr(exc), "traceback": traceback.format_exc()})
        save()

    save()
    count = _counter(result)
    print(f"progress case={case_id} stage=generate", flush=True)
    try:
        result["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        result["source_files_sha256"] = {str(path): sha(path) for path in _source_files()}
        generated = generate_case_terminals(result, config, case, count)
        for name in ("initial_noise_fingerprint", "state44_fingerprint", "scheduler44_fingerprint"):
            if name in generated:
                result["fresh_generation"][name] = generated[name]
        for arm in case_arms(case):
            result["progress"] = {"case": case_id, "stage": "generate", "arm": arm}
            print(f"progress case={case_id} stage=generate arm={arm}", flush=True)
            item = result["videos"][arm]
            try:
                branch = generated["arms"][arm]
                if branch["status"] != "GENERATED":
                    raise RuntimeError(branch["error"])
                terminal = branch["terminal"]
                torch.save(terminal, output / f"{arm}_terminal.pt")
                tensor_root = output / "control_tensors" / arm
                tensor_root.mkdir(parents=True, exist_ok=True)
                for index, tensors in branch["control_tensors"].items():
                    torch.save(tensors, tensor_root / f"step{index}.pt")
                writer_objective = branch_writer_objective(branch, case, generated)
                if writer_objective is None and case["role"] == "calibration_off":
                    writer_objective = None
                item.update(
                    status="GENERATED", writer_objective=writer_objective,
                    nominal_terminal=branch["nominal_terminal"],
                    control_steps=branch["control_steps"], clean_gradients=branch["clean_gradients"],
                    unit_probes=branch["unit_probes"],
                    cumulative_native_response=branch["cumulative_native_response"],
                    terminal_fingerprint=branch["terminal_fingerprint"], terminal_path=f"{arm}_terminal.pt",
                )
            except Exception as exc:
                item["status"] = "FAILED_GENERATION"
                fail("generate/" + arm, exc)
            save()
    except Exception as exc:
        fail("generation_setup", exc)
    finally:
        release()
    result["status"] = (
        "GENERATION_COMPLETE"
        if all(item["status"] == "GENERATED" for item in result["videos"].values()) and not result["failures"]
        else "WITH_RETAINED_FAILURES"
    )
    result["resources_generation"] = {
        "cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    }
    save()
    dump(Path(result["_progress_path"]), {
        "status": result["status"], "current": result.get("progress"),
        "actual_calls": dict(result["actual_calls"]),
    })
    return result


def _attack_rgb(name, rgb, full_path, count):
    if name == "FULL":
        return rgb
    if name == "DELETE90":
        return torch.cat((rgb[:90], rgb[91:]), dim=0)
    if name == "SPEED5_4":
        indices = torch.floor(torch.arange(0, len(rgb), 1.25)).long().clamp(max=len(rgb) - 1)
        return rgb[indices]
    raise ValueError("unknown fixed attack view")


def _receiver_rows(observations, phases, key, spec_sha):
    complete = all(phases.get(str(phase), {}).get("status") == "COMPLETE" for phase in PHASES)
    try:
        original = fixed_key.read(observations, fixed_key.codebook(key))
    except Exception as exc:
        original = {"status": "INVALID", "reason": "receiver exception", "error": repr(exc)}
    try:
        split = split_receiver.read(observations, key, spec_sha)
    except Exception as exc:
        split = {"status": "INVALID", "reason": "receiver exception", "error": repr(exc)}
    original_status = "SCORED" if complete and original.get("status") == "SCORED" else "PARTIAL_OR_FAILED"
    split_status = "SCORED" if complete and split.get("status") == "SCORED" else "PARTIAL_OR_FAILED"
    return {
        "ORIGINAL": {"status": original_status, "detection": original, "decision": {"status": "NOT_RUN", "detected": None}},
        "C1_MATCHED_CONFIRM": {"status": split_status, "detection": copy.deepcopy(split), "decision": {"status": "NOT_RUN", "detected": None}},
        "C2_STATE_CONFIRM": {"status": split_status, "detection": copy.deepcopy(split), "decision": {"status": "NOT_RUN", "detected": None}},
    }


def media_case(case_id, config_path, output):
    output = Path(output)
    config = load(config_path)
    validate_manifest(config)
    case = next(row for row in config["cases"] if row["id"] == case_id)
    generation_path = output / "generation.json"
    result = load(generation_path) if generation_path.exists() else empty_case(case)
    result["status"] = "MEDIA_RUNNING"
    result["_progress_path"] = str(output / "progress.json")
    result.setdefault("failures", [])
    for arm in case_arms(case):
        result["videos"].setdefault(arm, empty_arm(arm))
        result["videos"][arm]["views"] = {view: empty_view() for view in VIEWS}
        result["videos"][arm]["receiver_source_decisions"] = {
            receiver: {"status": "NOT_RUN", "detected": None} for receiver in RECEIVERS
        }

    def save():
        dump(output / "result.json", {key: value for key, value in result.items() if key != "_progress_path"})

    def fail(stage, exc):
        result["failures"].append({"stage": stage, "error": repr(exc), "traceback": traceback.format_exc()})
        save()

    count = _counter(result)
    vae = None
    print(f"progress case={case_id} stage=media", flush=True)
    save()
    try:
        protocol = fixed_key_core.load_protocol()
        fixed_key_core.validate_protocol(protocol)
        vae = load_frozen_vae(protocol)
        key = config["key_utf8"].encode()
        spec_sha = sha(SPEC)
        for arm in case_arms(case):
            result["progress"] = {"case": case_id, "stage": "media", "arm": arm}
            print(f"progress case={case_id} stage=media arm={arm}", flush=True)
            item = result["videos"][arm]
            rgb = None
            try:
                terminal = torch.load(output / f"{arm}_terminal.pt", map_location="cpu", weights_only=True)
                count("vae_decode", False)
                rgb = decode_normalized_latent(vae, terminal.to(next(vae.parameters()).device)).detach().cpu()
                count("vae_decode", True)
                decoded_path = output / "decoded_rgb" / f"{arm}.pt"
                decoded_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(rgb, decoded_path)
            except Exception as exc:
                item["status"] = "FAILED_DECODE"
                fail("decode/" + arm, exc)
                continue
            full_path = output / "received_videos" / arm / "FULL.mp4"
            for view in VIEWS:
                result["progress"] = {"case": case_id, "stage": "media", "arm": arm, "view": view}
                print(f"progress case={case_id} stage=media arm={arm} view={view}", flush=True)
                view_row = item["views"][view]
                pixels = None
                try:
                    attacked = _attack_rgb(view, rgb, full_path, count)
                    path = output / "received_videos" / arm / f"{view}.mp4"
                    count("mp4_save", False)
                    encode_rgb(attacked, path, config["generation"]["fps"], 18)
                    count("mp4_save", True)
                    view_row.update(
                        status="VIDEO_PERSISTED", path=str(path.relative_to(output)),
                        sha256=sha(path), frames=int(attacked.shape[0]),
                    )
                    count("mp4_read", False)
                    pixels = read_mp4(path)
                    count("mp4_read", True)

                    def persist_phase(phase, encoded, phase_row):
                        print(
                            f"progress case={case_id} stage=media arm={arm} view={view} phase={phase}",
                            flush=True,
                        )
                        latent_path = output / "observations" / arm / view / f"g{phase}.pt"
                        latent_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(encoded, latent_path)
                        phase_row["path"] = str(latent_path.relative_to(output))
                        phase_row["sha256"] = sha(latent_path)

                    observations, phases = encode_four_phases(pixels, vae, count, persist_phase)
                    view_row["observations"] = phases
                    view_row["receivers"] = _receiver_rows(observations, phases, key, spec_sha)
                    for phase, phase_row in phases.items():
                        if phase_row.get("status") != "COMPLETE":
                            result["failures"].append({
                                "stage": f"receive/{arm}/{view}/g{phase}",
                                "error": phase_row.get("error", "phase failed"),
                            })
                    for receiver, receiver_row in view_row["receivers"].items():
                        dump(output / "detections" / arm / view / f"{receiver}.json", receiver_row["detection"])
                        if receiver_row["status"] != "SCORED":
                            result["failures"].append({
                                "stage": f"receiver/{arm}/{view}/{receiver}",
                                "error": (receiver_row.get("detection") or {}).get("reason", "receiver failed"),
                            })
                    view_row["status"] = (
                        "SCORED"
                        if all(row["status"] == "SCORED" for row in view_row["receivers"].values())
                        else "PARTIAL_OR_FAILED"
                    )
                except Exception as exc:
                    view_row.update(status="FAILED", error=repr(exc))
                    fail(f"media/{arm}/{view}", exc)
                finally:
                    pixels = None
                    _clear_cache(vae)
                    release()
                    save()
            item["status"] = (
                "MEDIA_COMPLETE"
                if all(row["status"] == "SCORED" for row in item["views"].values())
                else "PARTIAL_OR_FAILED"
            )
            rgb = None
            release()
    except Exception as exc:
        fail("media_setup", exc)
    finally:
        vae = None
        release()
    result["status"] = (
        "EXECUTION_COMPLETE"
        if all(item["status"] == "MEDIA_COMPLETE" for item in result["videos"].values()) and not result["failures"]
        else "WITH_RETAINED_FAILURES"
    )
    result["resources_media"] = {
        "cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    }
    save()
    dump(Path(result["_progress_path"]), {
        "status": result["status"], "current": result.get("progress"),
        "actual_calls": dict(result["actual_calls"]),
    })
    return result


def _original_formal(view_row):
    receiver = view_row.get("receivers", {}).get("ORIGINAL", empty_receiver())
    return eligible_detection(
        receiver.get("status"), view_row.get("observations", {}),
        receiver.get("detection") or {"status": "INVALID"},
    )


def _source_record(item, receiver, spec_sha, key_id):
    if receiver == "ORIGINAL":
        return fixed_key.source_max_statistic({
            view: _original_formal(item.get("views", {}).get(view, {})) for view in VIEWS
        })
    return split_receiver.source_statistic(
        {
            view: (
                item.get("views", {}).get(view, {}).get("receivers", {}).get(receiver, {}).get("detection") or {}
                if item.get("views", {}).get(view, {}).get("receivers", {}).get(receiver, {}).get("status") == "SCORED"
                and all(
                    item.get("views", {}).get(view, {}).get("observations", {}).get(str(phase), {}).get("status") == "COMPLETE"
                    for phase in PHASES
                )
                else {}
            )
            for view in VIEWS
        },
        receiver, spec_sha, key_id,
    )


def _margin(decision):
    score, threshold = decision.get("statistic"), decision.get("threshold")
    if not isinstance(score, (int, float)) or not isinstance(threshold, (int, float)):
        return None
    if not math.isfinite(score) or not math.isfinite(threshold):
        return None
    return score - threshold


def _attach_receiver_decisions(item, calibrations, spec_sha, key_id, *, marked, calibration_sample):
    role = "THRESHOLD_CONSTRUCTION_SAMPLE_NOT_HELDOUT_FPR" if calibration_sample else "EVALUATION"
    for receiver in RECEIVERS:
        source = _source_record(item, receiver, spec_sha, key_id)
        for view in VIEWS:
            view_row = item["views"][view]
            receiver_row = view_row["receivers"][receiver]
            if receiver == "ORIGINAL":
                detection = _original_formal(view_row)
                decision = fixed_key.decide(detection, calibrations[receiver])
                alignment = fixed_key.alignment_report(receiver_row.get("detection") or {}, view)
            else:
                detection = receiver_row.get("detection") or {}
                score = detection.get("confirmation_scores", {}).get(receiver)
                decision = split_receiver.decide(
                    score, receiver_row.get("status"), calibrations[receiver], receiver, spec_sha, key_id,
                )
                alignment = detection.get("alignment_reporting_only", {}).get(view, {"status": "UNAVAILABLE"})
            decision["signed_margin"] = _margin(decision)
            receiver_row["decision"] = decision
            receiver_row["alignment_reporting_only"] = alignment
            receiver_row["reporting_only"] = {
                "marked": marked, "decision_role": role,
                "detected_marked": None if not marked else decision.get("status") == "DETECTED",
            }
        if receiver == "ORIGINAL":
            if source.get("status") == "SCORED" and source.get("winning_view") in VIEWS:
                source_decision = fixed_key.decide(
                    _original_formal(item["views"][source["winning_view"]]), calibrations[receiver],
                )
            else:
                source_decision = {"status": "INVALID", "detected": None}
        else:
            source_decision = split_receiver.decide(
                source.get("statistic"), source.get("status"), calibrations[receiver],
                receiver, spec_sha, key_id,
            )
        source_decision.update(
            winning_view=source.get("winning_view"),
            source_max_statistic=source.get("statistic"),
        )
        source_decision["signed_margin"] = _margin(source_decision)
        item["receiver_source_decisions"][receiver] = source_decision
    item["reporting_only"] = {"marked": marked, "decision_role": role}


def _paired_comparisons(result):
    comparisons = []
    for case_id, case in result["cases"].items():
        if case.get("role") != "evaluation":
            continue
        for receiver in RECEIVERS:
            for schedule in ("SINGLE46", "MULTI44_46"):
                legacy = case["videos"]["LEGACY_" + schedule]
                mse = case["videos"]["MSE_" + schedule]
                source_legacy = legacy["receiver_source_decisions"][receiver]
                source_mse = mse["receiver_source_decisions"][receiver]
                shared_off = case["videos"]["OFF"]
                view_pairs = {}
                for view in VIEWS:
                    legacy_row = legacy["views"][view]["receivers"][receiver]
                    mse_row = mse["views"][view]["receivers"][receiver]
                    legacy_decision = legacy_row["decision"]
                    mse_decision = mse_row["decision"]
                    reference_name = {
                        "FULL": "IDENTITY", "DELETE90": "DELETE90_REFERENCE",
                        "SPEED5_4": "SPEED5_4_REFERENCE",
                    }[view]
                    if receiver == "ORIGINAL":
                        legacy_reference = (legacy_row.get("detection") or {}).get("fixed_path_diagnostics", {}).get(reference_name, {}).get("score")
                        mse_reference = (mse_row.get("detection") or {}).get("fixed_path_diagnostics", {}).get(reference_name, {}).get("score")
                    else:
                        legacy_reference = (legacy_row.get("detection") or {}).get("fixed_path_diagnostics", {}).get(reference_name, {}).get("scores", {}).get(receiver)
                        mse_reference = (mse_row.get("detection") or {}).get("fixed_path_diagnostics", {}).get(reference_name, {}).get("scores", {}).get(receiver)
                    view_pairs[view] = {
                        "LEGACY": {"blind": legacy_decision, "fixed_reference_score": legacy_reference},
                        "MSE": {"blind": mse_decision, "fixed_reference_score": mse_reference},
                        "blind_score_difference_mse_minus_legacy": (
                            None
                            if legacy_decision.get("statistic") is None or mse_decision.get("statistic") is None
                            else mse_decision["statistic"] - legacy_decision["statistic"]
                        ),
                        "blind_margin_difference_mse_minus_legacy": (
                            None
                            if legacy_decision.get("signed_margin") is None or mse_decision.get("signed_margin") is None
                            else mse_decision["signed_margin"] - legacy_decision["signed_margin"]
                        ),
                        "fixed_reference_difference_mse_minus_legacy": (
                            None
                            if legacy_reference is None or mse_reference is None
                            else mse_reference - legacy_reference
                        ),
                    }
                comparisons.append({
                    "case_id": case_id,
                    "receiver": receiver,
                    "schedule": schedule,
                    "shared_initial_noise_fingerprint": case.get("fresh_generation", {}).get("initial_noise_fingerprint"),
                    "shared_state44_fingerprint": case.get("fresh_generation", {}).get("state44_fingerprint"),
                    "shared_scheduler44_fingerprint": case.get("fresh_generation", {}).get("scheduler44_fingerprint"),
                    "source_arm": {"LEGACY": source_legacy, "MSE": source_mse},
                    "source_score_difference_mse_minus_legacy": (
                        None
                        if source_legacy.get("statistic") is None or source_mse.get("statistic") is None
                        else source_mse["statistic"] - source_legacy["statistic"]
                    ),
                    "source_margin_difference_mse_minus_legacy": (
                        None
                        if source_legacy.get("signed_margin") is None or source_mse.get("signed_margin") is None
                        else source_mse["signed_margin"] - source_legacy["signed_margin"]
                    ),
                    "shared_off_reference": {
                        "source_arm": shared_off["receiver_source_decisions"][receiver],
                        "views": {
                            view: shared_off["views"][view]["receivers"][receiver]["decision"]
                            for view in VIEWS
                        },
                        "meaning": "one physical OFF for both writer comparisons; not an independent sample per writer",
                    },
                    "views": view_pairs,
                })
    return comparisons


def _run_child(command, log_path):
    with Path(log_path).open("w", encoding="utf-8") as log:
        child = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert child.stdout is not None
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return child.wait()


def run_all(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = load(config_path)
    validate_manifest(config)
    if (output / "result.json").exists() or (output / "manifest.json").exists():
        raise FileExistsError("run output already contains an experiment; use a new timestamp")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / "manifest.json")
    shutil.copyfile(SPEC, output / "paired_spec.md")
    result = {
        "status": "RUNNING",
        "fixed_denominator": {
            "fresh_sources": 4, "physical_source_arms": 12, "saved_attack_views": 36,
            "receiver_phase_encodes": 144, "calibration_sources": 2, "evaluation_sources": 2,
            "receivers": 3, "view_decision_slots": 108, "source_arm_decision_slots": 36,
            "unique_evaluation_marked_views": 24, "unique_evaluation_marked_source_arms": 8,
            "shared_evaluation_off_views": 6, "shared_evaluation_off_source_arms": 2,
        },
        "fixed_calls": TOTAL_PLAN,
        "cases": {case["id"]: empty_case(case) for case in config["cases"]},
        "calibrations": {receiver: {"status": "NOT_RUN", "threshold": None} for receiver in RECEIVERS},
        "paired_comparisons": [],
        "failures": [],
    }
    versions = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "numpy", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result["environment"] = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "packages": versions,
    }
    dump(output / "environment.json", result["environment"])
    dump(output / "result.json", result)

    def execute_case(case):
        case_root = output / case["id"]
        for stage in ("generate", "media"):
            log_path = output / f"{case['id']}.{stage}.log"
            command = [
                sys.executable, "-u", "-m", MODULE, "--config", str(config_path),
                "--output", str(case_root), "--case-id", case["id"], "--stage", stage,
            ]
            print(f"progress case={case['id']} stage={stage} log={log_path}", flush=True)
            previous = result["cases"][case["id"]]
            previous.setdefault("stage_logs", {})[stage] = str(log_path.relative_to(output))
            try:
                returncode = _run_child(command, log_path)
            except Exception as exc:
                failure = {
                    "case_id": case["id"], "stage": stage,
                    "log": str(log_path.relative_to(output)), "error": repr(exc),
                    "kind": "CHILD_SPAWN_OR_TEE_FAILURE",
                }
                result["failures"].append(failure)
                previous.setdefault("parent_failures", []).append(failure)
                previous[stage + "_exit_code"] = None
                previous["status"] = "FAILED_LAUNCH_OR_RESULT"
                dump(output / "result.json", result)
                continue
            record_path = case_root / ("generation.json" if stage == "generate" else "result.json")
            if record_path.exists():
                try:
                    current = load(record_path)
                except Exception as exc:
                    failure = {
                        "case_id": case["id"], "stage": stage,
                        "log": str(log_path.relative_to(output)),
                        "error": repr(exc), "kind": "CHILD_RESULT_INVALID",
                    }
                    result["failures"].append(failure)
                    previous.setdefault("parent_failures", []).append(failure)
                    previous["status"] = "FAILED_LAUNCH_OR_RESULT"
                else:
                    for key, value in previous.items():
                        if key.endswith("_exit_code") or key in ("stage_logs", "parent_failures"):
                            current[key] = value
                    result["cases"][case["id"]] = current
            else:
                result["cases"][case["id"]]["status"] = "FAILED_LAUNCH_OR_RESULT"
                failure = {
                    "case_id": case["id"], "stage": stage,
                    "log": str(log_path.relative_to(output)),
                    "error": "child result file missing", "kind": "CHILD_RESULT_MISSING",
                }
                result["failures"].append(failure)
                result["cases"][case["id"]].setdefault("parent_failures", []).append(failure)
            result["cases"][case["id"]][stage + "_exit_code"] = returncode
            dump(output / "result.json", result)

    calibration_cases = [case for case in config["cases"] if case["role"] == "calibration_off"]
    evaluation_cases = [case for case in config["cases"] if case["role"] == "evaluation"]
    for case in calibration_cases:
        execute_case(case)
    spec_sha, key_id = sha(SPEC), fixed_key.key_identifier(config["key_utf8"].encode())
    original_sources = []
    split_sources = {receiver: {} for receiver in split_receiver.CANDIDATES}
    for case in calibration_cases:
        item = result["cases"][case["id"]]["videos"]["OFF"]
        original = _source_record(item, "ORIGINAL", spec_sha, key_id)
        original_sources.append(original | {"case_id": case["id"]})
        for receiver in split_receiver.CANDIDATES:
            split_sources[receiver][case["id"]] = _source_record(item, receiver, spec_sha, key_id)
    result["calibrations"] = {
        "ORIGINAL": fixed_key.freeze_calibration(
            original_sources, config["calibration"]["guard"],
            protocol_id=config["receiver_protocol_id"], key_id=key_id,
        ),
        **{
            receiver: split_receiver.calibrate(split_sources[receiver], receiver, spec_sha, key_id)
            for receiver in split_receiver.CANDIDATES
        },
    }
    dump(output / "calibrations.json", result["calibrations"])
    for case in calibration_cases:
        case_result = result["cases"][case["id"]]
        _attach_receiver_decisions(
            case_result["videos"]["OFF"], result["calibrations"], spec_sha, key_id,
            marked=False, calibration_sample=True,
        )
        dump(output / case["id"] / "result.json", case_result)
    dump(output / "result.json", result)
    for case in evaluation_cases:
        execute_case(case)
    for case in evaluation_cases:
        case_result = result["cases"][case["id"]]
        for arm, item in case_result["videos"].items():
            _attach_receiver_decisions(
                item, result["calibrations"], spec_sha, key_id,
                marked=arm != "OFF", calibration_sample=False,
            )
        dump(output / case["id"] / "result.json", case_result)
    result["paired_comparisons"] = _paired_comparisons(result)
    result["actual_calls_observed"] = {
        key + "_" + status: sum(
            case.get("actual_calls", {}).get(key + "_" + status, 0)
            for case in result["cases"].values()
        )
        for key in TOTAL_PLAN for status in ("attempted", "completed")
    }
    result["call_accounting"] = {
        key: {
            "expected": expected,
            "attempted": result["actual_calls_observed"][key + "_attempted"],
            "completed": result["actual_calls_observed"][key + "_completed"],
            "status": "EXACT" if all(
                result["actual_calls_observed"][key + "_" + status] == expected
                for status in ("attempted", "completed")
            ) else "MISMATCH",
        }
        for key, expected in TOTAL_PLAN.items()
    }
    if any(row["status"] != "EXACT" for row in result["call_accounting"].values()):
        result["failures"].append({"stage": "call_accounting", "kind": "EXPECTED_CALL_MISMATCH"})
    result["status"] = (
        "EXECUTION_COMPLETE"
        if all(calibration.get("status") == "FROZEN" for calibration in result["calibrations"].values())
        and not result["failures"]
        and all(
            case.get("status") == "EXECUTION_COMPLETE"
            and case.get("generate_exit_code") == 0
            and case.get("media_exit_code") == 0
            for case in result["cases"].values()
        )
        else "WITH_RETAINED_FAILURES"
    )
    result["evidence_ceiling"] = config["evidence_ceiling"]
    dump(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(MANIFEST))
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--stage", choices=("generate", "media"))
    arguments = parser.parse_args()
    if arguments.case_id and arguments.stage:
        final = (
            generate_case(arguments.case_id, arguments.config, arguments.output)
            if arguments.stage == "generate"
            else media_case(arguments.case_id, arguments.config, arguments.output)
        )
    elif not arguments.case_id and not arguments.stage:
        final = run_all(arguments.config, arguments.output)
    else:
        parser.error("--case-id and --stage must be supplied together")
    if final["status"] not in ("GENERATION_COMPLETE", "EXECUTION_COMPLETE"):
        raise SystemExit(1)
