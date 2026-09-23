"""Fixed new-source Content-Background-Existence-V1 user-run experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
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

from experiments.wan_state_clock import window_state_mse_paired_run as shared
from main.tube_state import content_background_existence as candidate
from main.tube_state import fixed_key
from main.tube_state import fixed_key_split_receiver as split_receiver
from runtime.wan import (
    fixed_key_control, fixed_key_core, generation, integrated_core, io, payload_control,
    trajectory, vae,
)

MANIFEST = Path(__file__).parent / "configs" / "content_background_existence_v1.json"
SPEC = Path(__file__).parents[2] / "docs" / "content_background_existence_v1_spec.md"
MODULE = "experiments.wan_state_clock.content_background_existence_run"
VIEWS = fixed_key.FIXED_VIEWS
PHASES = (0, 1, 2, 3)
RECEIVERS = ("ORIGINAL", *split_receiver.CANDIDATES, candidate.RECEIVER_ID)
EVALUATION_ARMS = ("OFF", "SINGLE46", "MULTI44_46")
TOTAL_PLAN = {
    "generation": 8,
    "transformer": 896,
    "scheduler_step": 448,
    "zero_shadow_step": 12,
    "unit_response_probe_step": 12,
    "clean_leaf_backward": 12,
    "vae_decode": 16,
    "mp4_save": 48,
    "mp4_read": 48,
    "vae_encode": 192,
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_arms(case):
    return ("OFF",) if case["role"] == "calibration_off" else EVALUATION_ARMS


def validate_manifest(config):
    if config != load(MANIFEST):
        raise ValueError("fixed method and new-source roster must match the canonical manifest")
    protocol = fixed_key_core.load_protocol()
    fixed_key_core.validate_protocol(protocol)
    for field in ("model", "generation"):
        if config[field] != protocol[field]:
            raise ValueError("fixed experiment and public protocol mismatch: " + field)
    if tuple(config["views"]) != VIEWS or tuple(config["receivers"]) != RECEIVERS:
        raise ValueError("fixed view or receiver roster mismatch")
    if tuple(config["calibration_arms"]) != ("OFF",) or tuple(config["evaluation_arms"]) != EVALUATION_ARMS:
        raise ValueError("fixed physical arm roster mismatch")
    if config["receiver_protocol_id"] != candidate.PROTOCOL_ID:
        raise ValueError("candidate receiver protocol mismatch")
    if config["original_receiver_protocol_id"] != fixed_key.PROTOCOL_ID:
        raise ValueError("original receiver protocol mismatch")
    if config["split_receiver_protocol_id"] != split_receiver.PROTOCOL_ID:
        raise ValueError("split receiver protocol mismatch")
    if config["writer_objectives"] != {
        "OFF": "OFF_NO_WRITER_OBJECTIVE",
        "SINGLE46": fixed_key_control.DEFAULT_OBJECTIVE,
        "MULTI44_46": fixed_key_control.DEFAULT_OBJECTIVE,
    }:
        raise ValueError("legacy writer objective identity mismatch")
    if any(config["control"].get(key) != value for key, value in protocol["control"].items()):
        raise ValueError("fixed R* control mismatch")
    if config["fixed_denominator"] != {
        "fresh_sources": 8, "calibration_sources": 4, "evaluation_sources": 4,
        "physical_source_arms": 16, "saved_attack_views": 48,
        "receiver_phase_encodes": 192, "evaluation_OFF_views": 12,
        "evaluation_marked_views": 24, "evaluation_OFF_source_arms": 4,
        "evaluation_marked_source_arms": 8,
    }:
        raise ValueError("fixed denominator mismatch")
    if config["expected_complete_calls"] != TOTAL_PLAN:
        raise ValueError("fixed call plan mismatch")


def _configure_shared():
    replacements = {
        "MANIFEST": MANIFEST, "SPEC": SPEC, "MODULE": MODULE, "VIEWS": VIEWS,
        "PHASES": PHASES, "RECEIVERS": RECEIVERS, "EVALUATION_ARMS": EVALUATION_ARMS,
        "TOTAL_PLAN": TOTAL_PLAN, "validate_manifest": validate_manifest,
        "case_arms": case_arms, "_receiver_rows": _receiver_rows,
        "_source_files": _source_files,
    }
    originals = {name: getattr(shared, name) for name in replacements}
    for name, value in replacements.items():
        setattr(shared, name, value)
    return originals


def _restore_shared(originals):
    for name, value in originals.items():
        setattr(shared, name, value)


def _source_files():
    return (
        Path(__file__), Path(shared.__file__), MANIFEST, SPEC, Path(candidate.__file__), Path(fixed_key.__file__),
        Path(split_receiver.__file__), Path(fixed_key_core.__file__), fixed_key_core.PROTOCOL_PATH,
        Path(fixed_key_control.__file__), Path(payload_control.__file__), Path(trajectory.__file__),
        Path(generation.__file__),
        Path(integrated_core.__file__), Path(io.__file__), Path(vae.__file__),
        Path(fixed_key.carrier.__file__), Path(fixed_key.state_clock.__file__),
    )


def _receiver_rows(observations, phases, key, spec_sha):
    complete = set(observations) == set(PHASES) and all(
        phases.get(str(phase), {}).get("status") == "COMPLETE" for phase in PHASES
    )
    try:
        normalized = candidate.read(observations, key, spec_sha)
    except Exception as exc:
        normalized = {"status": "INVALID", "reason": "candidate exception", "error": repr(exc)}
    original = normalized.get("correct_detection") or {
        "status": "INVALID", "reason": "correct-key search unavailable from candidate receiver"
    }
    try:
        split = split_receiver.read(observations, key, spec_sha)
    except Exception as exc:
        split = {"status": "INVALID", "reason": "split receiver exception", "error": repr(exc)}
    original_status = "SCORED" if complete and original.get("status") == "SCORED" else "PARTIAL_OR_FAILED"
    split_status = "SCORED" if complete and split.get("status") == "SCORED" else "PARTIAL_OR_FAILED"
    candidate_status = "SCORED" if complete and normalized.get("status") == "SCORED" else "PARTIAL_OR_FAILED"
    empty_decision = {"status": "NOT_RUN", "detected": None}
    return {
        "ORIGINAL": {"status": original_status, "detection": original, "decision": dict(empty_decision)},
        "C1_MATCHED_CONFIRM": {"status": split_status, "detection": copy.deepcopy(split), "decision": dict(empty_decision)},
        "C2_STATE_CONFIRM": {"status": split_status, "detection": copy.deepcopy(split), "decision": dict(empty_decision)},
        candidate.RECEIVER_ID: {"status": candidate_status, "detection": normalized, "decision": dict(empty_decision)},
    }


def _generate_case_configured(case_id, config_path, output):
    output = Path(output)
    config = load(config_path)
    validate_manifest(config)
    case = next(row for row in config["cases"] if row["id"] == case_id)
    output.mkdir(parents=True, exist_ok=False)
    result = shared.empty_case(case)
    result.update(
        status="RUNNING", config=shared.case_config(config, case), source_commit=None,
        source_files_sha256={},
        fresh_generation={"source_cache_inputs": [], "prompt": case["prompt"], "seed": case["seed"], "initial_noise": "NEW"},
        fixed_calls={
            "generation": 1,
            "transformer": 100 if case["role"] == "calibration_off" else 124,
            "scheduler_step": 50 if case["role"] == "calibration_off" else 62,
            "zero_shadow_step": 0 if case["role"] == "calibration_off" else 3,
            "unit_response_probe_step": 0 if case["role"] == "calibration_off" else 3,
            "clean_leaf_backward": 0 if case["role"] == "calibration_off" else 3,
        },
        _progress_path=str(output / "progress.json"),
    )

    def save():
        shared.dump(output / "generation.json", {key: value for key, value in result.items() if key != "_progress_path"})

    def fail(stage, exc):
        result["failures"].append({"stage": stage, "error": repr(exc), "traceback": traceback.format_exc()})
        save()

    save()
    count = shared._counter(result)
    try:
        result["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        result["source_files_sha256"] = {str(path): sha(path) for path in _source_files()}
        arms = case_arms(case)
        generated = fixed_key_core.generate_key_terminals(
            result["config"], config["key_utf8"].encode("utf-8"), arms, count,
        )
        for name in ("initial_noise_fingerprint", "state44_fingerprint", "scheduler44_fingerprint"):
            if name in generated:
                result["fresh_generation"][name] = generated[name]
        for arm in arms:
            result["progress"] = {"case": case_id, "stage": "generate", "arm": arm}
            item = result["videos"][arm]
            try:
                branch = generated["arms"][arm]
                if branch["status"] != "GENERATED":
                    raise RuntimeError(branch["error"])
                expected_objective = config["writer_objectives"][arm]
                observed_objective = None if arm == "OFF" else generated.get("writer_objective")
                observed_objective_id = None if observed_objective is None else observed_objective.get("id")
                if arm != "OFF" and observed_objective_id != expected_objective:
                    raise RuntimeError(f"writer objective mismatch: {observed_objective!r}")
                terminal = branch["terminal"]
                torch.save(terminal, output / f"{arm}_terminal.pt")
                tensor_root = output / "control_tensors" / arm
                tensor_root.mkdir(parents=True, exist_ok=True)
                for index, tensors in branch["control_tensors"].items():
                    torch.save(tensors, tensor_root / f"step{index}.pt")
                item.update(
                    status="GENERATED",
                    writer_objective_identity={"expected": expected_objective, "observed": observed_objective_id, "arm": arm},
                    writer_objective=observed_objective,
                    nominal_terminal=branch["nominal_terminal"], control_steps=branch["control_steps"],
                    clean_gradients=branch["clean_gradients"], unit_probes=branch["unit_probes"],
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
        shared.release()
    result["status"] = "GENERATION_COMPLETE" if all(
        item["status"] == "GENERATED" for item in result["videos"].values()
    ) and not result["failures"] else "WITH_RETAINED_FAILURES"
    result["resources_generation"] = {
        "cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    }
    save()
    shared.dump(Path(result["_progress_path"]), {
        "status": result["status"], "current": result.get("progress"), "actual_calls": dict(result["actual_calls"]),
    })
    return result


def generate_case(case_id, config_path, output):
    originals = _configure_shared()
    try:
        return _generate_case_configured(case_id, config_path, output)
    finally:
        _restore_shared(originals)


def media_case(case_id, config_path, output):
    originals = _configure_shared()
    try:
        return shared.media_case(case_id, config_path, output)
    finally:
        _restore_shared(originals)


def _original_formal(view_row):
    receiver = view_row.get("receivers", {}).get("ORIGINAL", {})
    return integrated_core.eligible_detection(
        receiver.get("status"), view_row.get("observations", {}),
        receiver.get("detection") or {"status": "INVALID"},
    )


def _source_record(item, receiver, spec_sha, key_id):
    if receiver == "ORIGINAL":
        return fixed_key.source_max_statistic({view: _original_formal(item["views"][view]) for view in VIEWS})
    if receiver in split_receiver.CANDIDATES:
        return split_receiver.source_statistic(
            {
                view: (
                    item["views"][view]["receivers"][receiver].get("detection") or {}
                    if item["views"][view]["receivers"][receiver].get("status") == "SCORED"
                    and all(
                        item["views"][view].get("observations", {}).get(str(phase), {}).get("status") == "COMPLETE"
                        for phase in PHASES
                    )
                    else {}
                )
                for view in VIEWS
            },
            receiver, spec_sha, key_id,
        )
    return candidate.source_statistic(
        {
            view: (
                item["views"][view]["receivers"][receiver].get("detection") or {}
                if item["views"][view]["receivers"][receiver].get("status") == "SCORED"
                and all(
                    item["views"][view].get("observations", {}).get(str(phase), {}).get("status") == "COMPLETE"
                    for phase in PHASES
                )
                else {}
            )
            for view in VIEWS
        },
        spec_sha, key_id,
    )


def _freeze_four(receiver, sources, expected_ids, spec_sha, key_id):
    if receiver == candidate.RECEIVER_ID:
        return candidate.calibrate(sources, expected_ids, spec_sha, key_id)
    binding = (
        {"receiver_protocol_id": fixed_key.PROTOCOL_ID, "key_id": key_id}
        if receiver == "ORIGINAL"
        else {"receiver_protocol_id": split_receiver.PROTOCOL_ID, "candidate_id": receiver, "spec_sha256": spec_sha, "key_id": key_id}
    )
    ok = set(sources) == set(expected_ids) and len(expected_ids) == 4 and all(
        row.get("status") == "SCORED"
        and math.isfinite(row.get("statistic", math.nan))
        and all(row.get(key) == value for key, value in binding.items())
        for row in sources.values()
    )
    return {
        "status": "FROZEN" if ok else "UNCALIBRATED",
        "threshold": max(row["statistic"] for row in sources.values()) + 1e-6 if ok else None,
        "guard": 1e-6, "source_count": 4, "rank_resolution": "1/5", "sources": sources,
        **binding,
    }


def _decide(receiver, view_row, calibration, spec_sha, key_id):
    receiver_row = view_row["receivers"][receiver]
    if receiver == "ORIGINAL":
        return fixed_key.decide(_original_formal(view_row), calibration)
    if receiver in split_receiver.CANDIDATES:
        detection = receiver_row.get("detection") or {}
        return split_receiver.decide(
            detection.get("confirmation_scores", {}).get(receiver), receiver_row.get("status"),
            calibration, receiver, spec_sha, key_id,
        )
    return candidate.decide(receiver_row.get("detection") or {}, calibration)


def _attach_decisions(item, calibrations, spec_sha, key_id, *, marked, calibration_sample):
    role = "THRESHOLD_CONSTRUCTION_SAMPLE_NOT_HELDOUT_FPR" if calibration_sample else "EVALUATION"
    for receiver in RECEIVERS:
        source = _source_record(item, receiver, spec_sha, key_id)
        for view in VIEWS:
            row = item["views"][view]["receivers"][receiver]
            decision = _decide(receiver, item["views"][view], calibrations[receiver], spec_sha, key_id)
            if decision.get("statistic") is not None and decision.get("threshold") is not None:
                decision["signed_margin"] = decision["statistic"] - decision["threshold"]
            row["decision"] = decision
            row["reporting_only"] = {"marked": marked, "decision_role": role}
        if source.get("winning_view") in VIEWS:
            source_decision = _decide(
                receiver, item["views"][source["winning_view"]], calibrations[receiver], spec_sha, key_id,
            )
        else:
            source_decision = {"status": "INVALID", "detected": None, "statistic": source.get("statistic")}
        source_decision.update(winning_view=source.get("winning_view"), source_max_statistic=source.get("statistic"))
        if source_decision.get("statistic") is not None and source_decision.get("threshold") is not None:
            source_decision["signed_margin"] = source_decision["statistic"] - source_decision["threshold"]
        item["receiver_source_decisions"][receiver] = source_decision
    item["reporting_only"] = {"marked": marked, "decision_role": role}


def _evaluation_summary(result):
    summary = {receiver: {arm: {"views": {}, "sources": {}} for arm in EVALUATION_ARMS} for receiver in RECEIVERS}
    evaluation = [case for case in result["cases"].values() if case.get("role") == "evaluation"]
    for receiver in RECEIVERS:
        for arm in EVALUATION_ARMS:
            view_statuses, source_statuses = {}, {}
            for case in evaluation:
                item = case["videos"][arm]
                status = item["receiver_source_decisions"][receiver].get("status", "NOT_RUN")
                source_statuses[status] = source_statuses.get(status, 0) + 1
                for view in VIEWS:
                    status = item["views"][view]["receivers"][receiver]["decision"].get("status", "NOT_RUN")
                    view_statuses[status] = view_statuses.get(status, 0) + 1
            summary[receiver][arm] = {
                "views": {"denominator": 12, "statuses": view_statuses},
                "sources": {"denominator": 4, "statuses": source_statuses},
            }
    paired_reporting = {}
    for receiver in RECEIVERS:
        grid = {name: 0 for name in ("marked_detected_off_rejected", "both_detected", "both_rejected", "marked_rejected_off_detected", "invalid")}
        rows = []
        for case in evaluation:
            off = case["videos"]["OFF"]
            for arm in ("SINGLE46", "MULTI44_46"):
                marked = case["videos"][arm]
                for view in VIEWS:
                    marked_decision = marked["views"][view]["receivers"][receiver]["decision"]
                    off_decision = off["views"][view]["receivers"][receiver]["decision"]
                    if marked_decision.get("detected") is None or off_decision.get("detected") is None:
                        category = "invalid"
                    elif marked_decision["detected"] and not off_decision["detected"]:
                        category = "marked_detected_off_rejected"
                    elif marked_decision["detected"] and off_decision["detected"]:
                        category = "both_detected"
                    elif not marked_decision["detected"] and not off_decision["detected"]:
                        category = "both_rejected"
                    else:
                        category = "marked_rejected_off_detected"
                    grid[category] += 1
                    marked_stat = marked_decision.get("statistic")
                    off_stat = off_decision.get("statistic")
                    rows.append({
                        "case_id": case["case_id"], "schedule": arm, "view": view,
                        "marked": marked_decision, "same_source_off": off_decision,
                        "decision_category": category,
                        "marked_minus_off_statistic_reporting_only": (
                            marked_stat - off_stat
                            if isinstance(marked_stat, (int, float)) and isinstance(off_stat, (int, float))
                            and math.isfinite(marked_stat) and math.isfinite(off_stat)
                            else None
                        ),
                    })
        paired_reporting[receiver] = {
            "denominator": 24, "decision_grid": grid, "rows": rows,
            "meaning": "same-source descriptive reporting only; not a deployment input or threshold",
        }
    overlaps = {}
    for baseline in ("ORIGINAL", *split_receiver.CANDIDATES):
        scopes = {}
        for scope, arms in (("evaluation_off_views", ("OFF",)), ("evaluation_marked_views", ("SINGLE46", "MULTI44_46"))):
            counts = {name: 0 for name in ("both", "candidate_only", "baseline_only", "neither", "invalid")}
            for case in evaluation:
                for arm in arms:
                    for view in VIEWS:
                        a = case["videos"][arm]["views"][view]["receivers"][candidate.RECEIVER_ID]["decision"]
                        b = case["videos"][arm]["views"][view]["receivers"][baseline]["decision"]
                        if a.get("detected") is None or b.get("detected") is None:
                            counts["invalid"] += 1
                        elif a["detected"] and b["detected"]:
                            counts["both"] += 1
                        elif a["detected"]:
                            counts["candidate_only"] += 1
                        elif b["detected"]:
                            counts["baseline_only"] += 1
                        else:
                            counts["neither"] += 1
            scopes[scope] = {"denominator": 12 if scope == "evaluation_off_views" else 24, "counts": counts}
        overlaps[baseline] = scopes
    return {
        "per_receiver_and_arm": summary,
        "same_source_marked_off_reporting_only": paired_reporting,
        "decision_overlap_candidate_vs_baselines": overlaps,
    }


def _run_all_configured(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = load(config_path)
    validate_manifest(config)
    if (output / "result.json").exists() or (output / "manifest.json").exists():
        raise FileExistsError("run output already contains an experiment; use a new timestamp")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / "manifest.json")
    shutil.copyfile(SPEC, output / "method_spec.md")
    result = {
        "status": "RUNNING", "fixed_denominator": config["fixed_denominator"],
        "candidate_search_accounting": config["candidate_search_accounting"],
        "fixed_calls": TOTAL_PLAN,
        "cases": {case["id"]: shared.empty_case(case) for case in config["cases"]},
        "calibrations": {receiver: {"status": "NOT_RUN", "threshold": None} for receiver in RECEIVERS},
        "failures": [],
    }
    versions = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "numpy", "Pillow"):
        try:
            versions[name] = __import__("importlib.metadata").metadata.version(name)
        except Exception:
            versions[name] = None
    result["environment"] = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": versions}
    shared.dump(output / "environment.json", result["environment"])
    shared.dump(output / "result.json", result)

    def execute_case(case):
        case_root = output / case["id"]
        for stage in ("generate", "media"):
            log_path = output / f"{case['id']}.{stage}.log"
            command = [sys.executable, "-u", "-m", MODULE, "--config", str(config_path), "--output", str(case_root), "--case-id", case["id"], "--stage", stage]
            previous = result["cases"][case["id"]]
            previous.setdefault("stage_logs", {})[stage] = str(log_path.relative_to(output))
            try:
                returncode = shared._run_child(command, log_path)
            except Exception as exc:
                returncode = None
                failure = {"case_id": case["id"], "stage": stage, "error": repr(exc), "kind": "CHILD_SPAWN_OR_TEE_FAILURE"}
                result["failures"].append(failure)
                previous.setdefault("parent_failures", []).append(failure)
            record = case_root / ("generation.json" if stage == "generate" else "result.json")
            if record.exists():
                try:
                    current = load(record)
                    for key, value in previous.items():
                        if key.endswith("_exit_code") or key in ("stage_logs", "parent_failures"):
                            current[key] = value
                    result["cases"][case["id"]] = current
                except Exception as exc:
                    result["failures"].append({"case_id": case["id"], "stage": stage, "error": repr(exc), "kind": "CHILD_RESULT_INVALID"})
            else:
                result["failures"].append({"case_id": case["id"], "stage": stage, "error": "child result missing", "kind": "CHILD_RESULT_MISSING"})
            result["cases"][case["id"]][stage + "_exit_code"] = returncode
            shared.dump(output / "result.json", result)

    calibration_cases = [case for case in config["cases"] if case["role"] == "calibration_off"]
    evaluation_cases = [case for case in config["cases"] if case["role"] == "evaluation"]
    for case in calibration_cases:
        execute_case(case)
    spec_sha, key_id = sha(SPEC), fixed_key.key_identifier(config["key_utf8"].encode("utf-8"))
    calibration_ids = tuple(case["id"] for case in calibration_cases)
    for receiver in RECEIVERS:
        sources = {
            case["id"]: _source_record(result["cases"][case["id"]]["videos"]["OFF"], receiver, spec_sha, key_id)
            for case in calibration_cases
        }
        result["calibrations"][receiver] = _freeze_four(receiver, sources, calibration_ids, spec_sha, key_id)
    shared.dump(output / "calibrations.json", result["calibrations"])
    for case in calibration_cases:
        case_result = result["cases"][case["id"]]
        _attach_decisions(case_result["videos"]["OFF"], result["calibrations"], spec_sha, key_id, marked=False, calibration_sample=True)
        shared.dump(output / case["id"] / "result.json", case_result)
    shared.dump(output / "result.json", result)
    for case in evaluation_cases:
        execute_case(case)
    for case in evaluation_cases:
        case_result = result["cases"][case["id"]]
        for arm, item in case_result["videos"].items():
            _attach_decisions(item, result["calibrations"], spec_sha, key_id, marked=arm != "OFF", calibration_sample=False)
        shared.dump(output / case["id"] / "result.json", case_result)
    result["actual_calls_observed"] = {
        key + "_" + status: sum(case.get("actual_calls", {}).get(key + "_" + status, 0) for case in result["cases"].values())
        for key in TOTAL_PLAN for status in ("attempted", "completed")
    }
    result["call_accounting"] = {
        key: {
            "expected": expected,
            "attempted": result["actual_calls_observed"][key + "_attempted"],
            "completed": result["actual_calls_observed"][key + "_completed"],
            "status": "EXACT" if all(result["actual_calls_observed"][key + "_" + status] == expected for status in ("attempted", "completed")) else "MISMATCH",
        }
        for key, expected in TOTAL_PLAN.items()
    }
    if any(row["status"] != "EXACT" for row in result["call_accounting"].values()):
        result["failures"].append({"stage": "call_accounting", "kind": "EXPECTED_CALL_MISMATCH"})
    result["summary"] = _evaluation_summary(result)
    result["status"] = "EXECUTION_COMPLETE" if (
        all(row.get("status") == "FROZEN" for row in result["calibrations"].values())
        and not result["failures"]
        and all(case.get("status") == "EXECUTION_COMPLETE" and case.get("generate_exit_code") == 0 and case.get("media_exit_code") == 0 for case in result["cases"].values())
    ) else "WITH_RETAINED_FAILURES"
    result["evidence_ceiling"] = config["evidence_ceiling"]
    shared.dump(output / "result.json", result)
    return result


def run_all(config_path, output):
    originals = _configure_shared()
    try:
        return _run_all_configured(config_path, output)
    finally:
        _restore_shared(originals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(MANIFEST))
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--stage", choices=("generate", "media"))
    arguments = parser.parse_args()
    if arguments.case_id and arguments.stage:
        final = generate_case(arguments.case_id, arguments.config, arguments.output) if arguments.stage == "generate" else media_case(arguments.case_id, arguments.config, arguments.output)
    elif not arguments.case_id and not arguments.stage:
        final = run_all(arguments.config, arguments.output)
    else:
        parser.error("--case-id and --stage must be supplied together")
    if final["status"] not in ("GENERATION_COMPLETE", "EXECUTION_COMPLETE"):
        raise SystemExit(1)
