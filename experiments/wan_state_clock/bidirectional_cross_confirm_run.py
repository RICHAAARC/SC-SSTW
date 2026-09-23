"""Fixed new-source legacy run for bidirectional cross-confirmation."""
from __future__ import annotations

import argparse
import copy
import contextlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from experiments.wan_state_clock import window_state_mse_paired_run as shared
from main.tube_state import bidirectional_cross_confirm_receiver as bidirectional
from main.tube_state import fixed_key
from main.tube_state import fixed_key_split_receiver as split_receiver
from runtime.wan import fixed_key_control, fixed_key_core
from runtime.wan.integrated_core import eligible_detection
from runtime.wan.io import dump

MANIFEST = Path(__file__).parent / "configs" / "bidirectional_cross_confirm_v1.json"
SPEC = Path(__file__).parents[2] / "docs" / "bidirectional_cross_confirm_v1_spec.md"
MODULE = "experiments.wan_state_clock.bidirectional_cross_confirm_run"
VIEWS = fixed_key.FIXED_VIEWS
PHASES = (0, 1, 2, 3)
RECEIVERS = ("ORIGINAL", *split_receiver.CANDIDATES, bidirectional.CANDIDATE_ID)
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
    return shared.sha(path)


def case_arms(case):
    return ("OFF",) if case["role"] == "calibration_off" else EVALUATION_ARMS


def validate_manifest(config):
    if config != load(MANIFEST):
        raise ValueError("fixed route roster and method configuration must match the canonical manifest")
    protocol = fixed_key_core.load_protocol()
    fixed_key_core.validate_protocol(protocol)
    for field in ("model", "generation"):
        if config[field] != protocol[field]:
            raise ValueError("fixed route and public protocol mismatch: " + field)
    expected_protocols = {
        "receiver_protocol_id": fixed_key.PROTOCOL_ID,
        "split_receiver_protocol_id": split_receiver.PROTOCOL_ID,
        "candidate_receiver_protocol_id": bidirectional.PROTOCOL_ID,
    }
    if any(config[key] != value for key, value in expected_protocols.items()):
        raise ValueError("receiver protocol mismatch")
    if tuple(config["views"]) != VIEWS or tuple(config["receivers"]) != RECEIVERS:
        raise ValueError("fixed view or receiver roster mismatch")
    if tuple(config["calibration_arms"]) != ("OFF",):
        raise ValueError("calibration must be OFF only")
    if tuple(config["evaluation_arms"]) != EVALUATION_ARMS:
        raise ValueError("evaluation arm roster mismatch")
    if config["writer_objective"] != fixed_key_control.DEFAULT_OBJECTIVE:
        raise ValueError("legacy writer objective mismatch")
    if config["expected_complete_calls"] != TOTAL_PLAN:
        raise ValueError("fixed call plan mismatch")
    if any(config["control"].get(key) != value for key, value in protocol["control"].items()):
        raise ValueError("fixed control mismatch")
    if tuple(case["id"] for case in config["cases"][:4]) != bidirectional.CALIBRATION_SOURCE_IDS:
        raise ValueError("four-source calibration roster mismatch")


def _source_files():
    return (
        Path(__file__), MANIFEST, SPEC, Path(shared.__file__),
        Path(fixed_key.__file__), Path(split_receiver.__file__), Path(bidirectional.__file__),
        Path(fixed_key_core.__file__), fixed_key_core.PROTOCOL_PATH,
        Path(fixed_key_control.__file__), Path(shared.payload_control.__file__),
        Path(shared.trajectory.__file__), Path(shared.generation.__file__),
        Path(shared.integrated_core.__file__), Path(shared.io.__file__), Path(shared.vae.__file__),
        Path(fixed_key.carrier.__file__), Path(fixed_key.state_clock.__file__),
    )


def _receiver_rows(observations, phases, key, spec_sha):
    complete = all(phases.get(str(phase), {}).get("status") == "COMPLETE" for phase in PHASES)
    detections = {}
    for name, reader in (
        ("ORIGINAL", lambda: fixed_key.read(observations, fixed_key.codebook(key))),
        ("SPLIT", lambda: split_receiver.read(observations, key, spec_sha)),
        (bidirectional.CANDIDATE_ID, lambda: bidirectional.read(observations, key, spec_sha)),
    ):
        try:
            detections[name] = reader()
        except Exception as exc:
            detections[name] = {
                "status": "INVALID", "reason": "receiver exception", "error": repr(exc),
            }
    rows = {
        "ORIGINAL": {
            "status": "SCORED" if complete and detections["ORIGINAL"].get("status") == "SCORED" else "PARTIAL_OR_FAILED",
            "detection": detections["ORIGINAL"],
            "decision": {"status": "NOT_RUN", "detected": None},
        },
        bidirectional.CANDIDATE_ID: {
            "status": "SCORED" if complete and detections[bidirectional.CANDIDATE_ID].get("status") == "SCORED" else "PARTIAL_OR_FAILED",
            "detection": detections[bidirectional.CANDIDATE_ID],
            "decision": {"status": "NOT_RUN", "detected": None},
        },
    }
    split_status = (
        "SCORED" if complete and detections["SPLIT"].get("status") == "SCORED"
        else "PARTIAL_OR_FAILED"
    )
    for candidate in split_receiver.CANDIDATES:
        rows[candidate] = {
            "status": split_status,
            "detection": copy.deepcopy(detections["SPLIT"]),
            "decision": {"status": "NOT_RUN", "detected": None},
        }
    return {receiver: rows[receiver] for receiver in RECEIVERS}


def _legacy_generation_plan(case):
    return {
        "generation": 1,
        "transformer": 100 if case["role"] == "calibration_off" else 124,
        "scheduler_step": 50 if case["role"] == "calibration_off" else 62,
        "zero_shadow_step": 0 if case["role"] == "calibration_off" else 3,
        "unit_response_probe_step": 0 if case["role"] == "calibration_off" else 3,
        "clean_leaf_backward": 0 if case["role"] == "calibration_off" else 3,
    }


def _legacy_generate_case_terminals(result, config, case, count):
    arms = ("OFF",) if case["role"] == "calibration_off" else EVALUATION_ARMS
    return fixed_key_core.generate_key_terminals(
        result["config"], config["key_utf8"].encode(), arms, count,
        objective=fixed_key_control.DEFAULT_OBJECTIVE,
    )


def _legacy_branch_writer_objective(branch, case, generated):
    return None if branch is generated["arms"].get("OFF") else generated["writer_objective"]


_SHARED_OVERRIDES = {
    "MANIFEST": MANIFEST,
    "SPEC": SPEC,
    "MODULE": MODULE,
    "VIEWS": VIEWS,
    "PHASES": PHASES,
    "RECEIVERS": RECEIVERS,
    "EVALUATION_ARMS": EVALUATION_ARMS,
    "TOTAL_PLAN": TOTAL_PLAN,
    "case_arms": case_arms,
    "validate_manifest": validate_manifest,
    "_source_files": _source_files,
    "_receiver_rows": _receiver_rows,
    "generation_call_plan": _legacy_generation_plan,
    "generate_case_terminals": _legacy_generate_case_terminals,
    "branch_writer_objective": _legacy_branch_writer_objective,
}


@contextlib.contextmanager
def _shared_context():
    previous = {name: getattr(shared, name) for name in _SHARED_OVERRIDES}
    try:
        for name, value in _SHARED_OVERRIDES.items():
            setattr(shared, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(shared, name, value)


def generate_case(case_id, config_path, output):
    with _shared_context():
        result = shared.generate_case(case_id, config_path, output)
    return result


def media_case(case_id, config_path, output):
    with _shared_context():
        return shared.media_case(case_id, config_path, output)


def empty_case(case):
    with _shared_context():
        return shared.empty_case(case)


def _original_formal(view_row):
    receiver = view_row.get("receivers", {}).get("ORIGINAL", shared.empty_receiver())
    return eligible_detection(
        receiver.get("status"), view_row.get("observations", {}),
        receiver.get("detection") or {"status": "INVALID"},
    )


def _source_record(item, receiver, spec_sha, key_id):
    if receiver == "ORIGINAL":
        return fixed_key.source_max_statistic({
            view: _original_formal(item.get("views", {}).get(view, {})) for view in VIEWS
        })
    views = {
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
    }
    if receiver == bidirectional.CANDIDATE_ID:
        return bidirectional.source_statistic(views, spec_sha, key_id)
    return split_receiver.source_statistic(views, receiver, spec_sha, key_id)


def _freeze_four_sources(sources, receiver, spec_sha, key_id, guard):
    if receiver == bidirectional.CANDIDATE_ID:
        return bidirectional.calibrate(sources, spec_sha, key_id, guard)
    if receiver == "ORIGINAL":
        binding = {"receiver_protocol_id": fixed_key.PROTOCOL_ID, "key_id": key_id}
    else:
        binding = {
            "candidate_id": receiver, "spec_sha256": spec_sha, "key_id": key_id,
            "receiver_protocol_id": split_receiver.PROTOCOL_ID,
        }
    ok = set(sources) == set(bidirectional.CALIBRATION_SOURCE_IDS) and all(
        row.get("status") == "SCORED"
        and math.isfinite(row.get("statistic", math.nan))
        and all(row.get(key) == value for key, value in binding.items())
        for row in sources.values()
    )
    return dict(
        status="FROZEN" if ok else "UNCALIBRATED",
        threshold=max(row["statistic"] for row in sources.values()) + guard if ok else None,
        guard=guard, sources=sources, source_count=4,
        empirical_rank_resolution="1/5",
        claim="functional OFF calibration; not a low-FPR estimate",
        **binding,
    )


def _margin(decision):
    score, threshold = decision.get("statistic"), decision.get("threshold")
    if not isinstance(score, (int, float)) or not isinstance(threshold, (int, float)):
        return None
    return score - threshold if math.isfinite(score) and math.isfinite(threshold) else None


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
            elif receiver in split_receiver.CANDIDATES:
                detection = receiver_row.get("detection") or {}
                score = detection.get("confirmation_scores", {}).get(receiver)
                decision = split_receiver.decide(
                    score, receiver_row.get("status"), calibrations[receiver],
                    receiver, spec_sha, key_id,
                )
                alignment = detection.get("alignment_reporting_only", {}).get(
                    view, {"status": "UNAVAILABLE"},
                )
            else:
                detection = receiver_row.get("detection") or {}
                decision = bidirectional.decide(
                    detection.get("candidate_score"), receiver_row.get("status"),
                    calibrations[receiver], spec_sha, key_id,
                )
                alignment = detection.get("alignment_reporting_only", {}).get(
                    view, {"status": "UNAVAILABLE"},
                )
            decision["signed_margin"] = _margin(decision)
            receiver_row["decision"] = decision
            receiver_row["alignment_reporting_only"] = alignment
            receiver_row["reporting_only"] = {
                "marked": marked, "decision_role": role,
                "detected_marked": None if not marked else decision.get("status") == "DETECTED",
            }
        if source.get("status") == "SCORED":
            if receiver == "ORIGINAL":
                source_decision = fixed_key.decide(
                    _original_formal(item["views"][source["winning_view"]]), calibrations[receiver],
                )
            elif receiver in split_receiver.CANDIDATES:
                source_decision = split_receiver.decide(
                    source["statistic"], source["status"], calibrations[receiver],
                    receiver, spec_sha, key_id,
                )
            else:
                source_decision = bidirectional.decide(
                    source["statistic"], source["status"], calibrations[receiver], spec_sha, key_id,
                )
        else:
            source_decision = {"status": "INVALID", "detected": None}
        source_decision.update(
            winning_view=source.get("winning_view"),
            source_max_statistic=source.get("statistic"),
        )
        source_decision["signed_margin"] = _margin(source_decision)
        item["receiver_source_decisions"][receiver] = source_decision
    item["reporting_only"] = {"marked": marked, "decision_role": role}


def _run_child(command, log_path):
    return shared._run_child(command, log_path)


def _comparison_records(result):
    reference_names = {
        "FULL": "IDENTITY",
        "DELETE90": "DELETE90_REFERENCE",
        "SPEED5_4": "SPEED5_4_REFERENCE",
    }
    rows = []
    for case_id, case in result["cases"].items():
        if case.get("role") != "evaluation":
            continue
        for arm, item in case["videos"].items():
            for view in VIEWS:
                view_row = item["views"][view]
                reference_name = reference_names[view]
                single = view_row["receivers"]["C2_STATE_CONFIRM"]
                single_detection = single.get("detection") or {}
                single_blind = single_detection.get("confirmation_scores", {}).get("C2_STATE_CONFIRM")
                single_reference = (
                    single_detection.get("fixed_path_diagnostics", {})
                    .get(reference_name, {}).get("scores", {}).get("C2_STATE_CONFIRM")
                )
                candidate = view_row["receivers"][bidirectional.CANDIDATE_ID]
                candidate_detection = candidate.get("detection") or {}
                directional_reference = {
                    name: (
                        direction.get("fixed_path_diagnostics", {})
                        .get(reference_name, {}).get("scores", {}).get("C2_STATE_CONFIRM")
                    )
                    for name, direction in candidate_detection.get("directions", {}).items()
                }
                candidate_reference = (
                    sum(directional_reference.values()) / 2.0
                    if len(directional_reference) == 2
                    and all(value is not None and math.isfinite(value) for value in directional_reference.values())
                    else None
                )
                candidate_blind = candidate_detection.get("candidate_score")
                single_threshold = result["calibrations"]["C2_STATE_CONFIRM"].get("threshold")
                candidate_threshold = result["calibrations"][bidirectional.CANDIDATE_ID].get("threshold")
                rows.append({
                    "case_id": case_id,
                    "arm": arm,
                    "view": view,
                    "marked_reporting_only": arm != "OFF",
                    "reference_path_reporting_only": reference_name,
                    "C2_STATE_CONFIRM": {
                        "blind_score": single_blind,
                        "fixed_reference_score": single_reference,
                        "blind_minus_reference_gap": (
                            None if single_blind is None or single_reference is None
                            else single_blind - single_reference
                        ),
                        "blind_margin_vs_own_threshold": single.get("decision", {}).get("signed_margin"),
                        "fixed_reference_margin_vs_own_threshold": (
                            None if single_reference is None or single_threshold is None
                            else single_reference - single_threshold
                        ),
                    },
                    bidirectional.CANDIDATE_ID: {
                        "blind_score": candidate_blind,
                        "directional_blind_C2": candidate_detection.get("c2_candidate_components", {}),
                        "fixed_reference_score": candidate_reference,
                        "directional_fixed_reference_C2": directional_reference,
                        "blind_minus_reference_gap": (
                            None if candidate_blind is None or candidate_reference is None
                            else candidate_blind - candidate_reference
                        ),
                        "blind_margin_vs_own_threshold": candidate.get("decision", {}).get("signed_margin"),
                        "fixed_reference_margin_vs_own_threshold": (
                            None if candidate_reference is None or candidate_threshold is None
                            else candidate_reference - candidate_threshold
                        ),
                    },
                })
    return rows


def run_all(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = load(config_path)
    validate_manifest(config)
    if (output / "result.json").exists() or (output / "manifest.json").exists():
        raise FileExistsError("run output already contains an experiment; use a new timestamp")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / "manifest.json")
    shutil.copyfile(SPEC, output / "method_spec.md")
    result = {
        "status": "RUNNING",
        "fixed_denominator": config["fixed_denominator"] | {
            "receivers": 4, "view_decision_slots": 192, "source_arm_decision_slots": 64,
        },
        "fixed_calls": TOTAL_PLAN,
        "cases": {case["id"]: empty_case(case) for case in config["cases"]},
        "calibrations": {receiver: {"status": "NOT_RUN", "threshold": None} for receiver in RECEIVERS},
        "comparison_records": [],
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
            previous = result["cases"][case["id"]]
            previous.setdefault("stage_logs", {})[stage] = str(log_path.relative_to(output))
            print(f"progress case={case['id']} stage={stage} log={log_path}", flush=True)
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
            try:
                current = load(record_path)
            except Exception as exc:
                failure = {
                    "case_id": case["id"], "stage": stage,
                    "log": str(log_path.relative_to(output)), "error": repr(exc),
                    "kind": "CHILD_RESULT_INVALID" if record_path.exists() else "CHILD_RESULT_MISSING",
                }
                result["failures"].append(failure)
                previous.setdefault("parent_failures", []).append(failure)
                previous["status"] = "FAILED_LAUNCH_OR_RESULT"
            else:
                for key, value in previous.items():
                    if key.endswith("_exit_code") or key in ("stage_logs", "parent_failures"):
                        current[key] = value
                result["cases"][case["id"]] = current
            result["cases"][case["id"]][stage + "_exit_code"] = returncode
            dump(output / "result.json", result)

    calibration_cases = [case for case in config["cases"] if case["role"] == "calibration_off"]
    evaluation_cases = [case for case in config["cases"] if case["role"] == "evaluation"]
    for case in calibration_cases:
        execute_case(case)
    spec_sha = sha(SPEC)
    key_id = fixed_key.key_identifier(config["key_utf8"].encode())
    sources = {receiver: {} for receiver in RECEIVERS}
    for case in calibration_cases:
        item = result["cases"][case["id"]]["videos"]["OFF"]
        for receiver in RECEIVERS:
            sources[receiver][case["id"]] = _source_record(item, receiver, spec_sha, key_id)
    result["calibrations"] = {
        receiver: _freeze_four_sources(
            sources[receiver], receiver, spec_sha, key_id, config["calibration"]["guard"],
        )
        for receiver in RECEIVERS
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
    result["comparison_records"] = _comparison_records(result)
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
