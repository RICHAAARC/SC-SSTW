"""Fresh-generation fixed-key SC-SSTW candidate with fixed attacks and blind receive.

This is an unrun GPU candidate.  It keeps every planned arm/view/phase row and
separates independent OFF calibration, existence, and source existence.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import importlib.metadata
import platform
import math
import resource
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import torch

from main.tube_state import fixed_key
from runtime.wan import fixed_key_core, fixed_key_control, trajectory
from runtime.wan.generation import load_frozen_vae
from runtime.wan.io import dump, encode_rgb, read_mp4
from runtime.wan.vae import _clear_cache, decode_normalized_latent

MANIFEST = Path(__file__).parent / "configs" / "flow_fixed_key_v1.json"
MODULE = "experiments.wan_state_clock.flow_fixed_key_run"
VIEWS = fixed_key.FIXED_VIEWS
PHASES = (0, 1, 2, 3)
MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
R_STAR = fixed_key_control.R_STAR
TOTAL_PLAN = {
    "generation": 4,
    "transformer": 448,
    "scheduler_step": 224,
    "zero_shadow_step": 6,
    "unit_response_probe_step": 6,
    "clean_leaf_backward": 6,
    "vae_decode": 8,
    "mp4_save": 24,
    "mp4_read": 24,
    "vae_encode": 96,
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


def case_arms(case: dict) -> tuple[str, ...]:
    if case["role"] == "calibration_off":
        return ("OFF",)
    return ("OFF", "SINGLE46", "MULTI44_46")


def empty_view() -> dict:
    return {"status": "NOT_RUN", "observations": {str(phase): {"status": "NOT_RUN"} for phase in PHASES}, "detection": None}


def empty_case(case: dict) -> dict:
    return {
        "status": "NOT_RUN",
        "case_id": case["id"],
        "role": case["role"],
        "videos": {arm: {"status": "NOT_RUN", "views": {view: empty_view() for view in VIEWS}} for arm in case_arms(case)},
        "actual_calls": {},
        "failures": [],
    }


def validate_manifest(config: dict) -> None:
    canonical = load(MANIFEST)
    if config != canonical:
        raise ValueError('fixed development roster and method configuration must match the canonical manifest')
    protocol = fixed_key_core.load_protocol()
    fixed_key_core.validate_protocol(protocol)
    for field in ('model', 'generation', 'marker'):
        if config[field] != protocol[field]:
            raise ValueError('fixed experiment and public protocol mismatch: ' + field)
    if config['receiver_protocol_id'] != fixed_key.PROTOCOL_ID or tuple(config['views']) != VIEWS:
        raise ValueError('fixed receiver/view mismatch')
    if any(config['control'].get(k) != v for k, v in protocol['control'].items()):
        raise ValueError('fixed control mismatch')


def case_config(config: dict, case: dict) -> dict:
    result = copy.deepcopy(config)
    result["generation"].update(prompt=case["prompt"], seed=case["seed"])
    return result


def _counter(result: dict):
    def snapshot():
        return {
            "status": result["status"],
            "current": result.get("progress"),
            "actual_calls": dict(result["actual_calls"]),
        }

    def count(kind: str, completed: bool):
        key = kind + ("_completed" if completed else "_attempted")
        result["actual_calls"][key] = result["actual_calls"].get(key, 0) + 1
        dump(Path(result["_progress_path"]), snapshot())
        if kind == "transformer" and completed and result["actual_calls"][key] % 20 == 0:
            print(f"progress model_step_pairs_completed={result['actual_calls'][key] // 2}", flush=True)
    return count


def generate_case(case_id: str, config_path, output) -> dict:
    output = Path(output)
    config = load(config_path)
    validate_manifest(config)
    case = next(row for row in config["cases"] if row["id"] == case_id)
    output.mkdir(parents=True, exist_ok=False)
    result = empty_case(case)
    result.update(status="RUNNING", config=case_config(config, case), source_commit=None,
                  fresh_generation={"source_cache_inputs": [], "prompt": case["prompt"], "seed": case["seed"], "initial_noise": "NEW"},
                  fixed_calls={"generation": 1,
                               "transformer": 100 if case["role"] == "calibration_off" else 124,
                               "scheduler_step": 50 if case["role"] == "calibration_off" else 62,
                               "zero_shadow_step": 0 if case["role"] == "calibration_off" else 3,
                               "unit_response_probe_step": 0 if case["role"] == "calibration_off" else 3,
                               "clean_leaf_backward": 0 if case["role"] == "calibration_off" else 3},
                  _progress_path=str(output / "progress.json"))
    dump(output / "generation.json", {key: value for key, value in result.items() if key != "_progress_path"})

    def save():
        dump(output / "generation.json", {key: value for key, value in result.items() if key != "_progress_path"})

    def fail(stage, exc):
        result["failures"].append({"stage": stage, "error": repr(exc), "traceback": traceback.format_exc()})
        save()

    count = _counter(result)
    print(f"progress case={case_id} stage=generate", flush=True)
    try:
        result["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        result["source_files_sha256"] = {
            str(path): sha(path) for path in (
                Path(__file__), MANIFEST, Path(fixed_key.__file__), Path(fixed_key_control.__file__),
                Path(trajectory.__file__), Path(fixed_key_core.__file__), fixed_key_core.PROTOCOL_PATH,
                Path(fixed_key.carrier.__file__), Path(fixed_key.state_clock.__file__),
                Path(fixed_key_control.prepare_direction.__code__.co_filename),
            )
        }
        arm_names = {arm: ("OFF" if arm == "OFF" else ("SINGLE46" if arm.startswith("SINGLE") else "MULTI44_46")) for arm in case_arms(case)}
        generated = fixed_key_core.generate_key_terminals(
            result["config"], config["key_utf8"].encode(), tuple(arm_names.values()), count,
        )
        result["fresh_generation"]["initial_noise_fingerprint"] = generated["initial_noise_fingerprint"]
        result["fresh_generation"]["state44_fingerprint"] = generated["state44_fingerprint"]
        for arm in case_arms(case):
            result["progress"] = {"case": case_id, "stage": "generate", "arm": arm}
            print(f"progress case={case_id} stage=generate arm={arm}", flush=True)
            item = result["videos"][arm]
            try:
                branch = generated["arms"][arm_names[arm]]
                if branch["status"] != "GENERATED":
                    raise RuntimeError(branch["error"])
                terminal = branch["terminal"]
                torch.save(terminal, output / f"{arm}_terminal.pt")
                array_path = output / "control_tensors" / arm
                array_path.mkdir(parents=True, exist_ok=True)
                for index, tensors in branch["control_tensors"].items():
                    torch.save(tensors, array_path / f"step{index}.pt")
                item.update(status="GENERATED", nominal_terminal=branch["nominal_terminal"],
                            control_steps=branch["control_steps"], clean_gradients=branch["clean_gradients"],
                            unit_probes=branch["unit_probes"],
                            cumulative_native_response=branch["cumulative_native_response"],
                            terminal_fingerprint=branch["terminal_fingerprint"],
                            terminal_path=f"{arm}_terminal.pt")
            except Exception as exc:
                item["status"] = "FAILED_GENERATION"
                fail("generate/" + arm, exc)
            save()
    except Exception as exc:
        fail("generation_setup", exc)
    finally:
        release()
    result["status"] = "GENERATION_COMPLETE" if all(item["status"] == "GENERATED" for item in result["videos"].values()) and not result["failures"] else "WITH_RETAINED_FAILURES"
    result["resources_generation"] = {
        "cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    }
    save()
    dump(Path(result["_progress_path"]), {"status": result["status"], "current": result.get("progress"),
                                           "actual_calls": dict(result["actual_calls"])})
    return result


def _attack_rgb(name: str, rgb, full_path: Path, count):
    if name == "FULL":
        return rgb
    if name == "DELETE90":
        return torch.cat((rgb[:90], rgb[91:]), dim=0)
    if name == "SPEED5_4":
        indices = torch.floor(torch.arange(0, len(rgb), 1.25)).long().clamp(max=len(rgb) - 1)
        return rgb[indices]
    raise ValueError("unknown fixed attack view")


def media_case(case_id: str, config_path, output) -> dict:
    output = Path(output)
    config = load(config_path)
    validate_manifest(config)
    case = next(row for row in config["cases"] if row["id"] == case_id)
    generation_path = output / "generation.json"
    result = load(generation_path) if generation_path.exists() else empty_case(case)
    result["status"] = "MEDIA_RUNNING"
    result["_progress_path"] = str(output / "progress.json")
    for arm in case_arms(case):
        result["videos"].setdefault(arm, {"status": "NOT_RUN"})
        result["videos"][arm]["views"] = {view: empty_view() for view in VIEWS}

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
                    view_row.update(status="VIDEO_PERSISTED", path=str(path.relative_to(output)), frames=int(attacked.shape[0]))
                    def persist_phase(phase, encoded, phase_row):
                        print(f"progress case={case_id} stage=media arm={arm} view={view} phase={phase}", flush=True)
                        latent_path = output / "observations" / arm / view / f"g{phase}.pt"
                        latent_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(encoded, latent_path)
                        phase_row["path"] = str(latent_path.relative_to(output))

                    received = fixed_key_core.receive_mp4(
                        path, key, protocol, None, vae=vae, count=count, on_phase=persist_phase,
                    )
                    view_row["observations"] = received["observations"]
                    view_row["detection"] = received["detection"]
                    for phase, phase_row in received["observations"].items():
                        if phase_row.get("status") != "COMPLETE":
                            result["failures"].append({"stage": f"receive/{arm}/{view}/g{phase}",
                                                       "error": phase_row.get("error", "phase failed")})
                    dump(output / "detections" / arm / f"{view}.json", received["detection"])
                    view_row["status"] = received["status"]
                except Exception as exc:
                    view_row.update(status="FAILED", error=repr(exc))
                    fail(f"media/{arm}/{view}", exc)
                finally:
                    pixels = None
                    _clear_cache(vae)
                    release()
                    save()
            item["status"] = "MEDIA_COMPLETE" if all(row["status"] == "SCORED" for row in item["views"].values()) else "PARTIAL_OR_FAILED"
            rgb = None
            release()
    except Exception as exc:
        fail("media_setup", exc)
    finally:
        vae = None
        release()
    result["status"] = "EXECUTION_COMPLETE" if all(item["status"] == "MEDIA_COMPLETE" for item in result["videos"].values()) and not result["failures"] else "WITH_RETAINED_FAILURES"
    result["resources_media"] = {
        "cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    }
    save()
    dump(Path(result["_progress_path"]), {"status": result["status"], "current": result.get("progress"),
                                           "actual_calls": dict(result["actual_calls"])})
    return result


def eligible_detection(view_row: dict) -> dict:
    """Adapt raw ranking to the protocol: all four phase rows must complete."""
    raw = view_row.get("detection") or {"status": "INVALID"}
    return fixed_key_core.eligible_detection(view_row.get("status"), view_row.get("observations", {}), raw)


def _arm_source_record(item: dict) -> dict:
    return fixed_key.source_max_statistic({view: eligible_detection(item.get('views', {}).get(view, {})) for view in VIEWS})


def _attach_decisions(item: dict, calibration: dict, *, marked: bool, calibration_sample: bool) -> None:
    source = _arm_source_record(item)
    item['source_max_statistic'] = source
    role = 'THRESHOLD_CONSTRUCTION_SAMPLE_NOT_HELDOUT_FPR' if calibration_sample else 'EVALUATION'
    item['decision_role'] = role
    for view in VIEWS:
        row = item['views'][view]
        formal = eligible_detection(row)
        row['decision'] = fixed_key.decide(formal, calibration)
        row['alignment_reporting_only'] = fixed_key.alignment_report(row.get('detection') or {}, view)
        row['reporting_only'] = dict(marked=marked, decision_role=role,
            detected_marked=None if not marked else row['decision']['status']=='DETECTED')
    if source['status'] == 'SCORED':
        decision = fixed_key.decide(eligible_detection(item['views'][source['winning_view']]), calibration)
        decision.update(winning_view=source['winning_view'], source_max_statistic=source['statistic'])
    else:
        decision = dict(status='INVALID', detected=None)
    item['decision'] = decision
    item['reporting_only'] = dict(marked=marked, decision_role=role,
        detected_marked=None if not marked else decision['status']=='DETECTED')


def _run_child(command: list[str], log_path: Path) -> int:
    """Tee child progress to the notebook/stdout while retaining an exact log."""
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert child.stdout is not None
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return child.wait()


def run_all(config_path, output) -> dict:
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = load(config_path)
    validate_manifest(config)
    if (output / "result.json").exists() or (output / "manifest.json").exists():
        raise FileExistsError("run output already contains an experiment; use a new timestamp")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output / "manifest.json")
    result = {
        "status": "RUNNING",
        "fixed_denominator": {"fresh_cases": 4, "generated_arms": 8, "saved_attack_views": 24, "receiver_encodes": 96,
                              "calibration_sources": 2, "evaluation_sources": 2},
        "fixed_calls": TOTAL_PLAN,
        "cases": {case["id"]: empty_case(case) for case in config["cases"]},
        "calibration": {"status": "NOT_RUN", "threshold": None},
        "failures": [],
    }
    versions = {}
    for name in ('torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'numpy', 'Pillow'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    result['environment'] = dict(python=sys.version, executable=sys.executable, platform=platform.platform(), packages=versions)
    dump(output / 'environment.json', result['environment'])
    dump(output / "result.json", result)
    def execute_case(case):
        case_root = output / case["id"]
        for stage in ("generate", "media"):
            log_path = output / f"{case['id']}.{stage}.log"
            command = [sys.executable, "-u", "-m", MODULE, "--config", str(config_path), "--output", str(case_root),
                       "--case-id", case["id"], "--stage", stage]
            print(f"progress case={case['id']} stage={stage} log={log_path}", flush=True)
            previous = result["cases"][case["id"]]
            previous.setdefault("stage_logs", {})[stage] = str(log_path.relative_to(output))
            try:
                returncode = _run_child(command, log_path)
            except Exception as exc:
                failure = {"case_id": case["id"], "stage": stage, "log": str(log_path.relative_to(output)),
                           "error": repr(exc), "kind": "CHILD_SPAWN_OR_TEE_FAILURE"}
                result["failures"].append(failure)
                previous.setdefault("parent_failures", []).append(failure)
                previous[stage + "_exit_code"] = None
                previous["status"] = "FAILED_LAUNCH_OR_RESULT"
                dump(output / "result.json", result)
                continue
            record_path = case_root / ("generation.json" if stage == "generate" else "result.json")
            if record_path.exists():
                current = load(record_path)
                for key, value in previous.items():
                    if key.endswith("_exit_code") or key in ("stage_logs", "parent_failures"):
                        current[key] = value
                result["cases"][case["id"]] = current
            else:
                result["cases"][case["id"]]["status"] = "FAILED_LAUNCH_OR_RESULT"
                failure = {"case_id": case["id"], "stage": stage, "log": str(log_path.relative_to(output)),
                           "error": "child result file missing", "kind": "CHILD_RESULT_MISSING"}
                result["failures"].append(failure)
                result["cases"][case["id"]].setdefault("parent_failures", []).append(failure)
            result["cases"][case["id"]][stage + "_exit_code"] = returncode
            dump(output / "result.json", result)

    calibration_cases = [case for case in config["cases"] if case["role"] == "calibration_off"]
    evaluation_cases = [case for case in config["cases"] if case["role"] == "evaluation"]
    for case in calibration_cases:
        execute_case(case)
    calibration_rows = []
    for case in calibration_cases:
        case_result = result["cases"][case["id"]]
        for arm, item in case_result["videos"].items():
            source_row = _arm_source_record(item)
            item["source_max_statistic"] = source_row
            if arm == "OFF":
                calibration_rows.append(source_row | {"case_id": case["id"]})
    calibration = fixed_key.freeze_calibration(
        calibration_rows, config["calibration"]["guard"],
        protocol_id=config["receiver_protocol_id"], key_id=fixed_key.key_identifier(config["key_utf8"].encode()),
    )
    result["calibration"] = calibration
    dump(output / "calibration.json", calibration)
    for case in calibration_cases:
        case_result = result["cases"][case["id"]]
        for item in case_result["videos"].values():
            _attach_decisions(item, calibration, marked=False, calibration_sample=True)
        dump(output / case["id"] / "result.json", case_result)
    dump(output / "result.json", result)
    for case in evaluation_cases:
        execute_case(case)
    for case in evaluation_cases:
        case_result = result["cases"][case["id"]]
        for arm, item in case_result["videos"].items():
            _attach_decisions(item, calibration, marked=arm != "OFF", calibration_sample=False)
        dump(output / case["id"] / "result.json", case_result)
    result["actual_calls_observed"] = {
        key + "_" + status: sum(
            case.get("actual_calls", {}).get(key + "_" + status, 0) for case in result["cases"].values()
        )
        for key in TOTAL_PLAN for status in ("attempted", "completed")
    }
    result["status"] = "EXECUTION_COMPLETE" if (
        calibration.get("status") == "FROZEN"
        and not result["failures"]
        and all(case.get("status") == "EXECUTION_COMPLETE" and case.get("generate_exit_code") == 0 and case.get("media_exit_code") == 0 for case in result["cases"].values())
    ) else "WITH_RETAINED_FAILURES"
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
        final = generate_case(arguments.case_id, arguments.config, arguments.output) if arguments.stage == "generate" else media_case(arguments.case_id, arguments.config, arguments.output)
    elif not arguments.case_id and not arguments.stage:
        final = run_all(arguments.config, arguments.output)
    else:
        parser.error("--case-id and --stage must be supplied together")
    if final["status"] not in ("GENERATION_COMPLETE", "EXECUTION_COMPLETE"):
        raise SystemExit(1)
