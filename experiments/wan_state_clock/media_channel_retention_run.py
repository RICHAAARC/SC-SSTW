"""Read-only four-layer retention diagnostic over one fixed retained run."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path

import torch

from main.tube_state import fixed_key, fixed_key_split_receiver
from runtime.wan import fixed_key_core, media_channel_retention as retention, trajectory
from runtime.wan.generation import load_frozen_vae
from runtime.wan.io import dump
from runtime.wan.vae import _clear_cache, quantize_rgb8_no_codec, reencode_rgb24_readback

MANIFEST = Path(__file__).parent / "configs" / "media_channel_retention_v1.json"
SPEC = Path(__file__).parents[2] / "docs" / "media_channel_retention_v1_spec.md"
MODULE = "experiments.wan_state_clock.media_channel_retention_run"
LAYERS = (
    "TERMINAL", "FLOAT_RGB_REENCODE", "RGB8_NO_CODEC_REENCODE", "EXISTING_MP4_G0",
)
ADJACENT_TRANSITIONS = (
    ("TERMINAL", "FLOAT_RGB_REENCODE", "OLD_VAE_DECODE_CLAMP_PLUS_CURRENT_POSTERIOR_MODE_ENCODE_ROUNDTRIP"),
    ("FLOAT_RGB_REENCODE", "RGB8_NO_CODEC_REENCODE", "RGB8_QUANTIZATION_PROPAGATED_THROUGH_SAME_VAE"),
    ("RGB8_NO_CODEC_REENCODE", "EXISTING_MP4_G0", "RGB24_TO_YUV420_H264_READBACK_MEDIA_CHAIN_ADDITIONAL_CHANGE"),
)
DIRECT_COMBINED_TRANSITION = (
    "TERMINAL", "RGB8_NO_CODEC_REENCODE", "VAE_PLUS_RGB8_QUANTIZATION_COMBINED_CHANGE",
)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def release(vae=None):
    if vae is not None:
        _clear_cache(vae)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_manifest(config):
    canonical = load(MANIFEST)
    if config != canonical:
        raise ValueError("fixed media-retention configuration must match the canonical manifest")
    if config["input_root"] != (
        "/content/drive/MyDrive/Video-WM/Window-State-MSE-V1/"
        "window_state_mse_v1_20260922T174349464768Z"
    ):
        raise ValueError("input root is not the fixed retained run")
    if tuple(config["cases"]) != ("eval_p2_s2", "eval_p3_s3"):
        raise ValueError("fixed evaluation cases changed")
    if config["input_seeds"] != {"eval_p2_s2": 20260919, "eval_p3_s3": 20260920}:
        raise ValueError("fixed input seeds changed")
    if tuple(config["arms"]) != (
        "OFF", "LEGACY_SINGLE46", "MSE_SINGLE46", "LEGACY_MULTI44_46", "MSE_MULTI44_46",
    ):
        raise ValueError("fixed arm roster changed")
    if tuple(config["marked_arms"]) != tuple(config["arms"][1:]):
        raise ValueError("fixed marked arm roster changed")
    if tuple(config["layers"]) != LAYERS or tuple(config["partitions"]) != retention.PARTITIONS:
        raise ValueError("fixed layer or partition roster changed")
    if config["identity_path"] != fixed_key.REFERENCE_PATHS["IDENTITY"]:
        raise ValueError("identity path changed")
    if config["call_plan"] != {"vae_encode": 20}:
        raise ValueError("fixed VAE encode budget changed")
    if config["fixed_denominator"] != {
        "trajectories": 10,
        "layer_partition_slots": 120,
        "marked_off_increment_slots": 96,
        "adjacent_increment_change_slots": 72,
    }:
        raise ValueError("fixed denominator changed")
    retention.partition_contract()


def _source_files():
    from runtime.wan import generation, io, vae

    return (
        Path(__file__), MANIFEST, SPEC, Path(retention.__file__), Path(fixed_key.__file__),
        Path(fixed_key_split_receiver.__file__),
        Path(fixed_key.carrier.__file__), Path(fixed_key.state_clock.__file__),
        Path(fixed_key_core.__file__), fixed_key_core.PROTOCOL_PATH,
        Path(generation.__file__), Path(io.__file__), Path(vae.__file__), Path(trajectory.__file__),
    )


def _empty_layer(layer):
    return {
        "layer": layer,
        "status": "NOT_RUN",
        "artifact": None,
        "measurements": {
            name: retention.empty_measurement() | {"partition": name}
            for name in retention.PARTITIONS
        },
        "failure": None,
    }


def _empty_trajectory(case_id, arm):
    return {
        "case_id": case_id,
        "arm": arm,
        "status": "NOT_RUN",
        "input_audit": {
            name: {"status": "NOT_RUN"} for name in ("terminal", "decoded_rgb", "full_mp4", "existing_g0")
        },
        "layers": {layer: _empty_layer(layer) for layer in LAYERS},
        "raw_layer_changes": [],
        "failures": [],
    }


def _empty_case(config, case_id):
    return {
        "case_id": case_id,
        "status": "NOT_RUN",
        "trajectories": {arm: _empty_trajectory(case_id, arm) for arm in config["arms"]},
        "call_accounting": {
            "vae_encode": {"expected": 10, "attempted": 0, "completed": 0, "status": "NOT_RUN"},
            "vae_load": {"attempted": 0, "completed": 0},
        },
        "resources": {
            "status": "NOT_MEASURED",
            "claim": "No CPU, RAM, GPU, VRAM, or timing requirement was measured by this diagnostic",
        },
        "failures": [],
    }


def _root_input_audit(config, input_root):
    result_path = input_root / "result.json"
    audit = {
        "status": "NOT_RUN",
        "path": str(result_path),
        "expected_sha256": config["input_result"]["sha256"],
        "actual_sha256": None,
        "expected_size_bytes": config["input_result"]["size_bytes"],
        "actual_size_bytes": None,
        "expected_source_commit": config["input_source_commit"],
        "recorded_case_source_commits": {},
        "recorded_status": None,
        "recorded_experiment_id": None,
        "failures": [],
    }
    record = None
    try:
        audit["actual_size_bytes"] = result_path.stat().st_size
        audit["actual_sha256"] = sha256(result_path)
        if audit["actual_size_bytes"] != audit["expected_size_bytes"]:
            audit["failures"].append("INPUT_RESULT_SIZE_MISMATCH")
        if audit["actual_sha256"] != audit["expected_sha256"]:
            audit["failures"].append("INPUT_RESULT_SHA256_MISMATCH")
        record = load(result_path)
        audit["recorded_status"] = record.get("status")
        audit["recorded_experiment_id"] = (
            record.get("experiment_id") or (record.get("config") or {}).get("experiment_id")
        )
        for case_id in config["cases"]:
            case = record.get("cases", {}).get(case_id)
            if not isinstance(case, dict):
                audit["recorded_case_source_commits"][case_id] = None
                audit["failures"].append("MISSING_INPUT_CASE:" + case_id)
                continue
            source_commit = case.get("source_commit")
            audit["recorded_case_source_commits"][case_id] = source_commit
            if source_commit != config["input_source_commit"]:
                audit["failures"].append("INPUT_SOURCE_COMMIT_MISMATCH:" + case_id)
            missing = [arm for arm in config["arms"] if arm not in case.get("videos", {})]
            if missing:
                audit["failures"].append("MISSING_INPUT_ARMS:" + case_id + ":" + ",".join(missing))
    except Exception as exc:
        audit["failures"].append("INPUT_RESULT_UNREADABLE:" + repr(exc))
    audit["status"] = "VERIFIED" if not audit["failures"] else "FAILED"
    return audit, record


def _file_audit(path, expected_sha=None, historical_hash_available=True):
    try:
        exists = path.is_file()
    except Exception as exc:
        return {
            "status": "FAILED", "path": str(path), "exists": None,
            "expected_sha256": expected_sha, "actual_sha256": None,
            "historical_hash_verification": "FILE_EXISTENCE_PROBE_FAILED",
            "error": repr(exc),
        }
    row = {
        "status": "NOT_RUN", "path": str(path), "exists": exists,
        "expected_sha256": expected_sha, "actual_sha256": None,
        "historical_hash_verification": None,
    }
    if not row["exists"]:
        row.update(status="MISSING", historical_hash_verification="FAILED_MISSING")
        return row
    try:
        row["actual_sha256"] = sha256(path)
    except Exception as exc:
        row.update(
            status="FAILED", error=repr(exc),
            historical_hash_verification="FILE_HASH_READ_FAILED",
        )
        return row
    if not historical_hash_available:
        row.update(status="PRESENT", historical_hash_verification="UNAVAILABLE_NO_RECORDED_HASH")
    elif not expected_sha:
        row.update(status="FAILED", historical_hash_verification="UNAVAILABLE_MISSING_EXPECTED_HASH_FIELD")
    elif row["actual_sha256"] == expected_sha:
        row.update(status="VERIFIED", historical_hash_verification="MATCH")
    else:
        row.update(status="FAILED", historical_hash_verification="MISMATCH")
    return row


def _terminal_audit(path, expected_fingerprint):
    row = _file_audit(path, historical_hash_available=False)
    row.update(expected_tensor_fingerprint=expected_fingerprint, actual_tensor_fingerprint=None)
    tensor = None
    if not row["exists"] or row["status"] == "FAILED":
        return row, tensor
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        row["actual_tensor_fingerprint"] = trajectory.fingerprint(tensor)
        if not expected_fingerprint:
            row.update(status="FAILED", historical_hash_verification="UNAVAILABLE_MISSING_EXPECTED_FINGERPRINT")
        elif row["actual_tensor_fingerprint"] == expected_fingerprint:
            row.update(status="VERIFIED", historical_hash_verification="TENSOR_FINGERPRINT_MATCH")
        else:
            row.update(status="FAILED", historical_hash_verification="TENSOR_FINGERPRINT_MISMATCH")
    except Exception as exc:
        row.update(status="FAILED", error=repr(exc), historical_hash_verification="TENSOR_LOAD_FAILED")
    return row, tensor


def _old_artifact_references(old_arm):
    """Extract only fields present in the retained Window-State-MSE-V1 schema."""
    full = old_arm.get("views", {}).get("FULL", {})
    return {
        "terminal_fingerprint": old_arm.get("terminal_fingerprint"),
        "terminal_path": old_arm.get("terminal_path"),
        "full_mp4_sha256": full.get("sha256"),
        "full_mp4_path": full.get("path"),
        "full_mp4_frames": full.get("frames"),
        "full_g0_sha256": full.get("observations", {}).get("0", {}).get("sha256"),
        "full_g0_path": full.get("observations", {}).get("0", {}).get("path"),
        "full_g0_status": full.get("observations", {}).get("0", {}).get("status"),
        "full_g0_frames_used": full.get("observations", {}).get("0", {}).get("frames_used"),
        "full_g0_tail_discarded": full.get("observations", {}).get("0", {}).get("tail_discarded"),
    }


def _prepare_output(output, input_root, result_name):
    output = Path(output).resolve()
    input_root = Path(input_root).resolve()
    if output == input_root or input_root in output.parents:
        raise ValueError("output must not be the fixed input root or any directory below it")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / result_name
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {result_path}")
    return output


def _recorded_metadata_audit(arm, references):
    expected = {
        "terminal_path": f"{arm}_terminal.pt",
        "full_mp4_path": f"received_videos/{arm}/FULL.mp4",
        "full_mp4_frames": 181,
        "full_g0_path": f"observations/{arm}/FULL/g0.pt",
        "full_g0_status": "COMPLETE",
        "full_g0_frames_used": 181,
        "full_g0_tail_discarded": 0,
    }
    mismatches = {
        name: {"expected": value, "recorded": references.get(name)}
        for name, value in expected.items() if references.get(name) != value
    }
    return {
        "status": "VERIFIED" if not mismatches else "FAILED",
        "expected": expected,
        "recorded": {name: references.get(name) for name in expected},
        "mismatches": mismatches,
    }


def _score_layer(layer_row, tensor, config, book):
    try:
        scored = retention.score_identity_layer(
            tensor, book, config["identity_path"], config["expected_shapes"]["latent"],
        )
        layer_row.update(status="SCORED", measurements=scored["partitions"], failure=None)
    except Exception as exc:
        _fail_layer(layer_row, repr(exc))


def _fail_layer(layer_row, error):
    layer_row.update(status="FAILED", failure=error)
    for partition in retention.PARTITIONS:
        layer_row["measurements"][partition].update(status="FAILED", failure=error)


def _measurement(case_result, arm, layer, partition):
    return case_result["trajectories"][arm]["layers"][layer]["measurements"][partition]


def _attach_raw_changes(case_result, config):
    epsilon = config["numeric_near_zero_epsilon"]
    transitions = (*ADJACENT_TRANSITIONS, DIRECT_COMBINED_TRANSITION)
    for arm in config["arms"]:
        rows = []
        for before, after, meaning in transitions:
            for partition in retention.PARTITIONS:
                rows.append({
                    "before_layer": before,
                    "after_layer": after,
                    "partition": partition,
                    "meaning": meaning,
                    "comparison": retention.compare_measurements(
                        _measurement(case_result, arm, after, partition),
                        _measurement(case_result, arm, before, partition),
                        epsilon,
                    ),
                })
        case_result["trajectories"][arm]["raw_layer_changes"] = rows


def run_case(case_id, config_path, output):
    config = load(config_path)
    validate_manifest(config)
    if case_id not in config["cases"]:
        raise ValueError("case is outside the fixed evaluation roster")
    input_root = Path(config["input_root"])
    output = _prepare_output(output, input_root, "case_result.json")
    case_result = _empty_case(config, case_id)
    case_result["status"] = "RUNNING"
    case_result["input_seed"] = config["input_seeds"][case_id]
    case_result["output_path"] = str(Path(output))
    root_audit, old_result = _root_input_audit(config, input_root)
    case_result["root_input_audit"] = root_audit
    case_result["source_files_sha256"] = {str(path): sha256(path) for path in _source_files()}
    result_path = output / "case_result.json"

    def save():
        dump(result_path, case_result)

    def fail(stage, exc):
        case_result["failures"].append({
            "stage": stage, "error": repr(exc), "traceback": traceback.format_exc(),
        })
        save()

    save()
    if root_audit["status"] != "VERIFIED" or old_result is None:
        error = RuntimeError("fixed input result audit failed")
        fail("root_input_audit", error)
        case_result["status"] = "WITH_RETAINED_FAILURES"
        case_result["call_accounting"]["vae_encode"]["status"] = "MISMATCH"
        _attach_raw_changes(case_result, config)
        save()
        return case_result

    old_case = old_result["cases"][case_id]
    book = fixed_key.codebook(config["key_utf8"].encode())
    vae_holder = {"value": None, "failure": None}

    def get_vae():
        if vae_holder["failure"] is not None:
            raise RuntimeError("cached VAE load failure") from vae_holder["failure"]
        if vae_holder["value"] is None:
            case_result["call_accounting"]["vae_load"]["attempted"] += 1
            try:
                vae_holder["value"] = load_frozen_vae(fixed_key_core.load_protocol())
            except Exception as exc:
                vae_holder["failure"] = exc
                raise
            case_result["call_accounting"]["vae_load"]["completed"] += 1
            save()
        return vae_holder["value"]

    for arm in config["arms"]:
        print(f"progress case={case_id} arm={arm}", flush=True)
        item = case_result["trajectories"][arm]
        old_arm = old_case["videos"].get(arm, {})
        case_root = input_root / case_id
        terminal_path = case_root / f"{arm}_terminal.pt"
        decoded_path = case_root / "decoded_rgb" / f"{arm}.pt"
        mp4_path = case_root / "received_videos" / arm / "FULL.mp4"
        g0_path = case_root / "observations" / arm / "FULL" / "g0.pt"

        references = _old_artifact_references(old_arm)
        metadata_audit = _recorded_metadata_audit(arm, references)
        terminal_audit, terminal = _terminal_audit(
            terminal_path, references["terminal_fingerprint"],
        )
        decoded_audit = _file_audit(decoded_path, historical_hash_available=False)
        mp4_audit = _file_audit(mp4_path, references["full_mp4_sha256"])
        g0_audit = _file_audit(g0_path, references["full_g0_sha256"])
        item["input_audit"].update(
            terminal=terminal_audit, decoded_rgb=decoded_audit, full_mp4=mp4_audit, existing_g0=g0_audit,
            recorded_metadata=metadata_audit,
        )
        if metadata_audit["status"] != "VERIFIED":
            item["failures"].append({
                "stage": "recorded_metadata", "error": metadata_audit["mismatches"],
            })
            terminal_audit.update(status="FAILED", metadata_binding="MISMATCH")
            mp4_audit.update(status="FAILED", metadata_binding="MISMATCH")
            g0_audit.update(status="FAILED", metadata_binding="MISMATCH")

        terminal_layer = item["layers"]["TERMINAL"]
        terminal_layer["artifact"] = {"path": str(terminal_path), "audit": terminal_audit}
        if terminal_audit["status"] == "VERIFIED" and terminal is not None:
            _score_layer(terminal_layer, terminal, config, book)
        else:
            _fail_layer(terminal_layer, "terminal input audit failed")

        g0_layer = item["layers"]["EXISTING_MP4_G0"]
        g0_layer["artifact"] = {
            "path": str(g0_path), "audit": g0_audit,
            "full_mp4_binding_audit": mp4_audit,
        }
        if g0_audit["status"] == "VERIFIED" and mp4_audit["status"] == "VERIFIED":
            try:
                g0 = torch.load(g0_path, map_location="cpu", weights_only=True)
                _score_layer(g0_layer, g0, config, book)
            except Exception as exc:
                _fail_layer(g0_layer, repr(exc))
        else:
            _fail_layer(g0_layer, "existing FULL MP4/g0 input audit failed")

        rgb = None
        if decoded_audit["status"] == "PRESENT":
            try:
                rgb = torch.load(decoded_path, map_location="cpu", weights_only=True)
                if tuple(rgb.shape) != tuple(config["expected_shapes"]["decoded_rgb"]):
                    raise ValueError(
                        f"decoded RGB shape mismatch: {tuple(rgb.shape)} != "
                        f"{tuple(config['expected_shapes']['decoded_rgb'])}"
                    )
                if not bool(torch.isfinite(rgb).all()):
                    raise ValueError("decoded RGB contains nonfinite values")
                if bool((rgb < 0).any()) or bool((rgb > 1).any()):
                    raise ValueError("decoded RGB is outside [0,1]")
            except Exception as exc:
                item["failures"].append({"stage": "decoded_rgb_load", "error": repr(exc)})
                rgb = None
        else:
            item["failures"].append({"stage": "decoded_rgb_audit", "error": "decoded RGB missing"})

        for layer, quantized in (
            ("FLOAT_RGB_REENCODE", False), ("RGB8_NO_CODEC_REENCODE", True),
        ):
            row = item["layers"][layer]
            row["artifact"] = {
                "source_path": str(decoded_path),
                "source_sha256": decoded_audit.get("actual_sha256"),
                "historical_source_hash_verification": decoded_audit.get("historical_hash_verification"),
                "rgb8_quantized_before_encode": quantized,
                "persisted": False,
            }
            if rgb is None:
                _fail_layer(row, "decoded RGB unavailable")
                continue
            try:
                vae = get_vae()
                encode_input = quantize_rgb8_no_codec(rgb).float().div(255.0) if quantized else rgb
                case_result["call_accounting"]["vae_encode"]["attempted"] += 1
                encoded = reencode_rgb24_readback(vae, encode_input).detach().cpu()
                case_result["call_accounting"]["vae_encode"]["completed"] += 1
                _score_layer(row, encoded, config, book)
                encoded = None
                encode_input = None
            except Exception as exc:
                _fail_layer(row, repr(exc))
                item["failures"].append({"stage": layer, "error": repr(exc)})
            finally:
                release(vae_holder["value"])
                save()

        terminal = None
        rgb = None
        item["status"] = (
            "SCORED" if all(row["status"] == "SCORED" for row in item["layers"].values())
            else "WITH_RETAINED_FAILURES"
        )
        if item["status"] != "SCORED":
            item["failures"].append({"stage": "trajectory", "error": "one or more fixed layers failed"})
        save()
        release(vae_holder["value"])

    _attach_raw_changes(case_result, config)
    accounting = case_result["call_accounting"]["vae_encode"]
    accounting["status"] = (
        "EXACT" if accounting["attempted"] == accounting["expected"] == accounting["completed"] else "MISMATCH"
    )
    if accounting["status"] != "EXACT":
        case_result["failures"].append({
            "stage": "call_accounting", "error": "case VAE encode count is not exactly 10/10",
        })
    case_result["status"] = (
        "EXECUTION_COMPLETE"
        if not case_result["failures"]
        and all(item["status"] == "SCORED" for item in case_result["trajectories"].values())
        else "WITH_RETAINED_FAILURES"
    )
    vae_holder["value"] = None
    release()
    save()
    return case_result


def _run_child(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return child.wait()


def _attach_pair_comparisons(result, config):
    epsilon = config["numeric_near_zero_epsilon"]
    paired = []
    adjacent_changes = []
    direct_combined = []
    for case_id in config["cases"]:
        case = result["cases"][case_id]
        for arm in config["marked_arms"]:
            increments = {}
            for layer in LAYERS:
                increments[layer] = {}
                for partition in retention.PARTITIONS:
                    comparison = retention.compare_measurements(
                        _measurement(case, arm, layer, partition),
                        _measurement(case, "OFF", layer, partition),
                        epsilon,
                    )
                    increments[layer][partition] = comparison
                    paired.append({
                        "case_id": case_id, "marked_arm": arm, "off_arm": "OFF",
                        "layer": layer, "partition": partition,
                        "comparison": comparison,
                    })
            for before, after, meaning in ADJACENT_TRANSITIONS:
                for partition in retention.PARTITIONS:
                    adjacent_changes.append({
                        "case_id": case_id, "marked_arm": arm,
                        "before_layer": before, "after_layer": after,
                        "partition": partition, "meaning": meaning,
                        "comparison": retention.compare_increments(
                            increments[after][partition], increments[before][partition], epsilon,
                        ),
                    })
            before, after, meaning = DIRECT_COMBINED_TRANSITION
            for partition in retention.PARTITIONS:
                direct_combined.append({
                    "case_id": case_id, "marked_arm": arm,
                    "before_layer": before, "after_layer": after,
                    "partition": partition, "meaning": meaning,
                    "comparison": retention.compare_increments(
                        increments[after][partition], increments[before][partition], epsilon,
                    ),
                })
    result["marked_off_increments"] = paired
    result["adjacent_increment_changes"] = adjacent_changes
    result["terminal_to_rgb8_combined_increment_changes"] = direct_combined


def _slot_accounting(result, config):
    layer_rows = [
        trajectory_row["layers"][layer]["measurements"][partition]
        for case in result["cases"].values()
        for trajectory_row in case["trajectories"].values()
        for layer in LAYERS
        for partition in retention.PARTITIONS
    ]
    groups = {
        "layer_partition_slots": layer_rows,
        "marked_off_increment_slots": [row["comparison"] for row in result["marked_off_increments"]],
        "adjacent_increment_change_slots": [row["comparison"] for row in result["adjacent_increment_changes"]],
    }
    accounting = {}
    for name, rows in groups.items():
        expected = config["fixed_denominator"][name]
        complete = sum(row.get("status") in ("SCORED", "COMPARED") for row in rows)
        accounting[name] = {
            "expected": expected, "retained": len(rows), "complete": complete,
            "failed_or_undefined": len(rows) - complete,
            "status": "RETAINED" if len(rows) == expected else "DENOMINATOR_MISMATCH",
        }
    return accounting


def run_all(config_path, output):
    config = load(config_path)
    validate_manifest(config)
    input_root = Path(config["input_root"])
    output = _prepare_output(output, input_root, "result.json")
    root_audit, _ = _root_input_audit(config, input_root)
    result = {
        "experiment_id": config["experiment_id"],
        "status": "RUNNING",
        "config": config,
        "seed": config["input_seeds"],
        "output_path": str(output),
        "source_commit": None,
        "source_files_sha256": {},
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "input_root_audit": root_audit,
        "cases": {case_id: _empty_case(config, case_id) for case_id in config["cases"]},
        "failures": [],
        "resource_measurement": "NOT_MEASURED_AS_A_REQUIREMENT",
    }
    try:
        result["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
    except Exception as exc:
        result["failures"].append({"stage": "source_commit", "error": repr(exc)})
    result["source_files_sha256"] = {str(path): sha256(path) for path in _source_files()}
    dump(output / "result.json", result)

    for case_id in config["cases"]:
        case_output = output / case_id
        case_output.mkdir(parents=True, exist_ok=True)
        log_path = output / f"{case_id}.log"
        command = [
            sys.executable, "-u", "-m", MODULE, "--config", str(config_path),
            "--output", str(case_output), "--case-id", case_id,
        ]
        print(f"progress case={case_id} log={log_path}", flush=True)
        try:
            returncode = _run_child(command, log_path)
        except Exception as exc:
            returncode = None
            result["failures"].append({
                "case_id": case_id, "stage": "child_launch", "error": repr(exc),
            })
        child_path = case_output / "case_result.json"
        if child_path.exists():
            try:
                result["cases"][case_id] = load(child_path)
            except Exception as exc:
                result["failures"].append({
                    "case_id": case_id, "stage": "child_result", "error": repr(exc),
                })
        else:
            result["failures"].append({
                "case_id": case_id, "stage": "child_result", "error": "case_result.json missing",
            })
        result["cases"][case_id]["subprocess"] = {
            "returncode": returncode,
            "log": str(log_path.relative_to(output)),
            "result": str(child_path.relative_to(output)),
        }
        if returncode != 0:
            result["failures"].append({
                "case_id": case_id, "stage": "child_exit", "returncode": returncode,
            })
        dump(output / "result.json", result)

    _attach_pair_comparisons(result, config)
    result["slot_accounting"] = _slot_accounting(result, config)
    attempted = sum(
        case["call_accounting"]["vae_encode"]["attempted"] for case in result["cases"].values()
    )
    completed = sum(
        case["call_accounting"]["vae_encode"]["completed"] for case in result["cases"].values()
    )
    expected = config["call_plan"]["vae_encode"]
    result["call_accounting"] = {
        "vae_encode": {
            "expected": expected, "attempted": attempted, "completed": completed,
            "status": "EXACT" if expected == attempted == completed else "MISMATCH",
        },
        "vae_load": {
            "attempted": sum(case["call_accounting"]["vae_load"]["attempted"] for case in result["cases"].values()),
            "completed": sum(case["call_accounting"]["vae_load"]["completed"] for case in result["cases"].values()),
            "meaning": "one lazy reused VAE load per successful case subprocess",
        },
    }
    if result["call_accounting"]["vae_encode"]["status"] != "EXACT":
        result["failures"].append({"stage": "call_accounting", "error": "VAE encode budget mismatch"})
    if any(row["status"] != "RETAINED" for row in result["slot_accounting"].values()):
        result["failures"].append({"stage": "slot_accounting", "error": "fixed denominator was not retained"})
    result["status"] = (
        "EXECUTION_COMPLETE"
        if not result["failures"]
        and root_audit["status"] == "VERIFIED"
        and all(case["status"] == "EXECUTION_COMPLETE" for case in result["cases"].values())
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
    arguments = parser.parse_args()
    final = (
        run_case(arguments.case_id, arguments.config, arguments.output)
        if arguments.case_id else run_all(arguments.config, arguments.output)
    )
    if final["status"] != "EXECUTION_COMPLETE":
        raise SystemExit(1)
