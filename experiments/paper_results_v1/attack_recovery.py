"""Resume the fixed two-pilot attack evaluation from its saved media.

The source run is never modified.  Recovery copies its fixed-denominator state
to a new directory, validates/decodes already-published media, and attempts only
the missing work.  Completed or interrupted expensive-call receipts are never
silently retried.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.paper_results_v1 import attack_eval as base
from experiments.paper_results_v1.attack_matrix import apply_edit_rgb8, recipe_receipt


SOURCE_RUN_ID = "20261010T031617441784Z-2536172d"
RECOVERY_PHASES = (
    "index-saved-media", "receiver-clock", "receiver-read",
    "baseline-extract-videoseal", "baseline-extract-rivagan", "quality", "evaluate",
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rewrite(value, old_root: str, new_root: str):
    if isinstance(value, dict):
        return {key: _rewrite(item, old_root, new_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, old_root, new_root) for item in value]
    if isinstance(value, str) and value.startswith(old_root):
        return new_root + value[len(old_root):]
    return value


def _old_root(state):
    path = state.get("evaluation_report", {}).get("path")
    if not path:
        raise ValueError("source attack state does not identify its run directory")
    return str(Path(path).parent.parent)


def _terminal_receiver(row):
    return row.get("status") in ("EVALUATED", "UNSUPPORTED")


def _reset_case_phase(state, phase):
    row = state["phases"][phase]
    for case_id in base.ATTEMPT_CASES:
        row["cases"][case_id] = {"status": "PLANNED", "failures": []}
    row["status"] = "PLANNED"


def initialize(config_path, source_run, output, *, temp_root=None):
    """Create or reopen the single recovery directory for one source run."""
    output = Path(output)
    state_path = output / "attack_run_state.json"
    if state_path.is_file():
        return _read(state_path)
    config = base.validate_config(_read(config_path))
    source_run = Path(source_run).resolve()
    source_state_path = source_run / "run_state" / "attack_run_state.json"
    if not source_state_path.is_file():
        source_state_path = source_run / "attack_run_state.json"
    source_state = _read(source_state_path)
    old_root = _old_root(source_state)
    state = _rewrite(copy.deepcopy(source_state), old_root, str(source_run))
    output.mkdir(parents=True, exist_ok=False)
    base.atomic_json(output / "source_attack_state.json", source_state)
    state["source_phase_history"] = copy.deepcopy(state["phases"])
    state["source_evaluation_report"] = copy.deepcopy(state.get("evaluation_report"))
    state.pop("evaluation_report", None)
    state["status"] = "RECOVERY_INITIALIZED"
    state["recovery"] = {
        "schema_version": "paper-results-v1-temporal-attack-recovery-v1",
        "source_run_id": SOURCE_RUN_ID, "source_run": str(source_run),
        "source_recorded_root": old_root,
        "source_state": str(source_state_path), "created_at_unix": time.time(),
        "retain_pending_on_interrupt": True,
        "phases": {phase: {"status": "PLANNED", "failures": []} for phase in RECOVERY_PHASES},
        "source_actual_calls": {},
        "budget": {
            "reused_published_mp4": 116, "new_baseline_temporal_codec_roundtrips": 56,
            "new_main_framewise_clock_encodes": 112, "new_main_logical_reads": 448,
            "new_main_physical_wan_encode_upper_bound": 336,
            "new_baseline_extracts": 60, "new_quality_rows": 4,
        },
    }
    for case_id in base.ATTEMPT_CASES:
        records = state.setdefault("records", {}).setdefault(case_id, {})
        for field in ("framewise_clock_encodes", "physical_wan_encodes"):
            if field in records:
                state["recovery"]["source_actual_calls"][f"{case_id}/{field}"] = records.pop(field)
        for phase in (
            "receiver-clock", "receiver-read", "baseline-extract-videoseal",
            "baseline-extract-rivagan", "quality",
        ):
            _reset_case_phase(state, phase)
    state["phases"]["evaluate"] = {"status": "PLANNED", "failures": []}
    for artifact in state["artifacts"]:
        if artifact["case_id"] not in base.ATTEMPT_CASES:
            continue
        parts = artifact["artifact_id"].split("/")
        if artifact["kind"] == "TEMPORAL_OBSERVATION" and parts[2] != "full":
            artifact["source_run_status"] = artifact.get("status")
            artifact["source_run_reason"] = artifact.get("reason")
            artifact.update(status="RECOVERY_PENDING", reason=None)
        elif artifact["kind"] == "TEMPORAL_OBSERVATION" and parts[1] in base.BASELINES:
            artifact["source_run_status"] = artifact.get("status")
            artifact.update(status="RECOVERY_PENDING", reason=None)
        elif artifact["artifact_id"].endswith("/NATIVE_POST"):
            artifact["source_run_status"] = artifact.get("status")
            artifact["source_run_reason"] = artifact.get("reason")
            artifact.update(status="RECOVERY_PENDING", reason=None)
    for row in state["receiver_rows"]:
        if row["case_id"] in base.ATTEMPT_CASES and row["attack_id"] != "full":
            row["source_run_status"] = row.get("status")
            row["source_run_reason"] = row.get("reason")
            row.update(status="PLANNED", reason=None)
    for row in state["baseline_rows"]:
        if row["case_id"] in base.ATTEMPT_CASES:
            row["source_run_status"] = row.get("status")
            row["source_run_reason"] = row.get("reason")
            row.update(status="PLANNED", reason=None)
    for row in state["quality_rows"]:
        if row["case_id"] in base.ATTEMPT_CASES and row["candidate"] in base.BASELINES:
            row["source_run_status"] = row.get("status")
            row["source_run_reason"] = row.get("reason")
            row.update(status="PLANNED", reason=None)
    base.atomic_json(state_path, state)
    return state


def _phase(store, name, function, *, rematerialize=False):
    row = store.data["recovery"]["phases"].setdefault(name, {"status": "PLANNED", "failures": []})
    if row["status"] == "COMPLETE" and not rematerialize:
        return {"status": "ALREADY_COMPLETE"}
    row.update(status="RUNNING", started_at_unix=time.time())
    store.save()
    try:
        value = function()
        row.update(status="COMPLETE", finished_at_unix=time.time())
        store.save()
        return value
    except BaseException as exc:
        row.update(status="PARTIAL", finished_at_unix=time.time())
        row.setdefault("failures", []).append(f"{type(exc).__name__}: {exc}")
        store.save()
        raise


def _aggregate_case_phase(store, name):
    states = [
        store.data["recovery"]["phases"].get(f"{name}/{case_id}", {}).get("status", "PLANNED")
        for case_id in base.ATTEMPT_CASES
    ]
    aggregate = (
        "RUNNING" if "RUNNING" in states else
        "PARTIAL" if "PARTIAL" in states else
        "COMPLETE" if all(state == "COMPLETE" for state in states) else
        "PARTIAL" if any(state == "COMPLETE" for state in states) else "PLANNED"
    )
    store.data["recovery"]["phases"][name]["status"] = aggregate
    store.save()


def _mp4_for(source_run, case_id, method, attack_id):
    if method in base.BASELINES and attack_id == "native_codec":
        return Path(source_run) / "run_state" / "media" / case_id / method / "native_codec" / "published.mp4"
    return Path(source_run) / "run_state" / "media" / case_id / method / attack_id / "published.mp4"


def _decode(mp4, target, expected_frames):
    from runtime.wan.variable_rgb_media import decode_published
    _, receipt = decode_published(mp4, target, expected_frames=expected_frames)
    return base._available_media(receipt)


def _materialize_main_post(store, case_id, arm):
    artifact = store.artifact(f"{case_id}/{arm}/POST")
    path = Path(artifact.get("path", ""))
    if path.is_file():
        return artifact
    mp4 = Path(artifact.get("mp4", {}).get("path", ""))
    receipt = _decode(mp4, store.temp_root / case_id / arm / "source_post.rgb8", 181)
    artifact.update(**receipt, recovered_from_published_mp4=str(mp4))
    store.save()
    return artifact


def _validate_npz(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as arrays:
        for key in arrays.files:
            _ = arrays[key].shape


def _resolve_document_paths(value, record_dir):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "path" and isinstance(item, str) and not Path(item).is_absolute():
                result[key] = str((Path(record_dir) / item).resolve())
            else:
                result[key] = _resolve_document_paths(item, record_dir)
        return result
    if isinstance(value, list):
        return [_resolve_document_paths(item, record_dir) for item in value]
    return value


def _validate_reused_rows(store, case_id):
    recorded_root = store.data["recovery"]["source_recorded_root"]
    source_root = store.data["recovery"]["source_run"]
    for row in base._receiver_rows(store, case_id, attack_id="full"):
        if not _terminal_receiver(row):
            continue
        try:
            for field in ("clock_record", "detail_record"):
                receipt = row.get(field)
                if not receipt:
                    continue
                path = Path(receipt["path"])
                document = _resolve_document_paths(
                    _rewrite(_read(path), recorded_root, source_root), path.parent,
                )
                for candidate in (
                    document.get("evidence", {}).get("path"),
                    document.get("vote_sidecar", {}).get("path"),
                ):
                    if candidate:
                        _validate_npz(candidate)
                copied = store.output / "reused_records" / case_id / field / (row["slot_id"].replace("/", "_") + ".json")
                receipt.update(base.atomic_json(copied, document))
            row["reused_evidence_validation"] = "READABLE"
        except Exception as exc:
            row.update(
                status="FAILED", missing_bits=32,
                reason=f"REUSED_EVIDENCE_UNREADABLE: {type(exc).__name__}: {exc}",
                reused_evidence_validation="FAILED",
            )
    for row in store.data["quality_rows"]:
        if row["case_id"] != case_id or row.get("status") not in ("EVALUATED", "PARTIAL"):
            continue
        try:
            if row.get("record"):
                source_record = Path(row["record"]["path"])
                document = _resolve_document_paths(
                    _rewrite(_read(source_record), recorded_root, source_root), source_record.parent,
                )
                copied = store.output / "reused_records" / case_id / "quality" / (row["quality_id"].replace("/", "_") + ".json")
                row["record"].update(base.atomic_json(copied, document))
            video = row.get("side_by_side", {}).get("path")
            if video:
                command = [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,nb_frames", "-of", "json", video,
                ]
                result = subprocess.run(command, capture_output=True, check=False)
                if result.returncode:
                    raise RuntimeError(f"reused side-by-side probe exited {result.returncode}")
                stream = json.loads(result.stdout)["streams"][0]
                if (int(stream["width"]), int(stream["height"]), int(stream["nb_frames"])) != (1024, 320, 181):
                    raise ValueError("reused side-by-side geometry mismatch")
                decode = subprocess.run(
                    ["ffmpeg", "-v", "error", "-noautorotate", "-i", video,
                     "-map", "0:v:0", "-f", "null", "-"],
                    capture_output=True, check=False,
                )
                if decode.returncode:
                    raise RuntimeError(f"reused side-by-side decode exited {decode.returncode}")
            row["reused_evidence_validation"] = "READABLE"
        except Exception as exc:
            row.update(status="FAILED", reason=f"REUSED_QUALITY_UNREADABLE: {type(exc).__name__}: {exc}")
    store.save()


def phase_index_saved_media(store, config, case_id):
    def execute():
        from runtime.wan.variable_rgb_media import roundtrip
        source_run = store.data["recovery"]["source_run"]
        attacks = {row["attack_id"]: row for row in config["attacks"]}
        _validate_reused_rows(store, case_id)
        for arm in base.MAIN_ARMS:
            try:
                _materialize_main_post(store, case_id, arm)
            except Exception as exc:
                store.artifact(f"{case_id}/{arm}/POST")["recovery_materialization_error"] = f"{type(exc).__name__}: {exc}"
                store.save()
            for attack_id, spec in attacks.items():
                if attack_id == "full":
                    continue
                target = store.artifact(f"{case_id}/{arm}/{attack_id}/RECEIVED")
                if target["status"] == "AVAILABLE" and Path(target.get("path", "")).is_file():
                    continue
                try:
                    truth = recipe_receipt(spec)
                    receipt = _decode(
                        _mp4_for(source_run, case_id, arm, attack_id),
                        store.temp_root / case_id / arm / attack_id / "received.rgb8",
                        truth["output_frames"],
                    )
                    target.update(
                        **receipt, recovered_existing_publication=True,
                        publication_count_from_pre=2, edit_publication_added=True,
                        weighted_truth_record=base.atomic_json(
                            base._record_path(store, case_id, arm, attack_id, "weighted_truth.json"), truth,
                        ),
                        weighted_truth_sha256=truth["weighted_truth_sha256"],
                        output_frames=truth["output_frames"], truth_hidden_from_receiver=True,
                    )
                except Exception as exc:
                    target.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
                store.save()
        for method in base.BASELINES:
            native = store.artifact(f"{case_id}/{method}/NATIVE_POST")
            if native["status"] != "AVAILABLE" or not Path(native.get("path", "")).is_file():
                try:
                    native.update(**_decode(
                        _mp4_for(source_run, case_id, method, "native_codec"),
                        store.temp_root / case_id / method / "native_post.rgb8", 181,
                    ), recovered_existing_publication=True)
                except Exception as exc:
                    native.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
                store.save()
            for attack_id, spec in attacks.items():
                target = store.artifact(f"{case_id}/{method}/{attack_id}/RECEIVED")
                if target["status"] == "AVAILABLE" and Path(target.get("path", "")).is_file():
                    continue
                try:
                    if native["status"] != "AVAILABLE":
                        raise ValueError(f"baseline native POST is {native['status']}")
                    truth = recipe_receipt(spec)
                    if attack_id == "full":
                        target.update(
                            status="AVAILABLE", path=native["path"], shape=native["shape"],
                            dtype="uint8", bytes=native["bytes"], full_alias_of=native["artifact_id"],
                            publication_count_from_pre=1, edit_publication_added=False,
                        )
                    else:
                        saved_mp4 = store.output / "media" / case_id / method / attack_id / "published.mp4"
                        if saved_mp4.is_file():
                            target.update(**_decode(
                                saved_mp4, store.temp_root / case_id / method / attack_id / "received.rgb8",
                                truth["output_frames"],
                            ), recovered_interrupted_publication=True)
                        else:
                            edited, _ = apply_edit_rgb8(base._load_rgb8(native), spec)
                            calls = store.data["recovery"].setdefault("actual_calls", {}).setdefault("baseline_edit_codecs", [])
                            call_id = f"{case_id}/{method}/{attack_id}"
                            previous = next((row for row in calls if row["call_id"] == call_id), None)
                            if previous is not None:
                                raise RuntimeError("PRIOR_BASELINE_EDIT_CODEC_ATTEMPT_RETAINED_NO_REPEAT")
                            call = {"call_id": call_id, "status": "RUNNING", "started_at_unix": time.time()}
                            calls.append(call); store.save()
                            try:
                                _, codec = roundtrip(
                                    edited, store.temp_root / case_id / method / attack_id,
                                    media_output=store.output / "media" / case_id / method / attack_id,
                                )
                                call.update(status="COMPLETE", finished_at_unix=time.time()); store.save()
                            except BaseException as exc:
                                if isinstance(exc, Exception):
                                    call.update(status="FAILED", finished_at_unix=time.time(), reason=f"{type(exc).__name__}: {exc}")
                                    store.save()
                                raise
                            target.update(
                                **base._available_media(codec["received_rgb"]),
                                codec_record=base.atomic_json(
                                    base._record_path(store, case_id, method, attack_id, "codec.json"), codec,
                                ),
                            )
                        target.update(publication_count_from_pre=2, edit_publication_added=True)
                    target.update(
                        weighted_truth_record=base.atomic_json(
                            base._record_path(store, case_id, method, attack_id, "weighted_truth.json"), truth,
                        ), weighted_truth_sha256=truth["weighted_truth_sha256"],
                        output_frames=truth["output_frames"], truth_hidden_from_receiver=True,
                    )
                except Exception as exc:
                    target.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
                store.save()
        return {"case_id": case_id, "artifact_status_counts": base._counts(
            row["status"] for row in store.data["artifacts"] if row["case_id"] == case_id
        )}
    try:
        return _phase(store, f"index-saved-media/{case_id}", execute, rematerialize=True)
    finally:
        _aggregate_case_phase(store, "index-saved-media")


def _case_recovery_phase(store, name, case_id, function):
    recovery_key = f"{name}/{case_id}"
    if store.data["recovery"]["phases"].get(recovery_key, {}).get("status") == "COMPLETE":
        return {"status": "ALREADY_COMPLETE"}
    phase = store.data["phases"][name]
    phase["cases"][case_id]["status"] = "PLANNED"
    phase["status"] = "PLANNED"
    store.save()
    try:
        return _phase(store, recovery_key, function)
    finally:
        _aggregate_case_phase(store, name)


def run_case_phase(store, config, phase, case_id):
    if phase == "index-saved-media":
        return phase_index_saved_media(store, config, case_id)
    dispatch = {
        "receiver-clock": lambda: base.phase_receiver_clock(store, config, case_id),
        "receiver-read": lambda: base.phase_receiver_read(store, config, case_id),
        "baseline-extract-videoseal": lambda: base.phase_baseline_extract(store, config, case_id, "videoseal"),
        "baseline-extract-rivagan": lambda: base.phase_baseline_extract(store, config, case_id, "rivagan"),
        "quality": lambda: base.phase_quality(store, config, case_id),
    }
    if phase not in dispatch:
        raise ValueError(f"unknown recovery phase {phase}")
    return _case_recovery_phase(store, phase, case_id, dispatch[phase])


def evaluate(store, config):
    def execute():
        store.data["phases"]["evaluate"] = {"status": "PLANNED", "failures": []}
        report = base.phase_evaluate(store, config)
        store.data["status"] = "RECOVERY_COMPLETE_WITH_ITEM_STATUSES"
        store.data["recovery"]["item_status_counts"] = {
            "receiver": base._counts(row["status"] for row in store.data["receiver_rows"]),
            "baseline": base._counts(row["status"] for row in store.data["baseline_rows"]),
            "quality": base._counts(row["status"] for row in store.data["quality_rows"]),
        }
        store.save()
        return report
    return _phase(store, "evaluate", execute, rematerialize=True)


def record_external_failure(store, phase, case_id, reason):
    key = phase if phase == "evaluate" else f"{phase}/{case_id}"
    row = store.data["recovery"]["phases"].setdefault(key, {"status": "PLANNED", "failures": []})
    if row["status"] == "RUNNING":
        row.update(status="PARTIAL", finished_at_unix=time.time())
        row.setdefault("failures", []).append(reason)
    if phase == "quality":
        base._seal_interrupted_quality_rows(store, case_id, reason)
    store.save()
    return row
