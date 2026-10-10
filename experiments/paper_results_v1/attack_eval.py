"""Independent fixed temporal-attack evaluation over saved two-pilot media.

This module never regenerates a main arm.  It imports saved PRE/POST receipts,
creates each external baseline from OFF PRE once, publishes each baseline once,
then derives the fixed fifteen temporal observations from every decoded POST.
Receiver selection is blind; weighted edit truth is joined only in evaluation.
"""
from __future__ import annotations

import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from experiments.paper_results_v1.attack_matrix import (
    apply_edit_rgb8,
    expand_weighted_truth,
    fixed_attack_specs,
    path_error_rows,
    recipe_receipt,
)


MAIN_ARMS = (
    "OFF_NATIVE", "PAYLOAD_NATIVE", "PAYLOAD_FRAMEWISE_RECON", "PAYLOAD_FRAMEWISE_M05",
)
BASELINES = ("videoseal", "rivagan")
MODES = ("RAW", "BLIND_PATH")
KEY_LABELS = ("K0", "K1")
ATTEMPT_CASES = ("pilot_01", "pilot_02")
CONFIRMATION_CASES = tuple(f"confirm_{index:02d}" for index in range(1, 9))
PHASES = (
    "import-main", "baseline-embed-videoseal", "baseline-embed-rivagan",
    "baseline-codec", "attack-media", "receiver-clock", "receiver-read",
    "baseline-extract-videoseal", "baseline-extract-rivagan", "quality", "evaluate",
)
QUALITY_PAIRS = (
    ("PAYLOAD_NATIVE", "OFF_NATIVE"),
    ("PAYLOAD_FRAMEWISE_RECON", "PAYLOAD_NATIVE"),
    ("PAYLOAD_FRAMEWISE_RECON", "OFF_NATIVE"),
    ("PAYLOAD_FRAMEWISE_M05", "PAYLOAD_FRAMEWISE_RECON"),
    ("PAYLOAD_FRAMEWISE_M05", "OFF_NATIVE"),
    ("videoseal", "OFF_NATIVE"),
    ("rivagan", "OFF_NATIVE"),
)


class AttackEvalError(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return {"path": str(path), "bytes": path.stat().st_size}


def _observe(path):
    path = Path(path)
    try:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {"actual": digest, "status": "OBSERVED", "error": None, "blocking": False}
    except OSError as exc:
        return {
            "actual": None, "status": "OBSERVATION_UNAVAILABLE",
            "error": f"{type(exc).__name__}: {exc}", "blocking": False,
        }


def validate_config(config):
    if not isinstance(config, dict):
        raise AttackEvalError("attack config must be an object")
    required = (
        "schema_version", "study_id", "execution_scope", "source_run", "payload_bits",
        "keys", "models", "codec", "main_arms", "baselines", "attacks", "receiver",
        "quality", "cases",
    )
    missing = [field for field in required if field not in config]
    if missing:
        raise AttackEvalError(f"attack config missing {missing}")
    if config["main_arms"] != list(MAIN_ARMS) or config["baselines"] != list(BASELINES):
        raise AttackEvalError("fixed six-method roster changed")
    if config["receiver"].get("modes") != list(MODES):
        raise AttackEvalError("fixed RAW/BLIND_PATH receiver modes changed")
    if config["receiver"].get("baseline_reducers") != {
        "videoseal": "CHANNEL_J_MOD_32_REPEAT_MEAN_STRICT_GT_ZERO_NATIVE_TIE_RETAINED",
        "rivagan": "ALL_DECLARED_FRAMES_EQUAL_LOGIT_MEAN_GE_ZERO_NATIVE_TIE_RETAINED",
    }:
        raise AttackEvalError("adopted external baseline reducers changed")
    if config["execution_scope"] != {
        "attempt_cases": list(ATTEMPT_CASES), "not_executed_cases": list(CONFIRMATION_CASES),
    }:
        raise AttackEvalError("fixed two-pilot execution scope changed")
    if config["attacks"] != fixed_attack_specs():
        raise AttackEvalError("fixed fifteen-instance temporal matrix changed")
    if not isinstance(config["payload_bits"], list) or len(config["payload_bits"]) != 32 or any(
        type(bit) is not int or bit not in (0, 1) for bit in config["payload_bits"]
    ):
        raise AttackEvalError("payload_bits must contain exactly 32 integer bits")
    if config["models"]["rivagan"].get("native_message_bits") != config["payload_bits"]:
        raise AttackEvalError("RivaGAN fixed 32-bit message changed")
    if config["models"]["videoseal"].get("native_message_length") != 256 or config["models"]["videoseal"].get("native_message_bits") != config["payload_bits"] * 8:
        raise AttackEvalError("VideoSeal adopted j-mod-32 repeated message changed")
    if set(config["keys"]) != set(KEY_LABELS) or any(
        not isinstance(config["keys"][key], str) or not config["keys"][key] for key in KEY_LABELS
    ):
        raise AttackEvalError("fixed K0/K1 values are required")
    case_ids = [case.get("case_id") for case in config["cases"]]
    if case_ids != [*ATTEMPT_CASES, *CONFIRMATION_CASES]:
        raise AttackEvalError("fixed two-pilot plus eight-confirmation roster changed")
    if config["codec"] != {"codec": "libx264", "fps": 8, "crf": 18, "pix_fmt": "yuv420p"}:
        raise AttackEvalError("fixed publication codec changed")
    for case in config["cases"]:
        for field in ("case_id", "cohort", "prompt", "negative_prompt", "seed"):
            if field not in case:
                raise AttackEvalError(f"case {case.get('case_id')} missing {field}")
    return copy.deepcopy(config)


def _case(config, case_id):
    try:
        return next(case for case in config["cases"] if case["case_id"] == case_id)
    except StopIteration as exc:
        raise AttackEvalError(f"unknown case {case_id!r}") from exc


def build_plan(config):
    validate_config(config)
    attacks = [row["attack_id"] for row in config["attacks"]]
    artifacts = []
    receiver_rows = []
    baseline_rows = []
    quality_rows = []
    for case in config["cases"]:
        case_id = case["case_id"]
        attempted = case_id in ATTEMPT_CASES
        initial = "PLANNED" if attempted else "NOT_EXECUTED_BY_NOTEBOOK"
        for arm in MAIN_ARMS:
            for phase in ("PRE", "POST"):
                artifacts.append({
                    "artifact_id": f"{case_id}/{arm}/{phase}", "case_id": case_id,
                    "kind": "IMPORTED_MAIN", "status": initial,
                })
        for baseline in BASELINES:
            for phase in ("NATIVE_PRE", "NATIVE_POST"):
                artifacts.append({
                    "artifact_id": f"{case_id}/{baseline}/{phase}", "case_id": case_id,
                    "kind": "NEW_BASELINE", "status": initial,
                })
        for method in (*MAIN_ARMS, *BASELINES):
            for attack_id in attacks:
                artifacts.append({
                    "artifact_id": f"{case_id}/{method}/{attack_id}/RECEIVED",
                    "case_id": case_id, "kind": "TEMPORAL_OBSERVATION", "status": initial,
                })
        for candidate, reference in QUALITY_PAIRS:
            quality_rows.append({
                "quality_id": f"{case_id}/{candidate}_vs_{reference}", "case_id": case_id,
                "cohort": case["cohort"], "candidate": candidate, "reference": reference,
                "phase": "ORIGINAL_POST", "status": initial,
            })
        for arm in MAIN_ARMS:
            for attack_id in attacks:
                observation_id = f"{case_id}/{arm}/{attack_id}"
                for key_label in KEY_LABELS:
                    for mode in MODES:
                        receiver_rows.append({
                            "slot_id": f"{observation_id}/{key_label}/{mode}",
                            "case_id": case_id, "cohort": case["cohort"], "arm": arm,
                            "attack_id": attack_id, "key_label": key_label, "mode": mode,
                            "status": initial, "planned_bits": 32,
                        })
        for baseline in BASELINES:
            for attack_id in attacks:
                baseline_rows.append({
                    "slot_id": f"{case_id}/{baseline}/{attack_id}", "case_id": case_id,
                    "cohort": case["cohort"], "method": baseline, "attack_id": attack_id,
                    "status": initial, "planned_bits": 32,
                })
    return {
        "artifacts": artifacts, "receiver_rows": receiver_rows,
        "baseline_rows": baseline_rows, "quality_rows": quality_rows,
        "fixed_denominator": {
            "cases": 10, "attack_instances_per_method_case": 15,
            "attempt_main_receiver_rows": 480, "attempt_main_receiver_bits": 15360,
            "attempt_baseline_rows": 60, "confirmation_main_receiver_rows": 1920,
            "confirmation_baseline_rows": 240,
            "all_main_receiver_rows": 2400, "all_baseline_rows": 300,
            "attempt_quality_rows": 14, "confirmation_quality_rows": 56,
            "all_quality_rows": 70,
            "attempt_framewise_clock_encodes": 120,
            "attempt_logical_wan_reads": 480,
            "attempt_physical_wan_encode_upper_bound": 360,
            "attempt_new_codec_roundtrips": 172,
            "attempt_edited_and_received_rgb8_bytes": 29727129600,
        },
    }


class AttackRunStore:
    def __init__(self, output, config, *, create=False, temp_root=None):
        self.output = Path(output).resolve()
        self.path = self.output / "attack_run_state.json"
        self.config = validate_config(config)
        self.temp_root = Path(temp_root).resolve() if temp_root else self.output / "temporary_rgb"
        if create:
            self.output.mkdir(parents=True, exist_ok=False)
            plan = build_plan(config)
            cases = {
                case["case_id"]: {
                    "status": "PLANNED" if case["case_id"] in ATTEMPT_CASES else "NOT_EXECUTED_BY_NOTEBOOK",
                    "failures": [],
                }
                for case in config["cases"]
            }
            self.data = {
                "schema_version": config["schema_version"], "study_id": config["study_id"],
                "status": "INITIALIZED", "source_run": config["source_run"],
                "phases": {
                    phase: ({"status": "PLANNED", "failures": []} if phase == "evaluate" else {
                        "status": "PLANNED", "cases": copy.deepcopy(cases),
                    }) for phase in PHASES
                },
                "records": {}, "costs": [], **plan,
            }
            self.save()
        else:
            self.data = read_json(self.path)

    def save(self):
        atomic_json(self.path, self.data)

    def artifact(self, artifact_id):
        return next(row for row in self.data["artifacts"] if row["artifact_id"] == artifact_id)

    def phase_start(self, phase, case_id=None):
        row = self.data["phases"][phase]
        target = row if case_id is None else row["cases"][case_id]
        if target["status"] != "PLANNED":
            raise AttackEvalError(f"{phase}/{case_id} already attempted as {target['status']}")
        target.update(status="RUNNING", started_at_unix=time.time())
        row["status"] = "RUNNING"
        self.save()

    def phase_finish(self, phase, case_id=None):
        row = self.data["phases"][phase]
        target = row if case_id is None else row["cases"][case_id]
        target.update(status="COMPLETE", finished_at_unix=time.time())
        if case_id is None:
            row["status"] = "COMPLETE"
        else:
            states = [item["status"] for item in row["cases"].values()]
            row["status"] = "FAILED" if "FAILED" in states else "COMPLETE" if all(
                state in ("COMPLETE", "NOT_EXECUTED_BY_NOTEBOOK") for state in states
            ) else "PARTIAL"
        self.save()

    def phase_failure(self, phase, exc, case_id=None):
        reason = f"{type(exc).__name__}: {exc}"
        row = self.data["phases"][phase]
        target = row if case_id is None else row["cases"][case_id]
        target.update(status="FAILED", finished_at_unix=time.time())
        target.setdefault("failures", []).append(reason)
        row["status"] = "FAILED"
        if case_id is not None and not self.data.get("recovery", {}).get("retain_pending_on_interrupt"):
            for item in self.data["artifacts"]:
                if item["case_id"] == case_id and item["status"] == "PLANNED" and _phase_owns_artifact(phase, item):
                    item.update(status="FAILED", reason=reason)
            for collection in ("receiver_rows", "baseline_rows", "quality_rows"):
                for item in self.data[collection]:
                    if (
                        item["case_id"] == case_id
                        and item["status"] in ("PLANNED", "CLOCK_READY")
                        and _phase_owns(phase, collection, item)
                    ):
                        item.update(status="FAILED", reason=reason)
        self.save()
        return reason


def _phase_owns(phase, collection, row):
    if phase in ("receiver-clock", "receiver-read") and collection == "receiver_rows":
        return phase == "receiver-read" or row["mode"] == "BLIND_PATH"
    if phase == "quality" and collection == "quality_rows":
        return True
    if phase.startswith("baseline-extract-") and collection == "baseline_rows":
        return row["method"] == phase.removeprefix("baseline-extract-")
    return False


def _phase_owns_artifact(phase, row):
    artifact_id = row["artifact_id"]
    if phase == "import-main":
        return row["kind"] == "IMPORTED_MAIN"
    if phase.startswith("baseline-embed-"):
        return artifact_id.endswith(f"/{phase.removeprefix('baseline-embed-')}/NATIVE_PRE")
    if phase == "baseline-codec":
        return row["kind"] == "NEW_BASELINE" and artifact_id.endswith("/NATIVE_POST")
    if phase == "attack-media":
        return row["kind"] == "TEMPORAL_OBSERVATION"
    return False


def _timed(store, phase, case_id, function):
    store.phase_start(phase, case_id)
    started = time.perf_counter()
    try:
        value = function()
        store.data["costs"].append({
            "case_id": case_id, "phase": phase, "status": "COMPLETE",
            "seconds": time.perf_counter() - started,
        })
        store.phase_finish(phase, case_id)
        return value
    except BaseException as exc:
        store.data["costs"].append({
            "case_id": case_id, "phase": phase, "status": "FAILED",
            "seconds": time.perf_counter() - started, "reason": f"{type(exc).__name__}: {exc}",
        })
        store.phase_failure(phase, exc, case_id)
        raise


def _release():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def _load_rgb8(receipt):
    import numpy as np
    import torch

    path = Path(receipt["path"])
    shape = tuple(receipt["shape"])
    raw = path.read_bytes()
    if receipt.get("dtype") != "uint8" or len(shape) != 4 or shape[1:] != (320, 512, 3):
        raise ValueError("RGB8 receipt geometry mismatch")
    if len(raw) != math.prod(shape):
        raise ValueError("RGB8 receipt byte count mismatch")
    receipt["sha256_observation"] = _observe(path)
    return torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(shape).copy())


def _save_rgb8(path, value):
    import numpy as np

    candidate = value
    for method in ("detach", "cpu"):
        function = getattr(candidate, method, None)
        if callable(function):
            candidate = function()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    array = np.asarray(candidate)
    if array.dtype != np.uint8 or array.ndim != 4 or tuple(array.shape[1:]) != (320, 512, 3):
        raise ValueError("saved RGB8 must be uint8 [T,320,512,3]")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(array.tobytes(order="C"))
    os.replace(temporary, path)
    return {
        "path": str(path), "bytes": path.stat().st_size, "shape": list(array.shape),
        "dtype": "uint8", "sha256_observation": _observe(path),
    }


def _available_media(receipt):
    """Turn a codec RGB receipt into an AVAILABLE artifact without collision."""
    media = copy.deepcopy(receipt)
    media_status = media.pop("status", None)
    return {"status": "AVAILABLE", **media, "media_receipt_status": media_status}


def _record_path(store, case_id, *parts):
    return store.output / "records" / case_id / Path(*parts)


def phase_import_main(store, config, case_id):
    def execute():
        source_root = Path(config["source_run"]["run_dir"]).expanduser().resolve()
        source_state = read_json(source_root / config["source_run"].get("run_state", "run_state.json"))
        source_artifacts = {row["artifact_id"]: row for row in source_state.get("artifacts", [])}
        imported = {}
        for arm in MAIN_ARMS:
            for phase in ("PRE", "POST"):
                artifact_id = f"{case_id}/{arm}/{phase}"
                target = store.artifact(artifact_id)
                source = source_artifacts.get(artifact_id)
                try:
                    if not isinstance(source, dict) or source.get("status") != "AVAILABLE":
                        raise ValueError("saved main artifact receipt unavailable")
                    if source.get("shape") != [181, 320, 512, 3] or source.get("dtype") != "uint8":
                        raise ValueError("saved main artifact geometry mismatch")
                    path = Path(source["path"])
                    if not path.is_file():
                        raise FileNotFoundError(f"saved main artifact bytes unavailable: {path}")
                    receipt = {key: copy.deepcopy(value) for key, value in source.items() if key not in ("status",)}
                    receipt["source_run_state"] = str(source_root / config["source_run"].get("run_state", "run_state.json"))
                    receipt["identity_observation"] = _observe(path)
                    target.update(status="AVAILABLE", **receipt)
                    imported[artifact_id] = {"status": "AVAILABLE", "path": str(path)}
                except Exception as exc:
                    target.update(status="MISSING", reason=f"{type(exc).__name__}: {exc}")
                    imported[artifact_id] = {"status": "MISSING", "reason": target["reason"]}
                store.save()
        receipt = atomic_json(_record_path(store, case_id, "imported_main.json"), {
            "case_id": case_id, "regeneration_attempted": False,
            "source_run": config["source_run"], "artifacts": imported,
        })
        store.data["records"].setdefault(case_id, {})["imported_main"] = receipt
        store.save()

    return _timed(store, "import-main", case_id, execute)


def _native_store(root):
    def store(label, value, descriptor):
        import numpy as np

        arrays = {}
        def collect(prefix, item):
            if isinstance(item, dict):
                for key, child in item.items():
                    collect(f"{prefix}.{key}" if prefix else str(key), child)
                return
            candidate = item
            for method in ("detach", "cpu"):
                function = getattr(candidate, method, None)
                if callable(function):
                    candidate = function()
            numpy_method = getattr(candidate, "numpy", None)
            if callable(numpy_method):
                candidate = numpy_method()
            array = np.asarray(candidate)
            if array.dtype.kind not in "biuf":
                raise ValueError("native output sidecar requires numeric arrays")
            arrays[prefix or "value"] = array
        collect("", value)
        path = Path(root) / f"{label}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return {
            "lossless": True, "uri": str(path), "format": "npz",
            "sha256": _observe(path)["actual"], "bytes": path.stat().st_size,
            "arrays": {key: {"shape": list(array.shape), "dtype": str(array.dtype)} for key, array in arrays.items()},
            "descriptor": descriptor,
        }
    return store


def _adapter(config, method, sidecar_root):
    from experiments.paper_results_v1.real_backends import load_rivagan_adapter, load_videoseal_adapter
    if method == "videoseal":
        return load_videoseal_adapter(config["models"][method], native_output_store=_native_store(sidecar_root))
    if method == "rivagan":
        return load_rivagan_adapter(config["models"][method])
    raise ValueError(method)


def phase_baseline_embed(store, config, case_id, method):
    phase = f"baseline-embed-{method}"
    def execute():
        target = store.artifact(f"{case_id}/{method}/NATIVE_PRE")
        source = store.artifact(f"{case_id}/OFF_NATIVE/PRE")
        adapter = None
        try:
            if source["status"] != "AVAILABLE":
                raise ValueError(f"OFF PRE is {source['status']}")
            rgb = _load_rgb8(source)
            adapter = _adapter(config, method, _record_path(store, case_id, method, "embed_sidecars"))
            bits = config["models"][method]["native_message_bits"]
            media_input = rgb.numpy() if method == "rivagan" else rgb
            embedded, receipt = adapter.embed(media_input, bits)
            media = _save_rgb8(store.temp_root / case_id / method / "native_pre.rgb8", embedded)
            record = atomic_json(_record_path(store, case_id, method, "embed.json"), receipt)
            target.update(status="AVAILABLE", **media, embed_record=record)
            store.save()
        except Exception as exc:
            target.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
            store.save()
            raise
        finally:
            adapter = None
            _release()
    return _timed(store, phase, case_id, execute)


def phase_baseline_codec(store, config, case_id):
    def execute():
        from runtime.wan.variable_rgb_media import roundtrip
        for method in BASELINES:
            target = store.artifact(f"{case_id}/{method}/NATIVE_POST")
            source = store.artifact(f"{case_id}/{method}/NATIVE_PRE")
            try:
                if source["status"] != "AVAILABLE":
                    raise ValueError(f"{method} NATIVE_PRE is {source['status']}")
                rgb = _load_rgb8(source)
                received, receipt = roundtrip(
                    rgb, store.temp_root / case_id / method / "native_codec",
                    media_output=store.output / "media" / case_id / method / "native_codec",
                )
                media = _available_media(receipt["received_rgb"])
                record = atomic_json(_record_path(store, case_id, method, "codec.json"), receipt)
                target.update(**media, codec_record=record)
            except Exception as exc:
                target.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
            store.save()
    return _timed(store, "baseline-codec", case_id, execute)


def _post_artifact(store, case_id, method):
    phase = "POST" if method in MAIN_ARMS else "NATIVE_POST"
    return store.artifact(f"{case_id}/{method}/{phase}")


def phase_attack_media(store, config, case_id):
    def execute():
        from runtime.wan.variable_rgb_media import roundtrip
        for method in (*MAIN_ARMS, *BASELINES):
            source = _post_artifact(store, case_id, method)
            for spec in config["attacks"]:
                attack_id = spec["attack_id"]
                target = store.artifact(f"{case_id}/{method}/{attack_id}/RECEIVED")
                try:
                    if source["status"] != "AVAILABLE":
                        target.update(
                            status="MISSING" if source["status"] == "MISSING" else "FAILED",
                            reason=f"source POST is {source['status']}",
                        )
                        store.save()
                        continue
                    if spec["family"] == "FULL":
                        target.update(
                            status="AVAILABLE", path=source["path"], shape=source["shape"],
                            dtype="uint8", bytes=source["bytes"], full_alias_of=source["artifact_id"],
                            publication_count_from_pre=1, edit_publication_added=False,
                        )
                        receipt = recipe_receipt(spec)
                    else:
                        full = _load_rgb8(source)
                        edited, receipt = apply_edit_rgb8(full, spec)
                        received, codec = roundtrip(
                            edited, store.temp_root / case_id / method / attack_id,
                            media_output=store.output / "media" / case_id / method / attack_id,
                        )
                        media = _available_media(codec["received_rgb"])
                        target.update(
                            **media, publication_count_from_pre=2,
                            edit_publication_added=True,
                            codec_record=atomic_json(_record_path(store, case_id, method, attack_id, "codec.json"), codec),
                            edited_rgb_path=codec["edited_rgb"]["path"],
                        )
                    truth_record = atomic_json(
                        _record_path(store, case_id, method, attack_id, "weighted_truth.json"), receipt,
                    )
                    target.update(
                        weighted_truth_record=truth_record, weighted_truth_sha256=receipt["weighted_truth_sha256"],
                        output_frames=receipt["output_frames"], truth_hidden_from_receiver=True,
                    )
                except Exception as exc:
                    target.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
                store.save()
    return _timed(store, "attack-media", case_id, execute)


def _receiver_rows(store, case_id, *, arm=None, attack_id=None, key_label=None, mode=None):
    result = []
    for row in store.data["receiver_rows"]:
        if row["case_id"] != case_id:
            continue
        requested = {"arm": arm, "attack_id": attack_id, "key_label": key_label, "mode": mode}
        if all(value is None or row[field] == value for field, value in requested.items()):
            result.append(row)
    return result


def _clock_record_path(store, case_id, arm, attack_id, key_label):
    return _record_path(store, case_id, "clock", arm, attack_id, key_label)


def _persist_actual_calls(store, case_id, field, calls):
    """Persist started calls before, and outcomes immediately after, execution."""
    records = copy.deepcopy(calls)
    summary = {
        "attempted": len(records),
        "completed": sum(row["status"] == "COMPLETE" for row in records),
        "failed": sum(row["status"] == "FAILED" for row in records),
        "unfinished": sum(row["status"] == "RUNNING" for row in records),
        "records": records,
    }
    store.data["records"].setdefault(case_id, {})[field] = summary
    store.save()
    atomic_json(_record_path(store, case_id, field + ".json"), {
        "case_id": case_id, **summary,
    })
    return summary


def _raw_operation(frames):
    output_frames = ((frames - 1) // 4) * 4 + 1
    support = min(44, (output_frames - 1) // 4)
    return {
        "status": "SUPPORTED" if support >= 1 else "UNSUPPORTED",
        "reason": None if support >= 1 else "INSUFFICIENT_VISIBLE_WAN_SUPPORT",
        "output_frames": output_frames,
        "received_index_map": list(range(output_frames)),
        "source_coordinate_map": [None] * output_frames,
        "received_coordinate_map": list(range(output_frames)),
        "tail_discarded_received_frames": frames - output_frames,
        "wan_support": support,
        "coordinate_semantics": "RAW receiver-local coordinate; no source position estimated",
        "truth_inputs": False,
    }


def phase_receiver_clock(store, config, case_id):
    def execute():
        from experiments.paper_results_v1.real_backends import load_local_framewise_backend
        from main.tube_state import video_trajectory_temporal_edit_receiver_v1 as method

        backend = None
        try:
            # RAW is independent of the framewise clock and remains readable if
            # clock model setup or either key's DP fails.
            for arm in MAIN_ARMS:
                for spec in config["attacks"]:
                    source = store.artifact(f"{case_id}/{arm}/{spec['attack_id']}/RECEIVED")
                    for row in _receiver_rows(
                        store, case_id, arm=arm, attack_id=spec["attack_id"], mode="RAW",
                    ):
                        if row["status"] not in ("PLANNED", "RECOVERY_PENDING"):
                            continue
                        if source["status"] == "AVAILABLE":
                            row.update(
                                status="CLOCK_READY", reason=None,
                                operation=_raw_operation(int(source["shape"][0])),
                                clock_status="NOT_APPLICABLE_RAW", truth_inputs=False,
                            )
                        else:
                            row.update(
                                status="MISSING" if source["status"] == "MISSING" else "FAILED",
                                reason=f"received artifact is {source['status']}",
                            )
                    store.save()
            clock_calls = copy.deepcopy(
                store.data.get("records", {}).get(case_id, {}).get("framewise_clock_encodes", {}).get("records", [])
            )
            _persist_actual_calls(store, case_id, "framewise_clock_encodes", clock_calls)
            prior_clock_ids = {item.get("call_id") for item in clock_calls}
            for row in _receiver_rows(store, case_id, mode="BLIND_PATH"):
                call_id = f"{case_id}/{row['arm']}/{row['attack_id']}"
                if row["status"] in ("PLANNED", "RECOVERY_PENDING") and call_id in prior_clock_ids:
                    row.update(
                        status="FAILED", missing_bits=32,
                        reason="PRIOR_FRAMEWISE_ENCODE_ATTEMPT_RETAINED_NO_REPEAT",
                    )
            store.save()
            if not any(
                row["mode"] == "BLIND_PATH" and row["status"] in ("PLANNED", "RECOVERY_PENDING")
                for row in _receiver_rows(store, case_id)
            ):
                return
            backend = load_local_framewise_backend(config["models"]["framewise"])
            import numpy as np
            for arm in MAIN_ARMS:
                for spec in config["attacks"]:
                    attack_id = spec["attack_id"]
                    source = store.artifact(f"{case_id}/{arm}/{attack_id}/RECEIVED")
                    rows = _receiver_rows(store, case_id, arm=arm, attack_id=attack_id)
                    pending_blind = [
                        row for row in rows
                        if row["mode"] == "BLIND_PATH" and row["status"] in ("PLANNED", "RECOVERY_PENDING")
                    ]
                    if not pending_blind:
                        continue
                    try:
                        if source["status"] != "AVAILABLE":
                            status = "MISSING" if source["status"] == "MISSING" else "FAILED"
                            for row in pending_blind:
                                row.update(status=status, reason=f"received artifact is {source['status']}")
                            store.save()
                            continue
                        call_id = f"{case_id}/{arm}/{attack_id}"
                        previous = next((item for item in clock_calls if item.get("call_id") == call_id), None)
                        if previous is not None:
                            for row in pending_blind:
                                row.update(
                                    status="FAILED", missing_bits=32,
                                    reason="PRIOR_FRAMEWISE_ENCODE_ATTEMPT_RETAINED_NO_REPEAT",
                                )
                            store.save()
                            continue
                        rgb = _load_rgb8(source)
                        call = {
                            "call_id": call_id,
                            "artifact_id": source["artifact_id"], "status": "RUNNING",
                            "started_at_unix": time.time(),
                        }
                        clock_calls.append(call)
                        _persist_actual_calls(store, case_id, "framewise_clock_encodes", clock_calls)
                        try:
                            latent = backend.encode(rgb)
                        except Exception as exc:
                            call.update(
                                status="FAILED", finished_at_unix=time.time(),
                                reason=f"{type(exc).__name__}: {exc}",
                            )
                            _persist_actual_calls(store, case_id, "framewise_clock_encodes", clock_calls)
                            raise
                        call.update(status="COMPLETE", finished_at_unix=time.time())
                        _persist_actual_calls(store, case_id, "framewise_clock_encodes", clock_calls)
                        if int(latent.shape[0]) != int(source["shape"][0]):
                            raise ValueError("framewise encoder changed temporal length")
                        for key_label in KEY_LABELS:
                            key_rows = _receiver_rows(
                                store, case_id, arm=arm, attack_id=attack_id,
                                key_label=key_label, mode="BLIND_PATH",
                            )
                            key_rows = [
                                row for row in key_rows
                                if row["status"] in ("PLANNED", "RECOVERY_PENDING")
                            ]
                            if not key_rows:
                                continue
                            try:
                                root = _clock_record_path(store, case_id, arm, attack_id, key_label)
                                evidence = method.score_framewise(latent, config["keys"][key_label], method.PUBLIC)
                                estimate = method.solve_monotone(
                                    evidence["signed_projection"], evidence["rho"], method.PUBLIC,
                                )
                                operation = (
                                    method.decode_visible_span(estimate["path"], method.PUBLIC)
                                    if estimate["status"] == "ESTIMATED" else None
                                )
                                root.parent.mkdir(parents=True, exist_ok=True)
                                np.savez_compressed(
                                    root.with_suffix(".npz"),
                                    signed_projection=evidence["signed_projection"], rho=evidence["rho"],
                                )
                                estimate_record = atomic_json(root.with_suffix(".json"), {
                                    "case_id": case_id, "arm": arm, "attack_id": attack_id,
                                    "key_label": key_label, "status": estimate["status"],
                                    "estimate": estimate, "operation": operation,
                                    "evidence": {
                                        "path": str(root.with_suffix(".npz")),
                                        "shape": [evidence["received_frames"], evidence["source_frames"]],
                                        "arrays": ["signed_projection", "rho"], "truth_inputs": False,
                                    },
                                })
                                for row in key_rows:
                                    selected = operation
                                    row_status = "CLOCK_READY" if operation is not None and operation["status"] == "SUPPORTED" else (
                                        "UNSUPPORTED" if operation is not None else "UNRESOLVED"
                                    )
                                    row.update(
                                        status=row_status,
                                        reason=(None if row_status == "CLOCK_READY" else (
                                            selected.get("reason") if selected else estimate["reason"]
                                        )),
                                        clock_record=estimate_record,
                                        clock_status=estimate["status"],
                                        clock_score_gap=estimate.get("score_gap"),
                                        operation=selected,
                                        truth_inputs=False,
                                    )
                                store.save()
                            except Exception as exc:
                                reason = f"{type(exc).__name__}: {exc}"
                                for row in key_rows:
                                    if row["status"] in ("PLANNED", "RECOVERY_PENDING"):
                                        row.update(status="FAILED", reason=reason)
                                store.save()
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        for row in rows:
                            if row["status"] in ("PLANNED", "RECOVERY_PENDING"):
                                row.update(status="FAILED", reason=reason)
                        store.save()
        finally:
            if backend is not None:
                backend.close()
            _release()
    return _timed(store, "receiver-clock", case_id, execute)


def _read_record_path(store, row):
    return _record_path(
        store, row["case_id"], "reads", row["arm"], row["attack_id"],
        row["key_label"], row["mode"],
    )


def phase_receiver_read(store, config, case_id):
    def execute():
        import numpy as np
        from runtime.wan.generation import load_frozen_vae
        from runtime.wan.vae import reencode_rgb24_readback
        from runtime.wan.video_trajectory_temporal_edit_receiver_v1 import operate_map, read_payload_general

        vae = None
        cache = {}
        physical = copy.deepcopy(
            store.data.get("records", {}).get(case_id, {}).get("physical_wan_encodes", {}).get("records", [])
        )
        try:
            _persist_actual_calls(store, case_id, "physical_wan_encodes", physical)
            prior_physical_ids = {item.get("physical_id") for item in physical}
            for row in _receiver_rows(store, case_id):
                if row["status"] != "CLOCK_READY":
                    continue
                source = store.artifact(f"{case_id}/{row['arm']}/{row['attack_id']}/RECEIVED")
                cache_key = (source["artifact_id"], tuple(row["operation"]["received_index_map"]))
                if hashlib.sha256(repr(cache_key).encode()).hexdigest() in prior_physical_ids:
                    row.update(
                        status="FAILED", missing_bits=32,
                        reason="PRIOR_PHYSICAL_WAN_ENCODE_ATTEMPT_RETAINED_NO_REPEAT",
                    )
            store.save()
            if not any(row["status"] == "CLOCK_READY" for row in _receiver_rows(store, case_id)):
                return
            vae = load_frozen_vae(
                {"model": {"id": config["models"]["wan"]["local_snapshot_path"]}},
                device=config["models"]["wan"].get("device", "cuda"),
            )
            current_artifact = None
            for row in _receiver_rows(store, case_id):
                if row["status"] != "CLOCK_READY":
                    continue
                try:
                    source = store.artifact(f"{case_id}/{row['arm']}/{row['attack_id']}/RECEIVED")
                    operation = row["operation"]
                    map_key = tuple(operation["received_index_map"])
                    if current_artifact != source["artifact_id"]:
                        cache.clear()
                        current_artifact = source["artifact_id"]
                    cache_key = (source["artifact_id"], map_key)
                    if cache_key not in cache:
                        physical_id = hashlib.sha256(repr(cache_key).encode()).hexdigest()
                        previous = next((item for item in physical if item.get("physical_id") == physical_id), None)
                        if previous is not None:
                            raise RuntimeError("PRIOR_PHYSICAL_WAN_ENCODE_ATTEMPT_RETAINED_NO_REPEAT")
                        try:
                            received = _load_rgb8(source)
                            corrected = operate_map(received, list(map_key)).float().div(255.0)
                            call = {
                                "call_id": physical_id,
                                "physical_id": physical_id,
                                "artifact_id": source["artifact_id"],
                                "received_index_map": list(map_key),
                                "output_frames": operation["output_frames"],
                                "status": "RUNNING", "started_at_unix": time.time(),
                            }
                            physical.append(call)
                            _persist_actual_calls(store, case_id, "physical_wan_encodes", physical)
                            try:
                                encoded = reencode_rgb24_readback(vae, corrected)
                            except Exception as exc:
                                call.update(
                                    status="FAILED", finished_at_unix=time.time(),
                                    reason=f"{type(exc).__name__}: {exc}",
                                )
                                _persist_actual_calls(store, case_id, "physical_wan_encodes", physical)
                                raise
                            call.update(status="COMPLETE", finished_at_unix=time.time())
                            _persist_actual_calls(store, case_id, "physical_wan_encodes", physical)
                            normalized = encoded.detach().cpu()
                            cache[cache_key] = ("READY", normalized)
                        except Exception as exc:
                            cache[cache_key] = ("FAILED", f"{type(exc).__name__}: {exc}")
                    if cache[cache_key][0] != "READY":
                        raise RuntimeError(cache[cache_key][1])
                    result = read_payload_general(
                        cache[cache_key][1], config["keys"][row["key_label"]],
                        operation["output_frames"], operation["source_coordinate_map"],
                    )
                    root = _read_record_path(store, row)
                    root.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        root.with_suffix(".npz"), signed_votes=result.pop("signed_votes"),
                        zero_mask=result.pop("zero_mask"),
                    )
                    detail = atomic_json(root.with_suffix(".json"), {
                        **result,
                        "vote_sidecar": {
                            "path": str(root.with_suffix(".npz")),
                            "arrays": ["signed_votes", "zero_mask"],
                            "coordinate_order": "[payload_channel,receiver_latent,frequency_coordinate]",
                        },
                    })
                    row.update(
                        status="READ", reason=None, decoded_bits=result["decoded_bits"],
                        missing_bits=0, votes=result["votes"], bit_rows=result["bit_rows"],
                        detail_record=detail, R=result["R"], original_reader_match=result["original_reader_match"],
                    )
                except Exception as exc:
                    row.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}", missing_bits=32)
                store.save()
            _persist_actual_calls(store, case_id, "physical_wan_encodes", physical)
        finally:
            vae = None
            cache.clear()
            _release()
    return _timed(store, "receiver-read", case_id, execute)


def phase_baseline_extract(store, config, case_id, method):
    phase = f"baseline-extract-{method}"
    def execute():
        adapter = None
        try:
            calls = copy.deepcopy(
                store.data.get("records", {}).get(case_id, {}).get("baseline_extract_calls", {}).get("records", [])
            )
            _persist_actual_calls(store, case_id, "baseline_extract_calls", calls)
            prior_call_ids = {item.get("call_id") for item in calls}
            for row in store.data["baseline_rows"]:
                call_id = f"{case_id}/{method}/{row['attack_id']}"
                if (
                    row["case_id"] == case_id and row["method"] == method
                    and row["status"] in ("PLANNED", "RECOVERY_PENDING")
                    and call_id in prior_call_ids
                ):
                    row.update(
                        status="FAILED", missing_bits=32,
                        reason="PRIOR_BASELINE_EXTRACT_ATTEMPT_RETAINED_NO_REPEAT",
                    )
            store.save()
            pending = [
                row for row in store.data["baseline_rows"]
                if row["case_id"] == case_id and row["method"] == method
                and row["status"] in ("PLANNED", "RECOVERY_PENDING")
            ]
            if not pending:
                return
            adapter = _adapter(config, method, _record_path(store, case_id, method, "extract_sidecars"))
            for spec in config["attacks"]:
                attack_id = spec["attack_id"]
                row = next(item for item in store.data["baseline_rows"] if item["slot_id"] == f"{case_id}/{method}/{attack_id}")
                if row["status"] not in ("PLANNED", "RECOVERY_PENDING"):
                    continue
                source = store.artifact(f"{case_id}/{method}/{attack_id}/RECEIVED")
                try:
                    if source["status"] != "AVAILABLE":
                        raise ValueError(f"received baseline artifact is {source['status']}")
                    if method == "videoseal":
                        adapter.native_output_store = _native_store(
                            _record_path(store, case_id, method, attack_id, "sidecars"),
                        )
                    media_input = _load_rgb8(source)
                    if method == "rivagan":
                        media_input = media_input.numpy()
                    call_id = f"{case_id}/{method}/{attack_id}"
                    previous = next((item for item in calls if item.get("call_id") == call_id), None)
                    if previous is not None:
                        raise RuntimeError("PRIOR_BASELINE_EXTRACT_ATTEMPT_RETAINED_NO_REPEAT")
                    call = {"call_id": call_id, "status": "RUNNING", "started_at_unix": time.time()}
                    calls.append(call); _persist_actual_calls(store, case_id, "baseline_extract_calls", calls)
                    try:
                        record = adapter.extract(media_input)
                        call.update(status="COMPLETE", finished_at_unix=time.time())
                        _persist_actual_calls(store, case_id, "baseline_extract_calls", calls)
                    except BaseException as exc:
                        if isinstance(exc, Exception):
                            call.update(status="FAILED", finished_at_unix=time.time(), reason=f"{type(exc).__name__}: {exc}")
                            _persist_actual_calls(store, case_id, "baseline_extract_calls", calls)
                        raise
                    receipt = atomic_json(_record_path(store, case_id, method, attack_id, "extract.json"), record)
                    row.update(status="EXTRACTED", reason=None, extract_record=receipt, expected_frames=source["shape"][0])
                except Exception as exc:
                    row.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}", missing_bits=32)
                store.save()
        finally:
            adapter = None
            _release()
    return _timed(store, phase, case_id, execute)


def _psnr(reference, candidate):
    import numpy as np
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("quality pair RGB shape mismatch")
    mse = float(np.mean((left - right) ** 2, dtype=np.float64))
    return {
        "mse_rgb8": mse, "rmse_rgb8": math.sqrt(mse),
        "psnr_db": None if mse == 0.0 else 10.0 * math.log10((255.0 ** 2) / mse),
        "identical": mse == 0.0,
    }


def _lpips(reference, candidate):
    import torch
    import lpips
    model = lpips.LPIPS(net="alex").eval().to("cuda")
    values = []
    try:
        with torch.inference_mode():
            for start in range(0, len(reference), 8):
                left = reference[start:start + 8].permute(0, 3, 1, 2).to("cuda", torch.float32).div(127.5).sub(1.0)
                right = candidate[start:start + 8].permute(0, 3, 1, 2).to("cuda", torch.float32).div(127.5).sub(1.0)
                values.extend(float(item) for item in model(left, right).reshape(-1).cpu().tolist())
    finally:
        model = None
        _release()
    return {"mean": sum(values) / len(values), "frames": len(values), "values": values}


def _reference_flow_fluctuation(reference, candidate, segment_frames=8):
    """Backward reference flow (current->previous), sampled on current grid."""
    import cv2
    import numpy as np
    ref = np.asarray(reference, np.uint8)
    cand = np.asarray(candidate, np.uint8)
    if ref.shape != cand.shape or ref.ndim != 4 or ref.shape[-1] != 3:
        raise ValueError("reference-flow inputs must be shape-matched RGB video")
    residual = cand.astype(np.float32) - ref.astype(np.float32)
    h, w = ref.shape[1:3]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    rows = []
    for current in range(1, len(ref)):
        prior_gray = cv2.cvtColor(ref[current - 1], cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(ref[current], cv2.COLOR_RGB2GRAY)
        backward = cv2.calcOpticalFlowFarneback(
            current_gray, prior_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        map_x, map_y = grid_x + backward[..., 0], grid_y + backward[..., 1]
        valid = np.isfinite(map_x) & np.isfinite(map_y) & (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
        warped = cv2.remap(residual[current - 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        delta = np.abs(residual[current] - warped).mean(axis=2)
        rows.append({
            "transition": [current - 1, current], "coverage": float(valid.mean()),
            "mean_absolute_residual_delta": float(delta[valid].mean()) if bool(valid.any()) else None,
        })
    finite = [row["mean_absolute_residual_delta"] for row in rows if row["mean_absolute_residual_delta"] is not None]
    windows = []
    transitions = segment_frames - 1
    for start in range(max(0, len(rows) - transitions + 1)):
        window = rows[start:start + transitions]
        values = [row["mean_absolute_residual_delta"] for row in window]
        if len(values) == transitions and all(value is not None for value in values):
            windows.append({"start_frame": start, "end_frame": start + segment_frames - 1, "mean": sum(values) / len(values)})
    worst = max(windows, key=lambda row: row["mean"], default=None)
    return {
        "status": "EVALUATED" if finite else "NO_VALID_COVERAGE",
        "flow_direction": "backward reference flow current->previous; sample prior residual at x+B on current grid",
        "occlusion_mask": False, "transitions": rows,
        "mean": sum(finite) / len(finite) if finite else None,
        "worst_segment_frames": segment_frames, "worst_segment_transitions": transitions,
        "worst_segment": worst,
    }


def _side_by_side_mp4(reference, candidate, output):
    import numpy as np
    left = np.asarray(reference, np.uint8); right = np.asarray(candidate, np.uint8)
    if left.shape != right.shape or left.ndim != 4 or tuple(left.shape[1:]) != (320, 512, 3):
        raise ValueError("side-by-side inputs must be matched uint8 [T,320,512,3]")
    frames = int(left.shape[0]); raster = np.concatenate([left, right], axis=2)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-v", "error", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1024x320", "-r", "8", "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-crf", "18", "-pix_fmt", "yuv420p", "-frames:v", str(frames), "-n", str(output),
    ]
    child = subprocess.run(command, input=raster.tobytes(order="C"), capture_output=True, check=False)
    output.with_suffix(".stderr.txt").write_bytes(child.stderr)
    if child.returncode:
        raise RuntimeError(f"side-by-side ffmpeg exited {child.returncode}")
    return {"status": "SAVED", "path": str(output), "bytes": output.stat().st_size, "frames": frames, "command": command, "identity_observation": _observe(output)}


def phase_quality(store, config, case_id):
    def execute():
        for row in (item for item in store.data["quality_rows"] if item["case_id"] == case_id):
            if row["status"] not in ("PLANNED", "RECOVERY_PENDING"):
                continue
            row.update(status="RECOVERY_RUNNING", attempt_started_at_unix=time.time())
            store.save()
            try:
                reference = _load_rgb8(_post_artifact(store, case_id, row["reference"]))
                candidate = _load_rgb8(_post_artifact(store, case_id, row["candidate"]))
                metrics = {}
                for name, function in (
                    ("PSNR_RGB", lambda: _psnr(reference, candidate)),
                    ("LPIPS", lambda: _lpips(reference, candidate)),
                    ("REFERENCE_FLOW_RESIDUAL_WARP_FLUCTUATION", lambda: _reference_flow_fluctuation(
                        reference, candidate, config["quality"]["worst_segment_frames"],
                    )),
                ):
                    try:
                        value = function()
                        status = value.get("status", "EVALUATED") if isinstance(value, dict) else "EVALUATED"
                        metrics[name] = {"status": status, "value": value}
                    except Exception as exc:
                        metrics[name] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
                row.update(
                    status="EVALUATED" if all(item["status"] == "EVALUATED" for item in metrics.values()) else "PARTIAL",
                    metrics=metrics,
                )
                if (row["candidate"], row["reference"]) in (
                    ("PAYLOAD_FRAMEWISE_M05", "PAYLOAD_FRAMEWISE_RECON"),
                    ("PAYLOAD_FRAMEWISE_M05", "OFF_NATIVE"),
                ):
                    try:
                        row["side_by_side"] = _side_by_side_mp4(
                            reference, candidate,
                            store.output / "quality_media" / case_id / (row["candidate"] + "_vs_" + row["reference"] + ".mp4"),
                        )
                    except Exception as exc:
                        row["side_by_side"] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
                        row["status"] = "PARTIAL"
                row["record"] = atomic_json(_record_path(store, case_id, "quality", row["quality_id"].split("/", 1)[1] + ".json"), row)
            except Exception as exc:
                row.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
            store.save()
    return _timed(store, "quality", case_id, execute)


def _expected_for_key(config, key_label):
    # Wrong-key remains a negative control but is compared against the same fixed payload.
    return config["payload_bits"]


def _condition_group(attack_id):
    if attack_id.startswith("delete_"): return "DELETE_10_PERCENT"
    if attack_id.startswith("repeat_"): return "REPEAT_INSERT_10_PERCENT"
    if attack_id.startswith("interp_"): return "ADJACENT_EQUAL_INSERT_10_PERCENT"
    return attack_id.upper()


def _receiver_group_summary(rows):
    planned = len(rows); scorable = [row for row in rows if row.get("status") == "EVALUATED"]
    not_executed = planned > 0 and all(row.get("status") == "NOT_EXECUTED_BY_NOTEBOOK" for row in rows)
    return {
        "fixed_rows": planned, "fixed_bits": planned * 32,
        "scorable_rows": len(scorable), "scorable_bits": len(scorable) * 32,
        "missing_bits": (planned - len(scorable)) * 32,
        "bit_errors": sum(row["bit_errors"] for row in scorable),
        "exact_recoveries": sum(bool(row["exact_recovery"]) for row in scorable),
        "fixed_denominator_exact_recovery_rate": (
            None if not_executed or not planned else sum(bool(row["exact_recovery"]) for row in scorable) / planned
        ),
        "scorable_ber": (
            sum(row["bit_errors"] for row in scorable) / (len(scorable) * 32)
            if scorable else None
        ),
        "status_counts": _counts(row["status"] for row in rows),
    }


def _baseline_group_summary(rows):
    planned = len(rows); scorable = [row for row in rows if row.get("status") == "EVALUATED"]
    not_executed = planned > 0 and all(row.get("status") == "NOT_EXECUTED_BY_NOTEBOOK" for row in rows)
    errors = sum(row["bit_errors"] for row in scorable)
    return {
        "fixed_rows": planned, "fixed_bits": planned * 32,
        "scorable_rows": len(scorable), "scorable_bits": len(scorable) * 32,
        "missing_bits": (planned - len(scorable)) * 32,
        "bit_errors": errors,
        "exact_recoveries": sum(bool(row.get("exact_recovery")) for row in scorable),
        "fixed_denominator_exact_recovery_rate": (
            None if not_executed or not planned
            else sum(bool(row.get("exact_recovery")) for row in scorable) / planned
        ),
        "scorable_ber": errors / (len(scorable) * 32) if scorable else None,
        "status_counts": _counts(row["status"] for row in rows),
    }


def _paired_rows(receiver_rows):
    index = {
        (row["case_id"], row["arm"], row["attack_id"], row["key_label"], row["mode"]): row
        for row in receiver_rows
    }
    result = []
    for case_id in (*ATTEMPT_CASES, *CONFIRMATION_CASES):
        for attack_id in [row["attack_id"] for row in fixed_attack_specs()]:
            for key_label in KEY_LABELS:
                for mode in MODES:
                    m05 = index[(case_id, "PAYLOAD_FRAMEWISE_M05", attack_id, key_label, mode)]
                    for comparator in ("PAYLOAD_NATIVE", "PAYLOAD_FRAMEWISE_RECON"):
                        other = index[(case_id, comparator, attack_id, key_label, mode)]
                        evaluable = m05.get("status") == other.get("status") == "EVALUATED"
                        result.append({
                            "pair_id": f"{case_id}/{attack_id}/{key_label}/{mode}/M05_vs_{comparator}",
                            "case_id": case_id, "attack_id": attack_id,
                            "condition_group": _condition_group(attack_id),
                            "key_label": key_label, "mode": mode,
                            "candidate": "PAYLOAD_FRAMEWISE_M05", "comparator": comparator,
                            "status": (
                                "NOT_EXECUTED_BY_NOTEBOOK" if case_id in CONFIRMATION_CASES
                                else "PAIRED" if evaluable else "UNEVALUABLE_PAIR"
                            ),
                            "m05_bit_errors": m05.get("bit_errors"),
                            "comparator_bit_errors": other.get("bit_errors"),
                            "bit_error_gain_comparator_minus_m05": (
                                other["bit_errors"] - m05["bit_errors"] if evaluable else None
                            ),
                            "m05_status": m05.get("status"), "comparator_status": other.get("status"),
                        })
    return result


def _evaluate_baseline(config, row):
    from experiments.paper_results_v1.real_eval import (
        RIVAGAN_NATIVE_TIE_RULE, VIDEOSEAL_NATIVE_TIE_RULE,
        _rivagan_sequence_result, _videoseal_32_result,
    )
    record = read_json(row["extract_record"]["path"])
    expected = config["payload_bits"]
    if row["method"] == "videoseal":
        return _videoseal_32_result(record, expected, 256, row["expected_frames"], VIDEOSEAL_NATIVE_TIE_RULE)
    return _rivagan_sequence_result(record, expected, row["expected_frames"], RIVAGAN_NATIVE_TIE_RULE)


def phase_evaluate(store, config):
    store.phase_start("evaluate")
    try:
        attacks = {row["attack_id"]: row for row in config["attacks"]}
        for row in store.data["receiver_rows"]:
            if row["case_id"] not in ATTEMPT_CASES:
                continue
            if row["status"] == "READ":
                expected = _expected_for_key(config, row["key_label"])
                errors = sum(left != right for left, right in zip(row["decoded_bits"], expected))
                row.update(status="EVALUATED", bit_errors=errors, exact_recovery=errors == 0)
                if row["mode"] == "BLIND_PATH" and row.get("clock_status") == "ESTIMATED":
                    clock = read_json(row["clock_record"]["path"])
                    truth = expand_weighted_truth(attacks[row["attack_id"]])
                    row["path_error"] = path_error_rows(clock["estimate"]["best_path"], truth)
                    row["path_error"].update(
                        score_gap=clock["estimate"].get("score_gap"),
                        exact_tie=clock["estimate"].get("exact_tie"),
                        runner_up_is_distinct=(
                            clock["estimate"].get("runner_up_path") is None
                            or clock["estimate"].get("runner_up_path") != clock["estimate"].get("best_path")
                        ),
                    )
            elif row["status"] in ("UNRESOLVED", "UNSUPPORTED"):
                row.setdefault("missing_bits", 32)
            # Clock/path accuracy is post-hoc and remains evaluable when the
            # downstream Wan read failed.
            if (
                row["mode"] == "BLIND_PATH" and row.get("clock_status") == "ESTIMATED"
                and "path_error" not in row
            ):
                clock = read_json(row["clock_record"]["path"])
                truth = expand_weighted_truth(attacks[row["attack_id"]])
                row["path_error"] = path_error_rows(clock["estimate"]["best_path"], truth)
                row["path_error"].update(
                    score_gap=clock["estimate"].get("score_gap"), exact_tie=clock["estimate"].get("exact_tie"),
                    runner_up_is_distinct=clock["estimate"].get("runner_up_path") != clock["estimate"].get("best_path"),
                )
        for row in store.data["baseline_rows"]:
            if row["case_id"] not in ATTEMPT_CASES:
                continue
            if row["status"] == "EXTRACTED":
                try:
                    result = _evaluate_baseline(config, row)
                    row.update(**result, reason=result.get("reason"))
                    row.setdefault("missing_bits", 0 if result.get("status") == "EVALUATED" else 32)
                except Exception as exc:
                    row.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}", missing_bits=32)
        summaries = {}
        for case_id in (*ATTEMPT_CASES, *CONFIRMATION_CASES):
            summaries[case_id] = {
                "case_id": case_id,
                "execution_scope": "ATTEMPTED" if case_id in ATTEMPT_CASES else "NOT_EXECUTED_BY_NOTEBOOK",
                "main_status_counts": _counts(row["status"] for row in _receiver_rows(store, case_id)),
                "baseline_status_counts": _counts(row["status"] for row in store.data["baseline_rows"] if row["case_id"] == case_id),
                "main_fixed_summary": _receiver_group_summary(_receiver_rows(store, case_id)),
            }
        grouped = {}
        for execution_scope, case_ids in (
            ("PILOT_ATTEMPT", ATTEMPT_CASES), ("CONFIRMATION_NOT_EXECUTED", CONFIRMATION_CASES),
        ):
            for arm in MAIN_ARMS:
                for key_label in KEY_LABELS:
                    for mode in MODES:
                        for condition in (
                            "FULL", "CROP37_126", "SPEED075", "SPEED125", "MEAN3", "MEAN5",
                            "DELETE_10_PERCENT", "REPEAT_INSERT_10_PERCENT", "ADJACENT_EQUAL_INSERT_10_PERCENT",
                        ):
                            rows = [
                                row for row in store.data["receiver_rows"]
                                if row["case_id"] in case_ids and row["arm"] == arm
                                and row["key_label"] == key_label and row["mode"] == mode
                                and _condition_group(row["attack_id"]) == condition
                            ]
                            grouped[f"{execution_scope}|{arm}|{key_label}|{mode}|{condition}"] = _receiver_group_summary(rows)
        paired = _paired_rows(store.data["receiver_rows"])
        baseline_grouped = {}
        for execution_scope, case_ids in (
            ("PILOT_ATTEMPT", ATTEMPT_CASES), ("CONFIRMATION_NOT_EXECUTED", CONFIRMATION_CASES),
        ):
            for method in BASELINES:
                for condition in (
                    "FULL", "CROP37_126", "SPEED075", "SPEED125", "MEAN3", "MEAN5",
                    "DELETE_10_PERCENT", "REPEAT_INSERT_10_PERCENT", "ADJACENT_EQUAL_INSERT_10_PERCENT",
                ):
                    rows = [
                        row for row in store.data["baseline_rows"]
                        if row["case_id"] in case_ids and row["method"] == method
                        and _condition_group(row["attack_id"]) == condition
                    ]
                    baseline_grouped[f"{execution_scope}|{method}|{condition}"] = _baseline_group_summary(rows)
        report = {
            "schema_version": "paper-results-v1-temporal-attack-report-v1",
            "fixed_denominator": store.data["fixed_denominator"],
            "receiver_rows": store.data["receiver_rows"],
            "baseline_rows": store.data["baseline_rows"],
            "quality_rows": store.data["quality_rows"],
            "paired_m05_rows": paired,
            "receiver_group_summaries": grouped,
            "baseline_group_summaries": baseline_grouped,
            "source_summaries": summaries,
            "claim_ceiling": "two-pilot engineering/evaluation output; confirmation is not executed",
        }
        report_path = atomic_json(store.output / "evaluation_report.json", report)
        _write_csv(store.output / "receiver_rows.csv", store.data["receiver_rows"])
        _write_csv(store.output / "baseline_rows.csv", store.data["baseline_rows"])
        _write_csv(store.output / "quality_rows.csv", store.data["quality_rows"])
        _write_csv(store.output / "paired_m05_rows.csv", paired)
        store.data["evaluation_report"] = report_path
        store.save()
        store.phase_finish("evaluate")
        return report
    except BaseException as exc:
        store.phase_failure("evaluate", exc)
        raise


def _counts(values):
    result = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _write_csv(path, rows):
    flattened = []
    fields = set()
    for row in rows:
        value = {
            key: (json.dumps(item, sort_keys=True, allow_nan=False) if isinstance(item, (dict, list)) else item)
            for key, item in row.items()
        }
        flattened.append(value); fields.update(value)
    ordered = sorted(fields)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ordered)
        writer.writeheader(); writer.writerows(flattened)


def run_phase(config_path, output, phase, *, case_id=None, temp_root=None):
    config = validate_config(read_json(config_path))
    if phase == "plan":
        return atomic_json(output, build_plan(config))
    if phase == "init":
        return AttackRunStore(output, config, create=True, temp_root=temp_root).data
    store = AttackRunStore(output, config, temp_root=temp_root)
    if phase == "evaluate":
        return phase_evaluate(store, config)
    if case_id not in ATTEMPT_CASES:
        raise AttackEvalError("only fixed pilot_01/pilot_02 may be attempted")
    dispatch = {
        "import-main": lambda: phase_import_main(store, config, case_id),
        "baseline-embed-videoseal": lambda: phase_baseline_embed(store, config, case_id, "videoseal"),
        "baseline-embed-rivagan": lambda: phase_baseline_embed(store, config, case_id, "rivagan"),
        "baseline-codec": lambda: phase_baseline_codec(store, config, case_id),
        "attack-media": lambda: phase_attack_media(store, config, case_id),
        "receiver-clock": lambda: phase_receiver_clock(store, config, case_id),
        "receiver-read": lambda: phase_receiver_read(store, config, case_id),
        "baseline-extract-videoseal": lambda: phase_baseline_extract(store, config, case_id, "videoseal"),
        "baseline-extract-rivagan": lambda: phase_baseline_extract(store, config, case_id, "rivagan"),
        "quality": lambda: phase_quality(store, config, case_id),
    }
    if phase not in dispatch:
        raise AttackEvalError(f"unknown phase {phase}")
    return dispatch[phase]()


def record_external_failure(config_path, output, failed_phase, case_id, reason, *, temp_root=None):
    config = validate_config(read_json(config_path))
    store = AttackRunStore(output, config, temp_root=temp_root)
    if failed_phase not in PHASES or failed_phase == "evaluate":
        raise AttackEvalError("invalid case phase for external failure")
    target = store.data["phases"][failed_phase]["cases"][case_id]
    if target["status"] in ("PLANNED", "RUNNING"):
        store.phase_failure(failed_phase, RuntimeError(reason), case_id)
    return target
