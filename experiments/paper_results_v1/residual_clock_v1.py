"""Fixed RGB8 residual-clock PRE diagnostic and separate delete_g0 NO_SYNC read.

The two result groups deliberately share only model-loading infrastructure.
Residual PRE evaluates one fixed integer RGB8 construction against two saved
payload controls and describes blind CENTERED_U clock outputs.  The delete_g0
group reads an old received clip in its first 161 received frames without any
clock correction and compares it with the already saved blind result.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import time

from experiments.paper_results_v1.resume_state import StateJournal, atomic_bytes, read_state, unit_attempt

SCHEMA = "paper-results-v1-residual-clock-v1"
CHECKPOINT_RUN_ID = "20261012T032440550166Z-c3026cfb"
LAYER_RUN_ID = "20261014T003005600270Z-bad88775"
CASES = ("confirm_01", "confirm_02", "confirm_07", "confirm_03")
PAYLOAD_ARMS = ("NATIVE_PRE", "CURRENT_M05_PRE", "RESIDUAL_CLOCK_PRE")
CLOCK_INPUTS = ("RESIDUAL_CLOCK", "NATIVE_NO_CLOCK")
SYNC_ARMS = ("NO_SYNC", "SAVED_BLIND_CENTERED_U")
SYNC_ATTRIBUTION_LIMIT = (
    "NO_SYNC uses 161 received-order frames at R40 while saved blind uses 181 restored frames at R44; "
    "the delta measures the whole synchronization recovery operation, including restored support length, "
    "and cannot be attributed to the clock score alone"
)
IDENTITY_181 = list(range(181))
RECEIVED_ORDER_161 = list(range(161))
RGB_SHAPE = [181, 320, 512, 3]
ATTACK_RGB_SHAPE = [163, 320, 512, 3]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def reason(exc):
    return f"{type(exc).__name__}: {exc}"


def _fixed_slots():
    return [f"residual/{case}" for case in CASES] + [f"delete_g0/{case}" for case in CASES]


class Store:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.path = self.output / "residual_clock_state.json"
        self.temp_root = self.output / "restored_media"
        self.data = read_state(self.path)
        if self.data.get("schema") != SCHEMA:
            raise ValueError("residual-clock schema changed")
        if [row.get("slot_id") for row in self.data.get("rows", [])] != _fixed_slots():
            raise ValueError("fixed residual/delete_g0 denominator changed")
        self.journal = StateJournal(self.path)
        self.journal.seed_cache(self.data)
        # A newly opened process cannot own an in-flight call recorded by the
        # previous process.  Retire those attempts before deciding what to
        # retry, while leaving completed and failed attempts untouched.
        retired = []
        for phase, status in list(self.data.get("phases", {}).items()):
            if status == "RUNNING":
                self.data["phases"][phase] = "INTERRUPTED"
                retired.append("phase/" + phase)
        for row in self.data.get("rows", []):
            for name, unit in row.get("units", {}).items():
                if unit.get("status") == "RUNNING":
                    unit.update(status="INTERRUPTED", reason="previous process ended before unit completion")
                    retired.append(row["slot_id"] + "/" + name)
        for collection in ("wan_loads", "wan_encode_clips", "framewise_loads",
                           "framewise_encode_clips", "framewise_physical_batches"):
            for index, record in enumerate(self.data.get(collection, [])):
                if record.get("status") == "RUNNING":
                    record.update(status="INTERRUPTED", reason="previous process ended before completion",
                                  finished_at=time.time())
                    retired.append(f"{collection}/{index}")
        for unit_id, attempts in self.data.get("unit_attempts", {}).items():
            for index, attempt in enumerate(attempts):
                if attempt.get("status") == "RUNNING":
                    attempt.update(status="INTERRUPTED", reason="previous process ended before completion",
                                   finished_at=time.time())
                    retired.append(f"unit_attempts/{unit_id}/{index}")
        if retired:
            self.data.setdefault("resume_recoveries", []).append({"at": time.time(), "retired_running": retired})
            self.journal.save(self.data)

    def save(self):
        self.journal.save(self.data)

    def resolve(self, value, document=None):
        raw = str(value).replace("\\", "/")
        path = Path(value)
        candidates = [path] if path.is_absolute() else []
        if document is not None:
            candidates.append(Path(document).parent / (path.name if path.is_absolute() else path))
        for root_key in ("checkpoint_run", "layer_run"):
            root = Path(self.data[root_key])
            marker = "/" + root.name + "/"
            if marker in raw:
                candidates.append(root / raw.split(marker, 1)[1])
        checkpoint = Path(self.data["checkpoint_run"])
        for folder in ("run_state", "attack_run", "confirmation_receiver"):
            token = f"/{folder}/"
            if token in raw:
                candidates.append(checkpoint / folder / raw.split(token, 1)[1])
        layer = Path(self.data["layer_run"])
        if "/reads/" in raw:
            candidates.append(layer / "run_state/reads" / raw.split("/reads/", 1)[1])
        if not path.is_absolute():
            candidates.extend((checkpoint / path, layer / path))
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key not in seen and candidate.is_file():
                return candidate
            seen.add(key)
        raise FileNotFoundError(f"saved artifact unavailable: {value}")


def _find_file(root, names):
    for name in names:
        path = Path(root) / name
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("none of the required saved records is readable: " + ", ".join(names))


def initialize(checkpoint_run, layer_run, output, config):
    output = Path(output).resolve()
    state_path = output / "residual_clock_state.json"
    if state_path.is_file():
        return Store(output)
    checkpoint_run, layer_run = Path(checkpoint_run).resolve(), Path(layer_run).resolve()
    checkpoint_state = _find_file(checkpoint_run, ("run_state/run_state.json", "run_state.json"))
    attack_state = _find_file(checkpoint_run, ("attack_run/attack_run_state.json",))
    receiver_report = _find_file(checkpoint_run, ("confirmation_receiver/evaluation_report.json",))
    layer_report = _find_file(layer_run, ("run_state/evaluation_report.json", "evaluation_report.json"))
    source_config = _find_file(checkpoint_run, ("effective_confirmation_config.json",))
    cfg, source_cfg = read_json(config), read_json(source_config)
    key, payload = cfg.get("keys", {}).get("K0"), cfg.get("payload_bits")
    if not isinstance(key, str) or not key or key != source_cfg.get("keys", {}).get("K0"):
        raise ValueError("K0 must equal the saved source-run K0")
    if payload != source_cfg.get("payload_bits") or len(payload or []) != 32 or any(type(x) is not int or x not in (0, 1) for x in payload):
        raise ValueError("payload must equal the saved 32-bit message")
    for name in ("wan", "framewise"):
        model = cfg.get("models", {}).get(name)
        if not isinstance(model, dict) or not isinstance(model.get("local_snapshot_path"), str):
            raise ValueError(f"models.{name}.local_snapshot_path is required")
    rows = []
    for case in CASES:
        rows.append({
            "slot_id": f"residual/{case}", "group": "RESIDUAL_PRE", "case_id": case,
            "status": "PLANNED", "planned_bits": 96, "missing_bits": 96,
            "units": {name: {"status": "PENDING", "reason": None} for name in
                      ("residual_rgb", "payload_controls", "payload_candidate", "clock_candidate", "clock_native")},
        })
    for case in CASES:
        rows.append({
            "slot_id": f"delete_g0/{case}", "group": "DELETE_G0_SYNC_VALUE", "case_id": case,
            "status": "PLANNED", "planned_bits": 64, "missing_bits": 64,
            "units": {name: {"status": "PENDING", "reason": None} for name in
                      ("received_order_rgb", "payload_no_sync", "saved_blind")},
        })
    state = {
        "schema": SCHEMA, "status": "INITIALIZED", "checkpoint_run": str(checkpoint_run),
        "layer_run": str(layer_run), "checkpoint_state": str(checkpoint_state),
        "attack_state": str(attack_state), "receiver_report": str(receiver_report),
        "layer_report": str(layer_report), "source_config": str(source_config),
        "effective_config": str(Path(config).resolve()), "key": key, "models": cfg["models"],
        "fixed_denominator": {
            "cases": list(CASES),
            "residual_payload_rows": 12, "residual_payload_bits": 384,
            "residual_clock_rows": 8,
            "delete_g0_sync_rows": 8, "delete_g0_payload_bits": 256,
            "candidate_rows": 4, "no_sync_rows": 4,
            "payload_arms": list(PAYLOAD_ARMS), "clock_inputs": list(CLOCK_INPUTS),
            "sync_arms": list(SYNC_ARMS), "key_label": "K0",
        },
        "method": {
            "candidate": "clip_uint8(int16(NATIVE_PRE)+int16(M05_PRE)-int16(RECON_PRE))",
            "coefficient": 1, "candidate_frames": 181, "candidate_payload_R": 44,
            "clock_receiver": "CENTERED_U", "clock_presence_threshold": None,
            "delete_g0_no_sync": "first 161 received frames in received order", "delete_g0_R": 40,
            "payload_target": "4/4 x 32/32 suggested target; never a completion gate",
        },
        "planned_budget": {
            "wan_loads_uninterrupted": 1, "wan_encode_clips": 8,
            "candidate_wan_encode_clips": 4, "delete_g0_no_sync_wan_encode_clips": 4,
            "framewise_loads_uninterrupted": 1, "candidate_framewise_encode_clips": 4,
            "candidate_framewise_physical_batches": 92, "batch_frames": 8,
            "contingent_native_framewise_encode_clips": 4,
            "contingent_native_framewise_physical_batches": 92,
            "framewise_physical_batches_upper_bound": 184,
            "existing_mp4_unique_inputs": 4,
            "existing_mp4_readbacks_uninterrupted_upper_bound": 4,
            "existing_mp4_readback_resume_semantics": "interrupted readback attempts are retained and may make actual attempts exceed four",
            "wan_decode": 0, "generation_or_dit": 0, "vjp": 0, "codec_encode": 0,
        },
        "rows": rows, "phases": {}, "unit_attempts": {}, "errors": [],
        "wan_loads": [], "wan_encode_clips": [], "framewise_loads": [],
        "framewise_encode_clips": [], "framewise_physical_batches": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(state_path, state)
    store = Store(output); report(store)
    return store


def _row(store, group, case):
    return next(row for row in store.data["rows"] if row["group"] == group and row["case_id"] == case)


def _artifact(document, artifact_id):
    matches = [row for row in document.get("artifacts", []) if row.get("artifact_id") == artifact_id]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one saved artifact {artifact_id}; found {len(matches)}")
    receipt = matches[0]
    if receipt.get("status") != "AVAILABLE" or not isinstance(receipt.get("path"), str):
        raise FileNotFoundError(f"saved artifact is unavailable: {artifact_id}")
    return receipt


def _load_rgb_receipt(store, receipt, shape):
    import numpy as np
    import torch
    if receipt.get("shape") != shape or receipt.get("dtype") != "uint8":
        raise ValueError(f"RGB8 shape/dtype changed for {receipt.get('artifact_id')}")
    path = store.resolve(receipt["path"])
    expected = math.prod(shape)
    if path.stat().st_size != expected:
        raise ValueError(f"RGB8 byte count changed for {receipt.get('artifact_id')}")
    pixels = np.memmap(path, dtype=np.uint8, mode="r", shape=tuple(shape))
    return torch.from_numpy(np.asarray(pixels).copy())


def _source_rgb(store, case, arm):
    state = read_json(store.data["checkpoint_state"])
    return _load_rgb_receipt(store, _artifact(state, f"{case}/{arm}/PRE"), RGB_SHAPE)


def _residual_path(store, case):
    return store.output / "rgb" / case / "residual_clock_pre.rgb8"


def _quality_path(store, case):
    return store.output / "quality" / case / "residual_vs_native.json"


def residual_rgb8(native, recon, m05):
    """Apply the adopted unit-coefficient integer RGB8 residual formula."""
    import numpy as np
    native, recon, m05 = (np.asarray(value) for value in (native, recon, m05))
    if native.shape != recon.shape or native.shape != m05.shape or native.dtype != np.uint8 or recon.dtype != np.uint8 or m05.dtype != np.uint8:
        raise ValueError("residual inputs must be equal-shaped uint8 arrays")
    return np.clip(native.astype(np.int16) + m05.astype(np.int16) - recon.astype(np.int16), 0, 255).astype(np.uint8)


def _construct_residual(store, case):
    import numpy as np
    state = read_json(store.data["checkpoint_state"])
    receipts = {arm: _artifact(state, f"{case}/{arm}/PRE") for arm in
                ("PAYLOAD_NATIVE", "PAYLOAD_FRAMEWISE_RECON", "PAYLOAD_FRAMEWISE_M05")}
    paths = {arm: store.resolve(receipt["path"]) for arm, receipt in receipts.items()}
    expected = math.prod(RGB_SHAPE)
    if any(receipt.get("shape") != RGB_SHAPE or receipt.get("dtype") != "uint8" for receipt in receipts.values()):
        raise ValueError("residual inputs must be saved uint8 [181,320,512,3]")
    if any(path.stat().st_size != expected for path in paths.values()):
        raise ValueError("residual input RGB8 byte count changed")
    arrays = {name: np.memmap(path, dtype=np.uint8, mode="r", shape=tuple(RGB_SHAPE)) for name, path in paths.items()}
    destination = _residual_path(store, case); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    framewise, temporal = [], []
    squared_total = 0; max_abs = 0; peak_pixel = None
    previous = None; temporal_squared = 0; temporal_max = 0; temporal_peak_pixel = None
    with temporary.open("wb") as stream:
        for frame in range(181):
            native_u8 = np.asarray(arrays["PAYLOAD_NATIVE"][frame])
            candidate = residual_rgb8(native_u8, arrays["PAYLOAD_FRAMEWISE_RECON"][frame], arrays["PAYLOAD_FRAMEWISE_M05"][frame])
            stream.write(candidate.tobytes())
            delta = candidate.astype(np.int16) - native_u8.astype(np.int16)
            absolute = np.abs(delta); flat_peak = int(absolute.argmax()); peak = int(absolute.reshape(-1)[flat_peak])
            y, x, channel = (int(value) for value in np.unravel_index(flat_peak, absolute.shape))
            sq = int(np.square(delta, dtype=np.int64).sum(dtype=np.int64)); squared_total += sq
            if peak > max_abs:
                max_abs = peak; peak_pixel = {"frame": frame, "y": y, "x": x, "channel": channel, "absolute_error_rgb8": peak}
            framewise.append({"frame": frame, "rmse_rgb8": math.sqrt(sq / delta.size), "max_abs_rgb8": peak,
                              "peak_pixel": {"y": y, "x": x, "channel": channel, "absolute_error_rgb8": peak}})
            if previous is not None:
                dt = delta - previous
                dt_absolute = np.abs(dt); dt_flat = int(dt_absolute.argmax()); dt_peak = int(dt_absolute.reshape(-1)[dt_flat])
                dt_y, dt_x, dt_channel = (int(value) for value in np.unravel_index(dt_flat, dt_absolute.shape))
                dt_sq = int(np.square(dt, dtype=np.int64).sum(dtype=np.int64)); temporal_squared += dt_sq
                if dt_peak > temporal_max:
                    temporal_max = dt_peak
                    temporal_peak_pixel = {"from_frame": frame - 1, "to_frame": frame, "y": dt_y, "x": dt_x,
                                           "channel": dt_channel, "absolute_error_rgb8": dt_peak}
                temporal.append({"from_frame": frame - 1, "to_frame": frame,
                                 "rmse_rgb8": math.sqrt(dt_sq / dt.size), "max_abs_rgb8": dt_peak,
                                 "peak_pixel": {"y": dt_y, "x": dt_x, "channel": dt_channel, "absolute_error_rgb8": dt_peak}})
            previous = delta
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, destination)
    pixels_per_frame = math.prod(RGB_SHAPE[1:]); count = expected
    peak_events = []
    for kind, rows, field in (("frame_rmse", framewise, "rmse_rgb8"),
                              ("frame_max_abs", framewise, "max_abs_rgb8"),
                              ("temporal_rmse", temporal, "rmse_rgb8"),
                              ("temporal_max_abs", temporal, "max_abs_rgb8")):
        peak = max(rows, key=lambda item: item[field])
        left, right = peak.get("from_frame", peak.get("frame")), peak.get("to_frame", peak.get("frame"))
        peak_events.append({"kind": kind, "value": peak[field], "from_frame": left, "to_frame": right,
                            "neighborhood": list(range(max(0, left - 1), min(180, right + 1) + 1))})
    quality = {
        "definition": "candidate minus NATIVE_PRE in RGB8 integer units; temporal is adjacent residual change",
        "frames": 181, "elements": count, "rmse_rgb8": math.sqrt(squared_total / count),
        "max_abs_rgb8": max_abs, "temporal_residual_rmse_rgb8": math.sqrt(temporal_squared / (180 * pixels_per_frame)),
        "temporal_residual_max_abs_rgb8": temporal_max, "peak_pixel": peak_pixel,
        "temporal_peak_pixel": temporal_peak_pixel, "framewise": framewise,
        "temporal_framewise": temporal, "local_peak_events": peak_events,
    }
    atomic_json(_quality_path(store, case), quality)
    return destination, quality


def _control_payloads(store, case):
    report_doc = read_json(store.data["layer_report"])
    output = {}
    for layer, arm in (("NATIVE_PRE", "NATIVE_PRE"), ("M05_PRE", "CURRENT_M05_PRE")):
        matches = [row for row in report_doc.get("rows", []) if row.get("case_id") == case and row.get("layer") == layer]
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one saved layer row {case}/{layer}; found {len(matches)}")
        row = matches[0]
        if row.get("status") != "EVALUATED" or row.get("R") != 44 or len(row.get("decoded_bits") or []) != 32:
            raise ValueError(f"saved layer row is not a complete R44 result: {case}/{layer}")
        output[arm] = {key: row.get(key) for key in
                       ("slot_id", "status", "R", "decoded_bits", "votes", "bit_errors", "exact_recovery", "minimum_aligned_vote_margin")}
        output[arm]["evidence_origin"] = "SAVED_LAYER_READOUT"
    return output


def _saved_blind(store, case):
    document = read_json(store.data["receiver_report"])
    matches = [row for row in document.get("receiver_rows", []) if
               (row.get("case_id"), row.get("arm"), row.get("attack_id"), row.get("key_label"), row.get("receiver")) ==
               (case, "PAYLOAD_FRAMEWISE_M05", "delete_g0", "K0", "CENTERED_U")]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one saved blind delete_g0 row for {case}; found {len(matches)}")
    row = matches[0]
    operation = row.get("operation") or {}
    bits, votes = row.get("decoded_bits"), row.get("votes")
    if (row.get("status") != "EVALUATED" or row.get("R") != 44
            or not isinstance(bits, list) or len(bits) != 32 or any(type(bit) is not int or bit not in (0, 1) for bit in bits)
            or not isinstance(votes, list) or len(votes) != 32):
        raise ValueError("saved CENTERED_U row is not a complete EVALUATED R44/32-bit result")
    try:
        from main.tube_state.video_trajectory_temporal_edit_receiver_v1 import decode_visible_span
        config = read_json(store.data["source_config"])
        attack = next(item for item in config["attacks"] if item["attack_id"] == "delete_g0")
        truth_path = [index for index in IDENTITY_181 if index not in set(attack["positions"])]
        oracle_map = decode_visible_span(truth_path)["received_index_map"]
    except Exception as exc:
        map_equal, oracle_diagnostic = None, {"status": "UNKNOWN", "reason": reason(exc)}
    else:
        map_equal = operation.get("received_index_map") == oracle_map
        oracle_diagnostic = {"status": "AVAILABLE", "comparison": "POSTHOC_ONLY"}
    saved_path_error = row.get("path_metrics") if isinstance(row.get("path_metrics"), dict) else row.get("path_error")
    if isinstance(saved_path_error, dict):
        path_error = {**saved_path_error, "evidence_origin": "SAVED_RECEIVER_POSTHOC_PATH_METRICS"}
    else:
        path_error = {"status": "UNKNOWN", "reason": "saved blind row has no path-error record",
                      "evidence_origin": "NO_PATH_ERROR_IN_SAVED_ROW"}
    return {
        "arm": "SAVED_BLIND_CENTERED_U", "status": "EVALUATED", "R": 44,
        "decoded_bits": bits, "votes": votes, "bit_errors": row.get("bit_errors"),
        "exact_recovery": row.get("exact_recovery"), "operation": operation,
        "blind_map_equal_oracle": map_equal, "oracle_map_diagnostic": oracle_diagnostic,
        "path_error": path_error, "selector_record": row.get("selector_record"),
        "evidence_origin": "SAVED_CENTERED_U; no new clock or oracle encode",
    }


def _attack_received_rgb(store, case):
    attack = read_json(store.data["attack_state"])
    receipt = dict(_artifact(attack, f"{case}/PAYLOAD_FRAMEWISE_M05/delete_g0/RECEIVED"))
    try:
        value = _load_rgb_receipt(store, receipt, ATTACK_RGB_SHAPE)
        origin = "SAVED_RGB8_CACHE"
    except (OSError, ValueError):
        from experiments.paper_results_v1.resume_media import load_rgb
        if isinstance(receipt.get("codec_record"), dict) and receipt["codec_record"].get("path"):
            receipt["codec_record"] = {**receipt["codec_record"], "path": str(store.resolve(receipt["codec_record"]["path"]))}
        def loader(item):
            return _load_rgb_receipt(store, item, ATTACK_RGB_SHAPE)
        value = load_rgb(store, receipt, artifacts=attack["artifacts"], loader=loader, read_json=read_json)
        origin = "DECODED_EXISTING_SAVED_MP4"
    return value[:161].contiguous(), origin


def _received_path(store, case):
    return store.output / "rgb" / case / "delete_g0_received_order_161.rgb8"


def _atomic_rgb8(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    array = value.detach().cpu().contiguous().numpy()
    with temporary.open("wb") as stream:
        stream.write(array.tobytes()); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def prepare(store, *, attack_loader=None):
    store.data["phases"]["prepare"] = "RUNNING"; store.save()
    try:
        with store.journal.periodic():
            for case in CASES:
                row = _row(store, "RESIDUAL_PRE", case)
                unit = row["units"]["residual_rgb"]
                if unit["status"] == "PENDING":
                    try:
                        path, quality = _construct_residual(store, case)
                        unit.update(status="AVAILABLE", reason=None, path=str(path), shape=RGB_SHAPE, dtype="uint8",
                                    quality_record=str(_quality_path(store, case)), rmse_rgb8=quality["rmse_rgb8"])
                    except Exception as exc:
                        unit.update(status="MISSING" if isinstance(exc, FileNotFoundError) else "FAILED", reason=reason(exc))
                        store.data["errors"].append({"at": time.time(), "slot_id": row["slot_id"], "unit": "residual_rgb", "reason": reason(exc)})
                    store.save()
                unit = row["units"]["payload_controls"]
                if unit["status"] == "PENDING":
                    try:
                        controls = _control_payloads(store, case)
                        path = store.output / "controls" / case / "payload_controls.json"; atomic_json(path, controls)
                        unit.update(status="REUSED", reason=None, path=str(path), arms=list(PAYLOAD_ARMS[:2]))
                    except Exception as exc:
                        unit.update(status="MISSING" if isinstance(exc, FileNotFoundError) else "FAILED", reason=reason(exc))
                    store.save()
                sync = _row(store, "DELETE_G0_SYNC_VALUE", case)
                unit = sync["units"]["saved_blind"]
                if unit["status"] == "PENDING":
                    try:
                        blind = _saved_blind(store, case)
                        path = store.output / "sync" / case / "saved_blind.json"; atomic_json(path, blind)
                        unit.update(status="REUSED", reason=None, path=str(path), R=44)
                    except Exception as exc:
                        unit.update(status="MISSING" if isinstance(exc, FileNotFoundError) else "FAILED", reason=reason(exc))
                    store.save()
                unit = sync["units"]["received_order_rgb"]
                if unit["status"] == "PENDING":
                    try:
                        rgb, origin = (attack_loader(case) if attack_loader else _attack_received_rgb(store, case))
                        if tuple(rgb.shape) != (161, 320, 512, 3):
                            raise ValueError("delete_g0 NO_SYNC input must be first 161 received RGB8 frames")
                        path = _received_path(store, case); _atomic_rgb8(path, rgb)
                        unit.update(status="AVAILABLE", reason=None, path=str(path), shape=[161, 320, 512, 3], dtype="uint8",
                                    origin=origin, received_source_frames=163, tail_frames_discarded_for_wan_prefix=2)
                    except Exception as exc:
                        unit.update(status="MISSING" if isinstance(exc, FileNotFoundError) else "FAILED", reason=reason(exc))
                    store.save()
        store.data["phases"]["prepare"] = "COMPLETE"
    except BaseException as exc:
        store.data["phases"]["prepare"] = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"; raise
    finally:
        store.save(); report(store)
    return store


def _load_rgb_file(path, shape):
    import numpy as np
    import torch
    path = Path(path); expected = math.prod(shape)
    if not path.is_file() or path.stat().st_size != expected:
        raise FileNotFoundError(f"durable RGB8 file unavailable or incomplete: {path}")
    return torch.from_numpy(np.asarray(np.memmap(path, dtype=np.uint8, mode="r", shape=tuple(shape))).copy())


def _atomic_torch(path, tensor):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp"); torch.save(tensor, temporary)
    with temporary.open("rb") as stream: os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_torch(path, shape):
    import torch
    try: value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError: value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"durable latent must be finite {shape}")
    return value.float()


def _payload_paths(store, group, case):
    return store.output / "payload" / group / case / "read.json", store.output / "payload" / group / case / "votes.npz"


def _atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save_payload(store, group, case, result, origin):
    import numpy as np
    support = 44 if group == "candidate" else 40
    if result.get("status") != "READ" or result.get("R") != support or len(result.get("decoded_bits") or []) != 32:
        raise ValueError(f"payload result must be complete R{support}")
    signs, zeros = np.asarray(result["signed_votes"]), np.asarray(result["zero_mask"])
    if signs.shape != (4, support, 240) or zeros.shape != signs.shape or not np.array_equal(signs == 0, zeros):
        raise ValueError("payload vote sidecar shape/content changed")
    detail, sidecar = _payload_paths(store, group, case)
    _atomic_npz(sidecar, signed_votes=signs.astype(np.int8), zero_mask=zeros.astype(bool))
    document = {key: value for key, value in result.items() if key not in ("signed_votes", "zero_mask", "key_id")}
    document.update({"schema": SCHEMA, "group": group, "case_id": case, "key_label": "K0",
                     "evidence_origin": origin, "vote_sidecar": str(sidecar), "truth_inputs": False})
    atomic_json(detail, document)
    return detail, sidecar


def _wan_latent_path(store, group, case):
    return store.output / "latents" / "wan" / group / f"{case}.pt"


def _wan_units(store):
    units = []
    for case in CASES:
        residual = _row(store, "RESIDUAL_PRE", case)
        units.append(("candidate", residual, residual["units"]["payload_candidate"],
                      residual["units"]["residual_rgb"], RGB_SHAPE, 181, IDENTITY_181))
        sync = _row(store, "DELETE_G0_SYNC_VALUE", case)
        units.append(("no_sync", sync, sync["units"]["payload_no_sync"],
                      sync["units"]["received_order_rgb"], [161, 320, 512, 3], 161, RECEIVED_ORDER_161))
    return units


def run_wan(store, *, load_vae=None, encode=None, payload_read=None):
    from runtime.wan.generation import load_frozen_vae
    from runtime.wan.vae import reencode_rgb24_readback
    from runtime.wan.video_trajectory_temporal_edit_receiver_v1 import read_payload_general
    load_vae = load_vae or (lambda: load_frozen_vae({"model": {"id": store.data["models"]["wan"]["local_snapshot_path"]}},
                                                     device=store.data["models"]["wan"].get("device", "cuda")))
    encode, payload_read = encode or reencode_rgb24_readback, payload_read or read_payload_general
    store.data["phases"]["wan"] = "RUNNING"; store.save(); vae = None
    try:
        with store.journal.periodic():
            prepared = []
            for group, row, unit, rgb_unit, shape, frames, mapping in _wan_units(store):
                if unit["status"] not in ("PENDING", "RUNNING", "INTERRUPTED", "ENCODED"):
                    continue
                latent_path = _wan_latent_path(store, group, row["case_id"])
                if latent_path.is_file():
                    try:
                        latent = _load_torch(latent_path, [1, 16, (frames - 1) // 4 + 1, 40, 64])
                        result = payload_read(latent, store.data["key"], frames, mapping)
                        detail, sidecar = _save_payload(store, group, row["case_id"], result, "DURABLE_NEW_WAN_ENCODE")
                        unit.update(status="READ", reason=None, R=result["R"], decoded_bits=result["decoded_bits"], votes=result["votes"],
                                    detail_record=str(detail), vote_sidecar=str(sidecar), latent_path=str(latent_path))
                    except Exception as exc:
                        unit.update(status="MISSING" if isinstance(exc, FileNotFoundError) else "FAILED",
                                    reason="durable Wan latent/read failed: " + reason(exc), latent_path=str(latent_path))
                    store.save(); continue
                if unit["status"] == "ENCODED":
                    unit.update(status="MISSING", reason=f"durable encoded latent unavailable: {latent_path}"); store.save(); continue
                if rgb_unit.get("status") != "AVAILABLE":
                    unit.update(status="MISSING", reason="required durable RGB8 unit unavailable"); store.save(); continue
                try:
                    _load_rgb_file(rgb_unit["path"], shape)
                except Exception as exc:
                    unit.update(status="MISSING" if isinstance(exc, FileNotFoundError) else "FAILED", reason=reason(exc)); store.save()
                else:
                    prepared.append((group, row, unit, rgb_unit, shape, frames, mapping))
            if prepared:
                load = {"attempt": len(store.data["wan_loads"]) + 1, "status": "RUNNING", "started_at": time.time()}
                store.data["wan_loads"].append(load); store.save()
                try: vae = load_vae()
                except BaseException as exc:
                    status = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"
                    load.update(status=status, reason=reason(exc), finished_at=time.time())
                    for _group, _row_value, unit, _rgb_unit, _shape, _frames, _mapping in prepared:
                        unit.update(status=status, reason="Wan model load failed: " + reason(exc))
                    store.save(); raise
                load.update(status="COMPLETE", finished_at=time.time()); store.save()
            for group, row, unit, rgb_unit, shape, frames, mapping in prepared:
                rgb = latent = None; case = row["case_id"]; clip_id = f"{group}/{case}"
                try:
                    rgb = _load_rgb_file(rgb_unit["path"], shape); unit.update(status="RUNNING", reason=None); store.save()
                    with unit_attempt(store, "wan/" + clip_id, operation="wan_vae_encode_rgb8", frames=frames):
                        call = {"clip_id": clip_id, "status": "RUNNING", "started_at": time.time(), "frames": frames}
                        store.data["wan_encode_clips"].append(call); store.save()
                        def observer(event, error):
                            if event == "RETURNED": call.update(status="COMPLETE", finished_at=time.time())
                            elif event == "FAILED": call.update(status="FAILED" if isinstance(error, Exception) else "INTERRUPTED", reason=reason(error), finished_at=time.time())
                            elif event != "STARTED": raise ValueError("invalid Wan encode observer event")
                            store.save()
                        normalized = encode(vae, rgb.float().div(255.0), encode_observer=observer)
                        latent = normalized.detach().cpu().float().contiguous()
                        expected_shape = [1, 16, (frames - 1) // 4 + 1, 40, 64]
                        if list(latent.shape) != expected_shape:
                            raise ValueError(f"Wan encode returned {list(latent.shape)}, expected {expected_shape}")
                        path = _wan_latent_path(store, group, case); _atomic_torch(path, latent)
                        unit.update(status="ENCODED", latent_path=str(path), encode_finished_at=time.time()); store.save()
                    result = payload_read(latent, store.data["key"], frames, mapping)
                    detail, sidecar = _save_payload(store, group, case, result, "DURABLE_NEW_WAN_ENCODE")
                    unit.update(status="READ", reason=None, R=result["R"], decoded_bits=result["decoded_bits"], votes=result["votes"],
                                detail_record=str(detail), vote_sidecar=str(sidecar), latent_path=str(path)); store.save()
                except BaseException as exc:
                    unit.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED", reason=reason(exc)); store.save()
                    if not isinstance(exc, Exception): raise
                finally: rgb = latent = None
        store.data["phases"]["wan"] = "COMPLETE"
    except BaseException as exc:
        store.data["phases"]["wan"] = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"; raise
    finally:
        vae = None
        import gc; gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        finally: store.save(); report(store)
    return store


def _framewise_path(store, kind, case):
    return store.output / "latents" / "framewise" / kind / f"{case}.npy"


def _load_framewise_array(path):
    import numpy as np
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.shape != (181, 4, 40, 64) or not np.isfinite(value).all():
        raise ValueError("framewise latent must be finite float32 [181,4,40,64]")
    return value


def _atomic_npy(path, value):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(value, dtype=np.float32), allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _saved_native_latent(store, case):
    state = read_json(store.data["checkpoint_state"]); receipt = _artifact(state, f"{case}/SHARED_FRAMEWISE/LATENT")
    path = store.resolve(receipt["path"]); return _load_framewise_array(path), path


def _save_clock(store, kind, case, latent, origin):
    import numpy as np
    from main.tube_state import video_trajectory_temporal_edit_receiver_v1 as clock
    from main.tube_state.receiver_controls_v1 import receive_scores
    evidence = clock.score_framewise(latent, store.data["key"], clock.PUBLIC)
    q = evidence["signed_projection"] / evidence["rho"]
    background = np.median(q, axis=0); centered = q - background
    selection = receive_scores(evidence["signed_projection"], evidence["rho"], "CENTERED_U")
    folder = store.output / "clock" / kind / case
    scores = folder / "scores.npz"
    _atomic_npz(scores, signed_projection=evidence["signed_projection"], rho=evidence["rho"],
                raw_q=q, column_median=background, centered_q=centered)
    blind = {"schema": SCHEMA, "case_id": case, "input": kind, "receiver": "CENTERED_U",
             "selection": selection, "score_sidecar": str(scores), "latent_origin": origin,
             "truth_inputs": False, "clock_presence_threshold": None,
             "interpretation_limit": "path, gap, status, or SUPPORTED does not establish clock presence"}
    blind_path = folder / "blind_selection.json"; atomic_json(blind_path, blind)
    chosen = selection["estimate"].get("path")
    diagnostic_best = selection["estimate"].get("best_path")
    def error(path):
        if path is None: return None
        values = [abs(index - source) for index, source in enumerate(path)]
        return {"mean_absolute": sum(values) / len(values), "maximum_absolute": max(values), "count": len(values)}
    posthoc = {"truth_use": "POSTHOC_ONLY_AFTER_BLIND_SELECTION_SAVED", "identity_path": "0..180",
               "selected_path_error": error(chosen), "diagnostic_best_path_error": error(diagnostic_best)}
    truth_path = folder / "posthoc_path_error.json"; atomic_json(truth_path, posthoc)
    estimate = selection["estimate"]
    return {"status": "SCORED", "reason": None, "blind_record": str(blind_path), "score_sidecar": str(scores),
            "posthoc_path_error": str(truth_path), "selector_status": estimate["status"], "path": estimate.get("path"),
            "best_score": estimate.get("best_score"), "runner_up_score": estimate.get("runner_up_score"),
            "score_gap": estimate.get("score_gap"), "latent_origin": origin}


def run_clock(store, *, load_backend=None, encode_backend=None):
    from experiments.paper_results_v1.real_backends import load_local_framewise_backend
    load_backend = load_backend or (lambda: load_local_framewise_backend(store.data["models"]["framewise"]))
    store.data["phases"]["clock"] = "RUNNING"; store.save(); backend = None
    try:
        with store.journal.periodic():
            needs_encode = []
            for case in CASES:
                row = _row(store, "RESIDUAL_PRE", case)
                candidate = row["units"]["clock_candidate"]
                if candidate["status"] in ("PENDING", "RUNNING", "INTERRUPTED", "ENCODED"):
                    path = _framewise_path(store, "candidate", case)
                    if path.is_file():
                        try:
                            latent = _load_framewise_array(path)
                            candidate.update(**_save_clock(store, "candidate", case, latent, "NEW_CANDIDATE_FRAMEWISE_ENCODE"),
                                             latent_path=str(path))
                        except Exception as exc:
                            candidate.update(status="MISSING" if isinstance(exc, FileNotFoundError) else "FAILED",
                                             reason="durable candidate clock latent/read failed: " + reason(exc),
                                             latent_path=str(path))
                        store.save()
                    elif candidate["status"] == "ENCODED":
                        candidate.update(status="MISSING", reason=f"durable candidate latent unavailable: {path}"); store.save()
                    elif row["units"]["residual_rgb"].get("status") == "AVAILABLE":
                        needs_encode.append(("candidate", row, candidate, row["units"]["residual_rgb"], RGB_SHAPE))
                    else:
                        candidate.update(status="MISSING", reason="residual RGB8 unavailable"); store.save()
                native = row["units"]["clock_native"]
                if native["status"] in ("PENDING", "RUNNING", "INTERRUPTED", "ENCODED"):
                    fallback = _framewise_path(store, "native_fallback", case)
                    try:
                        latent, source_path = _saved_native_latent(store, case)
                    except Exception as reuse_exc:
                        if fallback.is_file():
                            try:
                                latent = _load_framewise_array(fallback)
                                native.update(**_save_clock(store, "native", case, latent, "CONTINGENT_NATIVE_FRAMEWISE_ENCODE"),
                                              latent_path=str(fallback), reuse_failure=reason(reuse_exc))
                            except Exception as exc:
                                native.update(status="MISSING" if isinstance(exc, FileNotFoundError) else "FAILED",
                                              reason="durable native fallback latent/read failed: " + reason(exc),
                                              latent_path=str(fallback), reuse_failure=reason(reuse_exc))
                            store.save()
                        elif native["status"] == "ENCODED":
                            native.update(status="MISSING", reason=f"durable fallback latent unavailable: {fallback}", reuse_failure=reason(reuse_exc)); store.save()
                        else:
                            try:
                                state = read_json(store.data["checkpoint_state"]); receipt = _artifact(state, f"{case}/PAYLOAD_NATIVE/PRE")
                                store.resolve(receipt["path"])
                            except Exception as exc:
                                native.update(status="MISSING", reason=reason(exc), reuse_failure=reason(reuse_exc)); store.save()
                            else:
                                needs_encode.append(("native_fallback", row, native,
                                                     {"path": str(store.resolve(receipt["path"])), "status": "AVAILABLE"}, RGB_SHAPE))
                    else:
                        try:
                            native.update(**_save_clock(store, "native", case, latent, "REUSED_SAVED_SHARED_FRAMEWISE_LATENT"),
                                          source_latent=str(source_path))
                        except Exception as exc:
                            native.update(status="MISSING" if isinstance(exc, FileNotFoundError) else "FAILED",
                                          reason="saved native clock latent/read failed: " + reason(exc),
                                          source_latent=str(source_path))
                        store.save()
            if needs_encode:
                load = {"attempt": len(store.data["framewise_loads"]) + 1, "status": "RUNNING", "started_at": time.time()}
                store.data["framewise_loads"].append(load); store.save()
                try: backend = load_backend()
                except BaseException as exc:
                    status = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"
                    load.update(status=status, reason=reason(exc), finished_at=time.time())
                    for _kind, _row_value, unit, _rgb_unit, _shape in needs_encode:
                        unit.update(status=status, reason="framewise model load failed: " + reason(exc))
                    store.save(); raise
                load.update(status="COMPLETE", finished_at=time.time()); store.save()
            for kind, row, unit, rgb_unit, shape in needs_encode:
                case = row["case_id"]; clip_id = f"{kind}/{case}"; latent = rgb = None
                try:
                    rgb = _load_rgb_file(rgb_unit["path"], shape); unit.update(status="RUNNING", reason=None); store.save()
                    clip = {"clip_id": clip_id, "status": "RUNNING", "frames": 181, "planned_physical_batches": 23, "started_at": time.time()}
                    store.data["framewise_encode_clips"].append(clip); store.save()
                    with unit_attempt(store, "framewise/" + clip_id, operation="framewise_vae_encode", frames=181, batch_frames=8):
                        active = {}
                        def observer(event, start, stop, error):
                            batch_id = f"{clip_id}/{start:03d}-{stop:03d}"
                            if event == "STARTED":
                                record = {"batch_id": batch_id, "clip_id": clip_id, "status": "RUNNING", "start": start, "stop": stop, "started_at": time.time()}
                                store.data["framewise_physical_batches"].append(record); active[batch_id] = record
                            else:
                                record = active[batch_id]
                                record.update(status="COMPLETE" if event == "RETURNED" else ("FAILED" if isinstance(error, Exception) else "INTERRUPTED"), finished_at=time.time())
                                if error is not None: record["reason"] = reason(error)
                            store.save()
                        if encode_backend is None:
                            latent = backend.encode(rgb, batch_observer=observer)
                        else:
                            latent = encode_backend(backend, rgb, observer)
                        # Count the physical clip call as complete at model
                        # return; later durable-save/read failures belong to
                        # the unit attempt and must not erase that call.
                        clip.update(status="COMPLETE", finished_at=time.time()); store.save()
                        path = _framewise_path(store, kind, case); _atomic_npy(path, latent)
                        unit.update(status="ENCODED", latent_path=str(path), encode_finished_at=time.time()); store.save()
                    unit.update(**_save_clock(store, "candidate" if kind == "candidate" else "native", case,
                                              _load_framewise_array(path), "NEW_CANDIDATE_FRAMEWISE_ENCODE" if kind == "candidate" else "CONTINGENT_NATIVE_FRAMEWISE_ENCODE")); store.save()
                except BaseException as exc:
                    clip = next((item for item in reversed(store.data["framewise_encode_clips"]) if item["clip_id"] == clip_id and item["status"] == "RUNNING"), None)
                    if clip: clip.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED", reason=reason(exc), finished_at=time.time())
                    unit.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED", reason=reason(exc)); store.save()
                    if not isinstance(exc, Exception): raise
                finally: latent = rgb = None
        store.data["phases"]["clock"] = "COMPLETE"
    except BaseException as exc:
        store.data["phases"]["clock"] = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"; raise
    finally:
        if backend is not None and callable(getattr(backend, "close", None)): backend.close()
        backend = None
        import gc; gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        finally: store.save(); report(store)
    return store


def _payload_summary(record, truth, arm):
    if record.get("status") not in ("READ", "EVALUATED"):
        return {"arm": arm, "status": record.get("status"), "reason": record.get("reason"), "missing_bits": 32}
    bits, votes = record["decoded_bits"], record["votes"]
    per_bit = []
    for index, (decoded, vote) in enumerate(zip(bits, votes)):
        ones, zeros = int(vote["ones"]), int(vote["zeros"]); count = int(vote.get("count", ones + zeros))
        margin = ((ones - zeros) if truth[index] else (zeros - ones)) / count
        per_bit.append({"bit_index": index, "truth": truth[index], "decoded": decoded,
                        "correct": decoded == truth[index], "ones": ones, "zeros": zeros,
                        "count": count, "aligned_vote_margin": margin})
    errors = sum(item["decoded"] != item["truth"] for item in per_bit)
    return {"arm": arm, "status": "EVALUATED", "R": record["R"], "decoded_bits": bits,
            "bit_errors": errors, "exact_recovery": errors == 0,
            "minimum_aligned_vote_margin": min(item["aligned_vote_margin"] for item in per_bit),
            "per_bit": per_bit, "missing_bits": 0}


def _load_unit_record(unit):
    return read_json(unit["detail_record"]) if unit.get("detail_record") else unit


def report(store):
    try:
        truth = read_json(store.data["source_config"])["payload_bits"]
        if len(truth) != 32: raise ValueError("truth payload length changed")
    except Exception as exc:
        truth = None; truth_error = reason(exc)
    else: truth_error = None
    residual_payload_rows, residual_clock_rows, sync_rows, sync_pairs = [], [], [], []
    residual_cases, sync_cases = [], []
    for case in CASES:
        row = _row(store, "RESIDUAL_PRE", case); case_payload = []
        controls_unit = row["units"]["payload_controls"]
        controls = read_json(controls_unit["path"]) if controls_unit.get("path") and Path(controls_unit["path"]).is_file() else {}
        for arm in PAYLOAD_ARMS[:2]:
            result = _payload_summary(controls.get(arm, {"status": controls_unit.get("status"), "reason": controls_unit.get("reason")}), truth, arm) if truth else {"arm": arm, "status": "MISSING_TRUTH", "reason": truth_error, "missing_bits": 32}
            result.update(case_id=case, group="RESIDUAL_PRE"); residual_payload_rows.append(result); case_payload.append(result)
        candidate_unit = row["units"]["payload_candidate"]
        candidate_record = _load_unit_record(candidate_unit)
        result = _payload_summary(candidate_record, truth, PAYLOAD_ARMS[2]) if truth else {"arm": PAYLOAD_ARMS[2], "status": "MISSING_TRUTH", "reason": truth_error, "missing_bits": 32}
        result.update(case_id=case, group="RESIDUAL_PRE"); residual_payload_rows.append(result); case_payload.append(result)
        case_clocks = []
        for field, name in (("clock_candidate", CLOCK_INPUTS[0]), ("clock_native", CLOCK_INPUTS[1])):
            unit = row["units"][field]
            item = {"case_id": case, "input": name, **{key: unit.get(key) for key in
                    ("status", "reason", "selector_status", "path", "best_score", "runner_up_score", "score_gap", "blind_record", "score_sidecar", "posthoc_path_error", "latent_origin")}}
            residual_clock_rows.append(item); case_clocks.append(item)
        quality = read_json(row["units"]["residual_rgb"]["quality_record"]) if row["units"]["residual_rgb"].get("quality_record") and Path(row["units"]["residual_rgb"]["quality_record"]).is_file() else None
        row["status"] = "COMPLETE_RECORD" if all(item["status"] == "EVALUATED" for item in case_payload) and all(item["status"] == "SCORED" for item in case_clocks) and quality else "PARTIAL_WITH_RETAINED_FAILURES"
        row["missing_bits"] = sum(item.get("missing_bits", 0) for item in case_payload)
        residual_cases.append({"case_id": case, "status": row["status"], "payload": case_payload, "clock": case_clocks, "quality": quality})

        sync = _row(store, "DELETE_G0_SYNC_VALUE", case); case_sync = []
        no_sync_unit = sync["units"]["payload_no_sync"]
        no_sync = _payload_summary(_load_unit_record(no_sync_unit), truth, "NO_SYNC") if truth else {"arm": "NO_SYNC", "status": "MISSING_TRUTH", "reason": truth_error, "missing_bits": 32}
        no_sync.update(case_id=case, group="DELETE_G0_SYNC_VALUE", input_frames=161, R_expected=40); sync_rows.append(no_sync); case_sync.append(no_sync)
        blind_unit = sync["units"]["saved_blind"]
        blind_record = read_json(blind_unit["path"]) if blind_unit.get("path") and Path(blind_unit["path"]).is_file() else blind_unit
        blind = _payload_summary(blind_record, truth, "SAVED_BLIND_CENTERED_U") if truth else {"arm": "SAVED_BLIND_CENTERED_U", "status": "MISSING_TRUTH", "reason": truth_error, "missing_bits": 32}
        blind.update(case_id=case, group="DELETE_G0_SYNC_VALUE", input_frames=181, R_expected=44,
                     blind_map_equal_oracle=blind_record.get("blind_map_equal_oracle"), path_error=blind_record.get("path_error")); sync_rows.append(blind); case_sync.append(blind)
        pair_available = no_sync.get("status") == "EVALUATED" and blind.get("status") == "EVALUATED"
        pair = {
            "case_id": case, "status": "AVAILABLE" if pair_available else "UNAVAILABLE",
            "no_sync_bit_errors": no_sync.get("bit_errors") if pair_available else None,
            "saved_blind_bit_errors": blind.get("bit_errors") if pair_available else None,
            "no_sync_minus_saved_blind_bit_errors": (
                no_sync["bit_errors"] - blind["bit_errors"] if pair_available else None
            ),
            "interpretation": "positive means fewer payload bit errors after the whole saved synchronization recovery operation",
            "attribution_limit": SYNC_ATTRIBUTION_LIMIT,
        }
        sync_pairs.append(pair)
        sync["status"] = "COMPLETE_RECORD" if all(item["status"] == "EVALUATED" for item in case_sync) else "PARTIAL_WITH_RETAINED_FAILURES"
        sync["missing_bits"] = sum(item.get("missing_bits", 0) for item in case_sync)
        sync_cases.append({"case_id": case, "status": sync["status"], "payload_comparison": case_sync,
                           "paired_bit_error_delta": pair,
                           "attribution_limit": SYNC_ATTRIBUTION_LIMIT + "; this group is independent of the residual writer"})
    store.data["status"] = "COMPLETE" if all(row["status"] == "COMPLETE_RECORD" for row in store.data["rows"]) else "PARTIAL_WITH_RETAINED_FAILURES"
    store.save()
    def counts(values): return dict(Counter(item.get("status", "UNKNOWN") for item in values))
    available_pairs = [item for item in sync_pairs if item["status"] == "AVAILABLE"]
    paired_effect = {
        "planned_pairs": len(CASES), "available_pairs": len(available_pairs),
        "unavailable_pairs": len(CASES) - len(available_pairs),
        "case_rows": sync_pairs,
        "no_sync_minus_saved_blind_bit_errors_total": (
            sum(item["no_sync_minus_saved_blind_bit_errors"] for item in available_pairs)
            if len(available_pairs) == len(CASES) else None
        ),
        "observed_available_pair_delta_sum": sum(
            item["no_sync_minus_saved_blind_bit_errors"] for item in available_pairs
        ),
        "complete_total_requires_all_four_pairs": True,
        "attribution_limit": SYNC_ATTRIBUTION_LIMIT,
    }
    summary = {
        "schema": SCHEMA, "status": store.data["status"], "fixed_denominator": store.data["fixed_denominator"],
        "residual_payload_status_counts": counts(residual_payload_rows), "residual_clock_status_counts": counts(residual_clock_rows),
        "delete_g0_sync_status_counts": counts(sync_rows),
        "residual_payload_exact_rows": sum(item.get("exact_recovery") is True for item in residual_payload_rows),
        "candidate_payload_exact_cases": sum(item.get("exact_recovery") is True for item in residual_payload_rows if item["arm"] == PAYLOAD_ARMS[2]),
        "suggested_target_only": "candidate 4/4 cases x 32/32 exact; not a completion gate",
        "clock_interpretation_limit": "No presence threshold: status, path, positive gap, or SUPPORTED is descriptive only.",
        "delete_g0_paired_sync_effect": paired_effect,
        "delete_g0_sync_attribution_limit": SYNC_ATTRIBUTION_LIMIT,
        "groups_are_causally_separate": True,
        "actual_calls": {
            "wan_loads": counts(store.data["wan_loads"]), "wan_encode_clips": counts(store.data["wan_encode_clips"]),
            "framewise_loads": counts(store.data["framewise_loads"]),
            "framewise_encode_clips": counts(store.data["framewise_encode_clips"]),
            "framewise_physical_batches": counts(store.data["framewise_physical_batches"]),
            "existing_mp4_readbacks": counts([
                attempt for unit_id, attempts in store.data.get("unit_attempts", {}).items()
                if unit_id.startswith("restore-rgb/") for attempt in attempts
            ]),
            "generation_or_dit": 0, "wan_decode": 0, "vjp": 0, "codec_encode": 0,
        },
        "pending_model_units": pending_model_units(store),
    }
    atomic_json(store.output / "summary.json", summary)
    atomic_json(store.output / "evaluation_report.json", {"summary": summary, "residual_cases": residual_cases,
                                                           "residual_payload_rows": residual_payload_rows,
                                                           "residual_clock_rows": residual_clock_rows,
                                                           "delete_g0_cases": sync_cases, "delete_g0_sync_rows": sync_rows})
    for filename, rows in (("residual_payload_rows.csv", residual_payload_rows),
                           ("residual_clock_rows.csv", residual_clock_rows), ("delete_g0_sync_rows.csv", sync_rows)):
        compact = [{key: value for key, value in row.items() if key not in ("per_bit", "path")} for row in rows]
        fields = sorted({key for row in compact for key in row})
        with (store.output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(compact)
    return summary


def pending_model_units(store):
    result = []
    for group, row, unit, rgb_unit, shape, frames, mapping in _wan_units(store):
        if unit["status"] in ("PENDING", "RUNNING", "INTERRUPTED") and not _wan_latent_path(store, group, row["case_id"]).is_file():
            result.append("wan/" + group + "/" + row["case_id"])
    for case in CASES:
        row = _row(store, "RESIDUAL_PRE", case)
        if row["units"]["clock_candidate"]["status"] in ("PENDING", "RUNNING", "INTERRUPTED") and not _framewise_path(store, "candidate", case).is_file():
            result.append("framewise/candidate/" + case)
        native = row["units"]["clock_native"]
        if native["status"] in ("PENDING", "RUNNING", "INTERRUPTED"):
            try: _saved_native_latent(store, case)
            except Exception:
                if not _framewise_path(store, "native_fallback", case).is_file(): result.append("framewise/native_fallback/" + case)
    return result


def record_external_failure(store, message):
    store.data["errors"].append({"at": time.time(), "phase": "external", "reason": str(message)})
    for row in store.data["rows"]:
        for unit in row["units"].values():
            if unit.get("status") == "RUNNING": unit.update(status="INTERRUPTED", reason=str(message))
    for phase, status in list(store.data["phases"].items()):
        if status == "RUNNING": store.data["phases"][phase] = "INTERRUPTED"
    store.save(); return report(store)
