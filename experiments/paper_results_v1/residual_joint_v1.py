"""Fixed clean-MP4/delete_g0 joint receiver for four saved residual candidates.

The writer is never reconstructed here.  Four saved RGB8 candidates are each
published once with the adopted codec.  The fixed delete_g0 observation is
derived from that same decoded MP4.  Both observations then follow the same
blind chain: received RGB -> framewise clock -> CENTERED_U -> visible-span
correction -> Wan payload read.  Attack truth and payload truth are joined only
by :func:`report`, after the blind selector and read records are durable.
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


SCHEMA = "paper-results-v1-residual-joint-v1"
CASES = ("confirm_01", "confirm_02", "confirm_07", "confirm_03")
GROUPS = ("CLEAN_MP4", "DELETE_G0")
RGB_SHAPE = [181, 320, 512, 3]
DELETE_SHAPE = [163, 320, 512, 3]
CODEC = {"codec": "libx264", "fps": 8, "crf": 18, "pix_fmt": "yuv420p"}
DELETE_POSITIONS = [0, 10, 42, 44, 59, 64, 65, 83, 84, 89, 91, 100, 102, 116, 131, 144, 160, 169]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def reason(exc):
    return f"{type(exc).__name__}: {exc}"


def delete_spec():
    return {
        "attack_id": "delete_g0", "family": "DELETE", "group": 0,
        "seed": 20261010, "positions": list(DELETE_POSITIONS),
    }


def validate_cases(cases):
    cases = tuple(cases)
    if (len(cases) != 4 or len(set(cases)) != 4 or
            any(not isinstance(case, str) or not case or not case.replace("_", "").isalnum() for case in cases)):
        raise ValueError("four distinct safe case IDs are required")
    return cases


def _slot_ids(cases=CASES):
    return [f"{group.lower()}/{case}" for group in GROUPS for case in cases]


def _new_rows(cases=CASES):
    rows = []
    for group in GROUPS:
        frames = 181 if group == "CLEAN_MP4" else 163
        for case in cases:
            rows.append({
                "slot_id": f"{group.lower()}/{case}", "group": group,
                "case_id": case, "status": "PENDING", "reason": None,
                "planned_bits": 32, "missing_bits": 32, "received_frames": frames,
                "media": {"status": "PENDING", "reason": None},
                "clock": {"status": "PENDING", "reason": None},
                "payload": {"status": "PENDING", "reason": None},
            })
    return rows


class Store:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.path = self.output / "residual_joint_state.json"
        self.data = read_state(self.path)
        if self.data.get("schema") != SCHEMA:
            raise ValueError("residual-joint schema changed")
        self.cases = validate_cases(self.data.get("fixed_denominator", {}).get("cases", CASES))
        if [row.get("slot_id") for row in self.data.get("rows", [])] != _slot_ids(self.cases):
            raise ValueError("fixed eight-row denominator changed")
        self.journal = StateJournal(self.path)
        self.data.setdefault("payload_read_calls", [])
        self.journal.seed_cache(self.data)
        retired = []
        for phase, status in list(self.data.get("phases", {}).items()):
            if status == "RUNNING":
                self.data["phases"][phase] = "INTERRUPTED"
                retired.append("phase/" + phase)
        for row in self.data["rows"]:
            for name in ("media", "clock", "payload"):
                if row[name].get("status") == "RUNNING":
                    row[name].update(status="INTERRUPTED", reason="previous process ended before unit completion")
                    retired.append(row["slot_id"] + "/" + name)
        for collection in (
            "quality_calls", "codec_calls", "edit_calls", "framewise_loads", "framewise_encode_clips",
            "framewise_physical_batches", "wan_loads", "wan_encode_clips", "payload_read_calls",
        ):
            for index, record in enumerate(self.data.get(collection, [])):
                if record.get("status") == "RUNNING":
                    record.update(status="INTERRUPTED", reason="previous process ended before completion", finished_at=time.time())
                    retired.append(f"{collection}/{index}")
        for unit_id, attempts in self.data.get("unit_attempts", {}).items():
            for index, attempt in enumerate(attempts):
                if attempt.get("status") == "RUNNING":
                    attempt.update(status="INTERRUPTED", reason="previous process ended before completion", finished_at=time.time())
                    retired.append(f"unit_attempts/{unit_id}/{index}")
        if retired:
            self.data.setdefault("resume_recoveries", []).append({"at": time.time(), "retired_running": retired})
            self.save()

    def save(self):
        self.journal.save(self.data)


def _validate_config(config):
    if config.get("codec") != CODEC:
        raise ValueError("codec must remain libx264/8fps/CRF18/yuv420p")
    if config.get("attacks") is not None:
        matches = [row for row in config["attacks"] if row.get("attack_id") == "delete_g0"]
        if len(matches) != 1 or matches[0] != delete_spec():
            raise ValueError("saved delete_g0 recipe changed")
    key = config.get("keys", {}).get("K0")
    bits = config.get("payload_bits")
    if not isinstance(key, str) or not key:
        raise ValueError("nonempty K0 is required")
    if not isinstance(bits, list) or len(bits) != 32 or any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise ValueError("payload_bits must contain 32 binary integers")
    for name in ("wan", "framewise"):
        model = config.get("models", {}).get(name)
        if not isinstance(model, dict) or not isinstance(model.get("local_snapshot_path"), str):
            raise ValueError(f"models.{name}.local_snapshot_path is required")
    if config["models"]["framewise"].get("batch_frames") != 8:
        raise ValueError("framewise batch_frames must remain 8")
    return key, list(bits)


def _candidate_path(source_run, case):
    return Path(source_run) / "run_state" / "rgb" / case / "residual_clock_pre.rgb8"


def initialize(source_run, output, config, *, cases=CASES):
    cases = validate_cases(cases)
    output = Path(output).resolve()
    state_path = output / "residual_joint_state.json"
    source_run = Path(source_run).resolve()
    source_state = source_run / "run_state" / "residual_clock_state.json"
    if not source_state.is_file():
        raise FileNotFoundError(source_state)
    source = read_json(source_state)
    if source.get("schema") != "paper-results-v1-residual-clock-v1":
        raise ValueError("source run is not a residual-clock-v1 result")
    cfg = read_json(config)
    key, payload_bits = _validate_config(cfg)
    if source.get("key") != key:
        raise ValueError("K0 differs from the saved residual run")
    source_config_candidates = [source_run / "effective_residual_clock_config.json"]
    if isinstance(source.get("effective_config"), str):
        source_config_candidates.append(Path(source["effective_config"]))
    source_effective_config = next((path for path in source_config_candidates if path.is_file()), None)
    if source_effective_config is None:
        raise FileNotFoundError("saved residual run effective config is unreadable")
    source_payload_bits = read_json(source_effective_config).get("payload_bits")
    if (not isinstance(source_payload_bits, list) or len(source_payload_bits) != 32
            or any(type(bit) is not int or bit not in (0, 1) for bit in source_payload_bits)):
        raise ValueError("saved residual run payload_bits are not 32 binary integers")
    if source_payload_bits != payload_bits:
        raise ValueError("payload_bits differ from the saved residual candidate method truth")
    if state_path.is_file():
        store = Store(output)
        if store.cases != cases:
            raise ValueError("initialized joint case roster differs")
        if store.data.get("key") != key:
            raise ValueError("K0 differs from the initialized joint run")
        return store
    expected_bytes = math.prod(RGB_SHAPE)
    rows = _new_rows(cases)
    sources = {}
    initialization_errors = []
    for case in cases:
        path = _candidate_path(source_run, case)
        failure = None
        if not path.is_file():
            failure = f"FileNotFoundError: {path}"
        elif path.stat().st_size != expected_bytes:
            failure = f"ValueError: candidate RGB8 byte count changed for {case}"
        sources[case] = {
            "path": str(path), "shape": list(RGB_SHAPE), "dtype": "uint8",
            "status": "FAILED" if failure else "AVAILABLE", "reason": failure,
        }
        if failure:
            clean = next(row for row in rows if row["group"] == "CLEAN_MP4" and row["case_id"] == case)
            attacked = next(row for row in rows if row["group"] == "DELETE_G0" and row["case_id"] == case)
            clean["media"].update(status="FAILED", reason=failure)
            clean.update(status="FAILED", reason=failure)
            blocked = "source candidate unavailable; clean MP4 cannot be produced"
            attacked["media"].update(status="BLOCKED", reason=blocked)
            attacked.update(status="BLOCKED", reason=blocked)
            initialization_errors.append({"at": time.time(), "phase": "initialize", "case_id": case, "reason": failure})
    state = {
        "schema": SCHEMA, "status": "INITIALIZED", "source_run": str(source_run),
        "source_state": str(source_state), "effective_config": str(Path(config).resolve()),
        "source_effective_config": str(source_effective_config.resolve()),
        "key": key, "sources": sources, "rows": rows,
        "fixed_denominator": {
            "cases": list(cases), "groups": list(GROUPS), "joint_rows": 8,
            "joint_bits": 256, "rows_per_group": 4, "bits_per_row": 32,
            "key_label": "K0", "receiver": "CENTERED_U",
        },
        "method": {
            "source": "four saved residual_clock_pre.rgb8 candidates; no writer reconstruction",
            "clean": "one fixed MP4 roundtrip per candidate",
            "attack": delete_spec(), "attack_input": "decoded RGB from the same clean MP4",
            "blind_chain": "received RGB -> framewise clock -> CENTERED_U -> decode_visible_span -> Wan payload",
            "truth_join": "after blind selection and payload read records are durable",
            "no_sync_control": "not included in this minimal joint-recovery validation",
            "thresholds": None,
        },
        "planned_budget": {
            "codec_encode_calls": 4, "codec_readback_calls": 4, "cpu_delete_edits": 4,
            "framewise_loads_uninterrupted": 1, "framewise_encode_clips": 8,
            "clean_framewise_physical_batches": 92, "delete_framewise_physical_batches": 84,
            "framewise_physical_batches": 176, "framewise_batch_frames": 8,
            "centered_u_selections": 8, "wan_loads_uninterrupted_upper_bound": 1,
            "wan_encode_clips_upper_bound": 8, "payload_reads_upper_bound": 8, "generation_or_dit": 0,
            "wan_decode": 0, "vjp": 0, "candidate_construction": 0,
            "peak_quality": "saved report event neighborhoods plus frames 179/180; CPU only and no gate",
        },
        "phases": {"quality": "PENDING", "media": "PENDING", "clock": "PENDING", "wan": "PENDING"},
        "unit_attempts": {}, "errors": initialization_errors, "quality_inspection": {"status": "PENDING"},
        "quality_calls": [], "codec_calls": [], "edit_calls": [],
        "framewise_loads": [], "framewise_encode_clips": [], "framewise_physical_batches": [],
        "wan_loads": [], "wan_encode_clips": [], "payload_read_calls": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(state_path, state)
    store = Store(output)
    report(store)
    return store


def _row(store, group, case):
    return next(row for row in store.data["rows"] if row["group"] == group and row["case_id"] == case)


def _shape(group):
    return RGB_SHAPE if group == "CLEAN_MP4" else DELETE_SHAPE


def _load_rgb_path(path, shape):
    import numpy as np
    import torch
    path = Path(path)
    expected = math.prod(shape)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected:
        raise ValueError(f"RGB8 byte count {path.stat().st_size} != {expected}")
    pixels = np.memmap(path, dtype=np.uint8, mode="r", shape=tuple(shape))
    return torch.from_numpy(np.asarray(pixels).copy())


def _atomic_rgb8(path, value, expected_shape):
    import numpy as np
    array = np.asarray(value)
    if array.dtype != np.uint8 or list(array.shape) != list(expected_shape):
        raise ValueError(f"RGB must be uint8 {expected_shape}")
    atomic_bytes(path, array.tobytes(order="C"))
    return {"path": str(Path(path).resolve()), "shape": list(expected_shape), "dtype": "uint8", "bytes": int(array.nbytes)}


def _source_checkpoint_state(store):
    source = read_json(store.data["source_state"])
    candidates = []
    if isinstance(source.get("checkpoint_state"), str):
        candidates.append(Path(source["checkpoint_state"]))
    if isinstance(source.get("checkpoint_run"), str):
        candidates.extend((Path(source["checkpoint_run"]) / "run_state/run_state.json",
                           Path(source["checkpoint_run"]) / "run_state.json"))
    for path in candidates:
        if path.is_file():
            return read_json(path), path
    raise FileNotFoundError("saved residual source does not expose a readable checkpoint state")


def _native_source_path(store, case):
    state, document = _source_checkpoint_state(store)
    matches = [row for row in state.get("artifacts", [])
               if row.get("artifact_id") == f"{case}/PAYLOAD_NATIVE/PRE"]
    if len(matches) != 1 or matches[0].get("status") != "AVAILABLE":
        raise FileNotFoundError(f"saved NATIVE_PRE artifact unavailable for {case}")
    path = Path(matches[0].get("path", ""))
    if not path.is_file():
        source = read_json(store.data["source_state"])
        root = Path(source.get("checkpoint_run", ""))
        raw = str(path).replace("\\", "/")
        marker = "/" + root.name + "/" if root.name else None
        if marker and marker in raw:
            path = root / raw.split(marker, 1)[1]
    if not path.is_file() or path.stat().st_size != math.prod(RGB_SHAPE):
        raise FileNotFoundError(f"saved NATIVE_PRE RGB8 is unreadable for {case}: {path}")
    return path, document


def _read_rgb_frames(path, indices):
    import numpy as np
    frame_bytes = math.prod(RGB_SHAPE[1:])
    frames = {}
    with Path(path).open("rb") as stream:
        for index in sorted(set(indices)):
            if type(index) is not int or index not in range(181):
                raise ValueError("quality frame index must be in 0..180")
            stream.seek(index * frame_bytes)
            raw = stream.read(frame_bytes)
            if len(raw) != frame_bytes:
                raise ValueError(f"short RGB8 frame read at {index}")
            frames[index] = np.frombuffer(raw, np.uint8).reshape(320, 512, 3).copy()
    return frames


def _quality_event_frames(quality):
    anchors = []
    peak = quality.get("peak_pixel") or {}
    temporal = quality.get("temporal_peak_pixel") or {}
    if type(peak.get("frame")) is int:
        anchors.append(("spatial_peak", peak["frame"]))
    if type(temporal.get("from_frame")) is int:
        anchors.append(("temporal_peak_from", temporal["from_frame"]))
    if type(temporal.get("to_frame")) is int:
        anchors.append(("temporal_peak_to", temporal["to_frame"]))
    anchors.extend((("tail_adjacent", 179), ("tail", 180)))
    events, seen = [], set()
    for label, anchor in anchors:
        for frame in range(max(0, anchor - 1), min(180, anchor + 1) + 1):
            key = (label, frame)
            if key not in seen:
                events.append({"event": label, "anchor": anchor, "frame": frame})
                seen.add(key)
    return events


def run_quality(store):
    """Recompute bounded peak neighborhoods and render a non-gating contact sheet."""
    if store.data.get("quality_inspection", {}).get("status") in ("COMPLETE", "FAILED"):
        return store
    call = {"status": "RUNNING", "started_at": time.time()}
    store.data["quality_calls"].append(call)
    store.data["quality_inspection"] = {"status": "RUNNING"}
    store.data["phases"]["quality"] = "RUNNING"
    store.save()
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        rows = []
        panels = []
        checkpoint_documents = set()
        for case in store.cases:
            quality_path = Path(store.data["source_run"]) / "run_state" / "quality" / case / "residual_vs_native.json"
            if not quality_path.is_file():
                raise FileNotFoundError(quality_path)
            quality = read_json(quality_path)
            events = _quality_event_frames(quality)
            frame_indices = [event["frame"] for event in events]
            candidate = _read_rgb_frames(store.data["sources"][case]["path"], frame_indices)
            native_path, checkpoint_document = _native_source_path(store, case)
            checkpoint_documents.add(str(checkpoint_document))
            native = _read_rgb_frames(native_path, frame_indices)
            anchor_labels = {(event["event"], event["anchor"]) for event in events}
            for event in events:
                frame = event["frame"]
                delta = candidate[frame].astype(np.int16) - native[frame].astype(np.int16)
                absolute = np.abs(delta)
                rows.append({
                    "case_id": case, **event, "rmse_rgb8": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
                    "max_abs_rgb8": int(absolute.max()), "changed_elements": int((absolute > 0).sum()),
                    "changed_pixels": int(np.any(absolute > 0, axis=2).sum()),
                    **{f"fraction_pixels_max_channel_abs_gt_{threshold}": float(np.mean(absolute.max(axis=2) > threshold))
                       for threshold in (8, 16, 32, 64)},
                    "candidate_boundary_zero_elements": int((candidate[frame] == 0).sum()),
                    "candidate_boundary_255_elements": int((candidate[frame] == 255).sum()),
                    "new_boundary_value_channel_fraction": float(np.mean(
                        ((candidate[frame] == 0) | (candidate[frame] == 255))
                        & ~((native[frame] == 0) | (native[frame] == 255))
                    )),
                })
            for label, anchor in sorted(anchor_labels, key=lambda item: (item[1], item[0])):
                left = Image.fromarray(native[anchor], "RGB").resize((256, 160))
                middle = Image.fromarray(candidate[anchor], "RGB").resize((256, 160))
                amplified = np.clip(np.abs(candidate[anchor].astype(np.int16) - native[anchor].astype(np.int16)) * 4, 0, 255).astype(np.uint8)
                right = Image.fromarray(amplified, "RGB").resize((256, 160))
                strip = Image.new("RGB", (768, 184), "white")
                strip.paste(left, (0, 24)); strip.paste(middle, (256, 24)); strip.paste(right, (512, 24))
                ImageDraw.Draw(strip).text((4, 4), f"{case} {label} frame {anchor} | native | candidate | 4x abs diff", fill="black")
                panels.append(strip)
        contact = Image.new("RGB", (768, 28 + 184 * len(panels)), "white")
        ImageDraw.Draw(contact).text((4, 6), "Saved residual peak inspection; display only, no quality gate", fill="black")
        for index, panel in enumerate(panels):
            contact.paste(panel, (0, 28 + 184 * index))
        output = store.output / "quality"
        output.mkdir(parents=True, exist_ok=True)
        contact_path = output / "peak_quality_contact_sheet.png"
        contact.save(contact_path)
        report_path = output / "peak_quality_report.json"
        report_value = {
            "status": "COMPLETE", "cases": list(store.cases), "rows": rows,
            "source_reports": "saved residual_vs_native quality JSON for each case",
            "checkpoint_documents": sorted(checkpoint_documents),
            "contact_sheet": str(contact_path), "quality_gate": None,
            "new_boundary_value_channel_fraction_definition": (
                "fraction of RGB channels newly equal to 0 or 255 relative to NATIVE_PRE; "
                "not an exact clipping measurement"
            ),
            "interpretation": "bounded CPU inspection of saved RGB8 event neighborhoods; not a perceptual score or PASS",
        }
        atomic_json(report_path, report_value)
        with (output / "peak_quality_rows.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
        store.data["quality_inspection"] = {
            "status": "COMPLETE", "report": str(report_path), "contact_sheet": str(contact_path),
            "rows": len(rows), "quality_gate": None,
        }
        call.update(status="COMPLETE", finished_at=time.time(), rows=len(rows))
        store.data["phases"]["quality"] = "COMPLETE"
    except BaseException as exc:
        terminal = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"
        store.data["quality_inspection"] = {"status": terminal, "reason": reason(exc), "quality_gate": None}
        call.update(status=terminal, reason=reason(exc), finished_at=time.time())
        store.data["phases"]["quality"] = terminal
        if not isinstance(exc, Exception):
            raise
    finally:
        store.save(); report(store)
    return store


def run_media(store, *, roundtrip=None, apply_edit=None, load_rgb=None):
    from experiments.paper_results_v1.attack_matrix import apply_edit_rgb8
    from runtime.wan.variable_rgb_media import roundtrip as fixed_roundtrip
    roundtrip = roundtrip or fixed_roundtrip
    apply_edit = apply_edit or apply_edit_rgb8
    load_rgb = load_rgb or _load_rgb_path
    store.data["phases"]["media"] = "RUNNING"
    store.save()
    with store.journal.periodic():
        for case in store.cases:
            clean = _row(store, "CLEAN_MP4", case)
            if clean["media"]["status"] not in ("AVAILABLE", "FAILED"):
                call = {"case_id": case, "status": "RUNNING", "started_at": time.time()}
                store.data["codec_calls"].append(call)
                clean["media"].update(status="RUNNING", reason=None)
                store.save()
                try:
                    source = load_rgb(store.data["sources"][case]["path"], RGB_SHAPE)
                    attempt_number = 1 + sum(item.get("case_id") == case for item in store.data["codec_calls"][:-1])
                    work = store.output / "media" / case / f"attempt-{attempt_number}"
                    published = store.output / "media" / case / f"published-attempt-{attempt_number}"
                    with unit_attempt(store, f"media/codec/{case}", operation="fixed_mp4_roundtrip"):
                        received, receipt = roundtrip(source, work, media_output=published)
                    received_path = work / "received.rgb8"
                    if list(received.shape) != RGB_SHAPE or not received_path.is_file():
                        raise ValueError("codec roundtrip did not return uint8 [181,320,512,3]")
                    clean["media"].update(status="AVAILABLE", reason=None, received_path=str(received_path),
                                           mp4_path=receipt["mp4"]["path"], receipt=receipt)
                    clean["status"] = "MEDIA_READY"
                    call.update(status="COMPLETE", finished_at=time.time(), mp4_path=receipt["mp4"]["path"])
                except BaseException as exc:
                    terminal = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"
                    clean["media"].update(status=terminal, reason=reason(exc))
                    clean.update(status=terminal, reason=reason(exc))
                    call.update(status=terminal, reason=reason(exc), finished_at=time.time())
                    store.save()
                    if not isinstance(exc, Exception):
                        raise
                store.save()
            attacked = _row(store, "DELETE_G0", case)
            if attacked["media"]["status"] in ("AVAILABLE", "FAILED"):
                continue
            if clean["media"]["status"] != "AVAILABLE":
                attacked["media"].update(status="BLOCKED", reason="clean MP4 received RGB unavailable")
                attacked.update(status="BLOCKED", reason=attacked["media"]["reason"])
                store.save()
                continue
            call = {"case_id": case, "status": "RUNNING", "started_at": time.time()}
            store.data["edit_calls"].append(call)
            attacked["media"].update(status="RUNNING", reason=None)
            store.save()
            try:
                clean_rgb = load_rgb(clean["media"]["received_path"], RGB_SHAPE)
                with unit_attempt(store, f"media/delete_g0/{case}", operation="fixed_delete_g0"):
                    edited, receipt = apply_edit(clean_rgb, delete_spec())
                    destination = store.output / "media" / case / "delete_g0_received.rgb8"
                    saved = _atomic_rgb8(destination, edited, DELETE_SHAPE)
                attacked["media"].update(status="AVAILABLE", reason=None, received_path=saved["path"],
                                          source_mp4=clean["media"]["mp4_path"], edit_receipt=receipt)
                attacked["status"] = "MEDIA_READY"
                call.update(status="COMPLETE", finished_at=time.time())
            except BaseException as exc:
                terminal = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"
                attacked["media"].update(status=terminal, reason=reason(exc))
                attacked.update(status=terminal, reason=reason(exc))
                call.update(status=terminal, reason=reason(exc), finished_at=time.time())
                store.save()
                if not isinstance(exc, Exception):
                    raise
            store.save()
    store.data["phases"]["media"] = "COMPLETE"
    store.save()
    report(store)
    return store


def _clock_latent_path(store, row):
    return store.output / "latents" / "framewise" / row["group"].lower() / (row["case_id"] + ".npy")


def _atomic_npy(path, value):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_npy(path, frames):
    import numpy as np
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.shape != (frames, 4, 40, 64) or not bool(np.isfinite(value).all()):
        raise ValueError("framewise latent must be finite float32 [N,4,40,64]")
    return value


def _default_framewise_loader(store):
    from experiments.paper_results_v1.real_backends import load_local_framewise_backend
    config = read_json(store.data["effective_config"])
    return load_local_framewise_backend(config["models"]["framewise"])


def _select_clock(latent, key):
    from main.tube_state import video_trajectory_temporal_edit_receiver_v1 as method
    from main.tube_state.receiver_controls_v1 import receive_scores
    evidence = method.score_framewise(latent, key, method.PUBLIC)
    selection = receive_scores(evidence["signed_projection"], evidence["rho"], "CENTERED_U")
    return evidence, selection


def run_clock(store, *, load_backend=None, load_rgb=None, select_clock=None):
    load_backend = load_backend or (lambda: _default_framewise_loader(store))
    load_rgb = load_rgb or _load_rgb_path
    select_clock = select_clock or _select_clock
    candidates = [row for row in store.data["rows"] if row["media"]["status"] == "AVAILABLE"
                  and row["clock"]["status"] not in ("CLOCK_READY", "UNRESOLVED", "UNSUPPORTED", "FAILED")]
    store.data["phases"]["clock"] = "RUNNING"
    store.save()
    backend = None
    with store.journal.periodic():
        try:
            for row in candidates:
                frames = row["received_frames"]
                latent_path = _clock_latent_path(store, row)
                try:
                    if latent_path.is_file():
                        latent = _load_npy(latent_path, frames)
                    else:
                        if backend is None:
                            load_record = {"status": "RUNNING", "started_at": time.time()}
                            store.data["framewise_loads"].append(load_record); store.save()
                            try:
                                backend = load_backend()
                            except BaseException as exc:
                                load_record.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED",
                                                   reason=reason(exc), finished_at=time.time())
                                store.save(); raise
                            load_record.update(status="COMPLETE", finished_at=time.time()); store.save()
                        rgb = load_rgb(row["media"]["received_path"], _shape(row["group"]))
                        row["clock"].update(status="RUNNING", reason=None)
                        call = {"slot_id": row["slot_id"], "frames": frames, "status": "RUNNING", "started_at": time.time()}
                        store.data["framewise_encode_clips"].append(call); store.save()
                        def batch_observer(event, start, stop, error):
                            if event == "STARTED":
                                record = {"slot_id": row["slot_id"], "start": start, "stop": stop,
                                          "status": "RUNNING", "started_at": time.time()}
                                store.data["framewise_physical_batches"].append(record)
                            else:
                                record = store.data["framewise_physical_batches"][-1]
                                record.update(status="COMPLETE" if event == "RETURNED" else (
                                    "FAILED" if isinstance(error, Exception) else "INTERRUPTED"), finished_at=time.time())
                                if error is not None:
                                    record["reason"] = reason(error)
                            store.save()
                        try:
                            with unit_attempt(store, "clock/" + row["slot_id"], operation="framewise_clock_encode"):
                                latent = backend.encode(rgb, batch_observer=batch_observer)
                                latent = _load_npy_from_value(latent, frames)
                                _atomic_npy(latent_path, latent)
                            call.update(status="COMPLETE", finished_at=time.time())
                        except BaseException as exc:
                            call.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED",
                                        reason=reason(exc), finished_at=time.time())
                            raise
                        finally:
                            store.save()
                    evidence, selection = select_clock(latent, store.data["key"])
                    folder = store.output / "clock" / row["group"].lower() / row["case_id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    import numpy as np
                    scores = folder / "scores.npz"
                    with scores.with_suffix(".npz.tmp").open("wb") as stream:
                        np.savez_compressed(stream, signed_projection=evidence["signed_projection"], rho=evidence["rho"])
                        stream.flush(); os.fsync(stream.fileno())
                    os.replace(scores.with_suffix(".npz.tmp"), scores)
                    blind = {
                        "schema": SCHEMA, "slot_id": row["slot_id"], "receiver": "CENTERED_U",
                        "received_frames": frames, "estimate": selection["estimate"],
                        "operation": selection["operation"], "truth_inputs": False,
                        "message_inputs": False, "attack_inputs": False,
                        "score_sidecar": str(scores),
                    }
                    blind_path = folder / "blind_selection.json"
                    atomic_json(blind_path, blind)
                    operation = selection["operation"]
                    status = "CLOCK_READY" if operation is not None and operation.get("status") == "SUPPORTED" else (
                        "UNSUPPORTED" if operation is not None else "UNRESOLVED")
                    row["clock"].update(status=status, reason=(selection.get("reason") if status != "CLOCK_READY" else None),
                                        selector_record=str(blind_path), latent_path=str(latent_path),
                                        selector_status=selection["estimate"]["status"],
                                        score_gap=selection["estimate"].get("score_gap"),
                                        output_frames=operation.get("output_frames") if operation else None,
                                        wan_support=operation.get("wan_support") if operation else None)
                    row["status"] = status
                except BaseException as exc:
                    terminal = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"
                    row["clock"].update(status=terminal, reason=reason(exc))
                    row.update(status=terminal, reason=reason(exc))
                    store.save()
                    if not isinstance(exc, Exception):
                        raise
                store.save()
        finally:
            if backend is not None:
                backend.close()
    store.data["phases"]["clock"] = "COMPLETE"
    store.save(); report(store)
    return store


def _load_npy_from_value(value, frames):
    import numpy as np
    result = np.asarray(value)
    if result.dtype != np.float32:
        result = result.astype(np.float32)
    if result.shape != (frames, 4, 40, 64) or not bool(np.isfinite(result).all()):
        raise ValueError("framewise encoder returned invalid latent")
    return np.ascontiguousarray(result)


def _wan_latent_path(store, row):
    return store.output / "latents" / "wan" / row["group"].lower() / (row["case_id"] + ".pt")


def _atomic_torch(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_torch(path, output_frames):
    import torch
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    times = (output_frames - 1) // 4 + 1
    if tuple(value.shape) != (1, 16, times, 40, 64) or not bool(torch.isfinite(value).all()):
        raise ValueError("Wan latent shape/nonfinite mismatch")
    return value.float().contiguous()


def _default_wan_loader(store):
    from runtime.wan.generation import load_frozen_vae
    config = read_json(store.data["effective_config"])
    return load_frozen_vae({"model": {"id": config["models"]["wan"]["local_snapshot_path"]}},
                           device=config["models"]["wan"].get("device", "cuda"))


def _default_wan_encode(vae, rgb, observer):
    from runtime.wan.vae import reencode_rgb24_readback
    return reencode_rgb24_readback(vae, rgb.float().div(255.0), encode_observer=observer)


def _default_payload_read(latent, key, output_frames, source_map):
    from runtime.wan.video_trajectory_temporal_edit_receiver_v1 import read_payload_general
    return read_payload_general(latent, key, output_frames, source_map)


def _save_payload(store, row, result, attempt_number):
    import numpy as np
    root = store.output / "payload" / row["group"].lower() / row["case_id"]
    root.mkdir(parents=True, exist_ok=True)
    signed, zeros = result.pop("signed_votes"), result.pop("zero_mask")
    sidecar = root / f"votes-attempt-{attempt_number}.npz"
    with sidecar.with_suffix(".npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, signed_votes=signed, zero_mask=zeros)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(sidecar.with_suffix(".npz.tmp"), sidecar)
    record = root / f"read-attempt-{attempt_number}.json"
    atomic_json(record, {**result, "vote_sidecar": {"path": str(sidecar), "arrays": ["signed_votes", "zero_mask"]}})
    row["payload"].update(status="READ", reason=None, read_record=str(record),
                          decoded_bits=result["decoded_bits"], votes=result["votes"], R=result["R"],
                          output_frames=result["output_frames"], original_reader_match=result["original_reader_match"])
    row.update(status="READ", reason=None, missing_bits=0)


def run_wan(store, *, load_vae=None, encode=None, payload_read=None, load_rgb=None):
    from runtime.wan.video_trajectory_temporal_edit_receiver_v1 import operate_map
    load_vae = load_vae or (lambda: _default_wan_loader(store))
    encode = encode or _default_wan_encode
    payload_read = payload_read or _default_payload_read
    load_rgb = load_rgb or _load_rgb_path
    candidates = [row for row in store.data["rows"] if row["clock"]["status"] == "CLOCK_READY"
                  and row["payload"]["status"] not in ("READ", "FAILED")]
    store.data["phases"]["wan"] = "RUNNING"
    store.save()
    vae = None
    with store.journal.periodic():
        try:
            for row in candidates:
                try:
                    blind = read_json(row["clock"]["selector_record"])
                    operation = blind.get("operation")
                    if not isinstance(operation, dict) or operation.get("status") != "SUPPORTED":
                        raise ValueError("durable blind selection lacks a supported operation")
                    output_frames = operation["output_frames"]
                    latent_path = _wan_latent_path(store, row)
                    if latent_path.is_file():
                        latent = _load_torch(latent_path, output_frames)
                    else:
                        if vae is None:
                            load_record = {"status": "RUNNING", "started_at": time.time()}
                            store.data["wan_loads"].append(load_record); store.save()
                            try:
                                vae = load_vae()
                            except BaseException as exc:
                                load_record.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED",
                                                   reason=reason(exc), finished_at=time.time())
                                store.save(); raise
                            load_record.update(status="COMPLETE", finished_at=time.time()); store.save()
                        received = load_rgb(row["media"]["received_path"], _shape(row["group"]))
                        corrected = operate_map(received, operation["received_index_map"])
                        call = {"slot_id": row["slot_id"], "input_frames": row["received_frames"],
                                "output_frames": output_frames, "status": "RUNNING", "started_at": time.time()}
                        store.data["wan_encode_clips"].append(call)
                        row["payload"].update(status="RUNNING", reason=None); store.save()
                        def observer(event, error):
                            if event == "RETURNED":
                                call.update(status="COMPLETE", finished_at=time.time())
                            elif event == "FAILED":
                                call.update(status="FAILED" if isinstance(error, Exception) else "INTERRUPTED",
                                            reason=reason(error), finished_at=time.time())
                            store.save()
                        try:
                            with unit_attempt(store, "wan/" + row["slot_id"], operation="wan_encode_blind_corrected_rgb"):
                                latent = encode(vae, corrected, observer).detach().cpu().float().contiguous()
                                times = (output_frames - 1) // 4 + 1
                                if (tuple(latent.shape) != (1, 16, times, 40, 64)
                                        or not bool(latent.isfinite().all())):
                                    raise ValueError("Wan encoder returned a shape/nonfinite mismatch")
                                _atomic_torch(latent_path, latent)
                            if call["status"] == "RUNNING":
                                call.update(status="COMPLETE", finished_at=time.time())
                        except BaseException as exc:
                            if call["status"] == "RUNNING":
                                call.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED",
                                            reason=reason(exc), finished_at=time.time())
                            raise
                        finally:
                            store.save()
                    read_call = {"slot_id": row["slot_id"], "status": "RUNNING", "started_at": time.time()}
                    store.data["payload_read_calls"].append(read_call)
                    attempt_number = sum(item.get("slot_id") == row["slot_id"]
                                         for item in store.data["payload_read_calls"])
                    store.save()
                    try:
                        with unit_attempt(store, "payload-read/" + row["slot_id"], operation="wan_payload_read"):
                            result = payload_read(latent, store.data["key"], output_frames,
                                                  operation["source_coordinate_map"])
                            bits = result.get("decoded_bits")
                            if (result.get("status") != "READ" or not isinstance(bits, list)
                                    or len(bits) != 32
                                    or any(type(bit) is not int or bit not in (0, 1) for bit in bits)):
                                raise ValueError("payload reader did not return 32 binary integer bits")
                            _save_payload(store, row, result, attempt_number)
                        read_call.update(status="COMPLETE", finished_at=time.time(),
                                         read_record=row["payload"]["read_record"])
                    except BaseException as exc:
                        read_call.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED",
                                         reason=reason(exc), finished_at=time.time())
                        raise
                    finally:
                        store.save()
                except BaseException as exc:
                    terminal = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"
                    row["payload"].update(status=terminal, reason=reason(exc))
                    row.update(status=terminal, reason=reason(exc), missing_bits=32)
                    store.save()
                    if not isinstance(exc, Exception):
                        raise
                store.save()
        finally:
            vae = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    store.data["phases"]["wan"] = "COMPLETE"
    store.save(); report(store)
    return store


def _posthoc_path_metrics(row):
    from experiments.paper_results_v1.attack_matrix import expand_weighted_truth, path_error_rows
    if not row["clock"].get("selector_record"):
        return {"status": "MISSING", "reason": "no durable blind selector record"}
    try:
        selected = read_json(row["clock"]["selector_record"])
        path = selected["estimate"].get("path")
        if path is None:
            return {"status": selected["estimate"]["status"], "reason": selected["estimate"].get("reason")}
        if row["group"] == "CLEAN_MP4":
            truth = [[{"source_index": index, "weight": 1.0}] for index in range(181)]
        else:
            truth = expand_weighted_truth(delete_spec())
        value = path_error_rows(path, truth)
        value["truth_join"] = "POSTHOC_ONLY_AFTER_BLIND_SELECTION"
        return value
    except Exception as exc:
        return {"status": "MISSING", "reason": reason(exc)}


def _count(records):
    return dict(sorted(Counter(item.get("status", "UNKNOWN") for item in records).items()))


def pending_model_units(store):
    result = []
    for row in store.data["rows"]:
        clock_may_resume = (
            row["media"]["status"] == "AVAILABLE"
            and row["clock"]["status"] in ("PENDING", "RUNNING", "INTERRUPTED")
        )
        if clock_may_resume:
            if not _clock_latent_path(store, row).is_file():
                result.append("framewise/" + row["slot_id"])
            if (row["payload"]["status"] in ("PENDING", "RUNNING", "INTERRUPTED")
                    and not _wan_latent_path(store, row).is_file()):
                result.append("wan-after-clock/" + row["slot_id"])
        if row["clock"]["status"] == "CLOCK_READY" and row["payload"]["status"] in ("PENDING", "RUNNING", "INTERRUPTED"):
            if not _wan_latent_path(store, row).is_file():
                result.append("wan/" + row["slot_id"])
    return result


def report(store):
    truth_ready = store.data.get("phases", {}).get("wan") == "COMPLETE"
    truth = None
    if truth_ready:
        try:
            truth = read_json(store.data["effective_config"])["payload_bits"]
            if len(truth) != 32 or any(type(bit) is not int or bit not in (0, 1) for bit in truth):
                raise ValueError("invalid payload truth")
        except Exception as exc:
            store.data["report_reason"] = reason(exc)
    rows = []
    for row in store.data["rows"]:
        payload = row["payload"]
        status = row["status"]
        decoded = payload.get("decoded_bits")
        bit_errors = None
        payload_exact = None
        clock_exact = None
        joint_exact = None
        if truth_ready and payload.get("status") == "READ" and truth is not None:
            bit_errors = sum(left != right for left, right in zip(decoded, truth))
            payload_exact = bit_errors == 0
            status = "EVALUATED"
        if truth_ready:
            path_metrics = _posthoc_path_metrics(row)
            if path_metrics.get("status") == "EVALUATED_POSTHOC":
                clock_exact = (
                    path_metrics.get("frames") == row["received_frames"]
                    and path_metrics.get("minimum_support_absolute_error_sum") == 0
                    and path_metrics.get("weighted_absolute_error_sum") == 0
                )
            joint_exact = payload_exact is True and clock_exact is True
        else:
            path_metrics = {"status": "NOT_JOINED_BEFORE_WAN_TERMINAL"}
        rows.append({
            "slot_id": row["slot_id"], "group": row["group"], "case_id": row["case_id"],
            "status": status, "reason": row.get("reason") or payload.get("reason") or row["clock"].get("reason"),
            "received_frames": row["received_frames"], "selector_status": row["clock"].get("selector_status"),
            "score_gap": row["clock"].get("score_gap"), "output_frames": row["clock"].get("output_frames"),
            "wan_support": row["clock"].get("wan_support"), "R": payload.get("R"),
            "decoded_bits": decoded, "bit_errors": bit_errors,
            "payload_exact": payload_exact, "clock_exact": clock_exact, "joint_exact": joint_exact,
            "exact_recovery": joint_exact,
            "missing_bits": 0 if status == "EVALUATED" else 32,
            "clock_record": row["clock"].get("selector_record"),
            "payload_record": payload.get("read_record"), "path_metrics": path_metrics,
            "truth_join": (
                "POSTHOC_COMPLETE_AFTER_WAN_TERMINAL"
                if truth_ready else "NOT_JOINED_BEFORE_WAN_TERMINAL"
            ),
        })
    groups = {}
    for group in GROUPS:
        selected = [row for row in rows if row["group"] == group]
        evaluated = [row for row in selected if row["status"] == "EVALUATED"]
        groups[group] = {
            "fixed_rows": 4, "fixed_bits": 128, "status_counts": _count(selected),
            "evaluated_rows": len(evaluated), "missing_rows": 4 - len(evaluated),
            "bit_errors": sum(row["bit_errors"] for row in evaluated),
            "payload_exact_recoveries": sum(row["payload_exact"] is True for row in selected),
            "clock_exact_recoveries": sum(row["clock_exact"] is True for row in selected),
            "joint_exact_recoveries": sum(row["joint_exact"] is True for row in selected),
            "exact_recoveries": sum(row["joint_exact"] is True for row in selected),
            "fixed_denominator_joint_exact_rate": (
                sum(row["joint_exact"] is True for row in selected) / 4 if truth_ready else None
            ),
            "scorable_ber": (sum(row["bit_errors"] for row in evaluated) / (32 * len(evaluated))) if evaluated else None,
        }
    summary = {
        "schema": SCHEMA,
        "status": (
            "IN_PROGRESS" if not truth_ready else
            ("COMPLETE" if all(row["status"] == "EVALUATED" for row in rows)
             else "COMPLETE_WITH_RETAINED_FAILURES")
        ),
        "fixed_denominator": store.data["fixed_denominator"], "groups": groups,
        "quality_inspection": store.data.get("quality_inspection"),
        "truth_is_posthoc_only": True,
        "truth_join_status": (
            "POSTHOC_COMPLETE_AFTER_WAN_TERMINAL"
            if truth_ready else "NOT_JOINED_BEFORE_WAN_TERMINAL"
        ),
        "claim_limit": "fixed four saved candidates under the adopted clean MP4 and delete_g0 conditions; no threshold, FPR, generalization, or scientific PASS",
        "actual_calls": {
            "peak_quality": _count(store.data["quality_calls"]),
            "codec_encode_and_readback": _count(store.data["codec_calls"]),
            "cpu_delete_edits": _count(store.data["edit_calls"]),
            "framewise_loads": _count(store.data["framewise_loads"]),
            "framewise_encode_clips": _count(store.data["framewise_encode_clips"]),
            "framewise_physical_batches": _count(store.data["framewise_physical_batches"]),
            "wan_loads": _count(store.data["wan_loads"]),
            "wan_encode_clips": _count(store.data["wan_encode_clips"]),
            "payload_reads": _count(store.data.get("payload_read_calls", [])),
            "generation_or_dit": 0, "wan_decode": 0, "vjp": 0, "candidate_construction": 0,
        },
        "pending_model_units": pending_model_units(store),
    }
    store.data["status"] = summary["status"]
    store.save()
    atomic_json(store.output / "summary.json", summary)
    atomic_json(store.output / "evaluation_report.json", {"summary": summary, "rows": rows})
    for group, filename in (("CLEAN_MP4", "clean_mp4_joint_rows.csv"), ("DELETE_G0", "delete_g0_joint_rows.csv")):
        compact = [{key: value for key, value in row.items() if key not in ("decoded_bits", "path_metrics")}
                   for row in rows if row["group"] == group]
        fields = list(dict.fromkeys(key for row in compact for key in row))
        with (store.output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(compact)
    return summary


def record_external_failure(store, message):
    store.data["errors"].append({"at": time.time(), "phase": "external", "reason": str(message)})
    for phase, status in list(store.data["phases"].items()):
        if status == "RUNNING":
            store.data["phases"][phase] = "INTERRUPTED"
    for row in store.data["rows"]:
        for name in ("media", "clock", "payload"):
            if row[name].get("status") == "RUNNING":
                row[name].update(status="INTERRUPTED", reason=str(message))
    store.save()
    return report(store)
